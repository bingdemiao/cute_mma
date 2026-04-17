#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <random>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error("cuBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

static cudaDataType_t torch_to_cublas_dtype(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf:     return CUDA_R_16F;
        case at::kBFloat16: return CUDA_R_16BF;
        case at::kFloat:    return CUDA_R_32F;
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuBLAS: ", dtype);
            return CUDA_R_16F; // unreachable
    }
}

template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float(half v) { return __half2float(v); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ half from_float(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }

// Deterministic hash-based uniform[0,1) for dropout masks. Same (seed, idx)
// produces the same value across forward and backward, so we can replay the
// mask without materializing a rand buffer.
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float uniform_from_hash(uint64_t seed, int64_t idx) {
    uint64_t h = splitmix64(seed + 0xdeadbeefULL) ^ splitmix64((uint64_t)idx);
    h = splitmix64(h);
    // Top 24 bits → [0, 1)
    return (float)((h >> 40) & 0xFFFFFF) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// cuBLAS GemmEx wrappers (f32 compute, f32 alpha/beta)
// ---------------------------------------------------------------------------

static void gemm_ex(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k,
    const void* A, cudaDataType_t Atype, int64_t lda,
    const void* B, cudaDataType_t Btype, int64_t ldb,
    void* C, cudaDataType_t Ctype, int64_t ldc,
    float alpha = 1.0f, float beta = 0.0f)
{
    GEMM_CHECK_CUBLAS(cublasGemmEx(
        handle, transa, transb,
        (int)m, (int)n, (int)k,
        &alpha,
        A, Atype, (int)lda,
        B, Btype, (int)ldb,
        &beta,
        C, Ctype, (int)ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void gemm_strided_batched_ex(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k,
    const void* A, cudaDataType_t Atype, int64_t lda, int64_t strideA,
    const void* B, cudaDataType_t Btype, int64_t ldb, int64_t strideB,
    void* C, cudaDataType_t Ctype, int64_t ldc, int64_t strideC,
    int64_t batchCount,
    float alpha = 1.0f, float beta = 0.0f)
{
    GEMM_CHECK_CUBLAS(cublasGemmStridedBatchedEx(
        handle, transa, transb,
        (int)m, (int)n, (int)k,
        &alpha,
        A, Atype, (int)lda, (long long)strideA,
        B, Btype, (int)ldb, (long long)strideB,
        &beta,
        C, Ctype, (int)ldc, (long long)strideC,
        (int)batchCount,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---------------------------------------------------------------------------
// Templated element-wise kernels
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Input shuffle helpers — butterfly shuffle via __shfl_xor_sync
// ---------------------------------------------------------------------------

// Butterfly shuffle: one warp = one chunk of 32 segments (64 fp16 elements).
// Each thread handles one segment (2 elements packed as uint32).
// Applies n_rounds rounds of conditional XOR-swap, each using one
// __shfl_xor_sync call.  Deltas are power-of-2 >= block_size (e.g. [8, 16]).
//
// Pair indexing: for XOR delta d = 2^r, the mask bit for thread i is at
// position pair_idx = (i & ((1<<r)-1)) | ((i >> (r+1)) << r), i.e. the
// lane index with bit r removed.  Both partners (i and i^d) map to the
// same pair_idx, so the swap decision is always consistent.

// Forward: load from A, shuffle, store to A_perm.
template<typename T>
__global__ void shuffle_forward_kernel(
    const T* __restrict__ A,           // (M, K) row-major
    T* __restrict__ A_perm,            // (M, K) permuted layout
    const int64_t* __restrict__ masks, // per-group: (n_chunks, n_rounds)
    int64_t M, int64_t K,
    int64_t n_chunks,
    int64_t n_rounds)                  // number of shuffle rounds
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int64_t row   = warp_id / n_chunks;
    int64_t chunk = warp_id % n_chunks;
    if (row >= M) return;

    // Each chunk has 32 segments of 2 elements = 64 elements
    int64_t col = chunk * 64 + lane * 2;
    uint32_t val = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(A) + (row * K + col) * sizeof(T));

    // Apply shuffle rounds.  Deltas are stored implicitly as the r-th
    // power of 2 >= block_size.  For reconn_sz=16, seg_sz=2, block_size=8:
    //   round 0 → delta=8  (log2=3)
    //   round 1 → delta=16 (log2=4)
    // These are hard-coded for 2-round / 4-block configuration.
    {
        // Round 0: delta = 8, r = 3
        int64_t m0 = masks[chunk * n_rounds + 0];
        uint32_t p0 = __shfl_xor_sync(0xFFFFFFFF, val, 8);
        int pi0 = (lane & 7) | ((lane >> 4) << 3);  // remove bit 3
        val = ((m0 >> pi0) & 1) ? p0 : val;
    }
    {
        // Round 1: delta = 16, r = 4
        int64_t m1 = masks[chunk * n_rounds + 1];
        uint32_t p1 = __shfl_xor_sync(0xFFFFFFFF, val, 16);
        int pi1 = (lane & 15);  // remove bit 4 (bit 4 is MSB for 5-bit lane)
        val = ((m1 >> pi1) & 1) ? p1 : val;
    }

    *reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(A_perm) + (row * K + col) * sizeof(T)) = val;
}

// Inverse shuffle + accumulate: applies the same masks in REVERSE order
// (delta=16 first, then delta=8), then adds the result to dA.
template<typename T>
__global__ void shuffle_inverse_add_kernel(
    T* __restrict__ dA,                 // (M, K) natural layout, accumulated
    const T* __restrict__ dA_perm,      // (M, K) permuted layout
    const int64_t* __restrict__ masks,  // per-group: (n_chunks, n_rounds)
    int64_t M, int64_t K,
    int64_t n_chunks,
    int64_t n_rounds)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int64_t row   = warp_id / n_chunks;
    int64_t chunk = warp_id % n_chunks;
    if (row >= M) return;

    int64_t col = chunk * 64 + lane * 2;
    uint32_t val = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(dA_perm) + (row * K + col) * sizeof(T));

    // Reverse order: round 1 first (delta=16), then round 0 (delta=8)
    {
        int64_t m1 = masks[chunk * n_rounds + 1];
        uint32_t p1 = __shfl_xor_sync(0xFFFFFFFF, val, 16);
        int pi1 = (lane & 15);
        val = ((m1 >> pi1) & 1) ? p1 : val;
    }
    {
        int64_t m0 = masks[chunk * n_rounds + 0];
        uint32_t p0 = __shfl_xor_sync(0xFFFFFFFF, val, 8);
        int pi0 = (lane & 7) | ((lane >> 4) << 3);
        val = ((m0 >> pi0) & 1) ? p0 : val;
    }

    // Unpack the two T values from uint32 and accumulate into dA via float
    const T* unpacked = reinterpret_cast<const T*>(&val);
    int64_t base = row * K + col;
    float prev0 = to_float(dA[base]);
    float prev1 = to_float(dA[base + 1]);
    dA[base]     = from_float<T>(prev0 + to_float(unpacked[0]));
    dA[base + 1] = from_float<T>(prev1 + to_float(unpacked[1]));
}

// Fused inverse shuffle + accumulate for gated backward.
// Inverse-shuffles both dA_perm and dA_gate_perm, adds both to dA.
template<typename T>
__global__ void shuffle_inverse_add_with_gate_kernel(
    T* __restrict__ dA,                     // (M, K) natural layout, accumulated
    const T* __restrict__ dA_perm,          // (M, K) permuted: dS_perm @ R_g
    const T* __restrict__ dA_gate_perm,     // (M, K) permuted: dh * gate(s)
    const int64_t* __restrict__ masks,      // per-group: (n_chunks, n_rounds)
    int64_t M, int64_t K,
    int64_t n_chunks,
    int64_t n_rounds)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;

    int64_t row   = warp_id / n_chunks;
    int64_t chunk = warp_id % n_chunks;
    if (row >= M) return;

    int64_t col = chunk * 64 + lane * 2;
    int64_t byte_off = (row * K + col) * sizeof(T);
    uint32_t v1 = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(dA_perm) + byte_off);
    uint32_t v2 = *reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(dA_gate_perm) + byte_off);

    // Reverse order: round 1 (delta=16), then round 0 (delta=8)
    {
        int64_t m1 = masks[chunk * n_rounds + 1];
        uint32_t p1a = __shfl_xor_sync(0xFFFFFFFF, v1, 16);
        uint32_t p1b = __shfl_xor_sync(0xFFFFFFFF, v2, 16);
        int pi1 = (lane & 15);
        bool swap1 = (m1 >> pi1) & 1;
        v1 = swap1 ? p1a : v1;
        v2 = swap1 ? p1b : v2;
    }
    {
        int64_t m0 = masks[chunk * n_rounds + 0];
        uint32_t p0a = __shfl_xor_sync(0xFFFFFFFF, v1, 8);
        uint32_t p0b = __shfl_xor_sync(0xFFFFFFFF, v2, 8);
        int pi0 = (lane & 7) | ((lane >> 4) << 3);
        bool swap0 = (m0 >> pi0) & 1;
        v1 = swap0 ? p0a : v1;
        v2 = swap0 ? p0b : v2;
    }

    // Unpack and accumulate both contributions
    const T* u1 = reinterpret_cast<const T*>(&v1);
    const T* u2 = reinterpret_cast<const T*>(&v2);
    int64_t base = row * K + col;
    float prev0 = to_float(dA[base]);
    float prev1 = to_float(dA[base + 1]);
    dA[base]     = from_float<T>(prev0 + to_float(u1[0]) + to_float(u2[0]));
    dA[base + 1] = from_float<T>(prev1 + to_float(u1[1]) + to_float(u2[1]));
}

// Fused forward gate: AR[i] = A[i] * SiLU(AR[i] + bias[i%K])  with optional
// dropout on the result.
//   bias == nullptr     → no bias add
//   dropout_seed == 0   → no dropout (no hash cost either; branch is uniform)
// For dropout, per-element mask is derived from (seed, idx) so backward can
// replay deterministically without a rand buffer.
template<typename T>
__global__ void silu_gate_bias_dropout_kernel(
    const T* A, T* AR, const T* bias,
    int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float a = to_float(A[idx]);
        float ar = to_float(AR[idx]);
        if (bias) ar += to_float(bias[idx % K]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        float h = a * silu_ar;
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            h = (u >= dropout_p) ? h * inv_keep : 0.0f;
        }
        AR[idx] = from_float<T>(h);
    }
}

// Fused backward: given recomputed AR (= A @ R^T), add bias, then compute
//   dA_gate[i] = dH[i] * silu(S[i])
//   dS[i]      = dH[i] * A[i] * silu'(S[i])
// where S = AR + bias. Replays dropout via the same (seed, idx) hash used
// in the forward kernel.
//
// Template ACCUMULATE:
//   false → dA_gate[idx] = dh*silu_s    (overwrite; used in shuffle mode
//           where dA_gate_perm is a fresh scratch buffer)
//   true  → dA_gate[idx] += dh*silu_s   (accumulate directly into dA;
//           used in non-shuffle mode to eliminate a temp + add_kernel pass.
//           Each output element is owned by exactly one thread so the RMW
//           needs no atomic.)
template<typename T, bool ACCUMULATE>
__global__ void silu_gate_backward_fused_kernel(
    const T* dH_in, const T* A, T* AR,    // AR is modified in-place to S
    T* dA_gate, T* dS,
    const T* bias, int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float s = to_float(AR[idx]);
        if (bias) s += to_float(bias[idx % K]);
        AR[idx] = from_float<T>(s);  // store S for later dR computation

        // Replay dropout on dH using the same hash as forward
        float dh = to_float(dH_in[idx]);
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            dh = (u >= dropout_p) ? dh * inv_keep : 0.0f;
        }

        float a = to_float(A[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_s = s * sigma;
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));

        float dA_gate_val = dh * silu_s;
        if (ACCUMULATE) {
            dA_gate[idx] = from_float<T>(to_float(dA_gate[idx]) + dA_gate_val);
        } else {
            dA_gate[idx] = from_float<T>(dA_gate_val);
        }
        dS[idx] = from_float<T>(dh * a * silu_prime);
    }
}

// Reduce-variant of silu_gate_backward_fused_kernel: in addition to writing
// dA_gate and dS, performs a column-wise sum of dS into an fp32 scratch
// buffer d_ib_f32 (shape (K,)). The element-wise work uses the standard
// 2D launch (one thread per element) to preserve element-wise latency
// hiding; a block-local tree reduction along the row (y) axis produces a
// partial sum per column, then one thread per column-block atomicAdds that
// partial to d_ib_f32[col]. A separate cast kernel finalizes the result.
//
// Launch:
//   block = dim3(BN, BM)  (threads.x = column, threads.y = row)
//   grid  = dim3(ceil(K/BN), ceil(M/BM))
// BM must be a power of 2 for the in-block tree reduction.
template<typename T, bool ACCUMULATE, int BM, int BN>
__global__ void silu_gate_backward_fused_reduce_kernel(
    const T* dH_in, const T* A, T* AR,
    T* dA_gate, T* dS,
    float* d_ib_f32,                 // (K,) fp32 accumulator (pre-zero'd)
    const T* bias, int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t col = (int64_t)blockIdx.x * BN + threadIdx.x;
    int64_t row = (int64_t)blockIdx.y * BM + threadIdx.y;

    float dS_val = 0.0f;
    if (row < M && col < K) {
        int64_t idx = row * K + col;
        float s = to_float(AR[idx]);
        if (bias) s += to_float(bias[col]);
        AR[idx] = from_float<T>(s);

        float dh = to_float(dH_in[idx]);
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            dh = (u >= dropout_p) ? dh * inv_keep : 0.0f;
        }

        float a = to_float(A[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_s = s * sigma;
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));

        float dA_gate_val = dh * silu_s;
        if (ACCUMULATE) {
            dA_gate[idx] = from_float<T>(to_float(dA_gate[idx]) + dA_gate_val);
        } else {
            dA_gate[idx] = from_float<T>(dA_gate_val);
        }
        dS_val = dh * a * silu_prime;
        dS[idx] = from_float<T>(dS_val);
    }

    // Block-wide column reduction along the row (y) axis.
    __shared__ float sm[BM * BN];
    int sm_idx = threadIdx.y * BN + threadIdx.x;
    sm[sm_idx] = dS_val;
    __syncthreads();

    #pragma unroll
    for (int stride = BM / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.y < stride) {
            sm[sm_idx] += sm[sm_idx + stride * BN];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && col < K) {
        atomicAdd(&d_ib_f32[col], sm[threadIdx.x]);
    }
}

// ---------------------------------------------------------------------------
// sigmoid_gate variants: forward H = A * 2*sigmoid(S), with 2*sigmoid(0) = 1
// matching silu(0)=0-slope behavior at zero and capping the gate in (0, 2).
// This bounds the elementwise dA contribution dH * 2*sigmoid(S), replacing
// the unbounded silu(S) term that grows linearly with |S| at init.
// ---------------------------------------------------------------------------

// Forward fused gate: AR[i] = A[i] * 2*sigmoid(AR[i] + bias[i%K])  with
// optional dropout on the result. Same interface and semantics as the
// silu_gate variant.
template<typename T>
__global__ void sigmoid_gate_bias_dropout_kernel(
    const T* A, T* AR, const T* bias,
    int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float a = to_float(A[idx]);
        float ar = to_float(AR[idx]);
        if (bias) ar += to_float(bias[idx % K]);
        float sigma = 1.0f / (1.0f + __expf(-ar));
        float h = a * (2.0f * sigma);
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            h = (u >= dropout_p) ? h * inv_keep : 0.0f;
        }
        AR[idx] = from_float<T>(h);
    }
}

// Fused backward for H = A * 2*sigmoid(S):
//   dA_gate[i] = dH[i] * 2*sigmoid(S[i])
//   dS[i]      = dH[i] * A[i] * 2*sigmoid(S[i]) * (1 - sigmoid(S[i]))
template<typename T, bool ACCUMULATE>
__global__ void sigmoid_gate_backward_fused_kernel(
    const T* dH_in, const T* A, T* AR,
    T* dA_gate, T* dS,
    const T* bias, int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float s = to_float(AR[idx]);
        if (bias) s += to_float(bias[idx % K]);
        AR[idx] = from_float<T>(s);

        float dh = to_float(dH_in[idx]);
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            dh = (u >= dropout_p) ? dh * inv_keep : 0.0f;
        }

        float a = to_float(A[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float two_sigma = 2.0f * sigma;
        float gate_prime = two_sigma * (1.0f - sigma);

        float dA_gate_val = dh * two_sigma;
        if (ACCUMULATE) {
            dA_gate[idx] = from_float<T>(to_float(dA_gate[idx]) + dA_gate_val);
        } else {
            dA_gate[idx] = from_float<T>(dA_gate_val);
        }
        dS[idx] = from_float<T>(dh * a * gate_prime);
    }
}

// Reduce-variant of sigmoid_gate_backward_fused_kernel; adds a column-wise
// sum of dS into d_ib_f32, analogous to the silu variant.
template<typename T, bool ACCUMULATE, int BM, int BN>
__global__ void sigmoid_gate_backward_fused_reduce_kernel(
    const T* dH_in, const T* A, T* AR,
    T* dA_gate, T* dS,
    float* d_ib_f32,
    const T* bias, int64_t M, int64_t K,
    uint64_t dropout_seed, float dropout_p, float inv_keep)
{
    int64_t col = (int64_t)blockIdx.x * BN + threadIdx.x;
    int64_t row = (int64_t)blockIdx.y * BM + threadIdx.y;

    float dS_val = 0.0f;
    if (row < M && col < K) {
        int64_t idx = row * K + col;
        float s = to_float(AR[idx]);
        if (bias) s += to_float(bias[col]);
        AR[idx] = from_float<T>(s);

        float dh = to_float(dH_in[idx]);
        if (dropout_seed) {
            float u = uniform_from_hash(dropout_seed, idx);
            dh = (u >= dropout_p) ? dh * inv_keep : 0.0f;
        }

        float a = to_float(A[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float two_sigma = 2.0f * sigma;
        float gate_prime = two_sigma * (1.0f - sigma);

        float dA_gate_val = dh * two_sigma;
        if (ACCUMULATE) {
            dA_gate[idx] = from_float<T>(to_float(dA_gate[idx]) + dA_gate_val);
        } else {
            dA_gate[idx] = from_float<T>(dA_gate_val);
        }
        dS_val = dh * a * gate_prime;
        dS[idx] = from_float<T>(dS_val);
    }

    __shared__ float sm[BM * BN];
    int sm_idx = threadIdx.y * BN + threadIdx.x;
    sm[sm_idx] = dS_val;
    __syncthreads();

    #pragma unroll
    for (int stride = BM / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.y < stride) {
            sm[sm_idx] += sm[sm_idx + stride * BN];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0 && col < K) {
        atomicAdd(&d_ib_f32[col], sm[threadIdx.x]);
    }
}

// Gate-kind enum shared between host dispatch and kernel selection.
// 0 = none (linear), 1 = silu_gate, 2 = sigmoid_gate.
enum : int { GATE_NONE = 0, GATE_SILU = 1, GATE_SIGMOID = 2 };

// Cast an fp32 buffer to T (half / bfloat16). Used once per group to
// finalize that group's d_internal_bias row after the fused reduction.
template<typename T>
__global__ void f32_to_t_kernel(const float* in, T* out, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) out[idx] = from_float<T>(in[idx]);
}

// ---------------------------------------------------------------------------
// Forward: AR mode  C = (A @ R^T) @ B^T  [gated: C = (A * SiLU(A @ R^T)) @ B^T]
// ---------------------------------------------------------------------------

// Dropout uses a hash-based per-element mask (see uniform_from_hash). A seed
// of 0 is reserved for "no dropout"; we coerce to 1 if the host RNG happens to
// produce 0 so the kernel branch stays consistent.
static inline uint64_t _nonzero_seed(uint64_t s) { return s ? s : 1ULL; }

template<typename scalar_t>
static void cublas_prism_ar_impl(
    const scalar_t* A_ptr, const scalar_t* B_ptr, const scalar_t* R_ptr, scalar_t* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt,
    torch::TensorOptions opts,
    int gate_kind, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    float dropout_p = 0.0f, bool training = false,
    int64_t* dropout_seeds_out = nullptr,
    const int64_t* shuffle_masks_ptr = nullptr,
    int64_t n_chunks = 0, int64_t n_rounds = 0)
{
    // Two-stream pipeline: producer (batched AR + gate) overlaps with
    // consumer (dense C_g GEMM) across groups. Double-buffer AR.
    constexpr int PIPE = 2;
    torch::Tensor AR_t[PIPE];
    scalar_t* AR_ptr[PIPE];
    for (int p = 0; p < PIPE; ++p) {
        AR_t[p] = torch::empty({m, k}, opts);
        AR_ptr[p] = static_cast<scalar_t*>(AR_t[p].data_ptr());
    }

    // Single-buffer A_perm for shuffle mode.
    const bool shuffle = shuffle_masks_ptr != nullptr;
    int64_t n_blocks = k / reconn_sz;
    torch::Tensor A_perm_t;
    scalar_t* A_perm_ptr = nullptr;
    if (shuffle) {
        A_perm_t = torch::empty({m, k}, opts);
        A_perm_ptr = static_cast<scalar_t*>(A_perm_t.data_ptr());
    }

    cudaStream_t stream_p;
    cudaEvent_t ready[PIPE];
    cudaStreamCreate(&stream_p);
    for (int p = 0; p < PIPE; ++p)
        cudaEventCreate(&ready[p]);

    cublasHandle_t handle_p;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle_p));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle_p, stream_p));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle_p, CUBLAS_DEFAULT_MATH));

    int threads = 256;
    bool apply_dropout = training && dropout_p > 0.0f;
    float inv_keep = apply_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    const bool gated = (gate_kind != GATE_NONE);
    auto device = opts.device();

    auto produce = [&](int64_t g, int p) {
        int64_t numel = m * k;
        int nblocks = (int)((numel + threads - 1) / threads);

        const scalar_t* A_gemm_in = A_ptr;
        if (shuffle) {
            int64_t total_warps = m * n_chunks;
            int shfl_blocks = (int)((total_warps * 32 + threads - 1) / threads);
            shuffle_forward_kernel<scalar_t><<<shfl_blocks, threads, 0, stream_p>>>(
                A_ptr, A_perm_ptr,
                shuffle_masks_ptr + g * n_chunks * n_rounds,
                m, k, n_chunks, n_rounds);
            A_gemm_in = A_perm_ptr;
        }

        gemm_strided_batched_ex(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_ptr + g * reconn_sz * k, dt, k, reconn_sz,
            A_gemm_in,                 dt, k, reconn_sz,
            AR_ptr[p],                 dt, k, reconn_sz,
            k / reconn_sz);

        if (gated) {
            // Unified fused gate: bias / dropout are optional (nullptr / seed=0
            // branches are uniform across the warp and compile out).
            uint64_t seed = 0;
            if (apply_dropout) {
                seed = _nonzero_seed((uint64_t)std::random_device{}());
                if (dropout_seeds_out) dropout_seeds_out[g] = (int64_t)seed;
            }
            const scalar_t* bias_g =
                internal_bias_ptr ? internal_bias_ptr + g * k : nullptr;
            if (gate_kind == GATE_SIGMOID) {
                sigmoid_gate_bias_dropout_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_gemm_in, AR_ptr[p], bias_g,
                    m, k, seed, dropout_p, inv_keep);
            } else {
                silu_gate_bias_dropout_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_gemm_in, AR_ptr[p], bias_g,
                    m, k, seed, dropout_p, inv_keep);
            }
        }

        cudaEventRecord(ready[p], stream_p);
    };

    auto consume = [&](int64_t g, int p) {
        cudaStreamWaitEvent(stream, ready[p]);

        gemm_ex(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            group_size, m, k,
            B_ptr + g * group_size * k, dt, k,
            AR_ptr[p],                  dt, k,
            C_ptr + g * group_size,     dt, n);
    };

    for (int64_t g = 0; g < n_groups; ++g) {
        int p = (int)(g % PIPE);

        if (g >= PIPE) {
            cudaEvent_t consumed;
            cudaEventCreate(&consumed);
            cudaEventRecord(consumed, stream);
            cudaStreamWaitEvent(stream_p, consumed);
            cudaEventDestroy(consumed);
        }

        produce(g, p);
        if (g > 0)
            consume(g - 1, (int)((g - 1) % PIPE));
    }
    if (n_groups > 0)
        consume(n_groups - 1, (int)((n_groups - 1) % PIPE));

    cudaStreamSynchronize(stream_p);
    GEMM_CHECK_CUBLAS(cublasDestroy(handle_p));
    for (int p = 0; p < PIPE; ++p)
        cudaEventDestroy(ready[p]);
    cudaStreamDestroy(stream_p);
}

// ---------------------------------------------------------------------------
// Forward: RW mode  C = A @ (R @ B)^T
// ---------------------------------------------------------------------------

template<typename scalar_t>
static void cublas_prism_rw_impl(
    const scalar_t* A_ptr, const scalar_t* B_ptr, const scalar_t* R_ptr, scalar_t* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt,
    torch::TensorOptions opts)
{
    auto Bp = torch::empty({n, k}, opts);
    auto* Bp_ptr = static_cast<scalar_t*>(Bp.data_ptr());

    for (int64_t i = 0; i < n_groups; ++i) {
        gemm_strided_batched_ex(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            reconn_sz, group_size, reconn_sz,
            R_ptr + i * reconn_sz * k,  dt, k, reconn_sz,
            B_ptr + i * group_size * k, dt, k, reconn_sz,
            Bp_ptr + i * group_size * k, dt, k, reconn_sz,
            k / reconn_sz);
    }

    // C = A @ Bp^T
    gemm_ex(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        Bp_ptr, dt, k,
        A_ptr,  dt, k,
        C_ptr,  dt, n);
}

// ---------------------------------------------------------------------------
// Backward dA + dR
// ---------------------------------------------------------------------------

template<typename scalar_t>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cublas_backward_dA_dR_impl(
    const scalar_t* dC_ptr, const scalar_t* A_ptr,
    const scalar_t* B_ptr, const scalar_t* R_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t compute_dt,
    torch::TensorOptions compute_opts,
    cudaDataType_t dR_dt, torch::TensorOptions dR_opts,
    int gate_kind, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    const int64_t* dropout_seeds = nullptr, float dropout_p = 0.0f,
    const int64_t* shuffle_masks_ptr = nullptr,
    int64_t n_chunks = 0, int64_t n_rounds = 0)
{
    const bool gated = (gate_kind != GATE_NONE);
    auto dA  = torch::zeros({m, k}, compute_opts);
    auto dR  = torch::zeros({n_groups * reconn_sz, k}, dR_opts);
    auto d_internal_bias = internal_bias_ptr
        ? torch::zeros({n_groups, k}, compute_opts)
        : torch::Tensor();
    auto AR  = torch::empty({m, k}, compute_opts);
    auto dAR = torch::empty({m, k}, compute_opts);

    auto* dA_ptr  = static_cast<scalar_t*>(dA.data_ptr());
    void* dR_raw  = dR.data_ptr();
    int64_t dR_es = dR.element_size();
    auto* AR_ptr  = static_cast<scalar_t*>(AR.data_ptr());
    auto* dAR_ptr = static_cast<scalar_t*>(dAR.data_ptr());

    // Shuffle-mode scratch buffers.
    const bool shuffle = shuffle_masks_ptr != nullptr;
    int64_t n_blocks = k / reconn_sz;
    torch::Tensor A_perm_t, dA_perm_t, dA_gate_perm_t;
    scalar_t* A_perm_ptr = nullptr;
    scalar_t* dA_perm_ptr = nullptr;
    scalar_t* dA_gate_perm_ptr = nullptr;
    if (shuffle) {
        A_perm_t = torch::empty({m, k}, compute_opts);
        dA_perm_t = torch::empty({m, k}, compute_opts);
        A_perm_ptr = static_cast<scalar_t*>(A_perm_t.data_ptr());
        dA_perm_ptr = static_cast<scalar_t*>(dA_perm_t.data_ptr());
        if (gated) {
            dA_gate_perm_t = torch::empty({m, k}, compute_opts);
            dA_gate_perm_ptr = static_cast<scalar_t*>(dA_gate_perm_t.data_ptr());
        }
    }

    // Parallel streams: overlap AR recompute with dH dense GEMM within each group
    cudaStream_t stream_ar;
    cudaEvent_t ar_ready;
    cudaStreamCreate(&stream_ar);
    cudaEventCreate(&ar_ready);

    cublasHandle_t handle_ar;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle_ar));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle_ar, stream_ar));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle_ar, CUBLAS_DEFAULT_MATH));

    int threads = 256;
    bool has_bias = internal_bias_ptr != nullptr;
    bool has_dropout = dropout_seeds != nullptr && dropout_p > 0.0f;
    float inv_keep = has_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    auto device = compute_opts.device();

    // Per-group fp32 scratch for d_internal_bias. Shared across groups — we
    // zero it at the start of each group iteration, let the fused reduce
    // kernel atomicAdd into it, then cast the result into the matching
    // d_internal_bias[g] row. This keeps scratch at O(K) instead of
    // O(n_groups * K).
    torch::Tensor d_ib_f32;
    float* d_ib_f32_ptr = nullptr;
    scalar_t* d_ib_base = nullptr;
    if (has_bias) {
        auto f32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        d_ib_f32 = torch::zeros({k}, f32_opts);
        d_ib_f32_ptr = d_ib_f32.data_ptr<float>();
        d_ib_base = static_cast<scalar_t*>(d_internal_bias.data_ptr());
    }

    for (int64_t g = 0; g < n_groups; ++g) {
        const scalar_t* dC_g = dC_ptr + g * group_size;
        const scalar_t* B_g  = B_ptr  + g * group_size * k;
        const scalar_t* R_g  = R_ptr  + g * reconn_sz * k;
        void* dR_g = static_cast<char*>(dR_raw) + g * reconn_sz * k * dR_es;

        int64_t numel = m * k;
        int nblocks = (int)((numel + threads - 1) / threads);

        // In shuffle mode, butterfly-shuffle A→A_perm. The AR GEMM and
        // downstream gate_bwd + dR GEMM all read A_perm instead of A.
        const scalar_t* A_use_ptr = A_ptr;
        if (shuffle) {
            int64_t total_warps = m * n_chunks;
            int shfl_blocks = (int)((total_warps * 32 + threads - 1) / threads);
            shuffle_forward_kernel<scalar_t><<<shfl_blocks, threads, 0, stream_ar>>>(
                A_ptr, A_perm_ptr,
                shuffle_masks_ptr + g * n_chunks * n_rounds,
                m, k, n_chunks, n_rounds);
            A_use_ptr = A_perm_ptr;
        }

        // Stream AR: recompute AR_g = A_use @ R_g^T
        gemm_strided_batched_ex(
            handle_ar, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_g,       compute_dt, k, reconn_sz,
            A_use_ptr, compute_dt, k, reconn_sz,
            AR_ptr,    compute_dt, k, reconn_sz,
            k / reconn_sz);
        cudaEventRecord(ar_ready, stream_ar);

        // Target buffer for the dS_g @ R_g (or dAR_g @ R_g non-gated) GEMM:
        // for shuffle we accumulate into a permuted scratch then unpermute;
        // otherwise we accumulate directly into dA with beta=1.
        scalar_t* dA_accum_ptr = shuffle ? dA_perm_ptr : dA_ptr;
        float dA_accum_beta = shuffle ? 0.0f : 1.0f;

        if (gated) {
            // Stream main: dH_g = dC_g @ B_g (runs in parallel with AR recompute)
            gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, m, group_size,
                B_g,     compute_dt, k,
                dC_g,    compute_dt, n,
                dAR_ptr, compute_dt, k);

            // Wait for AR recompute before using AR_ptr
            cudaStreamWaitEvent(stream, ar_ready);

            {
                // Target for the dA_gate contribution:
                //   non-shuffle → accumulate directly into dA (ACCUMULATE=true),
                //                 no temp buffer.
                //   shuffle     → write into dA_gate_perm scratch; the
                //                 shuffle_inverse_add_with_gate_kernel below
                //                 scatters it together with the dS@R contribution.
                scalar_t* dA_gate_dst = shuffle ? dA_gate_perm_ptr : dA_ptr;

                uint64_t seed = has_dropout ? (uint64_t)dropout_seeds[g] : 0ULL;
                const scalar_t* bias_g =
                    has_bias ? internal_bias_ptr + g * k : nullptr;

                // Fused gate backward. When has_bias, also reduces dS
                // column-wise into the per-group fp32 scratch d_ib_f32 via
                // 2D-grid element-wise + block-local reduction + atomicAdd.
                const bool is_sigmoid = (gate_kind == GATE_SIGMOID);
                if (has_bias) {
                    // Zero the K-element scratch before this group's pass.
                    cudaMemsetAsync(d_ib_f32_ptr, 0, k * sizeof(float), stream);

                    constexpr int BM = 16;
                    constexpr int BN = 16;
                    dim3 block(BN, BM);
                    dim3 grid((int)((k + BN - 1) / BN),
                              (int)((m + BM - 1) / BM));
                    if (shuffle) {
                        if (is_sigmoid) {
                            sigmoid_gate_backward_fused_reduce_kernel<
                                scalar_t, /*ACCUMULATE=*/false, BM, BN>
                                <<<grid, block, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    d_ib_f32_ptr, bias_g,
                                    m, k, seed, dropout_p, inv_keep);
                        } else {
                            silu_gate_backward_fused_reduce_kernel<
                                scalar_t, /*ACCUMULATE=*/false, BM, BN>
                                <<<grid, block, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    d_ib_f32_ptr, bias_g,
                                    m, k, seed, dropout_p, inv_keep);
                        }
                    } else {
                        if (is_sigmoid) {
                            sigmoid_gate_backward_fused_reduce_kernel<
                                scalar_t, /*ACCUMULATE=*/true, BM, BN>
                                <<<grid, block, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    d_ib_f32_ptr, bias_g,
                                    m, k, seed, dropout_p, inv_keep);
                        } else {
                            silu_gate_backward_fused_reduce_kernel<
                                scalar_t, /*ACCUMULATE=*/true, BM, BN>
                                <<<grid, block, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    d_ib_f32_ptr, bias_g,
                                    m, k, seed, dropout_p, inv_keep);
                        }
                    }

                    // Cast this group's scratch into d_internal_bias[g].
                    int cvt_threads = 256;
                    int cvt_blocks = (int)((k + cvt_threads - 1) / cvt_threads);
                    f32_to_t_kernel<scalar_t><<<cvt_blocks, cvt_threads, 0, stream>>>(
                        d_ib_f32_ptr, d_ib_base + g * k, k);
                } else {
                    if (shuffle) {
                        if (is_sigmoid) {
                            sigmoid_gate_backward_fused_kernel<
                                scalar_t, /*ACCUMULATE=*/false>
                                <<<nblocks, threads, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    bias_g, m, k, seed, dropout_p, inv_keep);
                        } else {
                            silu_gate_backward_fused_kernel<
                                scalar_t, /*ACCUMULATE=*/false>
                                <<<nblocks, threads, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    bias_g, m, k, seed, dropout_p, inv_keep);
                        }
                    } else {
                        if (is_sigmoid) {
                            sigmoid_gate_backward_fused_kernel<
                                scalar_t, /*ACCUMULATE=*/true>
                                <<<nblocks, threads, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    bias_g, m, k, seed, dropout_p, inv_keep);
                        } else {
                            silu_gate_backward_fused_kernel<
                                scalar_t, /*ACCUMULATE=*/true>
                                <<<nblocks, threads, 0, stream>>>(
                                    dAR_ptr, A_use_ptr, AR_ptr,
                                    dA_gate_dst, dAR_ptr,
                                    bias_g, m, k, seed, dropout_p, inv_keep);
                        }
                    }
                }
            }

            // dA_accum = dS_g @ R_g  (beta=1 non-shuffle, beta=0 shuffle).
            // Non-shuffle: dA already holds the dA_gate contribution from the
            // fused kernel above; beta=1 adds dS@R on top.
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz,
                R_g,          compute_dt, k, reconn_sz,
                dAR_ptr,      compute_dt, k, reconn_sz,
                dA_accum_ptr, compute_dt, k, reconn_sz,
                k / reconn_sz,
                1.0f, dA_accum_beta);

            if (shuffle) {
                int64_t total_warps = m * n_chunks;
                int shfl_blocks = (int)((total_warps * 32 + threads - 1) / threads);
                shuffle_inverse_add_with_gate_kernel<scalar_t>
                    <<<shfl_blocks, threads, 0, stream>>>(
                        dA_ptr, dA_perm_ptr, dA_gate_perm_ptr,
                        shuffle_masks_ptr + g * n_chunks * n_rounds,
                        m, k, n_chunks, n_rounds);
            }

            // dR_g[b] = dS_g^T @ A_use
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m,
                A_use_ptr, compute_dt, k, reconn_sz,
                dAR_ptr,   compute_dt, k, reconn_sz,
                dR_g,      dR_dt,      k, reconn_sz,
                k / reconn_sz);
        } else {
            // Non-gated: no gate recompute needed, just dH = dC @ B.
            // Still need to wait for AR stream because shuffle=true also
            // runs the gather there.
            cudaStreamWaitEvent(stream, ar_ready);

            // dAR_g = dC_g @ B_g
            gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, m, group_size,
                B_g,     compute_dt, k,
                dC_g,    compute_dt, n,
                dAR_ptr, compute_dt, k);

            // dA_accum = dAR_g @ R_g
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz,
                R_g,          compute_dt, k, reconn_sz,
                dAR_ptr,      compute_dt, k, reconn_sz,
                dA_accum_ptr, compute_dt, k, reconn_sz,
                k / reconn_sz,
                1.0f, dA_accum_beta);

            if (shuffle) {
                int64_t total_warps = m * n_chunks;
                int shfl_blocks = (int)((total_warps * 32 + threads - 1) / threads);
                shuffle_inverse_add_kernel<scalar_t><<<shfl_blocks, threads, 0, stream>>>(
                    dA_ptr, dA_perm_ptr,
                    shuffle_masks_ptr + g * n_chunks * n_rounds,
                    m, k, n_chunks, n_rounds);
            }

            // dR_g[b] = dAR_g^T @ A_use
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m,
                A_use_ptr, compute_dt, k, reconn_sz,
                dAR_ptr,   compute_dt, k, reconn_sz,
                dR_g,      dR_dt,      k, reconn_sz,
                k / reconn_sz);
        }
    }

    // (d_internal_bias is written directly by the fused reduce kernel —
    // no post-loop cast needed.)

    cudaStreamSynchronize(stream_ar);
    GEMM_CHECK_CUBLAS(cublasDestroy(handle_ar));
    cudaEventDestroy(ar_ready);
    cudaStreamDestroy(stream_ar);

    return {dA, dR, d_internal_bias};
}

// ---------------------------------------------------------------------------
// Backward dB
// ---------------------------------------------------------------------------

template<typename scalar_t>
static torch::Tensor cublas_backward_dB_impl(
    const scalar_t* dC_ptr, const scalar_t* A_ptr, const scalar_t* R_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t compute_dt,
    torch::TensorOptions compute_opts,
    cudaDataType_t dB_dt, torch::TensorOptions dB_opts,
    int gate_kind, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    const int64_t* dropout_seeds = nullptr, float dropout_p = 0.0f,
    const int64_t* shuffle_masks_ptr = nullptr,
    int64_t n_chunks = 0, int64_t n_rounds = 0)
{
    const bool gated = (gate_kind != GATE_NONE);
    auto dB = torch::zeros({n, k}, dB_opts);
    void* dB_raw  = dB.data_ptr();
    int64_t dB_es = dB.element_size();

    // Double-buffer AR for pipelined recompute + dB GEMM
    constexpr int PIPE = 2;
    torch::Tensor AR_t[PIPE];
    scalar_t* AR_ptr[PIPE];
    for (int p = 0; p < PIPE; ++p) {
        AR_t[p] = torch::empty({m, k}, compute_opts);
        AR_ptr[p] = static_cast<scalar_t*>(AR_t[p].data_ptr());
    }

    // Single-buffer A_perm for shuffle mode.
    const bool shuffle = shuffle_masks_ptr != nullptr;
    int64_t n_blocks = k / reconn_sz;
    torch::Tensor A_perm_t;
    scalar_t* A_perm_ptr = nullptr;
    if (shuffle) {
        A_perm_t = torch::empty({m, k}, compute_opts);
        A_perm_ptr = static_cast<scalar_t*>(A_perm_t.data_ptr());
    }

    cudaStream_t stream_p;
    cudaEvent_t ready[PIPE];
    cudaStreamCreate(&stream_p);
    for (int p = 0; p < PIPE; ++p)
        cudaEventCreate(&ready[p]);

    cublasHandle_t handle_p;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle_p));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle_p, stream_p));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle_p, CUBLAS_DEFAULT_MATH));

    int threads = 256;
    bool has_bias = internal_bias_ptr != nullptr;
    bool has_dropout = dropout_seeds != nullptr && dropout_p > 0.0f;
    float inv_keep = has_dropout ? 1.0f / (1.0f - dropout_p) : 1.0f;
    auto device = compute_opts.device();

    auto produce = [&](int64_t g, int p) {
        const scalar_t* R_g = R_ptr + g * reconn_sz * k;
        int64_t numel = m * k;
        int nblocks = (int)((numel + threads - 1) / threads);

        const scalar_t* A_gemm_in = A_ptr;
        if (shuffle) {
            int64_t total_warps = m * n_chunks;
            int shfl_blocks = (int)((total_warps * 32 + threads - 1) / threads);
            shuffle_forward_kernel<scalar_t><<<shfl_blocks, threads, 0, stream_p>>>(
                A_ptr, A_perm_ptr,
                shuffle_masks_ptr + g * n_chunks * n_rounds,
                m, k, n_chunks, n_rounds);
            A_gemm_in = A_perm_ptr;
        }

        gemm_strided_batched_ex(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_g,       compute_dt, k, reconn_sz,
            A_gemm_in, compute_dt, k, reconn_sz,
            AR_ptr[p], compute_dt, k, reconn_sz,
            k / reconn_sz);

        if (gated) {
            // Unified fused gate: nullptr bias and seed=0 are no-ops, so this
            // single kernel handles all four sub-cases (plain / bias / dropout
            // / bias+dropout).
            uint64_t seed = has_dropout ? (uint64_t)dropout_seeds[g] : 0ULL;
            const scalar_t* bias_g =
                has_bias ? internal_bias_ptr + g * k : nullptr;
            if (gate_kind == GATE_SIGMOID) {
                sigmoid_gate_bias_dropout_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_gemm_in, AR_ptr[p], bias_g,
                    m, k, seed, dropout_p, inv_keep);
            } else {
                silu_gate_bias_dropout_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_gemm_in, AR_ptr[p], bias_g,
                    m, k, seed, dropout_p, inv_keep);
            }
        }

        cudaEventRecord(ready[p], stream_p);
    };

    auto consume = [&](int64_t g, int p) {
        cudaStreamWaitEvent(stream, ready[p]);

        const scalar_t* dC_g = dC_ptr + g * group_size;
        void* dB_g = static_cast<char*>(dB_raw) + g * group_size * k * dB_es;

        gemm_ex(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            k, group_size, m,
            AR_ptr[p], compute_dt, k,
            dC_g,      compute_dt, n,
            dB_g,      dB_dt,      k);
    };

    for (int64_t g = 0; g < n_groups; ++g) {
        int p = (int)(g % PIPE);

        if (g >= PIPE) {
            cudaEvent_t consumed;
            cudaEventCreate(&consumed);
            cudaEventRecord(consumed, stream);
            cudaStreamWaitEvent(stream_p, consumed);
            cudaEventDestroy(consumed);
        }

        produce(g, p);
        if (g > 0)
            consume(g - 1, (int)((g - 1) % PIPE));
    }
    if (n_groups > 0)
        consume(n_groups - 1, (int)((n_groups - 1) % PIPE));

    cudaStreamSynchronize(stream_p);
    GEMM_CHECK_CUBLAS(cublasDestroy(handle_p));
    for (int p = 0; p < PIPE; ++p)
        cudaEventDestroy(ready[p]);
    cudaStreamDestroy(stream_p);

    return dB;
}

// ---------------------------------------------------------------------------
// Input validation helpers
// ---------------------------------------------------------------------------

static at::ScalarType validate_forward_inputs(
    torch::Tensor A, torch::Tensor B, torch::Tensor R,
    int64_t group_size, int64_t reconn_sz)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(R.is_cuda(), "R must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device() && A.device() == R.device(),
                "All tensors must be on the same CUDA device");

    auto dtype = A.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "A must be float16 or bfloat16, got ", dtype);
    TORCH_CHECK(B.scalar_type() == dtype && R.scalar_type() == dtype,
                "All tensors must have the same dtype (", dtype,
                "), but B is ", B.scalar_type(), " and R is ", R.scalar_type());

    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");
    TORCH_CHECK(R.is_contiguous(), "R must be contiguous (row-major)");

    TORCH_CHECK(A.dim() == 2, "A must be 2D (M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N, K)");
    TORCH_CHECK(R.dim() == 2, "R must be 2D (n_groups * reconn_sz, K)");

    int64_t k = A.size(1);
    int64_t n = B.size(0);
    TORCH_CHECK(B.size(1) == k, "K mismatch: A has K=", k, " but B has K=", B.size(1));
    TORCH_CHECK(R.size(1) == k, "K mismatch: A has K=", k, " but R has K=", R.size(1));
    TORCH_CHECK(n % group_size == 0,
                "N (", n, ") must be divisible by group_size (", group_size, ")");
    TORCH_CHECK(k % reconn_sz == 0,
                "K (", k, ") must be divisible by reconn_sz (", reconn_sz, ")");
    int64_t n_groups = n / group_size;
    TORCH_CHECK(R.size(0) == n_groups * reconn_sz,
                "R must have shape (n_groups * reconn_sz, K) = (", n_groups * reconn_sz,
                ", ", k, "), but got (", R.size(0), ", ", R.size(1), ")");

    return dtype;
}

static at::ScalarType validate_backward_inputs(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA tensors");
    auto dtype = dC.scalar_type();
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
                "dC must be float16 or bfloat16, got ", dtype);
    TORCH_CHECK(A.scalar_type() == dtype && R.scalar_type() == dtype,
                "All tensors must have the same dtype (", dtype, ")");
    TORCH_CHECK(dC.is_contiguous() && A.is_contiguous() && R.is_contiguous(),
                "All tensors must be contiguous");
    return dtype;
}

static void validate_grad_dtype(c10::optional<at::ScalarType> grad_dtype, const char* name) {
    if (grad_dtype.has_value()) {
        auto dt = grad_dtype.value();
        TORCH_CHECK(dt == at::kHalf || dt == at::kBFloat16 || dt == at::kFloat,
                     name, " must be float16, bfloat16, or float32, got ", dt);
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> cublas_prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode,
    int64_t gate_kind,
    c10::optional<torch::Tensor> internal_bias,
    double dropout_p,
    bool training,
    c10::optional<torch::Tensor> shuffle_masks)
{
    auto dtype = validate_forward_inputs(A, B, R, group_size, reconn_sz);
    auto cuda_dt = torch_to_cublas_dtype(dtype);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);
    int64_t n_groups = n / group_size;

    TORCH_CHECK(gate_kind == GATE_NONE || gate_kind == GATE_SILU || gate_kind == GATE_SIGMOID,
                "gate_kind must be 0 (none), 1 (silu_gate), or 2 (sigmoid_gate), got ", gate_kind);
    const bool gated = (gate_kind != GATE_NONE);
    TORCH_CHECK(!gated || !rw_mode,
                "Gated activation is only supported in AR mode, not RW mode");

    const int64_t* shuffle_masks_ptr = nullptr;
    int64_t n_chunks = 0, n_rounds = 0;
    if (shuffle_masks.has_value() && shuffle_masks->defined()) {
        TORCH_CHECK(!rw_mode, "shuffle_masks not supported in rw_mode");
        TORCH_CHECK(shuffle_masks->is_cuda() && shuffle_masks->device() == A.device(),
                    "shuffle_masks must be on the same CUDA device as A");
        TORCH_CHECK(shuffle_masks->scalar_type() == at::kLong, "shuffle_masks must be int64");
        TORCH_CHECK(shuffle_masks->is_contiguous(), "shuffle_masks must be contiguous");
        shuffle_masks_ptr = shuffle_masks->data_ptr<int64_t>();
        n_chunks = shuffle_masks->size(1);
        n_rounds = shuffle_masks->size(2);
    }

    auto C = torch::zeros({m, n}, A.options());

    auto seeds_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto dropout_seeds_t = torch::zeros({n_groups}, seeds_opts);
    int64_t* seeds_ptr = dropout_seeds_t.data_ptr<int64_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    float dp = (float)dropout_p;

    #define DISPATCH_FWD(scalar_type, cpp_type) \
    { \
        auto* A_p = static_cast<const cpp_type*>(A.data_ptr()); \
        auto* B_p = static_cast<const cpp_type*>(B.data_ptr()); \
        auto* R_p = static_cast<const cpp_type*>(R.data_ptr()); \
        auto* C_p = static_cast<cpp_type*>(C.data_ptr()); \
        const cpp_type* ib_p = (internal_bias.has_value() && internal_bias->defined()) \
            ? static_cast<const cpp_type*>(internal_bias->data_ptr()) : nullptr; \
        if (rw_mode) \
            cublas_prism_rw_impl<cpp_type>(A_p, B_p, R_p, C_p, m, n, k, \
                group_size, reconn_sz, n_groups, handle, cuda_dt, A.options()); \
        else \
            cublas_prism_ar_impl<cpp_type>(A_p, B_p, R_p, C_p, m, n, k, \
                group_size, reconn_sz, n_groups, handle, cuda_dt, A.options(), \
                (int)gate_kind, stream, ib_p, dp, training, seeds_ptr, \
                shuffle_masks_ptr, n_chunks, n_rounds); \
    }

    if (dtype == at::kHalf) {
        DISPATCH_FWD(at::kHalf, half)
    } else {
        DISPATCH_FWD(at::kBFloat16, __nv_bfloat16)
    }
    #undef DISPATCH_FWD

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return {C, dropout_seeds_t};
}

std::vector<torch::Tensor> cublas_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    int64_t gate_kind,
    c10::optional<at::ScalarType> dR_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> shuffle_masks)
{
    TORCH_CHECK(gate_kind == GATE_NONE || gate_kind == GATE_SILU || gate_kind == GATE_SIGMOID,
                "gate_kind must be 0 (none), 1 (silu_gate), or 2 (sigmoid_gate), got ", gate_kind);
    auto dtype = validate_backward_inputs(dC, A, R);
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == dtype && B.is_contiguous(),
                "B must be a contiguous CUDA tensor with dtype ", dtype);
    validate_grad_dtype(dR_dtype, "dR_dtype");

    auto compute_dt = torch_to_cublas_dtype(dtype);
    auto dR_scalar  = dR_dtype.value_or(R.scalar_type());
    auto dR_dt      = torch_to_cublas_dtype(dR_scalar);
    auto dR_opts    = A.options().dtype(dR_scalar);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);
    int64_t n_groups = n / group_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    float dp = (float)dropout_p;
    torch::Tensor seeds_cpu;
    const int64_t* seeds_ptr = nullptr;
    if (dropout_seeds.has_value() && dropout_seeds->defined()) {
        seeds_cpu = dropout_seeds->is_cpu() ? *dropout_seeds : dropout_seeds->cpu();
        seeds_ptr = seeds_cpu.data_ptr<int64_t>();
    }
    const int64_t* shuffle_masks_ptr = nullptr;
    int64_t n_chunks = 0, n_rounds = 0;
    if (shuffle_masks.has_value() && shuffle_masks->defined()) {
        TORCH_CHECK(shuffle_masks->is_cuda() && shuffle_masks->device() == A.device(),
                    "shuffle_masks must be on the same CUDA device as A");
        TORCH_CHECK(shuffle_masks->scalar_type() == at::kLong, "shuffle_masks must be int64");
        TORCH_CHECK(shuffle_masks->is_contiguous(), "shuffle_masks must be contiguous");
        shuffle_masks_ptr = shuffle_masks->data_ptr<int64_t>();
        n_chunks = shuffle_masks->size(1);
        n_rounds = shuffle_masks->size(2);
    }

    torch::Tensor dA, dR_out, d_ib;

    #define DISPATCH_BWD_DADR(scalar_type, cpp_type) \
    { \
        auto* dC_p = static_cast<const cpp_type*>(dC.data_ptr()); \
        auto* A_p  = static_cast<const cpp_type*>(A.data_ptr()); \
        auto* B_p  = static_cast<const cpp_type*>(B.data_ptr()); \
        auto* R_p  = static_cast<const cpp_type*>(R.data_ptr()); \
        const cpp_type* ib_p = (internal_bias.has_value() && internal_bias->defined()) \
            ? static_cast<const cpp_type*>(internal_bias->data_ptr()) : nullptr; \
        std::tie(dA, dR_out, d_ib) = cublas_backward_dA_dR_impl<cpp_type>( \
            dC_p, A_p, B_p, R_p, m, n, k, group_size, reconn_sz, n_groups, \
            handle, compute_dt, A.options(), dR_dt, dR_opts, (int)gate_kind, stream, \
            ib_p, seeds_ptr, dp, shuffle_masks_ptr, n_chunks, n_rounds); \
    }

    if (dtype == at::kHalf) {
        DISPATCH_BWD_DADR(at::kHalf, half)
    } else {
        DISPATCH_BWD_DADR(at::kBFloat16, __nv_bfloat16)
    }
    #undef DISPATCH_BWD_DADR

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return {dA, dR_out, d_ib};
}

torch::Tensor cublas_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    int64_t gate_kind,
    c10::optional<at::ScalarType> dB_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p,
    c10::optional<torch::Tensor> shuffle_masks)
{
    TORCH_CHECK(gate_kind == GATE_NONE || gate_kind == GATE_SILU || gate_kind == GATE_SIGMOID,
                "gate_kind must be 0 (none), 1 (silu_gate), or 2 (sigmoid_gate), got ", gate_kind);
    auto dtype = validate_backward_inputs(dC, A, R);
    validate_grad_dtype(dB_dtype, "dB_dtype");

    auto compute_dt = torch_to_cublas_dtype(dtype);
    auto dB_scalar  = dB_dtype.value_or(dtype);
    auto dB_dt      = torch_to_cublas_dtype(dB_scalar);
    auto dB_opts    = A.options().dtype(dB_scalar);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = dC.size(1);
    int64_t n_groups = n / group_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    float dp = (float)dropout_p;
    torch::Tensor seeds_cpu;
    const int64_t* seeds_ptr = nullptr;
    if (dropout_seeds.has_value() && dropout_seeds->defined()) {
        seeds_cpu = dropout_seeds->is_cpu() ? *dropout_seeds : dropout_seeds->cpu();
        seeds_ptr = seeds_cpu.data_ptr<int64_t>();
    }
    const int64_t* shuffle_masks_ptr = nullptr;
    int64_t n_chunks = 0, n_rounds = 0;
    if (shuffle_masks.has_value() && shuffle_masks->defined()) {
        TORCH_CHECK(shuffle_masks->is_cuda() && shuffle_masks->device() == A.device(),
                    "shuffle_masks must be on the same CUDA device as A");
        TORCH_CHECK(shuffle_masks->scalar_type() == at::kLong, "shuffle_masks must be int64");
        TORCH_CHECK(shuffle_masks->is_contiguous(), "shuffle_masks must be contiguous");
        shuffle_masks_ptr = shuffle_masks->data_ptr<int64_t>();
        n_chunks = shuffle_masks->size(1);
        n_rounds = shuffle_masks->size(2);
    }

    torch::Tensor dB;

    #define DISPATCH_BWD_DB(scalar_type, cpp_type) \
    { \
        auto* dC_p = static_cast<const cpp_type*>(dC.data_ptr()); \
        auto* A_p  = static_cast<const cpp_type*>(A.data_ptr()); \
        auto* R_p  = static_cast<const cpp_type*>(R.data_ptr()); \
        const cpp_type* ib_p = (internal_bias.has_value() && internal_bias->defined()) \
            ? static_cast<const cpp_type*>(internal_bias->data_ptr()) : nullptr; \
        dB = cublas_backward_dB_impl<cpp_type>( \
            dC_p, A_p, R_p, m, n, k, group_size, reconn_sz, n_groups, \
            handle, compute_dt, A.options(), dB_dt, dB_opts, (int)gate_kind, stream, \
            ib_p, seeds_ptr, dp, shuffle_masks_ptr, n_chunks, n_rounds); \
    }

    if (dtype == at::kHalf) {
        DISPATCH_BWD_DB(at::kHalf, half)
    } else {
        DISPATCH_BWD_DB(at::kBFloat16, __nv_bfloat16)
    }
    #undef DISPATCH_BWD_DB

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return dB;
}
