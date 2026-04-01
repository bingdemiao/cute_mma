#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

// ===========================================================================
// Shuffle PRISM forward kernel.
//
// Per group g:
//   1. Gather: A_perm = gather_segments(A, seg_pairs[g])
//   2. Block-diagonal GEMM: AR_perm = A_perm @ R_g^T  (cuBLAS strided batched)
//   3. SiLU gating in permuted space: H_perm = A_perm * SiLU(AR_perm)
//   4. Dense GEMM: C_g = H_perm @ B_g^T  (cuBLAS, column order irrelevant)
//
// No scatter needed — the dense GEMM sums over all K, so column order
// doesn't affect the result. B stays in natural order.
// ===========================================================================

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            throw std::runtime_error("cuBLAS: " + std::to_string(status));      \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

static cudaDataType_t torch_to_cublas(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf:     return CUDA_R_16F;
        case at::kBFloat16: return CUDA_R_16BF;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
            return CUDA_R_16F;
    }
}

__device__ __forceinline__ float scalar_to_float(c10::Half v) {
    return __half2float(*reinterpret_cast<const __half*>(&v));
}
__device__ __forceinline__ float scalar_to_float(c10::BFloat16 v) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}
template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ c10::Half from_float(float v) {
    __half h = __float2half(v);
    return *reinterpret_cast<c10::Half*>(&h);
}
template<> __device__ __forceinline__ c10::BFloat16 from_float(float v) {
    __nv_bfloat16 b = __float2bfloat16(v);
    return *reinterpret_cast<c10::BFloat16*>(&b);
}

// ---------------------------------------------------------------------------
// Gather segments of A into permuted layout for one group.
// seg_pairs: (n_blocks, 2) — for block b, reads segments seg_pairs[b,0] and
// seg_pairs[b,1] from A into contiguous columns [b*rs .. (b+1)*rs) in A_perm.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void gather_segments_kernel(
    const T* __restrict__ A,         // (M, K) row-major
    T* __restrict__ A_perm,          // (M, K) permuted layout
    const int64_t* __restrict__ seg_pairs,  // (n_blocks, 2)
    int64_t M, int64_t K, int64_t n_blocks, int64_t seg_sz)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total) return;

    int64_t col = idx % K;
    int64_t row = idx / K;

    int64_t block = col / (2 * seg_sz);
    int64_t within_block = col % (2 * seg_sz);
    int64_t seg_local = within_block / seg_sz;  // 0 or 1
    int64_t elem_in_seg = within_block % seg_sz;

    int64_t src_seg = seg_pairs[block * 2 + seg_local];
    int64_t src_col = src_seg * seg_sz + elem_in_seg;

    A_perm[idx] = A[row * K + src_col];
}

// ---------------------------------------------------------------------------
// SiLU gating: H_perm[i] = A_perm[i] * SiLU(AR_perm[i])
// ---------------------------------------------------------------------------

template<typename T>
__global__ void silu_gate_kernel(const T* A_perm, T* AR_perm, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float a = scalar_to_float(A_perm[idx]);
        float ar = scalar_to_float(AR_perm[idx]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        AR_perm[idx] = from_float<T>(a * silu_ar);
    }
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

template<typename scalar_t>
static void shuffle_prism_forward_impl(
    const scalar_t* A_ptr,
    const scalar_t* B_ptr,
    const scalar_t* R_ptr,
    scalar_t* C_ptr,
    const int64_t* seg_pairs_ptr,  // (n_groups, n_blocks, 2)
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt,
    torch::TensorOptions opts,
    bool gated, cudaStream_t stream)
{
    int64_t n_blocks = k / reconn_sz;
    int64_t seg_sz = reconn_sz / 2;

    // Two-stream pipeline: producer (gather + batched GEMM + gate) and
    // consumer (dense GEMM) overlap across groups. Double-buffer A_perm/AR_perm.
    constexpr int PIPE = 2;
    torch::Tensor A_perm_t[PIPE], AR_perm_t[PIPE];
    scalar_t* A_perm_ptr[PIPE];
    scalar_t* AR_perm_ptr[PIPE];
    for (int p = 0; p < PIPE; ++p) {
        A_perm_t[p] = torch::empty({m, k}, opts);
        AR_perm_t[p] = torch::empty({m, k}, opts);
        A_perm_ptr[p] = static_cast<scalar_t*>(A_perm_t[p].data_ptr());
        AR_perm_ptr[p] = static_cast<scalar_t*>(AR_perm_t[p].data_ptr());
    }


    // Create producer stream + event for synchronization
    cudaStream_t producer_stream;
    cudaEvent_t ready_event[PIPE];
    cudaStreamCreate(&producer_stream);
    for (int p = 0; p < PIPE; ++p)
        cudaEventCreate(&ready_event[p]);

    // Producer needs its own cuBLAS handle bound to producer_stream
    cublasHandle_t producer_handle;
    CHECK_CUBLAS(cublasCreate(&producer_handle));
    CHECK_CUBLAS(cublasSetStream(producer_handle, producer_stream));
    CHECK_CUBLAS(cublasSetMathMode(producer_handle, CUBLAS_DEFAULT_MATH));

    // Consumer uses the original handle + stream
    cublasHandle_t consumer_handle = handle;

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    // Kick off producer for group 0
    auto produce = [&](int64_t g, int p) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;

        // Gather
        int64_t numel = m * k;
        int blks = (int)((numel + threads - 1) / threads);
        gather_segments_kernel<scalar_t><<<blks, threads, 0, producer_stream>>>(
            A_ptr, A_perm_ptr[p], pairs_g, m, k, n_blocks, seg_sz);

        // Block-diagonal GEMM
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            producer_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_ptr + g * reconn_sz * k, dt, (int)k, (long long)reconn_sz,
            A_perm_ptr[p],             dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr[p],            dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // SiLU gating
        if (gated) {
            silu_gate_kernel<scalar_t><<<blks, threads, 0, producer_stream>>>(
                A_perm_ptr[p], AR_perm_ptr[p], numel);
        }

        // Signal that H_perm[p] is ready
        cudaEventRecord(ready_event[p], producer_stream);
    };

    auto consume = [&](int64_t g, int p) {
        // Wait for producer to finish H_perm[p]
        cudaStreamWaitEvent(stream, ready_event[p]);

        scalar_t* H_ptr = AR_perm_ptr[p];
        CHECK_CUBLAS(cublasGemmEx(
            consumer_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)group_size, (int)m, (int)k,
            &alpha,
            B_ptr + g * group_size * k, dt, (int)k,
            H_ptr,                      dt, (int)k,
            &beta,
            C_ptr + g * group_size,     dt, (int)n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Pipeline: produce g, consume g-1
    for (int64_t g = 0; g < n_groups; ++g) {
        int p = (int)(g % PIPE);

        // Before producing into slot p, ensure consumer finished reading it
        if (g >= PIPE) {
            // Consumer for g-PIPE used slot p; it was launched on `stream`
            // and producer_stream must wait for it before overwriting
            cudaEvent_t consumed;
            cudaEventCreate(&consumed);
            cudaEventRecord(consumed, stream);
            cudaStreamWaitEvent(producer_stream, consumed);
            cudaEventDestroy(consumed);
        }

        produce(g, p);

        if (g > 0) {
            consume(g - 1, (int)((g - 1) % PIPE));
        }
    }
    // Consume last group
    if (n_groups > 0) {
        consume(n_groups - 1, (int)((n_groups - 1) % PIPE));
    }

    // Sync and cleanup
    cudaStreamSynchronize(producer_stream);
    CHECK_CUBLAS(cublasDestroy(producer_handle));
    for (int p = 0; p < PIPE; ++p)
        cudaEventDestroy(ready_event[p]);
    cudaStreamDestroy(producer_stream);
}

// ---------------------------------------------------------------------------
// Scatter-add: inverse of gather. Adds dX_perm segments back to dX at
// the original positions specified by seg_pairs.
// ---------------------------------------------------------------------------

template<typename T>
__global__ void scatter_add_segments_kernel(
    const T* __restrict__ dX_perm,       // (M, K) permuted layout
    T* __restrict__ dX,                  // (M, K) natural layout, accumulated
    const int64_t* __restrict__ seg_pairs,  // (n_blocks, 2)
    int64_t M, int64_t K, int64_t n_blocks, int64_t seg_sz)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total) return;

    int64_t col = idx % K;
    int64_t row = idx / K;

    int64_t block = col / (2 * seg_sz);
    int64_t within_block = col % (2 * seg_sz);
    int64_t seg_local = within_block / seg_sz;
    int64_t elem_in_seg = within_block % seg_sz;

    int64_t dst_seg = seg_pairs[block * 2 + seg_local];
    int64_t dst_col = dst_seg * seg_sz + elem_in_seg;

    // Atomic add because multiple groups scatter to the same dA
    float val = scalar_to_float(dX_perm[idx]);
    atomicAdd(reinterpret_cast<float*>(&dX[row * K + dst_col]),
              val);
}

// Specialized scatter-add for float accumulator (dA is f32 for accuracy)
__global__ void scatter_add_segments_f32_kernel(
    const c10::Half* __restrict__ dX_perm,
    float* __restrict__ dX,
    const int64_t* __restrict__ seg_pairs,
    int64_t M, int64_t K, int64_t n_blocks, int64_t seg_sz)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total) return;

    int64_t col = idx % K;
    int64_t row = idx / K;

    int64_t block = col / (2 * seg_sz);
    int64_t within_block = col % (2 * seg_sz);
    int64_t seg_local = within_block / seg_sz;
    int64_t elem_in_seg = within_block % seg_sz;

    int64_t dst_seg = seg_pairs[block * 2 + seg_local];
    int64_t dst_col = dst_seg * seg_sz + elem_in_seg;

    float val = __half2float(*reinterpret_cast<const __half*>(&dX_perm[idx]));
    atomicAdd(&dX[row * K + dst_col], val);
}

__global__ void scatter_add_segments_f32_bf16_kernel(
    const c10::BFloat16* __restrict__ dX_perm,
    float* __restrict__ dX,
    const int64_t* __restrict__ seg_pairs,
    int64_t M, int64_t K, int64_t n_blocks, int64_t seg_sz)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total) return;

    int64_t col = idx % K;
    int64_t row = idx / K;

    int64_t block = col / (2 * seg_sz);
    int64_t within_block = col % (2 * seg_sz);
    int64_t seg_local = within_block / seg_sz;
    int64_t elem_in_seg = within_block % seg_sz;

    int64_t dst_seg = seg_pairs[block * 2 + seg_local];
    int64_t dst_col = dst_seg * seg_sz + elem_in_seg;

    float val = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&dX_perm[idx]));
    atomicAdd(&dX[row * K + dst_col], val);
}

// ---------------------------------------------------------------------------
// Gating backward element-wise kernels
// ---------------------------------------------------------------------------

// dS_perm[i] = dH_perm[i] * A_perm[i] * SiLU'(S_perm[i])
// where S_perm = AR_perm (pre-gating)
template<typename T>
__global__ void silu_gate_backward_ds_kernel(
    const T* dH_perm, const T* A_perm, const T* S_perm,
    T* dS_perm, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = scalar_to_float(dH_perm[idx]);
        float a = scalar_to_float(A_perm[idx]);
        float s = scalar_to_float(S_perm[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));
        dS_perm[idx] = from_float<T>(dh * a * silu_prime);
    }
}

// dA_gate_perm[i] = dH_perm[i] * SiLU(S_perm[i])
template<typename T>
__global__ void silu_gate_backward_da_kernel(
    const T* dH_perm, const T* S_perm, T* dA_gate_perm, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = scalar_to_float(dH_perm[idx]);
        float s = scalar_to_float(S_perm[idx]);
        float silu_s = s / (1.0f + __expf(-s));
        dA_gate_perm[idx] = from_float<T>(dh * silu_s);
    }
}

// dst[i] += src[i]
template<typename T>
__global__ void add_kernel(T* dst, const T* src, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dst[idx] = from_float<T>(scalar_to_float(dst[idx]) + scalar_to_float(src[idx]));
    }
}

// ---------------------------------------------------------------------------
// Backward: dA + dR
//
// Per group g (gated):
//   Recompute: A_perm = gather(A), AR_perm = A_perm @ R_g^T
//   dH_perm = dC_g @ B_g                     (dense GEMM)
//   dA_gate_perm = dH_perm * SiLU(AR_perm)   (gate path -> dA)
//   dS_perm = dH_perm * A_perm * SiLU'(AR_perm)
//   dA_gemm_perm = dS_perm @ R_g             (block-diagonal)
//   dA_perm = dA_gate_perm + dA_gemm_perm
//   dA += scatter_add(dA_perm, seg_pairs[g])
//   dR_g[b] = dS_perm_b^T @ A_perm_b        (per block)
// ---------------------------------------------------------------------------

template<typename scalar_t>
static std::tuple<torch::Tensor, torch::Tensor> shuffle_backward_dA_dR_impl(
    const scalar_t* dC_ptr, const scalar_t* A_ptr,
    const scalar_t* B_ptr, const scalar_t* R_ptr,
    const int64_t* seg_pairs_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt,
    torch::TensorOptions opts,
    bool gated, cudaStream_t stream)
{
    int64_t n_blocks = k / reconn_sz;
    int64_t seg_sz = reconn_sz / 2;

    auto dA_f32 = torch::zeros({m, k}, opts.dtype(at::kFloat));
    auto dR = torch::zeros({n_groups * reconn_sz, k}, opts);
    auto* dR_ptr = static_cast<scalar_t*>(dR.data_ptr());
    float* dA_f32_ptr = dA_f32.data_ptr<float>();

    // Double-buffer for producer-consumer overlap:
    // Producer (stream_p): gather + AR recompute for group g+1
    // Consumer (stream):   dH dense GEMM + gating grads + scatter for group g
    constexpr int PIPE = 2;
    torch::Tensor A_perm_t[PIPE], AR_perm_t[PIPE], dH_perm_t[PIPE], dS_perm_t[PIPE];
    scalar_t *A_perm_ptr[PIPE], *AR_perm_ptr[PIPE], *dH_perm_ptr[PIPE], *dS_perm_ptr[PIPE];
    for (int p = 0; p < PIPE; ++p) {
        A_perm_t[p]  = torch::empty({m, k}, opts);
        AR_perm_t[p] = torch::empty({m, k}, opts);
        dH_perm_t[p] = torch::empty({m, k}, opts);
        dS_perm_t[p] = torch::empty({m, k}, opts);
        A_perm_ptr[p]  = static_cast<scalar_t*>(A_perm_t[p].data_ptr());
        AR_perm_ptr[p] = static_cast<scalar_t*>(AR_perm_t[p].data_ptr());
        dH_perm_ptr[p] = static_cast<scalar_t*>(dH_perm_t[p].data_ptr());
        dS_perm_ptr[p] = static_cast<scalar_t*>(dS_perm_t[p].data_ptr());
    }

    cudaStream_t stream_p;
    cudaEvent_t recomp_ready[PIPE];
    cudaStreamCreate(&stream_p);
    for (int p = 0; p < PIPE; ++p)
        cudaEventCreate(&recomp_ready[p]);

    cublasHandle_t handle_p;
    CHECK_CUBLAS(cublasCreate(&handle_p));
    CHECK_CUBLAS(cublasSetStream(handle_p, stream_p));
    CHECK_CUBLAS(cublasSetMathMode(handle_p, CUBLAS_DEFAULT_MATH));

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    // Producer: recompute A_perm, AR_perm for group g
    auto recompute = [&](int64_t g, int p) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;
        const scalar_t* R_g = R_ptr + g * reconn_sz * k;

        int64_t numel = m * k;
        int blks = (int)((numel + threads - 1) / threads);
        gather_segments_kernel<scalar_t><<<blks, threads, 0, stream_p>>>(
            A_ptr, A_perm_ptr[p], pairs_g, m, k, n_blocks, seg_sz);

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_g,            dt, (int)k, (long long)reconn_sz,
            A_perm_ptr[p],  dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        cudaEventRecord(recomp_ready[p], stream_p);
    };

    // Consumer: compute gradients for group g using slot p
    auto grad_group = [&](int64_t g, int p) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;
        const scalar_t* dC_g = dC_ptr + g * group_size;
        const scalar_t* B_g  = B_ptr  + g * group_size * k;
        const scalar_t* R_g  = R_ptr  + g * reconn_sz * k;
        scalar_t* dR_g       = dR_ptr + g * reconn_sz * k;

        // Wait for recomputed A_perm, AR_perm
        cudaStreamWaitEvent(stream, recomp_ready[p]);

        // dH_perm = dC_g @ B_g
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)k, (int)m, (int)group_size,
            &alpha,
            B_g,            dt, (int)k,
            dC_g,           dt, (int)n,
            &beta,
            dH_perm_ptr[p], dt, (int)k,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (gated) {
            int64_t numel = m * k;
            int blk = (int)((numel + threads - 1) / threads);

            silu_gate_backward_da_kernel<scalar_t><<<blk, threads, 0, stream>>>(
                dH_perm_ptr[p], AR_perm_ptr[p], dS_perm_ptr[p], numel);

            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            silu_gate_backward_ds_kernel<scalar_t><<<blk, threads, 0, stream>>>(
                dH_perm_ptr[p], A_perm_ptr[p], AR_perm_ptr[p], dS_perm_ptr[p], numel);

            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)reconn_sz, (int)m, (int)reconn_sz,
                &alpha,
                R_g,            dt, (int)k, (long long)reconn_sz,
                dS_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                &beta,
                dH_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dH_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dH_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)reconn_sz, (int)reconn_sz, (int)m,
                &alpha,
                A_perm_ptr[p],  dt, (int)k, (long long)reconn_sz,
                dS_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                &beta,
                dR_g,           dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)reconn_sz, (int)m, (int)reconn_sz,
                &alpha,
                R_g,            dt, (int)k, (long long)reconn_sz,
                dH_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                &beta,
                dS_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            int64_t numel = m * k;
            int blk = (int)((numel + threads - 1) / threads);
            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr[p], dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)reconn_sz, (int)reconn_sz, (int)m,
                &alpha,
                A_perm_ptr[p],  dt, (int)k, (long long)reconn_sz,
                dH_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
                &beta,
                dR_g,           dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    };

    // Pipeline
    for (int64_t g = 0; g < n_groups; ++g) {
        int p = (int)(g % PIPE);

        if (g >= PIPE) {
            cudaEvent_t consumed;
            cudaEventCreate(&consumed);
            cudaEventRecord(consumed, stream);
            cudaStreamWaitEvent(stream_p, consumed);
            cudaEventDestroy(consumed);
        }

        recompute(g, p);

        if (g > 0)
            grad_group(g - 1, (int)((g - 1) % PIPE));
    }
    if (n_groups > 0)
        grad_group(n_groups - 1, (int)((n_groups - 1) % PIPE));

    cudaStreamSynchronize(stream_p);
    CHECK_CUBLAS(cublasDestroy(handle_p));
    for (int p = 0; p < PIPE; ++p)
        cudaEventDestroy(recomp_ready[p]);
    cudaStreamDestroy(stream_p);

    auto dA = dA_f32.to(opts.dtype_opt().value());
    return {dA, dR};
}

// ---------------------------------------------------------------------------
// Backward: dB
//
// Per group g:
//   Recompute: A_perm, AR_perm, H_perm
//   dB_g = dC_g^T @ H_perm  (dense, column order doesn't matter)
// ---------------------------------------------------------------------------

template<typename scalar_t>
static torch::Tensor shuffle_backward_dB_impl(
    const scalar_t* dC_ptr, const scalar_t* A_ptr, const scalar_t* R_ptr,
    const int64_t* seg_pairs_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt,
    torch::TensorOptions opts,
    bool gated, cudaStream_t stream)
{
    int64_t n_blocks = k / reconn_sz;
    int64_t seg_sz = reconn_sz / 2;

    auto dB = torch::zeros({n, k}, opts);
    auto* dB_ptr = static_cast<scalar_t*>(dB.data_ptr());

    // Double-buffer: producer recomputes H_perm, consumer does dB dense GEMM
    constexpr int PIPE = 2;
    torch::Tensor A_perm_t[PIPE], AR_perm_t[PIPE];
    scalar_t *A_perm_ptr[PIPE], *AR_perm_ptr[PIPE];
    for (int p = 0; p < PIPE; ++p) {
        A_perm_t[p]  = torch::empty({m, k}, opts);
        AR_perm_t[p] = torch::empty({m, k}, opts);
        A_perm_ptr[p]  = static_cast<scalar_t*>(A_perm_t[p].data_ptr());
        AR_perm_ptr[p] = static_cast<scalar_t*>(AR_perm_t[p].data_ptr());
    }

    cudaStream_t stream_p;
    cudaEvent_t ready[PIPE];
    cudaStreamCreate(&stream_p);
    for (int p = 0; p < PIPE; ++p)
        cudaEventCreate(&ready[p]);

    cublasHandle_t handle_p;
    CHECK_CUBLAS(cublasCreate(&handle_p));
    CHECK_CUBLAS(cublasSetStream(handle_p, stream_p));
    CHECK_CUBLAS(cublasSetMathMode(handle_p, CUBLAS_DEFAULT_MATH));

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    auto produce = [&](int64_t g, int p) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;
        const scalar_t* R_g = R_ptr + g * reconn_sz * k;

        int64_t numel = m * k;
        int blks = (int)((numel + threads - 1) / threads);
        gather_segments_kernel<scalar_t><<<blks, threads, 0, stream_p>>>(
            A_ptr, A_perm_ptr[p], pairs_g, m, k, n_blocks, seg_sz);

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_g,            dt, (int)k, (long long)reconn_sz,
            A_perm_ptr[p],  dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr[p], dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (gated) {
            silu_gate_kernel<scalar_t><<<blks, threads, 0, stream_p>>>(
                A_perm_ptr[p], AR_perm_ptr[p], numel);
        }

        cudaEventRecord(ready[p], stream_p);
    };

    auto consume = [&](int64_t g, int p) {
        cudaStreamWaitEvent(stream, ready[p]);

        const scalar_t* dC_g = dC_ptr + g * group_size;
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)k, (int)group_size, (int)m,
            &alpha,
            AR_perm_ptr[p], dt, (int)k,
            dC_g,           dt, (int)n,
            &beta,
            dB_ptr + g * group_size * k, dt, (int)k,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    CHECK_CUBLAS(cublasDestroy(handle_p));
    for (int p = 0; p < PIPE; ++p)
        cudaEventDestroy(ready[p]);
    cudaStreamDestroy(stream_p);

    return dB;
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor shuffle_prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && R.is_cuda() && seg_pairs.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && R.is_contiguous(),
                "A, B, R must be contiguous");
    TORCH_CHECK(A.scalar_type() == B.scalar_type() && A.scalar_type() == R.scalar_type(),
                "A, B, R must have same dtype");
    TORCH_CHECK(seg_pairs.scalar_type() == at::kLong, "seg_pairs must be int64");

    int64_t m = A.size(0), k = A.size(1), n = B.size(0);
    int64_t n_groups = n / group_size;

    auto dt = torch_to_cublas(A.scalar_type());
    auto opts = A.options();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    auto C = torch::zeros({m, n}, opts);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    AT_DISPATCH_SWITCH(A.scalar_type(), "shuffle_prism_forward",
        AT_DISPATCH_CASE(at::kHalf,
            [&] { shuffle_prism_forward_impl<scalar_t>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
                R.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
        AT_DISPATCH_CASE(at::kBFloat16,
            [&] { shuffle_prism_forward_impl<scalar_t>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
                R.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
    );

    CHECK_CUBLAS(cublasDestroy(handle));
    return C;
}

std::tuple<torch::Tensor, torch::Tensor> shuffle_prism_backward_dA_dR(
    torch::Tensor dC, torch::Tensor A, torch::Tensor B, torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size, int64_t reconn_sz, bool gated)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && B.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA");

    int64_t m = A.size(0), k = A.size(1), n = B.size(0);
    int64_t n_groups = n / group_size;

    auto dt = torch_to_cublas(A.scalar_type());
    auto opts = A.options();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    std::tuple<torch::Tensor, torch::Tensor> result;

    AT_DISPATCH_SWITCH(A.scalar_type(), "shuffle_backward_dA_dR",
        AT_DISPATCH_CASE(at::kHalf,
            [&] { result = shuffle_backward_dA_dR_impl<scalar_t>(
                dC.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
        AT_DISPATCH_CASE(at::kBFloat16,
            [&] { result = shuffle_backward_dA_dR_impl<scalar_t>(
                dC.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(), R.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
    );

    CHECK_CUBLAS(cublasDestroy(handle));
    return result;
}

torch::Tensor shuffle_prism_backward_dB(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R,
    torch::Tensor seg_pairs,
    int64_t group_size, int64_t reconn_sz, bool gated)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA");

    int64_t m = A.size(0), k = A.size(1), n = dC.size(1);
    int64_t n_groups = n / group_size;

    auto dt = torch_to_cublas(A.scalar_type());
    auto opts = A.options();
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    torch::Tensor result;

    AT_DISPATCH_SWITCH(A.scalar_type(), "shuffle_backward_dB",
        AT_DISPATCH_CASE(at::kHalf,
            [&] { result = shuffle_backward_dB_impl<scalar_t>(
                dC.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                R.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
        AT_DISPATCH_CASE(at::kBFloat16,
            [&] { result = shuffle_backward_dB_impl<scalar_t>(
                dC.data_ptr<scalar_t>(), A.data_ptr<scalar_t>(),
                R.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
    );

    CHECK_CUBLAS(cublasDestroy(handle));
    return result;
}
