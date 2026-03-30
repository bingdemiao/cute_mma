#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

// ===========================================================================
// Shuffle OFT forward kernel.
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
static void shuffle_oft_forward_impl(
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

    // A_perm: gathered A for the current group
    // AR_perm: result of block-diagonal GEMM, then becomes H_perm after gating
    auto A_perm_t = torch::empty({m, k}, opts);
    auto AR_perm_t = torch::empty({m, k}, opts);
    auto* A_perm_ptr = static_cast<scalar_t*>(A_perm_t.data_ptr());
    auto* AR_perm_ptr = static_cast<scalar_t*>(AR_perm_t.data_ptr());

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;

        // 1. Gather A segments into permuted layout
        {
            int64_t numel = m * k;
            int blocks = (int)((numel + threads - 1) / threads);
            gather_segments_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A_ptr, A_perm_ptr, pairs_g, m, k, n_blocks, seg_sz);
        }

        // 2. Block-diagonal GEMM: AR_perm = A_perm @ R_g^T
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_ptr + g * reconn_sz * k, dt, (int)k, (long long)reconn_sz,
            A_perm_ptr,                dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr,               dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // 3. SiLU gating: AR_perm becomes H_perm = A_perm * SiLU(AR_perm)
        if (gated) {
            int64_t numel = m * k;
            int blocks = (int)((numel + threads - 1) / threads);
            silu_gate_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A_perm_ptr, AR_perm_ptr, numel);
        }

        // 4. Dense GEMM: C_g = H_perm @ B_g^T
        //    Full reduction over K — column permutation doesn't affect the sum.
        scalar_t* H_ptr = gated ? AR_perm_ptr : AR_perm_ptr;

        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)group_size, (int)m, (int)k,
            &alpha,
            B_ptr + g * group_size * k, dt, (int)k,
            H_ptr,                      dt, (int)k,
            &beta,
            C_ptr + g * group_size,     dt, (int)n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
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

    // dA accumulated in f32 for precision (scatter-add from multiple groups)
    auto dA_f32 = torch::zeros({m, k}, opts.dtype(at::kFloat));
    auto dR = torch::zeros({n_groups * reconn_sz, k}, opts);

    auto A_perm_t  = torch::empty({m, k}, opts);
    auto AR_perm_t = torch::empty({m, k}, opts);
    auto dH_perm_t = torch::empty({m, k}, opts);
    auto dS_perm_t = torch::empty({m, k}, opts);  // reused for dA_perm

    auto* A_perm_ptr  = static_cast<scalar_t*>(A_perm_t.data_ptr());
    auto* AR_perm_ptr = static_cast<scalar_t*>(AR_perm_t.data_ptr());
    auto* dH_perm_ptr = static_cast<scalar_t*>(dH_perm_t.data_ptr());
    auto* dS_perm_ptr = static_cast<scalar_t*>(dS_perm_t.data_ptr());
    auto* dR_ptr      = static_cast<scalar_t*>(dR.data_ptr());
    float* dA_f32_ptr = dA_f32.data_ptr<float>();

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;
        const scalar_t* dC_g = dC_ptr + g * group_size;
        const scalar_t* B_g  = B_ptr  + g * group_size * k;
        const scalar_t* R_g  = R_ptr  + g * reconn_sz * k;
        scalar_t* dR_g       = dR_ptr + g * reconn_sz * k;

        // Recompute: gather A, compute AR_perm
        {
            int64_t numel = m * k;
            int blocks = (int)((numel + threads - 1) / threads);
            gather_segments_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A_ptr, A_perm_ptr, pairs_g, m, k, n_blocks, seg_sz);
        }

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_g,         dt, (int)k, (long long)reconn_sz,
            A_perm_ptr,  dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr, dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        // dH_perm = dC_g @ B_g  (dense, B in natural order = permuted for H_perm)
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            (int)k, (int)m, (int)group_size,
            &alpha,
            B_g,         dt, (int)k,
            dC_g,        dt, (int)n,
            &beta,
            dH_perm_ptr, dt, (int)k,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (gated) {
            int64_t numel = m * k;
            int blk = (int)((numel + threads - 1) / threads);

            // dA_gate_perm = dH_perm * SiLU(AR_perm) → store in dS_perm temp
            silu_gate_backward_da_kernel<scalar_t><<<blk, threads, 0, stream>>>(
                dH_perm_ptr, AR_perm_ptr, dS_perm_ptr, numel);

            // scatter-add dA_gate_perm to dA_f32
            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            // dS_perm = dH_perm * A_perm * SiLU'(AR_perm)
            silu_gate_backward_ds_kernel<scalar_t><<<blk, threads, 0, stream>>>(
                dH_perm_ptr, A_perm_ptr, AR_perm_ptr, dS_perm_ptr, numel);

            // dA_gemm_perm = dS_perm @ R_g (block-diagonal) → store in dH_perm
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)reconn_sz, (int)m, (int)reconn_sz,
                &alpha,
                R_g,         dt, (int)k, (long long)reconn_sz,
                dS_perm_ptr, dt, (int)k, (long long)reconn_sz,
                &beta,
                dH_perm_ptr, dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // scatter-add dA_gemm_perm to dA_f32
            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dH_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dH_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            // dR_g[b] = dS_perm_b^T @ A_perm_b (per block)
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)reconn_sz, (int)reconn_sz, (int)m,
                &alpha,
                A_perm_ptr,  dt, (int)k, (long long)reconn_sz,
                dS_perm_ptr, dt, (int)k, (long long)reconn_sz,
                &beta,
                dR_g,        dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else {
            // Non-gated: dH_perm is already dAR_perm
            // dA_perm = dAR_perm @ R_g
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)reconn_sz, (int)m, (int)reconn_sz,
                &alpha,
                R_g,         dt, (int)k, (long long)reconn_sz,
                dH_perm_ptr, dt, (int)k, (long long)reconn_sz,
                &beta,
                dS_perm_ptr, dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            // scatter-add to dA_f32
            int64_t numel = m * k;
            int blk = (int)((numel + threads - 1) / threads);
            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                scatter_add_segments_f32_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            } else {
                scatter_add_segments_f32_bf16_kernel<<<blk, threads, 0, stream>>>(
                    dS_perm_ptr, dA_f32_ptr, pairs_g, m, k, n_blocks, seg_sz);
            }

            // dR_g[b] = dAR_perm_b^T @ A_perm_b
            CHECK_CUBLAS(cublasGemmStridedBatchedEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)reconn_sz, (int)reconn_sz, (int)m,
                &alpha,
                A_perm_ptr,  dt, (int)k, (long long)reconn_sz,
                dH_perm_ptr, dt, (int)k, (long long)reconn_sz,
                &beta,
                dR_g,        dt, (int)k, (long long)reconn_sz,
                (int)n_blocks,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

    // Convert dA back to compute dtype
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
    auto A_perm_t  = torch::empty({m, k}, opts);
    auto AR_perm_t = torch::empty({m, k}, opts);

    auto* dB_ptr      = static_cast<scalar_t*>(dB.data_ptr());
    auto* A_perm_ptr  = static_cast<scalar_t*>(A_perm_t.data_ptr());
    auto* AR_perm_ptr = static_cast<scalar_t*>(AR_perm_t.data_ptr());

    int threads = 256;
    float alpha = 1.0f, beta = 0.0f;

    for (int64_t g = 0; g < n_groups; ++g) {
        const int64_t* pairs_g = seg_pairs_ptr + g * n_blocks * 2;
        const scalar_t* dC_g = dC_ptr + g * group_size;
        const scalar_t* R_g  = R_ptr  + g * reconn_sz * k;

        // Recompute: gather A, block-diagonal GEMM, gating
        {
            int64_t numel = m * k;
            int blocks = (int)((numel + threads - 1) / threads);
            gather_segments_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A_ptr, A_perm_ptr, pairs_g, m, k, n_blocks, seg_sz);
        }

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            (int)reconn_sz, (int)m, (int)reconn_sz,
            &alpha,
            R_g,         dt, (int)k, (long long)reconn_sz,
            A_perm_ptr,  dt, (int)k, (long long)reconn_sz,
            &beta,
            AR_perm_ptr, dt, (int)k, (long long)reconn_sz,
            (int)n_blocks,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        if (gated) {
            int64_t numel = m * k;
            int blocks = (int)((numel + threads - 1) / threads);
            silu_gate_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                A_perm_ptr, AR_perm_ptr, numel);
        }

        // H_perm is in AR_perm_ptr after gating
        // dB_g = dC_g^T @ H_perm
        CHECK_CUBLAS(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            (int)k, (int)group_size, (int)m,
            &alpha,
            AR_perm_ptr, dt, (int)k,
            dC_g,        dt, (int)n,
            &beta,
            dB_ptr + g * group_size * k, dt, (int)k,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    return dB;
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

torch::Tensor shuffle_oft_forward(
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

    AT_DISPATCH_SWITCH(A.scalar_type(), "shuffle_oft_forward",
        AT_DISPATCH_CASE(at::kHalf,
            [&] { shuffle_oft_forward_impl<scalar_t>(
                A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
                R.data_ptr<scalar_t>(), C.data_ptr<scalar_t>(),
                seg_pairs.data_ptr<int64_t>(),
                m, n, k, group_size, reconn_sz, n_groups,
                handle, dt, opts, gated, stream);
            })
        AT_DISPATCH_CASE(at::kBFloat16,
            [&] { shuffle_oft_forward_impl<scalar_t>(
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

std::tuple<torch::Tensor, torch::Tensor> shuffle_oft_backward_dA_dR(
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

torch::Tensor shuffle_oft_backward_dB(
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
