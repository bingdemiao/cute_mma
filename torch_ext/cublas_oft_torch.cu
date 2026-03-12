#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error("cuBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)

// Element-wise kernel: out[i] = A[i] * silu(AR[i])
// where silu(x) = x / (1 + exp(-x))
__global__ void silu_gate_kernel(const half* A, half* AR, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float a = __half2float(A[idx]);
        float ar = __half2float(AR[idx]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        AR[idx] = __float2half(a * silu_ar);
    }
}

// AR mode: C = (A @ R^T) @ B^T, computed per group
// With gated=true: C = (A * SiLU(A @ R^T)) @ B^T
static torch::Tensor cublas_oft_ar(
    const half* A_ptr, const half* B_ptr, const half* R_ptr, half* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle,
    torch::TensorOptions opts,
    bool gated,
    cudaStream_t stream)
{
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    auto AR = torch::empty({m, k}, opts);
    half* AR_ptr = reinterpret_cast<half*>(AR.data_ptr<at::Half>());

    for (int64_t i = 0; i < n_groups; ++i) {
        // Batched: AR_block = A_block @ R_block^T over K/reconn_sz blocks
        GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz, &alpha,
            R_ptr + i * reconn_sz * k, k,
            reconn_sz,
            A_ptr, k,
            reconn_sz,
            &beta,
            AR_ptr, k,
            reconn_sz,
            k / reconn_sz
        ));

        // Apply gating: AR = A * SiLU(AR)
        if (gated) {
            int64_t numel = m * k;
            int threads = 256;
            int blocks = (numel + threads - 1) / threads;
            silu_gate_kernel<<<blocks, threads, 0, stream>>>(A_ptr, AR_ptr, numel);
        }

        // C_group = AR @ B_group^T
        GEMM_CHECK_CUBLAS(cublasHgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            group_size, m, k, &alpha,
            B_ptr + i * group_size * k, k,
            AR_ptr, k,
            &beta,
            C_ptr + i * group_size, n
        ));
    }

    return {};  // unused, writes directly to C_ptr
}

// RW mode: C = A @ (R @ B)^T, computed by transforming B first
static void cublas_oft_rw(
    const half* A_ptr, const half* B_ptr, const half* R_ptr, half* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle,
    torch::TensorOptions opts)
{
    half alpha = __float2half(1.0f);
    half beta = __float2half(0.0f);

    // Transform B: B'[i*gs:(i+1)*gs, :] = R_i @ B[i*gs:(i+1)*gs, :]
    // Done block-wise: B'_block = R_block @ B_block over K/reconn_sz blocks per group
    auto Bp = torch::empty({n, k}, opts);
    half* Bp_ptr = reinterpret_cast<half*>(Bp.data_ptr<at::Half>());

    for (int64_t i = 0; i < n_groups; ++i) {
        GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            reconn_sz, group_size, reconn_sz, &alpha,
            R_ptr + i * reconn_sz * k, k,
            reconn_sz,
            B_ptr + i * group_size * k, k,
            reconn_sz,
            &beta,
            Bp_ptr + i * group_size * k, k,
            reconn_sz,
            k / reconn_sz
        ));
    }

    // C = A @ B'^T
    GEMM_CHECK_CUBLAS(cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha,
        Bp_ptr, k,
        A_ptr, k,
        &beta,
        C_ptr, n
    ));
}

// Element-wise kernel: dst[i] += src[i]
__global__ void half_add_kernel(half* dst, const half* src, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
    }
}

// Element-wise kernel for gated backward:
// dS[i] = dH[i] * A[i] * silu_prime(S[i])
// where silu_prime(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
__global__ void silu_gate_backward_ds_kernel(
    const half* dH, const half* A, const half* S, half* dS, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = __half2float(dH[idx]);
        float a = __half2float(A[idx]);
        float s = __half2float(S[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));
        dS[idx] = __float2half(dh * a * silu_prime);
    }
}

// Element-wise kernel for gated backward dA contribution:
// dA_gate[i] = dH[i] * silu(S[i])
// where silu(x) = x * sigmoid(x)
__global__ void silu_gate_backward_da_kernel(
    const half* dH, const half* S, half* dA_gate, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = __half2float(dH[idx]);
        float s = __half2float(S[idx]);
        float silu_s = s / (1.0f + __expf(-s));
        dA_gate[idx] = __float2half(dh * silu_s);
    }
}

// Backward pass: compute dA and dR
static std::tuple<torch::Tensor, torch::Tensor> cublas_backward_dA_dR_impl(
    const half* dC_ptr, const half* A_ptr, const half* B_ptr, const half* R_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle,
    torch::TensorOptions opts,
    bool gated,
    cudaStream_t stream)
{
    half alpha = __float2half(1.0f);
    half beta_zero = __float2half(0.0f);
    half beta_one = __float2half(1.0f);

    auto dA = torch::zeros({m, k}, opts);
    auto dR = torch::zeros({n_groups * reconn_sz, k}, opts);
    auto AR = torch::empty({m, k}, opts);

    half* dA_ptr = reinterpret_cast<half*>(dA.data_ptr<at::Half>());
    half* dR_ptr = reinterpret_cast<half*>(dR.data_ptr<at::Half>());
    half* AR_ptr = reinterpret_cast<half*>(AR.data_ptr<at::Half>());

    auto dAR = torch::empty({m, k}, opts);
    half* dAR_ptr = reinterpret_cast<half*>(dAR.data_ptr<at::Half>());

    int threads = 256;

    for (int64_t g = 0; g < n_groups; ++g) {
        const half* dC_g = dC_ptr + g * group_size;
        const half* B_g = B_ptr + g * group_size * k;
        const half* R_g = R_ptr + g * reconn_sz * k;

        // Recompute AR_g = A @ R_g^T
        GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz, &alpha,
            R_g, k, reconn_sz,
            A_ptr, k, reconn_sz,
            &beta_zero,
            AR_ptr, k, reconn_sz,
            k / reconn_sz
        ));

        if (gated) {
            // dH_g = dC_g @ B_g
            GEMM_CHECK_CUBLAS(cublasHgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, m, group_size, &alpha,
                B_g, k,
                dC_g, n,
                &beta_zero,
                dAR_ptr, k
            ));

            {
                int64_t numel = m * k;
                int blocks = (numel + threads - 1) / threads;
                // dA_gate = dH_g * SiLU(S_g), stored in temp
                auto temp = torch::empty({m, k}, opts);
                half* temp_ptr = reinterpret_cast<half*>(temp.data_ptr<at::Half>());
                silu_gate_backward_da_kernel<<<blocks, threads, 0, stream>>>(
                    dAR_ptr, AR_ptr, temp_ptr, numel);
                half_add_kernel<<<blocks, threads, 0, stream>>>(dA_ptr, temp_ptr, numel);

                // dS_g = dH_g * A * silu_prime(S_g)
                silu_gate_backward_ds_kernel<<<blocks, threads, 0, stream>>>(
                    dAR_ptr, A_ptr, AR_ptr, dAR_ptr, numel);
            }

            // dA += dS_g @ R_g
            GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz, &alpha,
                R_g, k, reconn_sz,
                dAR_ptr, k, reconn_sz,
                &beta_one,
                dA_ptr, k, reconn_sz,
                k / reconn_sz
            ));

            // dR_g[b] = dS_g^T @ A
            GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m, &alpha,
                A_ptr, k, reconn_sz,
                dAR_ptr, k, reconn_sz,
                &beta_zero,
                dR_ptr + g * reconn_sz * k, k, reconn_sz,
                k / reconn_sz
            ));
        } else {
            // dAR_g = dC_g @ B_g
            GEMM_CHECK_CUBLAS(cublasHgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, m, group_size, &alpha,
                B_g, k,
                dC_g, n,
                &beta_zero,
                dAR_ptr, k
            ));

            // dA += dAR_g @ R_g
            GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz, &alpha,
                R_g, k, reconn_sz,
                dAR_ptr, k, reconn_sz,
                &beta_one,
                dA_ptr, k, reconn_sz,
                k / reconn_sz
            ));

            // dR_g[b] = dAR_g^T @ A
            GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m, &alpha,
                A_ptr, k, reconn_sz,
                dAR_ptr, k, reconn_sz,
                &beta_zero,
                dR_ptr + g * reconn_sz * k, k, reconn_sz,
                k / reconn_sz
            ));
        }
    }

    return {dA, dR};
}

// Backward pass: compute dB
static torch::Tensor cublas_backward_dB_impl(
    const half* dC_ptr, const half* A_ptr, const half* R_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle,
    torch::TensorOptions opts,
    bool gated,
    cudaStream_t stream)
{
    half alpha = __float2half(1.0f);
    half beta_zero = __float2half(0.0f);

    auto dB = torch::zeros({n, k}, opts);
    auto AR = torch::empty({m, k}, opts);

    half* dB_ptr = reinterpret_cast<half*>(dB.data_ptr<at::Half>());
    half* AR_ptr = reinterpret_cast<half*>(AR.data_ptr<at::Half>());

    int threads = 256;

    for (int64_t g = 0; g < n_groups; ++g) {
        const half* dC_g = dC_ptr + g * group_size;
        const half* R_g = R_ptr + g * reconn_sz * k;

        // Recompute AR_g = A @ R_g^T
        GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz, &alpha,
            R_g, k, reconn_sz,
            A_ptr, k, reconn_sz,
            &beta_zero,
            AR_ptr, k, reconn_sz,
            k / reconn_sz
        ));

        if (gated) {
            int64_t numel = m * k;
            int blocks = (numel + threads - 1) / threads;
            // AR = A * SiLU(AR) in-place
            silu_gate_kernel<<<blocks, threads, 0, stream>>>(A_ptr, AR_ptr, numel);
        }

        // dB_g = dC_g^T @ AR_g
        GEMM_CHECK_CUBLAS(cublasHgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            k, group_size, m, &alpha,
            AR_ptr, k,
            dC_g, n,
            &beta_zero,
            dB_ptr + g * group_size * k, k
        ));
    }

    return dB;
}


torch::Tensor cublas_oft_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode,
    bool gated)
{
    // Validate devices
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(R.is_cuda(), "R must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device() && A.device() == R.device(),
                "All tensors must be on the same CUDA device");

    // Validate dtypes
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kHalf, "B must be float16");
    TORCH_CHECK(R.scalar_type() == torch::kHalf, "R must be float16");

    // Validate contiguity
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous (row-major)");
    TORCH_CHECK(R.is_contiguous(), "R must be contiguous (row-major)");

    // Validate dimensions
    TORCH_CHECK(A.dim() == 2, "A must be 2D (M, K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (N, K)");
    TORCH_CHECK(R.dim() == 2, "R must be 2D (n_groups * reconn_sz, K)");

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);

    TORCH_CHECK(B.size(1) == k, "K dimension mismatch: A has K=", k, " but B has K=", B.size(1));
    TORCH_CHECK(R.size(1) == k, "K dimension mismatch: A has K=", k, " but R has K=", R.size(1));
    TORCH_CHECK(n % group_size == 0,
                "N (", n, ") must be divisible by group_size (", group_size, ")");
    int64_t n_groups = n / group_size;
    TORCH_CHECK(R.size(0) == n_groups * reconn_sz,
                "R must have shape (n_groups * reconn_sz, K) = (", n_groups * reconn_sz,
                ", ", k, "), but got (", R.size(0), ", ", R.size(1), ")");
    TORCH_CHECK(k % reconn_sz == 0,
                "K (", k, ") must be divisible by reconn_sz (", reconn_sz, ")");

    // Allocate output
    auto C = torch::zeros({m, n}, A.options());

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Create cuBLAS handle on the current stream
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    const half* A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    const half* R_ptr = reinterpret_cast<const half*>(R.data_ptr<at::Half>());
    half* C_ptr = reinterpret_cast<half*>(C.data_ptr<at::Half>());

    TORCH_CHECK(!gated || !rw_mode, "Gated activation is only supported in AR mode, not RW mode");

    if (rw_mode) {
        cublas_oft_rw(A_ptr, B_ptr, R_ptr, C_ptr,
                      m, n, k, group_size, reconn_sz, n_groups,
                      handle, A.options());
    } else {
        cublas_oft_ar(A_ptr, B_ptr, R_ptr, C_ptr,
                      m, n, k, group_size, reconn_sz, n_groups,
                      handle, A.options(), gated, stream);
    }

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return C;
}


static void validate_cublas_backward_inputs(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(dC.scalar_type() == torch::kHalf && A.scalar_type() == torch::kHalf &&
                R.scalar_type() == torch::kHalf,
                "All tensors must be float16");
    TORCH_CHECK(dC.is_contiguous() && A.is_contiguous() && R.is_contiguous(),
                "All tensors must be contiguous");
}

std::vector<torch::Tensor> cublas_backward_dA_dR(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated)
{
    validate_cublas_backward_inputs(dC, A, R);
    TORCH_CHECK(B.is_cuda() && B.scalar_type() == torch::kHalf && B.is_contiguous(),
                "B must be a contiguous CUDA float16 tensor");

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);
    int64_t n_groups = n / group_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    const half* dC_ptr = reinterpret_cast<const half*>(dC.data_ptr<at::Half>());
    const half* A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* B_ptr = reinterpret_cast<const half*>(B.data_ptr<at::Half>());
    const half* R_ptr = reinterpret_cast<const half*>(R.data_ptr<at::Half>());

    auto [dA, dR] = cublas_backward_dA_dR_impl(
        dC_ptr, A_ptr, B_ptr, R_ptr,
        m, n, k, group_size, reconn_sz, n_groups,
        handle, A.options(), gated, stream);

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return {dA, dR};
}

torch::Tensor cublas_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated)
{
    validate_cublas_backward_inputs(dC, A, R);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = dC.size(1);
    int64_t n_groups = n / group_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
    cublasHandle_t handle;
    GEMM_CHECK_CUBLAS(cublasCreate(&handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    GEMM_CHECK_CUBLAS(cublasSetStream(handle, stream));

    const half* dC_ptr = reinterpret_cast<const half*>(dC.data_ptr<at::Half>());
    const half* A_ptr = reinterpret_cast<const half*>(A.data_ptr<at::Half>());
    const half* R_ptr = reinterpret_cast<const half*>(R.data_ptr<at::Half>());

    auto dB = cublas_backward_dB_impl(
        dC_ptr, A_ptr, R_ptr,
        m, n, k, group_size, reconn_sz, n_groups,
        handle, A.options(), gated, stream);

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return dB;
}
