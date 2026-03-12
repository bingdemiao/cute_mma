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

// AR mode: C = (A @ R^T) @ B^T, computed per group
static torch::Tensor cublas_oft_ar(
    const half* A_ptr, const half* B_ptr, const half* R_ptr, half* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle,
    torch::TensorOptions opts)
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

torch::Tensor cublas_oft_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool rw_mode)
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

    if (rw_mode) {
        cublas_oft_rw(A_ptr, B_ptr, R_ptr, C_ptr,
                      m, n, k, group_size, reconn_sz, n_groups,
                      handle, A.options());
    } else {
        cublas_oft_ar(A_ptr, B_ptr, R_ptr, C_ptr,
                      m, n, k, group_size, reconn_sz, n_groups,
                      handle, A.options());
    }

    GEMM_CHECK_CUBLAS(cublasDestroy(handle));
    return C;
}
