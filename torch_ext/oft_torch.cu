#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "cute_oft_coop_pc.hpp"

torch::Tensor oft_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz)
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

    // Validate shape consistency
    TORCH_CHECK(B.size(1) == k, "K dimension mismatch: A has K=", k, " but B has K=", B.size(1));
    TORCH_CHECK(R.size(1) == k, "K dimension mismatch: A has K=", k, " but R has K=", R.size(1));

    // Validate group/reconnect parameters
    TORCH_CHECK(n % group_size == 0,
                "N (", n, ") must be divisible by group_size (", group_size, ")");
    int64_t n_groups = n / group_size;
    TORCH_CHECK(R.size(0) == n_groups * reconn_sz,
                "R must have shape (n_groups * reconn_sz, K) = (", n_groups * reconn_sz,
                ", ", k, "), but got (", R.size(0), ", ", R.size(1), ")");
    TORCH_CHECK(k % reconn_sz == 0,
                "K (", k, ") must be divisible by reconn_sz (", reconn_sz, ")");

    // Validate against compiled constants
    TORCH_CHECK(group_size == CurrKernelParams::group_size,
                "group_size (", group_size, ") must match compiled OFT_GROUP_SIZE (",
                CurrKernelParams::group_size, "). Rebuild with -DGROUP_SIZE=", group_size);
    TORCH_CHECK(reconn_sz == CurrKernelParams::reconn_sz,
                "reconn_sz (", reconn_sz, ") must match compiled OFT_RECONN_SIZE (",
                CurrKernelParams::reconn_sz, "). Rebuild with -DRECONN_SIZE=", reconn_sz);

    // Allocate output
    auto C = torch::zeros({m, n}, A.options());

    // Extract strides (leading dimensions for row-major layout)
    int ldA = A.stride(0);  // == K for contiguous
    int ldB = B.stride(0);  // == K for contiguous
    int ldR = R.stride(0);  // == K for contiguous
    int ldC = C.stride(0);  // == N for contiguous

    // Get current CUDA stream from PyTorch
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Launch the kernel
    oft_tn<CurrKernelParams>(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        ldA,
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        ldB,
        reinterpret_cast<const half*>(R.data_ptr<at::Half>()),
        ldR,
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        ldC,
        stream
    );

    return C;
}
