#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "cute_oft_coop_pc.hpp"
#include "cute_oft_backward_db.hpp"

static void validate_backward_inputs(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R,
    int64_t group_size, int64_t reconn_sz)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(dC.scalar_type() == torch::kHalf && A.scalar_type() == torch::kHalf &&
                R.scalar_type() == torch::kHalf,
                "All tensors must be float16");
    TORCH_CHECK(dC.is_contiguous() && A.is_contiguous() && R.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(A.device() == R.device() && A.device() == dC.device(),
                "All tensors must be on the same CUDA device");
    TORCH_CHECK(group_size == CurrKernelParams::group_size,
                "group_size mismatch with compiled kernel");
    TORCH_CHECK(reconn_sz == CurrKernelParams::reconn_sz,
                "reconn_sz mismatch with compiled kernel");
}

torch::Tensor oft_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    bool gated)
{
    validate_backward_inputs(dC, A, R, group_size, reconn_sz);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = dC.size(1);
    int64_t n_groups = n / group_size;

    auto dB = torch::zeros({n, k}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    oft_backward_dB_launch(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        reinterpret_cast<const half*>(dC.data_ptr<at::Half>()),
        static_cast<int>(dC.stride(0)),
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        static_cast<int>(A.stride(0)),
        reinterpret_cast<const half*>(R.data_ptr<at::Half>()),
        static_cast<int>(R.stride(0)),
        reinterpret_cast<half*>(dB.data_ptr<at::Half>()),
        static_cast<int>(dB.stride(0)),
        gated,
        stream
    );

    return dB;
}
