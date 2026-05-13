#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cute_prism_coop_pc.hpp"
#include "cute_prism_backward_db.hpp"
#include "prism_shuffle.cuh"

#if PRISM_DTYPE == 0
constexpr c10::ScalarType k_prism_torch_dtype = torch::kHalf;
constexpr const char* k_prism_dtype_name = "float16";
#define PRISM_CONST_DATA(T) reinterpret_cast<const prism_native*>((T).data_ptr<at::Half>())
#define PRISM_DATA(T)       reinterpret_cast<prism_native*>((T).data_ptr<at::Half>())
#else
constexpr c10::ScalarType k_prism_torch_dtype = torch::kBFloat16;
constexpr const char* k_prism_dtype_name = "bfloat16";
#define PRISM_CONST_DATA(T) reinterpret_cast<const prism_native*>((T).data_ptr<at::BFloat16>())
#define PRISM_DATA(T)       reinterpret_cast<prism_native*>((T).data_ptr<at::BFloat16>())
#endif

static void validate_backward_inputs(
    torch::Tensor dC, torch::Tensor A, torch::Tensor R,
    int64_t group_size, int64_t reconn_sz)
{
    TORCH_CHECK(dC.is_cuda() && A.is_cuda() && R.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(dC.scalar_type() == k_prism_torch_dtype && A.scalar_type() == k_prism_torch_dtype &&
                R.scalar_type() == k_prism_torch_dtype,
                "All tensors must be ", k_prism_dtype_name);
    TORCH_CHECK(dC.is_contiguous() && A.is_contiguous() && R.is_contiguous(),
                "All tensors must be contiguous");
    TORCH_CHECK(A.device() == R.device() && A.device() == dC.device(),
                "All tensors must be on the same CUDA device");
    TORCH_CHECK(group_size == CurrKernelParams::group_size,
                "group_size mismatch with compiled kernel");
    TORCH_CHECK(reconn_sz == CurrKernelParams::reconn_sz,
                "reconn_sz mismatch with compiled kernel");
}

torch::Tensor prism_backward_dB(
    torch::Tensor dC,
    torch::Tensor A,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> shuffle_masks,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p)
{
    validate_backward_inputs(dC, A, R, group_size, reconn_sz);

    // Pin the current CUDA device to A's device for the duration of this call.
    // Same reasoning as the dA/dR binding: kernel launch helpers may call
    // cudaMalloc, which uses the current device.
    at::cuda::CUDAGuard device_guard(A.device());

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = dC.size(1);
    int64_t n_groups = n / group_size;

    const prism_native* bias_ptr = nullptr;
    if (CurrKernelParams::internal_bias) {
        TORCH_CHECK(internal_bias.has_value(),
                    "Kernel was compiled with PRISM_INTERNAL_BIAS=1 but no internal_bias tensor was passed");
        const auto& bias = internal_bias.value();
        TORCH_CHECK(bias.is_cuda() && bias.device() == A.device(),
                    "internal_bias must be on the same CUDA device as A");
        TORCH_CHECK(bias.scalar_type() == k_prism_torch_dtype,
                    "internal_bias must be ", k_prism_dtype_name);
        TORCH_CHECK(bias.is_contiguous(), "internal_bias must be contiguous (row-major)");
        TORCH_CHECK(bias.dim() == 2 && bias.size(0) == n_groups && bias.size(1) == k,
                    "internal_bias must have shape (n_groups, K)");
        bias_ptr = PRISM_CONST_DATA(bias);
    } else {
        TORCH_CHECK(!internal_bias.has_value(),
                    "Kernel was compiled with PRISM_INTERNAL_BIAS=0 but an internal_bias tensor was passed");
    }

    auto dB = torch::zeros({n, k}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Optional shuffle: build n_groups × (M, K) A_perm buffer.
    const prism_native* A_kernel_ptr = PRISM_CONST_DATA(A);
    int strideA_per_group = 0;
    torch::Tensor A_perm_buf;
    if (shuffle_masks.has_value()) {
        const auto& masks = shuffle_masks.value();
        TORCH_CHECK(masks.is_cuda() && masks.device() == A.device(),
                    "shuffle_masks must be on the same CUDA device as A");
        TORCH_CHECK(masks.scalar_type() == torch::kInt64, "shuffle_masks must be int64");
        TORCH_CHECK(masks.is_contiguous(), "shuffle_masks must be contiguous");
        TORCH_CHECK(masks.dim() == 3 && masks.size(0) == n_groups && masks.size(2) == 2,
                    "shuffle_masks must be (n_groups, n_chunks, 2)");
        int64_t n_chunks = masks.size(1);
        int64_t n_rounds = masks.size(2);
        TORCH_CHECK(n_chunks * 64 == k,
                    "shuffle expects n_chunks*64 == K");

        auto opts = torch::TensorOptions().dtype(k_prism_torch_dtype).device(A.device());
        A_perm_buf = torch::empty({n_groups, m, k}, opts);
        prism_native* A_perm_ptr = PRISM_DATA(A_perm_buf);
        const int64_t* masks_ptr = masks.data_ptr<int64_t>();

        int threads = 256;
        int64_t total_warps = m * n_chunks;
        int blocks = (int)((total_warps * 32 + threads - 1) / threads);
        for (int64_t g = 0; g < n_groups; ++g) {
            prism_shuffle::shuffle_forward_kernel<prism_native><<<blocks, threads, 0, stream>>>(
                PRISM_CONST_DATA(A),
                A_perm_ptr + g * m * k,
                masks_ptr + g * n_chunks * n_rounds,
                m, k, n_chunks, n_rounds);
        }
        A_kernel_ptr = A_perm_ptr;
        strideA_per_group = (int)(m * k);
    }

    // Dropout params (only consumed when PRISM_DROPOUT=1).
    const int64_t* dropout_seeds_ptr = nullptr;
    float dropout_p_f = 0.0f;
    float inv_keep_f = 1.0f;
#if PRISM_DROPOUT
    TORCH_CHECK(dropout_seeds.has_value(),
                "Kernel was compiled with PRISM_DROPOUT=1 but no dropout_seeds tensor was passed");
    const auto& seeds_t = dropout_seeds.value();
    TORCH_CHECK(seeds_t.is_cuda() && seeds_t.scalar_type() == torch::kInt64
                && seeds_t.is_contiguous() && seeds_t.dim() == 1 && seeds_t.size(0) == n_groups,
                "dropout_seeds must be CUDA int64 contiguous shape (n_groups,)");
    TORCH_CHECK(dropout_p > 0.0 && dropout_p < 1.0, "dropout_p must be in (0, 1)");
    dropout_seeds_ptr = seeds_t.data_ptr<int64_t>();
    dropout_p_f = (float)dropout_p;
    inv_keep_f = 1.0f / (1.0f - dropout_p_f);
#else
    TORCH_CHECK(!dropout_seeds.has_value(),
                "Kernel was compiled with PRISM_DROPOUT=0 but dropout_seeds was passed");
#endif

    prism_backward_dB_launch(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        PRISM_CONST_DATA(dC),
        static_cast<int>(dC.stride(0)),
        A_kernel_ptr,
        static_cast<int>(A.stride(0)),
        PRISM_CONST_DATA(R),
        static_cast<int>(R.stride(0)),
        bias_ptr,
        PRISM_DATA(dB),
        static_cast<int>(dB.stride(0)),
        strideA_per_group,
        dropout_seeds_ptr, dropout_p_f, inv_keep_f,
        stream
    );

    return dB;
}
