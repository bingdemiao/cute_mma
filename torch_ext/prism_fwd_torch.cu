#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cute_prism_coop_pc.hpp"
#include "prism_shuffle.cuh"

// Compile-time scalar dtype expected by this binary (set via PRISM_DTYPE in
// cute_prism_dtype.hpp). 0 → fp16 (kHalf), 1 → bf16 (kBFloat16).
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

torch::Tensor prism_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor R,
    int64_t group_size,
    int64_t reconn_sz,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> shuffle_masks,
    // dropout (only used when PRISM_DROPOUT=1)
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p)
{
    // Validate devices
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(R.is_cuda(), "R must be a CUDA tensor");
    TORCH_CHECK(A.device() == B.device() && A.device() == R.device(),
                "All tensors must be on the same CUDA device");

    // Validate dtypes (must match the dtype this binary was compiled for)
    TORCH_CHECK(A.scalar_type() == k_prism_torch_dtype, "A must be ", k_prism_dtype_name);
    TORCH_CHECK(B.scalar_type() == k_prism_torch_dtype, "B must be ", k_prism_dtype_name);
    TORCH_CHECK(R.scalar_type() == k_prism_torch_dtype, "R must be ", k_prism_dtype_name);

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
                "group_size (", group_size, ") must match compiled PRISM_GROUP_SIZE (",
                CurrKernelParams::group_size, "). Rebuild with -DGROUP_SIZE=", group_size);
    TORCH_CHECK(reconn_sz == CurrKernelParams::reconn_sz,
                "reconn_sz (", reconn_sz, ") must match compiled PRISM_RECONN_SIZE (",
                CurrKernelParams::reconn_sz, "). Rebuild with -DRECONN_SIZE=", reconn_sz);

    // Validate bias presence vs compile-time PRISM_INTERNAL_BIAS
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
        TORCH_CHECK(bias.dim() == 2, "internal_bias must be 2D (n_groups, K)");
        TORCH_CHECK(bias.size(0) == n_groups && bias.size(1) == k,
                    "internal_bias must have shape (", n_groups, ", ", k,
                    ") but got (", bias.size(0), ", ", bias.size(1), ")");
        bias_ptr = PRISM_CONST_DATA(bias);
    } else {
        TORCH_CHECK(!internal_bias.has_value(),
                    "Kernel was compiled with PRISM_INTERNAL_BIAS=0 but an internal_bias tensor was passed; "
                    "recompile with internal_bias=True");
    }

    // Optional input-shuffle: pre-build n_groups × (M, K) A_perm buffer.
    const prism_native* A_kernel_ptr = PRISM_CONST_DATA(A);
    int strideA_per_group = 0;
    torch::Tensor A_perm_buf;  // keeps the buffer alive during the kernel call
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
                    "shuffle expects n_chunks*64 == K (got n_chunks=", n_chunks, ", K=", k, ")");

        auto opts = torch::TensorOptions().dtype(k_prism_torch_dtype).device(A.device());
        A_perm_buf = torch::empty({n_groups, m, k}, opts);
        prism_native* A_perm_ptr = PRISM_DATA(A_perm_buf);
        const int64_t* masks_ptr = masks.data_ptr<int64_t>();

        cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();
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

    // Allocate output
    auto C = torch::zeros({m, n}, A.options());

    int ldA = (int)k;
    int ldB = B.stride(0);
    int ldR = R.stride(0);
    int ldC = C.stride(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Resolve dropout params. When PRISM_DROPOUT=0 the kernel ignores these,
    // but we still validate the call site to catch caller mistakes.
    const int64_t* dropout_seeds_ptr = nullptr;
    float dropout_p_f = 0.0f;
    float inv_keep_f = 1.0f;
#if PRISM_DROPOUT
    TORCH_CHECK(dropout_seeds.has_value(),
                "Kernel was compiled with PRISM_DROPOUT=1 but no dropout_seeds tensor was passed");
    const auto& seeds_t = dropout_seeds.value();
    TORCH_CHECK(seeds_t.is_cuda() && seeds_t.device() == A.device(),
                "dropout_seeds must be on the same CUDA device as A");
    TORCH_CHECK(seeds_t.scalar_type() == torch::kInt64, "dropout_seeds must be int64");
    TORCH_CHECK(seeds_t.is_contiguous() && seeds_t.dim() == 1 && seeds_t.size(0) == n_groups,
                "dropout_seeds must be shape (n_groups,)");
    TORCH_CHECK(dropout_p > 0.0 && dropout_p < 1.0, "dropout_p must be in (0, 1)");
    dropout_seeds_ptr = seeds_t.data_ptr<int64_t>();
    dropout_p_f = (float)dropout_p;
    inv_keep_f = 1.0f / (1.0f - dropout_p_f);
#else
    TORCH_CHECK(!dropout_seeds.has_value(),
                "Kernel was compiled with PRISM_DROPOUT=0 but dropout_seeds was passed");
#endif

    prism_tn<CurrKernelParams>(
        static_cast<int>(m),
        static_cast<int>(n),
        static_cast<int>(k),
        A_kernel_ptr,
        ldA,
        PRISM_CONST_DATA(B),
        ldB,
        PRISM_CONST_DATA(R),
        ldR,
        PRISM_DATA(C),
        ldC,
        bias_ptr,
        strideA_per_group,
        dropout_seeds_ptr,
        dropout_p_f,
        inv_keep_f,
        stream
    );

    return C;
}
