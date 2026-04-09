#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
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

// AR = A * SiLU(AR)
template<typename T>
__global__ void silu_gate_kernel(const T* A, T* AR, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float a = to_float(A[idx]);
        float ar = to_float(AR[idx]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        AR[idx] = from_float<T>(a * silu_ar);
    }
}

// Fused: AR[i] = A[i] * SiLU(AR[i] + bias[i%K])  with optional dropout on result
// rand_buf (if non-null) contains pre-generated uniform [0,1) values from torch::rand.
template<typename T>
__global__ void silu_gate_bias_dropout_kernel(
    const T* A, T* AR, const T* bias,
    int64_t M, int64_t K,
    const float* rand_buf, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float a = to_float(A[idx]);
        float ar = to_float(AR[idx]);
        if (bias) ar += to_float(bias[idx % K]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        float h = a * silu_ar;
        if (rand_buf) {
            h = (rand_buf[idx] >= dropout_p) ? h * inv_keep : 0.0f;
        }
        AR[idx] = from_float<T>(h);
    }
}

// dst[i] += src[i]
template<typename T>
__global__ void add_kernel(T* dst, const T* src, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        dst[idx] = from_float<T>(to_float(dst[idx]) + to_float(src[idx]));
    }
}

// dS[i] = dH[i] * A[i] * silu_prime(S[i])
template<typename T>
__global__ void silu_gate_backward_ds_kernel(
    const T* dH, const T* A, const T* S, T* dS, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = to_float(dH[idx]);
        float a = to_float(A[idx]);
        float s = to_float(S[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));
        dS[idx] = from_float<T>(dh * a * silu_prime);
    }
}

// dA_gate[i] = dH[i] * silu(S[i])
template<typename T>
__global__ void silu_gate_backward_da_kernel(
    const T* dH, const T* S, T* dA_gate, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float dh = to_float(dH[idx]);
        float s = to_float(S[idx]);
        float silu_s = s / (1.0f + __expf(-s));
        dA_gate[idx] = from_float<T>(dh * silu_s);
    }
}

// Fused backward: given recomputed AR (= A @ R^T), add bias, then compute
//   dA_gate[i] = dH[i] * silu(S[i])
//   dS[i]      = dH[i] * A[i] * silu'(S[i])
// where S = AR + bias. Replays dropout on dH using the saved rand_buf.
template<typename T>
__global__ void silu_gate_backward_fused_kernel(
    const T* dH_in, const T* A, T* AR,    // AR is modified in-place to S
    T* dA_gate, T* dS,
    const T* bias, int64_t M, int64_t K,
    const float* rand_buf, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float s = to_float(AR[idx]);
        if (bias) s += to_float(bias[idx % K]);
        AR[idx] = from_float<T>(s);  // store S for later dR computation

        // Replay dropout on dH
        float dh = to_float(dH_in[idx]);
        if (rand_buf) {
            dh = (rand_buf[idx] >= dropout_p) ? dh * inv_keep : 0.0f;
        }

        float a = to_float(A[idx]);
        float sigma = 1.0f / (1.0f + __expf(-s));
        float silu_s = s * sigma;
        float silu_prime = sigma * (1.0f + s * (1.0f - sigma));

        dA_gate[idx] = from_float<T>(dh * silu_s);
        dS[idx] = from_float<T>(dh * a * silu_prime);
    }
}

// Fused backward for dB: recompute H = A * SiLU(AR + bias), apply dropout
// Writes H into AR buffer (in-place). Uses saved rand_buf for mask replay.
template<typename T>
__global__ void silu_gate_bias_dropout_kernel_for_db(
    const T* A, T* AR, const T* bias,
    int64_t M, int64_t K,
    const float* rand_buf, float dropout_p, float inv_keep)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t numel = M * K;
    if (idx < numel) {
        float a = to_float(A[idx]);
        float ar = to_float(AR[idx]);
        if (bias) ar += to_float(bias[idx % K]);
        float silu_ar = ar / (1.0f + __expf(-ar));
        float h = a * silu_ar;
        if (rand_buf) {
            h = (rand_buf[idx] >= dropout_p) ? h * inv_keep : 0.0f;
        }
        AR[idx] = from_float<T>(h);
    }
}

// ---------------------------------------------------------------------------
// Forward: AR mode  C = (A @ R^T) @ B^T  [gated: C = (A * SiLU(A @ R^T)) @ B^T]
// ---------------------------------------------------------------------------

// Generate a rand buffer of shape (m, k) in float32 using a seeded generator.
// The seed is stored so backward can regenerate the same buffer.
static torch::Tensor _make_rand_buf(int64_t m, int64_t k,
                                     torch::Device device, int64_t seed) {
    auto gen = at::cuda::detail::createCUDAGenerator(device.index());
    gen.set_current_seed(seed);
    auto rand_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    return torch::rand({m, k}, gen, rand_opts);
}

template<typename scalar_t>
static void cublas_prism_ar_impl(
    const scalar_t* A_ptr, const scalar_t* B_ptr, const scalar_t* R_ptr, scalar_t* C_ptr,
    int64_t m, int64_t n, int64_t k,
    int64_t group_size, int64_t reconn_sz, int64_t n_groups,
    cublasHandle_t handle, cudaDataType_t dt, 
    torch::TensorOptions opts,
    bool gated, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    float dropout_p = 0.0f, bool training = false,
    int64_t* dropout_seeds_out = nullptr)
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
    auto device = opts.device();

    auto produce = [&](int64_t g, int p) {
        gemm_strided_batched_ex(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_ptr + g * reconn_sz * k, dt, k, reconn_sz,
            A_ptr,                     dt, k, reconn_sz,
            AR_ptr[p],                 dt, k, reconn_sz,
            k / reconn_sz);

        if (gated) {
            int64_t numel = m * k;
            int nblocks = (int)((numel + threads - 1) / threads);

            if (internal_bias_ptr || apply_dropout) {
                const float* rand_ptr = nullptr;
                torch::Tensor rand_buf;
                if (apply_dropout) {
                    int64_t seed = (int64_t)std::random_device{}();
                    if (dropout_seeds_out) dropout_seeds_out[g] = seed;
                    rand_buf = _make_rand_buf(m, k, device, seed);
                    rand_ptr = rand_buf.data_ptr<float>();
                }
                silu_gate_bias_dropout_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_ptr, AR_ptr[p],
                    internal_bias_ptr ? internal_bias_ptr + g * k : nullptr,
                    m, k, rand_ptr, dropout_p, inv_keep);
            } else {
                silu_gate_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_ptr, AR_ptr[p], numel);
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
    bool gated, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    const int64_t* dropout_seeds = nullptr, float dropout_p = 0.0f)
{
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

    for (int64_t g = 0; g < n_groups; ++g) {
        const scalar_t* dC_g = dC_ptr + g * group_size;
        const scalar_t* B_g  = B_ptr  + g * group_size * k;
        const scalar_t* R_g  = R_ptr  + g * reconn_sz * k;
        void* dR_g = static_cast<char*>(dR_raw) + g * reconn_sz * k * dR_es;

        // Stream AR: recompute AR_g = A @ R_g^T
        gemm_strided_batched_ex(
            handle_ar, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_g,    compute_dt, k, reconn_sz,
            A_ptr,  compute_dt, k, reconn_sz,
            AR_ptr, compute_dt, k, reconn_sz,
            k / reconn_sz);
        cudaEventRecord(ar_ready, stream_ar);

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
                int64_t numel = m * k;
                int nblocks = (int)((numel + threads - 1) / threads);

                auto temp = torch::empty({m, k}, compute_opts);
                auto* temp_ptr = static_cast<scalar_t*>(temp.data_ptr());

                // Regenerate dropout mask from saved seed
                const float* rand_ptr = nullptr;
                torch::Tensor rand_buf;
                if (has_dropout) {
                    rand_buf = _make_rand_buf(m, k, device, dropout_seeds[g]);
                    rand_ptr = rand_buf.data_ptr<float>();
                }

                // Fused: add bias to AR, replay dropout on dH, compute dA_gate and dS
                silu_gate_backward_fused_kernel<scalar_t><<<nblocks, threads, 0, stream>>>(
                    dAR_ptr, A_ptr, AR_ptr,
                    temp_ptr, dAR_ptr,
                    has_bias ? internal_bias_ptr + g * k : nullptr,
                    m, k, rand_ptr, dropout_p, inv_keep);

                // dA += dA_gate (= dH * silu(S))
                add_kernel<scalar_t><<<nblocks, threads, 0, stream>>>(
                    dA_ptr, temp_ptr, numel);

                // d_internal_bias[g] = dS.sum(dim=0)
                if (has_bias) {
                    // Simple reduction: iterate rows. For large M this could
                    // use a proper reduction kernel, but matches the PyTorch
                    // reference and M is typically modest.
                    auto dS_2d = torch::from_blob(dAR_ptr, {m, k}, compute_opts);
                    auto dib_g = dS_2d.sum(0);
                    auto* dib_ptr = static_cast<scalar_t*>(d_internal_bias.data_ptr());
                    cudaMemcpyAsync(dib_ptr + g * k, dib_g.data_ptr(),
                                    k * sizeof(scalar_t), cudaMemcpyDeviceToDevice, stream);
                }
            }

            // dA += dS_g @ R_g
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz,
                R_g,    compute_dt, k, reconn_sz,
                dAR_ptr, compute_dt, k, reconn_sz,
                dA_ptr, compute_dt, k, reconn_sz,
                k / reconn_sz,
                1.0f, 1.0f);   // beta = 1 for accumulation

            // dR_g[b] = dS_g^T @ A
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m,
                A_ptr,   compute_dt, k, reconn_sz,
                dAR_ptr, compute_dt, k, reconn_sz,
                dR_g,    dR_dt,      k, reconn_sz,
                k / reconn_sz);
        } else {
            // Non-gated: no AR recompute needed, just dH = dC @ B
            // Wait for AR stream (no-op if nothing was launched, but safe)
            cudaStreamWaitEvent(stream, ar_ready);

            // dAR_g = dC_g @ B_g
            gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, m, group_size,
                B_g,     compute_dt, k,
                dC_g,    compute_dt, n,
                dAR_ptr, compute_dt, k);

            // dA += dAR_g @ R_g
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                reconn_sz, m, reconn_sz,
                R_g,     compute_dt, k, reconn_sz,
                dAR_ptr, compute_dt, k, reconn_sz,
                dA_ptr,  compute_dt, k, reconn_sz,
                k / reconn_sz,
                1.0f, 1.0f);

            // dR_g[b] = dAR_g^T @ A
            gemm_strided_batched_ex(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                reconn_sz, reconn_sz, m,
                A_ptr,   compute_dt, k, reconn_sz,
                dAR_ptr, compute_dt, k, reconn_sz,
                dR_g,    dR_dt,      k, reconn_sz,
                k / reconn_sz);
        }
    }

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
    bool gated, cudaStream_t stream,
    const scalar_t* internal_bias_ptr = nullptr,
    const int64_t* dropout_seeds = nullptr, float dropout_p = 0.0f)
{
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

        gemm_strided_batched_ex(
            handle_p, CUBLAS_OP_T, CUBLAS_OP_N,
            reconn_sz, m, reconn_sz,
            R_g,       compute_dt, k, reconn_sz,
            A_ptr,     compute_dt, k, reconn_sz,
            AR_ptr[p], compute_dt, k, reconn_sz,
            k / reconn_sz);

        if (gated) {
            int64_t numel = m * k;
            int nblocks = (int)((numel + threads - 1) / threads);

            if (has_bias || has_dropout) {
                const float* rand_ptr = nullptr;
                torch::Tensor rand_buf;
                if (has_dropout) {
                    rand_buf = _make_rand_buf(m, k, device, dropout_seeds[g]);
                    rand_ptr = rand_buf.data_ptr<float>();
                }
                silu_gate_bias_dropout_kernel_for_db<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_ptr, AR_ptr[p],
                    has_bias ? internal_bias_ptr + g * k : nullptr,
                    m, k, rand_ptr, dropout_p, inv_keep);
            } else {
                silu_gate_kernel<scalar_t><<<nblocks, threads, 0, stream_p>>>(
                    A_ptr, AR_ptr[p], numel);
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
    bool gated,
    c10::optional<torch::Tensor> internal_bias,
    double dropout_p,
    bool training)
{
    auto dtype = validate_forward_inputs(A, B, R, group_size, reconn_sz);
    auto cuda_dt = torch_to_cublas_dtype(dtype);

    int64_t m = A.size(0);
    int64_t k = A.size(1);
    int64_t n = B.size(0);
    int64_t n_groups = n / group_size;

    TORCH_CHECK(!gated || !rw_mode,
                "Gated activation is only supported in AR mode, not RW mode");

    auto C = torch::zeros({m, n}, A.options());

    // Allocate dropout seeds tensor (returned to Python for backward replay)
    auto seeds_opts = torch::TensorOptions().dtype(torch::kInt64).device(A.device());
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
                gated, stream, ib_p, dp, training, seeds_ptr); \
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
    bool gated,
    c10::optional<at::ScalarType> dR_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p)
{
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
    const int64_t* seeds_ptr = (dropout_seeds.has_value() && dropout_seeds->defined())
        ? dropout_seeds->data_ptr<int64_t>() : nullptr;

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
            handle, compute_dt, A.options(), dR_dt, dR_opts, gated, stream, \
            ib_p, seeds_ptr, dp); \
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
    bool gated,
    c10::optional<at::ScalarType> dB_dtype,
    c10::optional<torch::Tensor> internal_bias,
    c10::optional<torch::Tensor> dropout_seeds,
    double dropout_p)
{
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
    const int64_t* seeds_ptr = (dropout_seeds.has_value() && dropout_seeds->defined())
        ? dropout_seeds->data_ptr<int64_t>() : nullptr;

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
            handle, compute_dt, A.options(), dB_dt, dB_opts, gated, stream, \
            ib_p, seeds_ptr, dp); \
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
