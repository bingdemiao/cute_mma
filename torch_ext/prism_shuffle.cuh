// Butterfly input-shuffle kernels — used by both cublas and cute torch_ext modules.
//
// Each shuffle is per-group: masks have shape (n_groups, n_chunks, n_rounds), and
// callers index into masks[g] before invoking these kernels.
//
// One warp = one chunk of 32 segments (64 elements). The kernels are templated on
// the element type (half or __nv_bfloat16) so each .cu file gets its own
// instantiation.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace prism_shuffle {

template<typename T> __device__ __forceinline__ float to_float(T v);
template<> __device__ __forceinline__ float to_float(half v)            { return __half2float(v); }
template<> __device__ __forceinline__ float to_float(__nv_bfloat16 v)   { return __bfloat162float(v); }

template<typename T> __device__ __forceinline__ T from_float(float v);
template<> __device__ __forceinline__ half          from_float(float v) { return __float2half(v); }
template<> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }

// Forward: load from A, shuffle per-group masks, store to A_perm.
template<typename T>
__global__ void shuffle_forward_kernel(
    const T* __restrict__ A,
    T* __restrict__ A_perm,
    const int64_t* __restrict__ masks,
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
        reinterpret_cast<const char*>(A) + (row * K + col) * sizeof(T));

    {
        // Round 0: delta = 8
        int64_t m0 = masks[chunk * n_rounds + 0];
        uint32_t p0 = __shfl_xor_sync(0xFFFFFFFF, val, 8);
        int pi0 = (lane & 7) | ((lane >> 4) << 3);
        val = ((m0 >> pi0) & 1) ? p0 : val;
    }
    {
        // Round 1: delta = 16
        int64_t m1 = masks[chunk * n_rounds + 1];
        uint32_t p1 = __shfl_xor_sync(0xFFFFFFFF, val, 16);
        int pi1 = (lane & 15);
        val = ((m1 >> pi1) & 1) ? p1 : val;
    }

    *reinterpret_cast<uint32_t*>(
        reinterpret_cast<char*>(A_perm) + (row * K + col) * sizeof(T)) = val;
}

// Inverse + accumulate: applies masks in reverse order, then atomically/non-
// atomically accumulates into dA. Single-thread-per-element (no atomic needed
// — each (row, col) is touched by one thread).
template<typename T>
__global__ void shuffle_inverse_add_kernel(
    T* __restrict__ dA,
    const T* __restrict__ dA_perm,
    const int64_t* __restrict__ masks,
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

    const T* unpacked = reinterpret_cast<const T*>(&val);
    int64_t base = row * K + col;
    float prev0 = to_float(dA[base]);
    float prev1 = to_float(dA[base + 1]);
    dA[base]     = from_float<T>(prev0 + to_float(unpacked[0]));
    dA[base + 1] = from_float<T>(prev1 + to_float(unpacked[1]));
}

}  // namespace prism_shuffle
