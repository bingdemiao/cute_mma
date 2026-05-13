// Element-type scaffolding for the cute Prism kernels.
//
// PRISM_DTYPE = 0 → fp16 (existing path, F16-accum MMAs available).
// PRISM_DTYPE = 1 → bf16 (only F32-accum MMAs available; silu computed in
// float, converted to bf16 at smem boundaries).

#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cute/numeric/numeric_types.hpp>
#include <cute/atom/mma_atom.hpp>

#ifndef PRISM_DTYPE
#define PRISM_DTYPE 0
#endif

// Compile-time dropout switch. 0 → dropout code path is preprocessed away
// (dropout_seeds_ptr in launcher signatures is unused / nullptr-only).
// 1 → silu/sigmoid sites apply a hash-derived mask on the gated activation,
// and dB/dadr replay the same mask on the recomputed gate / on dH.
#ifndef PRISM_DROPOUT
#define PRISM_DROPOUT 0
#endif

// Deterministic hash-based uniform[0,1) for dropout masks. **MUST match the
// implementation in cute_mma/torch_ext/cublas_prism_torch.cu** so the cublas
// and cute backends produce bit-identical masks for a given (seed, idx).
__device__ __forceinline__ uint64_t prism_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float prism_uniform_from_hash(uint64_t seed, int64_t idx) {
    uint64_t h = prism_splitmix64(seed + 0xdeadbeefULL) ^ prism_splitmix64((uint64_t)idx);
    h = prism_splitmix64(h);
    // Top 24 bits → [0, 1)
    return (float)((h >> 40) & 0xFFFFFF) * (1.0f / 16777216.0f);
}

#if PRISM_DTYPE == 0
// fp16: native CUDA half + CuTe half_t alias.
using prism_native  = __half;
using prism_native2 = __half2;
using prism_cute    = cute::half_t;
#elif PRISM_DTYPE == 1
using prism_native  = __nv_bfloat16;
using prism_native2 = __nv_bfloat162;
using prism_cute    = cute::bfloat16_t;
#else
#error "PRISM_DTYPE must be 0 (fp16) or 1 (bf16)"
#endif

// AR-style MMA atoms (fwd stage1, fwd stage2, dB AR producer, dB consumer,
// dadr dH producer): F16-accum for fp16, F32-accum for bf16.
template <int RS>
using prism_ar_atom = std::conditional_t<
    PRISM_DTYPE == 0,
    std::conditional_t<(RS < 16),
        cute::MMA_Atom<cute::SM80_16x8x8_F16F16F16F16_TN>,
        cute::MMA_Atom<cute::SM80_16x8x16_F16F16F16F16_TN>>,
    std::conditional_t<(RS < 16),
        cute::MMA_Atom<cute::SM80_16x8x8_F32BF16BF16F32_TN>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>
>;

// Same as prism_ar_atom but exposing the raw MMA op type (not wrapped in
// MMA_Atom). Used by `make_tiled_mma(op{}, ...)` call sites.
template <int RS>
using prism_ar_op = std::conditional_t<
    PRISM_DTYPE == 0,
    std::conditional_t<(RS < 16),
        cute::SM80_16x8x8_F16F16F16F16_TN,
        cute::SM80_16x8x16_F16F16F16F16_TN>,
    std::conditional_t<(RS < 16),
        cute::SM80_16x8x8_F32BF16BF16F32_TN,
        cute::SM80_16x8x16_F32BF16BF16F32_TN>
>;

// dA / dR MMAs (always F32-accum). For bf16 just swap operand type.
template <int K_DIM>
using prism_dA_atom_f32 = std::conditional_t<
    PRISM_DTYPE == 0,
    std::conditional_t<(K_DIM < 16),
        cute::MMA_Atom<cute::SM80_16x8x8_F32F16F16F32_TN>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>>,
    std::conditional_t<(K_DIM < 16),
        cute::MMA_Atom<cute::SM80_16x8x8_F32BF16BF16F32_TN>,
        cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>>
>;

// Scalar conversion helpers.
__device__ __forceinline__ float prism_to_float(prism_native v) {
#if PRISM_DTYPE == 0
    return __half2float(v);
#else
    return __bfloat162float(v);
#endif
}

__device__ __forceinline__ prism_native prism_from_float(float v) {
#if PRISM_DTYPE == 0
    return __float2half(v);
#else
    return __float2bfloat16(v);
#endif
}

__device__ __forceinline__ prism_native2 prism_pack(prism_native lo, prism_native hi) {
#if PRISM_DTYPE == 0
    return __halves2half2(lo, hi);
#else
    return __halves2bfloat162(lo, hi);
#endif
}
