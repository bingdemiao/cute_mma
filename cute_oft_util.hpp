#pragma once
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

// =============================================================================
// Packed f16x2 sigmoid / SiLU primitives
// sigmoid(x) = 0.5*(1+tanh(x*0.5))  — avoids reciprocal
// SiLU(x) = x * sigmoid(x)
// SiLU'(x) = sigma * (1 + x*(1-sigma))
// =============================================================================

/// Packed sigmoid: returns sigma(x) for two half values
__device__ __forceinline__
__half2 sigmoid_h2(__half2 x) {
    __half2 half_x = __hmul2(x, __float2half2_rn(0.5f));
    __half2 tanh_val;
    asm("tanh.approx.f16x2 %0, %1;"
        : "=r"(reinterpret_cast<uint32_t&>(tanh_val))
        : "r"(reinterpret_cast<const uint32_t&>(half_x)));
    return __hmul2(__float2half2_rn(0.5f),
                   __hadd2(__float2half2_rn(1.0f), tanh_val));
}

/// Packed SiLU: returns x * sigmoid(x)
__device__ __forceinline__
__half2 silu_h2(__half2 x) {
    return __hmul2(x, sigmoid_h2(x));
}

/// Packed SiLU derivative: returns sigma * (1 + x*(1-sigma))
__device__ __forceinline__
__half2 silu_prime_h2(__half2 x, __half2 sigma) {
    __half2 one = __float2half2_rn(1.0f);
    return __hmul2(sigma, __hfma2(x, __hsub2(one, sigma), one));
}

/// Load two half_t values from smem as __half2
__device__ __forceinline__
__half2 load_half2(const cute::half_t& a, const cute::half_t& b) {
    return __halves2half2(
        reinterpret_cast<const __half&>(a),
        reinterpret_cast<const __half&>(b));
}

/// Store __half2 low/high to two smem half_t locations
__device__ __forceinline__
void store_half2(cute::half_t& dst0, cute::half_t& dst1, __half2 val) {
    dst0 = cute::half_t(__low2half(val));
    dst1 = cute::half_t(__high2half(val));
}

template <class WARP_N, class N_GROUPS>
CUTE_HOST_DEVICE constexpr
auto warp_group_mapping(WARP_N warp_n, N_GROUPS n_groups) {
    using namespace cute;
    // would generate a layout with the size of (groups_per_warp, n_warps)
    if constexpr (n_groups >= warp_n) {
        // if the number of groups is greater or equal to the number of warps, then each warp will handle at least one group
        CUTE_STATIC_ASSERT(n_groups % warp_n == 0, "Number of groups must be divisible by number of warps.");
        return make_layout(make_shape(n_groups / warp_n, warp_n));
    } else {
        // if the number of groups is less than the number of warps, then a group would be handled by multiple warps
        CUTE_STATIC_ASSERT(warp_n % n_groups == 0, "Number of warps must be divisible by number of groups.");
        return make_layout(
            make_shape(_1{}, make_shape(warp_n / n_groups, n_groups)),
            make_stride(_0{}, make_stride(_0{}, _1{}))
        );
    }
}

template <class WARP_N, class N_GROUPS, class GROUP_SIZE>
CUTE_HOST_DEVICE constexpr
auto warp_in_group_mapping(WARP_N warp_n, N_GROUPS n_groups, GROUP_SIZE group_sz) {
    using namespace cute;
    // would generate a layout with size of (warp_responsible_size, warp_n)
    if constexpr(n_groups >= warp_n) {
        // if the number of groups is greater or equal to the number of warps, then each warp would handle a full group
        return make_layout(make_shape(group_sz, warp_n), make_stride(_1{}, _0{}));
    } else {
        auto warp_per_group = warp_n / n_groups;
        auto warp_responsible_size = group_sz / warp_per_group;
        return make_layout(
            make_shape(warp_responsible_size, make_shape(warp_per_group, n_groups)),
            make_stride(_1{}, make_stride(warp_responsible_size, _0{}))
        );
    }
}

// Create a tiled copy layout for cp.async (SM80_CP_ASYNC_CACHEALWAYS)
// Shared between forward and backward kernels.
template <typename copy_as_t, typename ele_t,
  typename _BM, typename _BK, typename _N_Threads>
constexpr auto cp_layout(_BM bm, _BK bk, _N_Threads _total_threads) {
    using namespace cute;
    auto vec_width = Int<sizeof(copy_as_t)>{} / Int<sizeof(ele_t)>();
    auto total_elements = bm * bk;
    auto needed_threads = total_elements / vec_width;
    CUTE_STATIC_ASSERT_V(total_elements % vec_width == _0{}, "total number of elements shall be divisible by the vector length");
    auto total_threads = min(_total_threads, needed_threads);
    auto elements_per_thread = total_elements / total_threads;
    CUTE_STATIC_ASSERT_V(total_elements % total_threads == _0{}, "total number of elements shall be divisible by the number of threads using");
    CUTE_STATIC_ASSERT_V(elements_per_thread % vec_width == _0{}, "number of elements handled by each thread should be divisible by the vector width");
    auto cp_width = vec_width;
    auto threads_along_k = max(bk / cp_width, _1{});
    auto threads_k_size = bk / threads_along_k;
    auto threads_m_size = max(cp_width / bk, _1{});
    auto threads_along_m = total_threads / threads_along_k;
    return make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<copy_as_t>, ele_t>{},
                        make_layout(make_shape(threads_along_m, threads_along_k), LayoutRight{}),
                        make_layout(make_shape(threads_m_size, threads_k_size)));
}

template <bool swizzle = true, int _k_width>
CUTE_HOST_DEVICE constexpr
auto get_smem_atom(cute::Int<_k_width>) {
    using namespace cute;
    /*
    k=64: 3,3
    k=32: 2,4
    k=16: 1,5
    k= 8: 0,6
    */
    Int<_k_width> k_width;
    CUTE_STATIC_ASSERT(k_width % 8 == 0);
    CUTE_STATIC_ASSERT_V(k_width == bit_floor(k_width)); // k_width must be a power of two
    constexpr auto n_blocks = k_width / _8{};
    constexpr auto permutation_bits = log_2(static_cast<unsigned int>(_k_width)) - _3{};
    constexpr auto base_layout = make_layout(
        make_shape(_8{}, make_shape(_8{}, n_blocks)),
        make_stride(_8{}, make_stride(_1{}, _64{}))
    );
    if constexpr (!swizzle) {
        return base_layout;
    } else {
        constexpr auto sw = Swizzle<permutation_bits, 6 - permutation_bits>{};
        return composition(sw, base_layout);
    }
}

// In-place 8x8 matrix transpose in registers via PTX movmatrix instruction.
// Each thread in a warp holds one u32 (= 2 f16 values) of the 8x8 matrix.
// After transpose, each thread's u32 holds the transposed elements.
// Works in-place (d == a is valid, verified empirically).
__device__ __forceinline__
void movmatrix_trans_b16(unsigned int& reg) {
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %0;" : "+r"(reg));
}

// In-place transpose of MMA fragment register tensor via movmatrix.
//
// Input shape:  (2, MMA_M, MMA_K) — 2 halfs (one u32) per 8x8 sub-block
// Output shape: (2, MMA_K, MMA_M) — dims 1 and 2 swapped
//
// The user must group modes into this 3D shape before calling.
// For example, a fragment ((2,2), 2, 4) should first be grouped via
// group_modes<0,1> to get (2*2, 2, 4) = (4, 2, 4), then the caller
// interprets the first mode as the u32 pairs.
//
// Actually: the innermost mode must be exactly 2 (one u32 = 2 f16 values).
// movmatrix transposes each 8x8 block distributed across the warp.
// Swapping modes 1↔2 reinterprets M-atoms as K-atoms and vice versa.
template <class Engine, class Layout>
__device__ __forceinline__
auto inplace_transpose(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected shape (2, MMA_M, MMA_K)");
    static_assert(size<0>(Layout{}) == _2{}, "Mode 0 must be 2 (one u32 = 2 f16)");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    // Apply movmatrix to every u32
    #pragma unroll
    for (int i = 0; i < size<1>(Layout{}); ++i) {
        #pragma unroll
        for (int j = 0; j < size<2>(Layout{}); ++j) {
            auto& reg = reinterpret_cast<unsigned int&>(tensor(0, i, j));
            movmatrix_trans_b16(reg);
        }
    }

    // Return view with modes 1 and 2 swapped: (2, MMA_M, MMA_K) → (2, MMA_K, MMA_M)
    auto final_layout = make_layout(get<0>(Layout{}), get<2>(Layout{}), get<1>(Layout{}));
    return make_tensor(&tensor(0), final_layout);
}
