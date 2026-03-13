#pragma once
#include <cute/tensor.hpp>

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