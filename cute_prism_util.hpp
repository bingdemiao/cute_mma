#pragma once
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

// =============================================================================
// CuTe extensions — placed in namespace cute so ADL finds them
// =============================================================================

namespace cute {

// -- SM80 f32→f16x2 conversion copy trait ------------------------------------

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#  define CUTE_PRISM_CVT_SM80_ENABLED
#endif

struct SM80_CVT_F32_F16
{
  using SRegisters = uint32_t[2];
  using DRegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src1,
       uint32_t const& src2,
       uint32_t& dst)
  {
#if defined(CUTE_PRISM_CVT_SM80_ENABLED)
    asm volatile ("cvt.rn.f16x2.f32 {%0}, {%1}, {%2};\n"
        : "=r"(dst)
        :  "r"(src1), "r"(src2));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use cvt without SM80+.");
#endif
  }
};

// -- print overload for __half -----------------------------------------------

CUTE_HOST_DEVICE
void
print(half a) {
  printf("%f", static_cast<float>(a));
}

} // namespace cute

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
    CUTE_STATIC_ASSERT_V(bk % vec_width == _0{}, "K dimension shall be divisible by the vector length");
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

// Select (permute) modes of a layout. Handles both plain Layout and ComposedLayout.
// For ComposedLayout (swizzled), permutes the inner layout_b while preserving the swizzle.
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr
auto select_layout(cute::Layout<Shape, Stride> const& layout) {
    return cute::select<Is...>(layout);
}

template <int... Is, class A, class O, class B>
CUTE_HOST_DEVICE constexpr
auto select_layout(cute::ComposedLayout<A, O, B> const& layout) {
    // Permute the inner layout (domain) while keeping the swizzle (layout_a) and offset intact
    return cute::make_composed_layout(layout.layout_a(), layout.offset(),
                                      cute::select<Is...>(layout.layout_b()));
}

// Select (permute) modes of a Tensor, analogous to cute::select on layouts.
// Usage: select<1,0>(tensor)  — swaps modes 0 and 1 (transpose)
//        select<2,0,1>(tensor) — arbitrary mode permutation
// Works with both plain and swizzled (ComposedLayout) smem tensors.
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE
auto select(cute::Tensor<Engine, Layout> const& tensor) {
    return cute::make_tensor(tensor.data(), select_layout<Is...>(tensor.layout()));
}

// Swap two outer modes of a Tensor.
// Usage: transpose<0,2>(tensor) — swaps modes 0 and 2
namespace detail {
template <int I, int J>
struct swap_index {
    static constexpr int apply(int k) { return k == I ? J : (k == J ? I : k); }
};

template <int I, int J, class Engine, class Layout, int... Ks>
CUTE_HOST_DEVICE
auto transpose_impl(cute::Tensor<Engine, Layout> const& tensor,
                     std::integer_sequence<int, Ks...>) {
    return select<swap_index<I, J>::apply(Ks)...>(tensor);
}
} // namespace detail

template <int I, int J, class Engine, class Layout>
CUTE_HOST_DEVICE
auto transpose(cute::Tensor<Engine, Layout> const& tensor) {
    constexpr int R = cute::rank(Layout{});
    static_assert(I >= 0 && I < R, "Mode index I out of range");
    static_assert(J >= 0 && J < R, "Mode index J out of range");
    static_assert(I != J, "Mode indices must be different");
    return detail::transpose_impl<I, J>(tensor, std::make_integer_sequence<int, R>{});
}

// Replace mode I of a plain Layout with a new sub-layout.
template <int I, class Shape, class Stride, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto replace_mode(cute::Layout<Shape, Stride> const& layout,
                  cute::Layout<NewShape, NewStride> const& new_mode) {
    return cute::make_layout(
        cute::replace<I>(layout.shape(),  new_mode.shape()),
        cute::replace<I>(layout.stride(), new_mode.stride())
    );
}

// Replace mode I of a ComposedLayout (swizzled) — modifies layout_b only.
template <int I, class A, class O, class B, class NewShape, class NewStride>
CUTE_HOST_DEVICE constexpr
auto replace_mode(cute::ComposedLayout<A, O, B> const& layout,
                  cute::Layout<NewShape, NewStride> const& new_mode) {
    return cute::make_composed_layout(
        layout.layout_a(), layout.offset(),
        replace_mode<I>(layout.layout_b(), new_mode)
    );
}

// Pad mode I of a tensor by appending a stride-0 repeat dimension.
// Shape of mode I becomes (original_shape, PadFactor) with stride (original_stride, 0).
// The logical size of mode I grows by PadFactor, but all new elements alias the originals.
// When PadFactor == 1, returns the tensor unchanged (complete no-op).
// Works with both plain and swizzled (ComposedLayout) smem tensors.

// PadFactor == 1: no-op, return original tensor unchanged (preserves layout type)
template <int I, int PadFactor, class Engine, class Layout>
CUTE_HOST_DEVICE
auto pad_mode_stride0(cute::Tensor<Engine, Layout> const& tensor)
    -> std::enable_if_t<PadFactor == 1, cute::Tensor<Engine, Layout> const&> {
    return tensor;
}

// PadFactor > 1, plain Layout: extract shape/stride directly
template <int I, int PadFactor, class Engine, class Shape, class Stride>
CUTE_HOST_DEVICE
auto pad_mode_stride0(cute::Tensor<Engine, cute::Layout<Shape, Stride>> const& tensor)
    -> std::enable_if_t<(PadFactor > 1), decltype(cute::make_tensor(tensor.data(),
        replace_mode<I>(tensor.layout(), cute::make_layout(
            cute::make_shape(cute::get<I>(tensor.layout().shape()), cute::Int<PadFactor>{}),
            cute::make_stride(cute::get<I>(tensor.layout().stride()), cute::_0{})))))> {
    using namespace cute;
    auto old_shape  = get<I>(tensor.layout().shape());
    auto old_stride = get<I>(tensor.layout().stride());
    auto new_mode = make_layout(
        make_shape(old_shape, Int<PadFactor>{}),
        make_stride(old_stride, _0{})
    );
    return make_tensor(tensor.data(), replace_mode<I>(tensor.layout(), new_mode));
}

// PadFactor > 1, ComposedLayout (swizzled): extract shape/stride from layout_b
template <int I, int PadFactor, class Engine, class A, class O, class B>
CUTE_HOST_DEVICE
auto pad_mode_stride0(cute::Tensor<Engine, cute::ComposedLayout<A, O, B>> const& tensor)
    -> std::enable_if_t<(PadFactor > 1), decltype(cute::make_tensor(tensor.data(),
        replace_mode<I>(tensor.layout(), cute::make_layout(
            cute::make_shape(cute::get<I>(tensor.layout().layout_b().shape()), cute::Int<PadFactor>{}),
            cute::make_stride(cute::get<I>(tensor.layout().layout_b().stride()), cute::_0{})))))> {
    using namespace cute;
    auto old_shape  = get<I>(tensor.layout().layout_b().shape());
    auto old_stride = get<I>(tensor.layout().layout_b().stride());
    auto new_mode = make_layout(
        make_shape(old_shape, Int<PadFactor>{}),
        make_stride(old_stride, _0{})
    );
    return make_tensor(tensor.data(), replace_mode<I>(tensor.layout(), new_mode));
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
    // Cap swizzle at 3 permutation bits (Swizzle<3,3>, covering 64 elements per row).
    // 8 segments × 4 banks each = 32 banks fully utilized. Beyond 64 elements,
    // additional swizzle bits just repeat the same bank pattern.
    constexpr unsigned raw_perm_bits = log_2(static_cast<unsigned int>(_k_width)) - 3u;
    constexpr unsigned permutation_bits = raw_perm_bits > 3u ? 3u : raw_perm_bits;
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

    // Apply movmatrix to every u32 via CuTe copy atom
    // Mode 0 (size 2) matches the atom's value width (1 u32 = 2 half_t)
    copy(Copy_Atom<SM75_U32x1_MOVM_T, typename Engine::value_type>{}, tensor, tensor);

    // Return view with modes 1 and 2 swapped: (2, MMA_M, MMA_K) → (2, MMA_K, MMA_M)
    return select<0, 2, 1>(tensor);
}

// Store a blockened register tensor (2, dim1_blocks, dim2_blocks) to a 2D smem tensor.
//
// Within each 8x8 block, the MMA C-fragment distribution is:
//   thread t holds row (t/4), columns (t%4)*2 and (t%4)*2+1
// This matches both the original C-fragment and the post-movmatrix layout.
//
// The smem tensor must have shape (dim1_blocks*8, dim2_blocks*8) with any layout.
// Addressing goes through the smem tensor's indexing to handle swizzled/non-RowMajor layouts.
template <class RegEngine, class RegLayout, class SmemEngine, class SmemLayout>
__device__ __forceinline__
void store_blocks_to_smem(const cute::Tensor<RegEngine, RegLayout>& tensor,
                          cute::Tensor<SmemEngine, SmemLayout>& smem,
                          int lane_idx) {
    using namespace cute;
    static_assert(rank(RegLayout{}) == _3{}, "Expected shape (2, dim1_blocks, dim2_blocks)");
    static_assert(size<0>(RegLayout{}) == _2{}, "Mode 0 must be 2 (one u32 = 2 f16)");

    int row_in_block = lane_idx / 4;
    int col_pair     = lane_idx % 4;

    #pragma unroll
    for (int b1 = 0; b1 < size<1>(RegLayout{}); ++b1) {
        #pragma unroll
        for (int b2 = 0; b2 < size<2>(RegLayout{}); ++b2) {
            int smem_row = b1 * 8 + row_in_block;
            int smem_col = b2 * 8 + col_pair * 2;
            smem(smem_row, smem_col)     = tensor(0, b1, b2);
            smem(smem_row, smem_col + 1) = tensor(1, b1, b2);
        }
    }
}

// Helper: build atom inner layout from (halves, mma_dim, mma_k) sub-layouts,
// dropping size-1 modes to match CuTe's MMA fragment convention.
// E.g. (2, 2, 1) → (2, 2);  (2, 1, 2) → (2, 2);  (2, 2, 2) → (2, 2, 2);  (2, 1, 1) → (2)
template <class L_halves, class L_dim, class L_k>
CUTE_HOST_DEVICE constexpr auto make_atom_inner(L_halves l_h, L_dim l_d, L_k l_k) {
    using namespace cute;
    if constexpr (size(L_dim{}) == 1 && size(L_k{}) == 1) {
        return l_h;
    } else if constexpr (size(L_dim{}) == 1) {
        return make_layout(l_h, l_k);
    } else if constexpr (size(L_k{}) == 1) {
        return make_layout(l_h, l_d);
    } else {
        return make_layout(l_h, l_d, l_k);
    }
}

// Blocken MMA A-operand fragment back to (2, M_blocks, K_blocks) for inplace_transpose.
// Inverse of construct_operand_A: merges per-atom sub-modes back into flat block dimensions.
//
// Input:  ((2, [MMA_M], [MMA_K]), outer_M, outer_K) — from CuTe fragment or construct_operand_A
// Output: (2, MMA_M * outer_M, MMA_K * outer_K)
template <class MMA_ATOM, class Engine, class Layout>
__device__ __forceinline__
auto blocken_operand_A(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected rank-3 fragment");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    using atom_shape = typename MMA_ATOM::Shape_MNK;
    constexpr int MMA_M = get<0>(atom_shape{}) / 8;
    constexpr int MMA_K = get<2>(atom_shape{}) / 8;

    auto dl = tensor.layout();
    if constexpr (MMA_M == 1 && MMA_K == 1) {
        // inner is (2), modes 1,2 are already the full block dims
        return tensor;
    } else if constexpr (MMA_K == 1) {
        // inner is (2, MMA_M) → merge MMA_M with outer_M
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                // halves
            make_layout(get<0,1>(dl), get<1>(dl)),       // (MMA_M, outer_M)
            get<2>(dl)                                   // outer_K
        ));
    } else if constexpr (MMA_M == 1) {
        // inner is (2, MMA_K) → merge MMA_K with outer_K
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                // halves
            get<1>(dl),                                  // outer_M
            make_layout(get<0,1>(dl), get<2>(dl))        // (MMA_K, outer_K)
        ));
    } else {
        // inner is (2, MMA_M, MMA_K) → merge both
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                    // halves
            make_layout(get<0,1>(dl), get<1>(dl)),           // (MMA_M, outer_M)
            make_layout(get<0,2>(dl), get<2>(dl))            // (MMA_K, outer_K)
        ));
    }
}

// Blocken MMA B-operand fragment back to (2, N_blocks, K_blocks) for inplace_transpose.
// Inverse of construct_operand_B: merges per-atom sub-modes back into flat block dimensions.
//
// Input:  ((2, [MMA_N], [MMA_K]), outer_N, outer_K) — from CuTe fragment or construct_operand_B
// Output: (2, MMA_N * outer_N, MMA_K * outer_K)
template <class MMA_ATOM, class Engine, class Layout>
__device__ __forceinline__
auto blocken_operand_B(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected rank-3 fragment");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    using atom_shape = typename MMA_ATOM::Shape_MNK;
    constexpr int MMA_N = get<1>(atom_shape{}) / 8;
    constexpr int MMA_K = get<2>(atom_shape{}) / 8;

    auto dl = tensor.layout();
    if constexpr (MMA_N == 1 && MMA_K == 1) {
        return tensor;
    } else if constexpr (MMA_K == 1) {
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),
            make_layout(get<0,1>(dl), get<1>(dl)),
            get<2>(dl)
        ));
    } else if constexpr (MMA_N == 1) {
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),
            get<1>(dl),
            make_layout(get<0,1>(dl), get<2>(dl))
        ));
    } else {
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),
            make_layout(get<0,1>(dl), get<1>(dl)),
            make_layout(get<0,2>(dl), get<2>(dl))
        ));
    }
}

// Blocken MMA C-operand fragment back to (2, M_blocks, N_blocks) for inplace_transpose.
// C-fragment has dimensions (M, N), with MMA_M = M_atom/8, MMA_N = N_atom/8.
//
// Input:  ((2, [MMA_M], [MMA_N]), outer_M, outer_N) — from CuTe C-fragment
// Output: (2, MMA_M * outer_M, MMA_N * outer_N)
template <class MMA_ATOM, class Engine, class Layout>
__device__ __forceinline__
auto blocken_operand_C(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected rank-3 fragment");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    using atom_shape = typename MMA_ATOM::Shape_MNK;
    constexpr int MMA_M = get<0>(atom_shape{}) / 8;
    constexpr int MMA_N = get<1>(atom_shape{}) / 8;

    auto dl = tensor.layout();
    if constexpr (MMA_M == 1 && MMA_N == 1) {
        return tensor;
    } else if constexpr (MMA_N == 1) {
        // inner is (2, MMA_M) → merge MMA_M with outer_M
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                // halves
            make_layout(get<0,1>(dl), get<1>(dl)),       // (MMA_M, outer_M)
            get<2>(dl)                                   // outer_N
        ));
    } else if constexpr (MMA_M == 1) {
        // inner is (2, MMA_N) → merge MMA_N with outer_N
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                // halves
            get<1>(dl),                                  // outer_M
            make_layout(get<0,1>(dl), get<2>(dl))        // (MMA_N, outer_N)
        ));
    } else {
        // inner is (2, MMA_M, MMA_N) → merge both
        return make_tensor(tensor.data(), make_layout(
            get<0,0>(dl),                                    // halves
            make_layout(get<0,1>(dl), get<1>(dl)),           // (MMA_M, outer_M)
            make_layout(get<0,2>(dl), get<2>(dl))            // (MMA_N, outer_N)
        ));
    }
}

// Construct MMA A-operand from a block tensor of shape (2, M_blocks, K_blocks).
// Groups 8x8 blocks into atom-sized chunks matching MMA_ATOM's A-operand layout.
//
// MMA_ATOM's A-operand has (M_atom, K_atom) = (16, K_atom). In 8x8 blocks:
//   MMA_M = M_atom / 8 = 2,  MMA_K = K_atom / 8
//
// Input:  (2, total_M_blocks, total_K_blocks)
// Output: ((2, [MMA_M], [MMA_K]), outer_M, outer_K) — size-1 modes dropped
template <class MMA_ATOM, class Engine, class Layout>
__device__ __forceinline__
auto construct_operand_A(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected shape (2, M_blocks, K_blocks)");
    static_assert(size<0>(Layout{}) == _2{}, "Mode 0 must be 2 (one u32 = 2 f16)");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    // Get atom A shape: (M_atom, K_atom) in elements, convert to 8x8 blocks
    using atom_shape = typename MMA_ATOM::Shape_MNK;
    constexpr int MMA_M = get<0>(atom_shape{}) / 8;  // 16/8 = 2 for SM80
    constexpr int MMA_K = get<2>(atom_shape{}) / 8;  // 8/8=1 or 16/8=2

    // Divide M and K block dimensions by per-atom block counts
    auto divided = logical_divide(tensor,
        make_tile(_, make_layout(Int<MMA_M>{}), make_layout(Int<MMA_K>{})));
    // Shape: (2, (MMA_M, outer_M), (MMA_K, outer_K))

    // Build atom inner layout: (2, [MMA_M], [MMA_K]) with size-1 modes dropped
    auto dl = divided.layout();
    auto atom_inner = make_atom_inner(get<0>(dl), get<1,0>(dl), get<2,0>(dl));
    return make_tensor(divided.data(), make_layout(atom_inner, get<1,1>(dl), get<2,1>(dl)));
}

// Construct MMA B-operand from a block tensor of shape (2, N_blocks, K_blocks).
// Groups 8x8 blocks into atom-sized chunks matching MMA_ATOM's B-operand layout.
//
// MMA_ATOM's B-operand has (N_atom, K_atom) = (8, K_atom). In 8x8 blocks:
//   MMA_N = N_atom / 8 = 1,  MMA_K = K_atom / 8
//
// Input:  (2, total_N_blocks, total_K_blocks)
// Output: ((2, [MMA_N], [MMA_K]), outer_N, outer_K) — size-1 modes dropped
template <class MMA_ATOM, class Engine, class Layout>
__device__ __forceinline__
auto construct_operand_B(cute::Tensor<Engine, Layout>& tensor) {
    using namespace cute;
    static_assert(rank(Layout{}) == _3{}, "Expected shape (2, N_blocks, K_blocks)");
    static_assert(size<0>(Layout{}) == _2{}, "Mode 0 must be 2 (one u32 = 2 f16)");
    static_assert(sizeof(typename Engine::value_type) == 2, "Value type must be 16-bit");

    // Get atom B shape: (N_atom, K_atom) in elements, convert to 8x8 blocks
    using atom_shape = typename MMA_ATOM::Shape_MNK;
    constexpr int MMA_N = get<1>(atom_shape{}) / 8;  // 8/8 = 1 for SM80
    constexpr int MMA_K = get<2>(atom_shape{}) / 8;  // 8/8=1 or 16/8=2

    // Divide N and K block dimensions by per-atom block counts
    auto divided = logical_divide(tensor,
        make_tile(_, make_layout(Int<MMA_N>{}), make_layout(Int<MMA_K>{})));
    // Shape: (2, (MMA_N, outer_N), (MMA_K, outer_K))

    // Build atom inner layout: (2, [MMA_N], [MMA_K]) with size-1 modes dropped
    auto dl = divided.layout();
    auto atom_inner = make_atom_inner(get<0>(dl), get<1,0>(dl), get<2,0>(dl));
    return make_tensor(divided.data(), make_layout(atom_inner, get<1,1>(dl), get<2,1>(dl)));
}
