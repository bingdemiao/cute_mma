#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// Backward dA + dR kernel (restructured)
//
// GEMM: all warps cooperate on dH(BLK_M, BLK_K) += dC @ B^T
//   Warp layout: 2-row × (n_warps/2)-column.
//   Fragment shape per warp: (MMA, M_atoms=2, N_atoms=2)
//   Column pairs: (W0,W1), (W2,W3), ... share same K ranges.
//   Each column pair owns 2 reconn blocks at non-consecutive K offsets
//   (stride = n_warps_K * atom_N due to CuTe interleaving).
//
// Epilogue: per column pair, loop over 2 reconn blocks:
//   1. Both warps write rDH(_,_,rb) to sDH_exch via make_tiled_copy_C
//   2. Pairwise sync (bar.sync 64 threads)
//   3. dA: each warp loads own M-half from sDH_exch + R → MMA
//   4. dR: each warp computes partial dR from own M-half → atomicAdd
//
// Grid: (K/BLK_K, M/BLK_M) with z-curve ordering
// =============================================================================

template <int BLK_M, int BLK_K, int BLK_N, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(
    (size(typename cute::BwdDAdRParams::warp_layout_arb{}) +
     size(typename cute::BwdDAdRParams::warp_layout_ar{})) * 32)
bwd_dadr_kernel(
    const half_t* __restrict__ dC_ptr,  int ldDC,
    const half_t* __restrict__ A_ptr,   int ldA,
    const half_t* __restrict__ B_ptr,   int ldB,
    const half_t* __restrict__ R_ptr,   int ldR,
    half_t* __restrict__ dA_ptr,        int ldDA,
    float* __restrict__ dR_partial,
    int M, int N, int K,
    int n_groups, int n_buf_slots)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    static_assert(BLK_K % rs == 0, "BLK_K must be divisible by reconn_sz");
    static_assert(gs % BLK_N == 0, "group_size must be divisible by BLK_N");
    constexpr int n_reconn_blocks = BLK_K / rs;
    constexpr int tiles_per_group = gs / BLK_N;

    using WarpLayoutProd = typename BwdDAdRParams::warp_layout_arb;
    using WarpLayoutCons = typename BwdDAdRParams::warp_layout_ar;
    constexpr int n_warps  = size(WarpLayoutProd{}) + size(WarpLayoutCons{});
    constexpr int n_threads = n_warps * 32;

    constexpr int n_warps_M = 2;
    constexpr int n_warps_K = n_warps / 2;
    static_assert(n_warps_M * n_warps_K == n_warps, "total warps must be even");
    static_assert(BLK_M % n_warps_M == 0 && BLK_K % n_warps_K == 0);
    static_assert(n_reconn_blocks == n_warps, "Need exactly n_warps reconn blocks");
    using GemmWarpLayout = Layout<Shape<Int<n_warps_M>, Int<n_warps_K>>>;

    constexpr int HALF_M = BLK_M / 2;
    constexpr int atom_N = 8;
    constexpr int reconn_per_col = 2;
    constexpr int col_k_stride = n_warps_K * atom_N;

    int lane_idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;
    int warp_m = warp_idx / n_warps_K;
    int warp_k = warp_idx % n_warps_K;
    int col_k_base = warp_k * atom_N;
    int pair_bar = warp_k + 1;

    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start  = get<0>(grid_coord) * BLK_K;
    int m_start  = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;
    int m_valid  = min(m_start + BLK_M, M) - m_start;

    // === Shared memory ===
    auto sdC_layout = tile_to_shape(get_smem_atom(Int<BLK_N>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_N>{}, _2{}));
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}), make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom, make_shape(Int<BLK_K>{}, Int<BLK_N>{}, _2{}));
    auto sA_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}), make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto sR_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}), make_shape(Int<rs>{}, Int<BLK_K>{}, _2{}));
    // sDH_exch uses same swizzle as sA for make_tiled_copy_C compatibility
    auto sDH_exch_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    constexpr int sDR_acc_elems = rs * BLK_K;

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC      = make_tensor(make_smem_ptr(p), sdC_layout);      p += cosize(sdC_layout);
    auto sB       = make_tensor(make_smem_ptr(p), sB_layout);       p += cosize(sB_layout);
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR       = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sDH_exch = make_tensor(make_smem_ptr(p), sDH_exch_layout); p += cosize(sDH_exch_layout);
    p += (reinterpret_cast<uintptr_t>(p) % 4) ? (4 - reinterpret_cast<uintptr_t>(p) % 4) / sizeof(half_t) : 0;
    float* sDR_acc = reinterpret_cast<float*>(p);

    // === GEMM MMA ===
    using mma_atom_t = std::conditional_t<(BLK_N < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;
    auto mma = make_tiled_mma(mma_atom_t{}, GemmWarpLayout{}, Tile<Int<BLK_M>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    auto rDH = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}))));
    clear(rDH);
    auto tCid = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{})));

    // R2S for writing rDH to sDH_exch via make_tiled_copy_C
    using r2s_c_atom = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s_c = make_tiled_copy_C(r2s_c_atom{}, mma);
    auto r2s_c_thr = r2s_c.get_slice(threadIdx.x);
    auto tXrDH_c = r2s_c_thr.retile_S(rDH);
    auto tXsDH_c = r2s_c_thr.partition_D(sDH_exch);

    // GEMM LDSM
    constexpr int K_atom_g = (BLK_N < 16) ? 8 : 16;
    constexpr int a_u32 = (BLK_M * K_atom_g) / (n_warps_M * 64);
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    constexpr int b_u16 = (BLK_K * K_atom_g) / 32;
    using s2r_B = std::conditional_t<(b_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, half_t>,
        std::conditional_t<(b_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, half_t>,
            Copy_Atom<SM75_U16x2_LDSM_T, half_t>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dC{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(threadIdx.x);
    auto tXsdC = s2r_a_thr.partition_S(sdC);
    auto rdC_frag = thr_mma.make_fragment_A(thr_mma.partition_A(sdC(_,_,_0{})));
    auto tXrdC = s2r_a_thr.retile_D(rdC_frag);

    auto s2r_b = make_tiled_copy_B(s2r_B{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(threadIdx.x);
    auto tXsB = s2r_b_thr.partition_S(sB);
    auto rB_frag = thr_mma.make_fragment_B(thr_mma.partition_B(sB(_,_,_0{})));
    auto tXrB = s2r_b_thr.retile_D(rB_frag);

    // cp.async
    auto copy_dC = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_N>{}, Int<n_threads>{});
    auto thr_copy_dC = copy_dC.get_slice(threadIdx.x);
    auto tCsdC_cp = thr_copy_dC.partition_D(sdC);

    constexpr int bt0 = BLK_K / 8;
    constexpr int b_copy_threads = (n_threads < bt0 * BLK_N) ? n_threads : bt0 * BLK_N;
    constexpr int bt1 = b_copy_threads / bt0;
    constexpr int bv1 = BLK_N / bt1;
    static_assert(bt1 > 0 && bv1 > 0);
    auto copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto thr_copy_B = copy_B.get_slice(threadIdx.x);
    auto tCsB_cp = thr_copy_B.partition_D(sB);

    // === Tiled sub-views for reconn access (preserves swizzle) ===
    // Divide K dimension of sR, sA, sDH_exch into rs-sized reconn blocks
    auto sR_tiled = logical_divide(sR,
        make_tile(_, make_layout(Int<rs>{})));
    // sR_tiled: (rs, (rs, n_reconn_blocks), 2)
    // Access: sR_tiled(_, make_coord(_, rb_idx), pipe) → (rs, rs) sub-view

    auto sA_tiled = logical_divide(sA,
        make_tile(make_layout(Int<HALF_M>{}), make_layout(Int<rs>{})));
    // sA_tiled: ((HALF_M, 2), (rs, n_reconn_blocks))
    // Access: sA_tiled(make_coord(_, wm), make_coord(_, rb_idx)) → (HALF_M, rs)

    auto sDH_tiled = logical_divide(sDH_exch,
        make_tile(make_layout(Int<HALF_M>{}), make_layout(Int<rs>{})));
    // sDH_tiled: ((HALF_M, 2), (rs, n_reconn_blocks))

    // === Reconn dA MMA ===
    constexpr int WARP_M_reconn = HALF_M;
    using cons_mma_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto cons_mma = make_tiled_mma(cons_mma_atom{}, Layout<Shape<_1, _1>>{},
                                    Tile<Int<WARP_M_reconn>, Int<rs>>{});
    auto cons_thr = cons_mma.get_slice(lane_idx);
    auto tCdA_id = cons_thr.partition_C(
        make_identity_tensor(make_shape(Int<WARP_M_reconn>{}, Int<rs>{})));

    // LDSM for dA A-operand from sDH_exch: (HALF_M, rs) sub-view
    constexpr int cons_a_u32 = (WARP_M_reconn * rs) / 64;
    using cons_s2r_A = std::conditional_t<(cons_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(cons_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    auto cons_s2r_a = make_tiled_copy_A(cons_s2r_A{}, cons_mma);
    auto cons_s2r_a_thr = cons_s2r_a.get_slice(lane_idx);

    // LDSM for dA B-operand (R): (rs, rs) sub-view
    constexpr int cons_b_u32 = (rs * rs) / 64;
    using cons_s2r_B = std::conditional_t<(cons_b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    auto cons_s2r_b = make_tiled_copy_B(cons_s2r_B{}, cons_mma);
    auto cons_s2r_b_thr = cons_s2r_b.get_slice(lane_idx);

    using frag_c_type = decltype(cons_thr.make_fragment_C(cons_thr.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<WARP_M_reconn>{}, Int<rs>{}), LayoutRight{})))));
    frag_c_type rDA[reconn_per_col];
    for (int i = 0; i < reconn_per_col; ++i) clear(rDA[i]);

    // === dR MMA: K=BLK_M reduction, only warp_m=0 computes ===
    auto dr_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
        Layout<Shape<_1, _1>>{}, Tile<Int<rs>, Int<rs>>{});
    auto dr_thr = dr_mma.get_slice(lane_idx);
    auto dr_C_dummy = make_tensor(static_cast<half_t*>(nullptr),
        make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}));
    auto tCdr_id = dr_thr.partition_C(make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));

    // dR LDSM: load (BLK_M, rs) sub-views naturally, then movmatrix to transpose
    // The dR MMA expects A(rs, BLK_M) but smem has (BLK_M, rs). LDSM_N loads
    // (BLK_M, rs) treating BLK_M as M-dim, then movmatrix transposes each 8x8 block.
    // dr_load_mma uses SM80_16x8x8 (K_atom=8=rs)
    // A(BLK_M, K_atom=8): per-atom A size
    constexpr int dr_K_atom = 8;
    constexpr int dr_a_u32 = (BLK_M * dr_K_atom) / (1 * 64);
    using dr_s2r_A = std::conditional_t<(dr_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(dr_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    // B(rs=8, K_atom=8): per-atom B size
    constexpr int dr_b_u32 = (rs * dr_K_atom) / 64;
    using dr_s2r_B = std::conditional_t<(dr_b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    // For dR, we create a helper MMA matching the smem orientation (BLK_M, rs)
    // to get correct LDSM partitions. Use K_atom=8 (=rs) to match the rs-column sub-view.
    auto dr_load_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>{},
        Layout<Shape<_1, _1>>{}, Tile<Int<BLK_M>, Int<rs>>{});
    auto dr_load_thr = dr_load_mma.get_slice(lane_idx);
    auto dr_s2r_a = make_tiled_copy_A(dr_s2r_A{}, dr_load_mma);
    auto dr_s2r_a_thr = dr_s2r_a.get_slice(lane_idx);
    auto dr_s2r_b = make_tiled_copy_B(dr_s2r_B{}, dr_load_mma);
    auto dr_s2r_b_thr = dr_s2r_b.get_slice(lane_idx);

    // Keep identity tensors for dR C-fragment write
    auto tAdr_id = dr_thr.partition_A(make_identity_tensor(make_shape(Int<rs>{}, Int<BLK_M>{})));
    auto tBdr_id = dr_thr.partition_B(make_identity_tensor(make_shape(Int<rs>{}, Int<BLK_M>{})));
    auto dr_A_dummy = make_tensor(static_cast<half_t*>(nullptr),
        make_layout(make_shape(Int<rs>{}, Int<BLK_M>{}), LayoutRight{}));
    auto dr_B_dummy = make_tensor(static_cast<half_t*>(nullptr),
        make_layout(make_shape(Int<rs>{}, Int<BLK_M>{}), LayoutRight{}));

    // === Load sA + sR ===
    for (int i = threadIdx.x; i < BLK_M * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sA(r, c) = (m_start + r < M && k_start + c < K) ?
            *(A_ptr + (m_start + r) * ldA + k_start + c) : half_t(0);
    }
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c, _0{}) = (k_start + c < K) ? *(R_ptr + r * ldR + k_start + c) : half_t(0);
    }
    __syncthreads();

    int r_pipe_w = 1, r_pipe_r = 0;
    int n_total_tiles = N / BLK_N;

    {
        auto gdC0 = make_tensor(make_gmem_ptr(dC_ptr + m_start * ldDC),
            make_layout(make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_stride(ldDC, Int<1>{})));
        copy(copy_dC, thr_copy_dC.partition_S(gdC0), tCsdC_cp(_,_,_,_0{}));
        auto gB0 = make_tensor(make_gmem_ptr(B_ptr + k_start),
            make_layout(make_shape(Int<BLK_K>{}, Int<BLK_N>{}), make_stride(Int<1>{}, ldB)));
        copy(copy_B, thr_copy_B.partition_S(gB0), tCsB_cp(_,_,_,_0{}));
        cp_async_fence();
    }

    int dc_b_pipe_w = 1, dc_b_pipe_r = 0;

    // === Main loop ===
    for (int nt = 0; nt < n_total_tiles; ++nt) {
        int tile_in_group = nt % tiles_per_group;

        if (nt + 1 < n_total_tiles) {
            int next_n = (nt + 1) * BLK_N;
            auto gdC_next = make_tensor(make_gmem_ptr(dC_ptr + m_start * ldDC + next_n),
                make_layout(make_shape(Int<BLK_M>{}, Int<BLK_N>{}), make_stride(ldDC, Int<1>{})));
            copy(copy_dC, thr_copy_dC.partition_S(gdC_next), tCsdC_cp(_,_,_,dc_b_pipe_w));
            auto gB_next = make_tensor(make_gmem_ptr(B_ptr + next_n * ldB + k_start),
                make_layout(make_shape(Int<BLK_K>{}, Int<BLK_N>{}), make_stride(Int<1>{}, ldB)));
            copy(copy_B, thr_copy_B.partition_S(gB_next), tCsB_cp(_,_,_,dc_b_pipe_w));
        }
        cp_async_fence();
        cp_async_wait<1>();
        __syncthreads();

        copy(s2r_dC{}, tXsdC(_,_,_,dc_b_pipe_r), tXrdC);
        copy(s2r_B{}, tXsB(_,_,_,dc_b_pipe_r), tXrB);
        gemm(mma, rdC_frag, rB_frag, rDH);

        dc_b_pipe_w = dc_b_pipe_r;
        dc_b_pipe_r = 1 - dc_b_pipe_r;

        // === Group boundary epilogue ===
        if (tile_in_group == tiles_per_group - 1) {
            int g = nt / tiles_per_group;

            if (g + 1 < n_groups) {
                for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
                    int r = i / BLK_K, c = i % BLK_K;
                    sR(r, c, r_pipe_w) = (k_start + c < K) ?
                        *(R_ptr + ((g + 1) * rs + r) * ldR + k_start + c) : half_t(0);
                }
            }

            for (int i = threadIdx.x; i < sDR_acc_elems; i += n_threads)
                sDR_acc[i] = 0.0f;
            __syncthreads();

            // === Per-column-pair reconn block loop ===
            for (int rb = 0; rb < reconn_per_col; ++rb) {
                int blk_k_off = col_k_base + rb * col_k_stride;

                // Write rDH sub-fragment to sDH_exch
                copy(r2s_c_atom{}, tXrDH_c(_,_,rb), tXsDH_c(_,_,rb));

                asm volatile("bar.sync %0, 64;\n" : : "r"(pair_bar));

                // --- dA ---
                {
                    int rb_idx = blk_k_off / rs;  // reconn block index in K

                    // LDSM load A-operand from sDH_exch via tiled sub-view
                    Tensor sDH_rb = sDH_tiled(make_coord(_, warp_m), make_coord(_, rb_idx));
                    auto tXsDH_rb = cons_s2r_a_thr.partition_S(sDH_rb);
                    auto rA_dA = cons_thr.make_fragment_A(cons_thr.partition_A(sDH_rb));
                    auto tXrA_dA = cons_s2r_a_thr.retile_D(rA_dA);
                    copy(cons_s2r_A{}, tXsDH_rb, tXrA_dA);

                    // LDSM load B-operand (R): reconn wants B[j,i] = sR(i, k_off+j) = R^T
                    // Load natural sR sub-tile via LDSM, then movmatrix to transpose
                    Tensor sR_rb = sR_tiled(_, make_coord(_, rb_idx), r_pipe_r);
                    auto tXsR_rb = cons_s2r_b_thr.partition_S(sR_rb);
                    auto rB_R = cons_thr.make_fragment_B(cons_thr.partition_B(sR_rb));
                    auto tXrB_R = cons_s2r_b_thr.retile_D(rB_R);
                    copy(cons_s2r_B{}, tXsR_rb, tXrB_R);
                    // movmatrix: transpose 8x8 in registers to get reconn convention
                    movmatrix_trans_b16(rB_R);

                    if constexpr (!GATED) {
                        gemm(cons_mma, rA_dA, rB_R, rDA[rb]);
                    } else {
                        // AR: LDSM load sA via tiled sub-view
                        Tensor sA_rb = sA_tiled(make_coord(_, warp_m), make_coord(_, rb_idx));
                        auto tXsA_rb = cons_s2r_a_thr.partition_S(sA_rb);
                        auto rA_sA = cons_thr.make_fragment_A(cons_thr.partition_A(sA_rb));
                        auto tXrA_sA = cons_s2r_a_thr.retile_D(rA_sA);
                        copy(cons_s2r_A{}, tXsA_rb, tXrA_sA);

                        // R for AR: B[r,k] = sR(r, k_off+k) — natural convention, no transpose
                        // LDSM loads directly
                        auto rB_R_ar = cons_thr.make_fragment_B(cons_thr.partition_B(sR_rb));
                        auto tXrB_ar = cons_s2r_b_thr.retile_D(rB_R_ar);
                        copy(cons_s2r_B{}, tXsR_rb, tXrB_ar);
                        frag_c_type rAR; clear(rAR);
                        gemm(cons_mma, rA_sA, rB_R_ar, rAR);

                        constexpr int n_frag = decltype(size(rAR))::value;
                        constexpr int n_pairs = n_frag / 2;
                        __half2 rSigma[n_pairs], rAR_h2[n_pairs];
                        for (int pp = 0; pp < n_pairs; ++pp) {
                            rAR_h2[pp] = __floats2half2_rn(rAR(pp * 2), rAR(pp * 2 + 1));
                            rSigma[pp] = sigmoid_h2(rAR_h2[pp]);
                        }

                        // Step A
                        for (int pp = 0; pp < n_pairs; ++pp) {
                            int idx = pp * 2;
                            auto c0 = tCdA_id(idx), c1 = tCdA_id(idx + 1);
                            int mi0 = get<0>(c0), mi1 = get<0>(c1);
                            if (mi0 >= HALF_M) continue;
                            int gm0 = warp_m * HALF_M + mi0, gm1 = warp_m * HALF_M + mi1;
                            __half2 dH2 = load_half2(sDH_exch(gm0, blk_k_off + get<1>(c0)),
                                                     sDH_exch(gm1, blk_k_off + get<1>(c1)));
                            __half2 result = __hmul2(__hmul2(dH2, rAR_h2[pp]), rSigma[pp]);
                            rDA[rb](idx)     += __half2float(__low2half(result));
                            rDA[rb](idx + 1) += __half2float(__high2half(result));
                        }

                        // Step B
                        for (int pp = 0; pp < n_pairs; ++pp) {
                            int idx = pp * 2;
                            auto c0 = tCdA_id(idx), c1 = tCdA_id(idx + 1);
                            int mi0 = get<0>(c0), mi1 = get<0>(c1);
                            int ri0 = get<1>(c0), ri1 = get<1>(c1);
                            if (mi0 >= HALF_M) continue;
                            int gm0 = warp_m * HALF_M + mi0, gm1 = warp_m * HALF_M + mi1;
                            __half2 sp = silu_prime_h2(rAR_h2[pp], rSigma[pp]);
                            __half2 dH2 = load_half2(sDH_exch(gm0, blk_k_off + ri0),
                                                     sDH_exch(gm1, blk_k_off + ri1));
                            __half2 a2 = load_half2(sA(gm0, blk_k_off + ri0),
                                                    sA(gm1, blk_k_off + ri1));
                            __half2 dS = __hmul2(__hmul2(dH2, a2), sp);
                            store_half2(sDH_exch(gm0, blk_k_off + ri0),
                                        sDH_exch(gm1, blk_k_off + ri1), dS);
                        }
                        asm volatile("bar.sync %0, 64;\n" : : "r"(pair_bar));

                        // Step C: reload dS from sDH_exch via LDSM
                        // sDH_rb still points to the right sub-view (Step B wrote in-place)
                        copy(cons_s2r_A{}, tXsDH_rb, tXrA_dA);  // reloads modified data
                        gemm(cons_mma, rA_dA, rB_R, rDA[rb]);
                    }
                }

                // --- dR: only warp_m=0 computes (full BLK_M reduction) ---
                if (warp_m == 0) {
                    auto rA_dr = dr_thr.make_fragment_A(dr_thr.partition_A(dr_A_dummy));
                    auto rB_dr = dr_thr.make_fragment_B(dr_thr.partition_B(dr_B_dummy));
                    auto rDR   = dr_thr.make_fragment_C(dr_thr.partition_C(dr_C_dummy));
                    clear(rDR);

                    int dr_rb_idx = blk_k_off / rs;

                    // dR operands: transposed access (scalar load)
                    // A[i, k] = sDH_exch[k, blk_k_off + i], B[j, k] = sA[k, blk_k_off + j]
                    auto rA_dr2 = dr_thr.make_fragment_A(dr_thr.partition_A(dr_A_dummy));
                    auto rB_dr2 = dr_thr.make_fragment_B(dr_thr.partition_B(dr_B_dummy));
                    for (int f = 0; f < size(rA_dr2); ++f) {
                        auto coord = tAdr_id(f);
                        int i = get<0>(coord), k = get<1>(coord);
                        rA_dr2(f) = (i < rs && k < m_valid) ? sDH_exch(k, blk_k_off + i) : half_t(0);
                    }
                    for (int f = 0; f < size(rB_dr2); ++f) {
                        auto coord = tBdr_id(f);
                        int j = get<0>(coord), k = get<1>(coord);
                        rB_dr2(f) = (j < rs && k < m_valid) ? sA(k, blk_k_off + j) : half_t(0);
                    }

                    gemm(dr_mma, rA_dr2, rB_dr2, rDR);

                    // Exclusive write — no atomicAdd needed
                    for (int f = 0; f < size(rDR); ++f) {
                        auto coord = tCdr_id(f);
                        int i = get<0>(coord), j = get<1>(coord);
                        if (i < rs && j < rs)
                            sDR_acc[i * BLK_K + blk_k_off + j] = rDR(f);
                    }
                }

                asm volatile("bar.sync %0, 64;\n" : : "r"(pair_bar));
            }

            __syncthreads();

            {
                int dR_rows = n_groups * rs;
                for (int idx = threadIdx.x; idx < sDR_acc_elems; idx += n_threads) {
                    float val = sDR_acc[idx];
                    if (val != 0.0f) {
                        int i = idx / BLK_K;
                        int kj = idx % BLK_K;
                        atomicAdd(&dR_partial[buf_slot * dR_rows * K +
                            (g * rs + i) * K + k_start + kj], val);
                    }
                }
            }

            clear(rDH);
            r_pipe_w = r_pipe_r;
            r_pipe_r = 1 - r_pipe_r;
            __syncthreads();
        }
    }

    // Write dA
    for (int rb = 0; rb < reconn_per_col; ++rb) {
        int blk_k_off = col_k_base + rb * col_k_stride;
        for (int f = 0; f < size(rDA[rb]); ++f) {
            auto coord = tCdA_id(f);
            int mi = get<0>(coord) + warp_m * HALF_M;
            int ki = get<1>(coord) + blk_k_off;
            if (m_start + mi < M && k_start + ki < K)
                *(dA_ptr + (m_start + mi) * ldDA + k_start + ki) = half_t(rDA[rb](f));
        }
    }
}


__global__ void dR_reduce_kernel(
    const float* __restrict__ dR_partial, half* __restrict__ dR,
    int dR_elements, int n_buf_slots)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dR_elements) return;
    float sum = 0.0f;
    for (int s = 0; s < n_buf_slots; ++s)
        sum += dR_partial[s * dR_elements + idx];
    dR[idx] = __float2half(sum);
}


void oft_backward_dA_dR_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* B,  int ldB,
    half const* R,  int ldR,
    half* dA, int ldDA,
    half* dR, int ldDR,
    cudaStream_t stream)
{
    constexpr int gs = CurrKernelParams::group_size;
    constexpr int rs = CurrKernelParams::reconn_sz;
    constexpr int BLK_M = cute::BwdDAdRParams::bM;
    constexpr int BLK_K = cute::BwdDAdRParams::bK;
    constexpr int BLK_N = cute::BwdDAdRParams::bK_inner;
    constexpr int n_buf_slots_param = cute::BwdDAdRParams::n_buf_slots;
    using wlp_t = typename cute::BwdDAdRParams::warp_layout_arb;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_ar;
    constexpr int n_threads = (size(wlp_t{}) + size(wlc_t{})) * 32;

    int n_groups = n / gs;
    int n_k_tiles = (k + BLK_K - 1) / BLK_K;
    int n_m_tiles = (m + BLK_M - 1) / BLK_M;
    int n_buf_slots = min(n_buf_slots_param, n_m_tiles);
    if (n_buf_slots < 1) n_buf_slots = 1;
    int dR_elements = n_groups * rs * k;

    float* dR_partial_buf = nullptr;
    cudaMalloc(&dR_partial_buf, (int64_t)n_buf_slots * dR_elements * sizeof(float));
    cudaMemsetAsync(dR_partial_buf, 0, (int64_t)n_buf_slots * dR_elements * sizeof(float), stream);

    auto smem_n = get_smem_atom(cute::Int<BLK_N>{});
    auto smem_k = get_smem_atom(cute::Int<BLK_K>{});
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(cute::Int<BLK_K>{}, _8{}),
                    make_stride(_1{}, cute::Int<BLK_K>{})));
    int smem = (cosize(tile_to_shape(smem_n, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_N>{}, _2{})))
                + cosize(tile_to_shape(sB_atom,
                    make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_N>{}, _2{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{}, _2{})))
                + BLK_M * BLK_K)              // sDH_exch
               * sizeof(half_t)
               + rs * BLK_K * sizeof(float)   // sDR_acc
               + 256;

    dim3 grid(n_k_tiles * n_m_tiles);
    dim3 block(n_threads);

#if OFT_GATED
    auto kernel = bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, true>;
#else
    auto kernel = bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, false>;
#endif
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<half_t const*>(dC), ldDC,
        reinterpret_cast<half_t const*>(A), ldA,
        reinterpret_cast<half_t const*>(B), ldB,
        reinterpret_cast<half_t const*>(R), ldR,
        reinterpret_cast<half_t*>(dA), ldDA,
        dR_partial_buf, m, n, k, n_groups, n_buf_slots);

    {
        int threads = 256;
        int blocks_r = (dR_elements + threads - 1) / threads;
        dR_reduce_kernel<<<blocks_r, threads, 0, stream>>>(
            dR_partial_buf, dR, dR_elements, n_buf_slots);
    }

    cudaStreamSynchronize(stream);
    cudaFree(dR_partial_buf);
}
