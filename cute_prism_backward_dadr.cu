#include "cute_prism_backward_dadr.hpp"
#include "cute_prism_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_prism_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// Backward dA + dR kernel — Producer-Consumer Architecture
//
// Grid: (K/BLK_K) × (M/BLK_M) with z-curve ordering.
// Producer warps: dH(BLK_M, BLK_K) = dC(BLK_M, BLK_N) @ B(BLK_K, BLK_N)^T per group
// Consumer warps: 1D along K. Each warp owns WARP_K = BLK_K / n_consumer_warps.
//   dA: per reconn block MMA (F32 accum), accumulated across groups
//   dR: LDSM load + in-register transpose + MMA M-reduction, atomicAdd to gmem
// =============================================================================

// =============================================================================
// Producer: dH GEMM with F16 accumulation
// =============================================================================
template <class TensorGDC, class SmemDC, class TiledCopyDC,
          class TensorGB,  class SmemB,  class TiledCopyB,
          class SmemDH, class TilesPerGroup, class ReconnectSize,
          class WarpLayoutProd, class WarpLayoutCons>
__device__ static inline
void dH_producer(
    TensorGDC const &gdC, SmemDC &sdC, TiledCopyDC copy_dc,
    TensorGB  const &gB,  SmemB  &sB,  TiledCopyB  copy_b,
    SmemDH &sdH, TilesPerGroup tiles_per_group, ReconnectSize,
    int thread_idx,
    WarpLayoutProd wl_prod, WarpLayoutCons wl_cons)
{
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;

    constexpr int BAR_READY_BASE = 0;
    constexpr int BAR_CONSUMED_BASE = bP_dh;
    constexpr uint32_t n_prod = size(WarpLayoutProd{}) * 32;
    constexpr uint32_t n_total = n_prod + size(WarpLayoutCons{}) * 32;

    auto lane_idx = thread_idx % 32;
    constexpr int _BLK_M = decltype(size<0>(sdC))::value;
    constexpr int _BLK_N = decltype(size<1>(sdC))::value;
    constexpr int _BLK_K = decltype(size<1>(sdH))::value;

    // -- cp.async partitions (gmem → smem) --
    ThrCopy thr_cp_dc = copy_dc.get_slice(thread_idx);
    Tensor tDCg = thr_cp_dc.partition_S(gdC);  // (CPY,...,n_tiles)
    Tensor tDCs = thr_cp_dc.partition_D(sdC);   // (CPY,...,PIPE)

    ThrCopy thr_cp_b = copy_b.get_slice(thread_idx);
    Tensor tBg = thr_cp_b.partition_S(gB);      // (CPY,...,n_tiles)
    Tensor tBs = thr_cp_b.partition_D(sB);       // (CPY,...,PIPE)

    auto PIPE_DC_B = size<3>(tDCs);  // pipeline depth for dC/B
    int n_tiles = size<3>(tDCg);     // total N tiles

    // -- MMA: F16 TN for dH(BLK_M, BLK_K) += dC(BLK_M, BLK_N) @ B(BLK_K, BLK_N)^T --
    using mma_atom_t = std::conditional_t<(_BLK_N < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;

    auto mma = make_tiled_mma(mma_atom_t{}, wl_prod, Tile<Int<_BLK_M>, Int<_BLK_K>>{});
    auto thr_mma = mma.get_slice(thread_idx);

    auto rDH = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<_BLK_M>{}, Int<_BLK_K>{}), LayoutRight{}))));

    // LDSM s2r for dC (A-operand, LDSM_N) and B (B-operand, LDSM_T)
    constexpr int K_atom = (_BLK_N < 16) ? 8 : 16;
    constexpr int a_u32 = (_BLK_M / size(WarpLayoutProd{}) * K_atom) / 64;
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;

    constexpr int b_u16 = (_BLK_K * K_atom) / 32;
    using s2r_B = std::conditional_t<(b_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, half_t>,
        std::conditional_t<(b_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, half_t>,
            Copy_Atom<SM75_U16x2_LDSM_T, half_t>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dC{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(thread_idx);
    Tensor tXsdC = s2r_a_thr.partition_S(sdC);
    auto rdC = thr_mma.make_fragment_A(thr_mma.partition_A(sdC(_,_,_0{})));
    auto tXrdC = s2r_a_thr.retile_D(rdC);

    auto s2r_b = make_tiled_copy_B(s2r_B{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(thread_idx);
    Tensor tXsB = s2r_b_thr.partition_S(sB);
    auto rB = thr_mma.make_fragment_B(thr_mma.partition_B(sB(_,_,_0{})));
    auto tXrB = s2r_b_thr.retile_D(rB);

    // R2S for writing dH to sdH via make_tiled_copy_C
    using r2s_atom = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s_c = make_tiled_copy_C(r2s_atom{}, mma);
    auto r2s_thr = r2s_c.get_slice(thread_idx);
    auto tXrDH_c = r2s_thr.retile_S(rDH);

    int dh_pipe_w = 0;
    int n_groups = n_tiles / tiles_per_group;

    for (int g = 0; g < n_groups; ++g) {
        clear(rDH);
        int pipe_w = PIPE_DC_B - 1, pipe_r = 0;
        int base_tile = g * tiles_per_group;

        // Prefill dC/B pipeline
        CUTE_UNROLL
        for (int p = 0; p < PIPE_DC_B - 1 && p < tiles_per_group; ++p) {
            copy(copy_dc, tDCg(_,_,_,base_tile + p), tDCs(_,_,_,p));
            copy(copy_b,  tBg(_,_,_,base_tile + p),  tBs(_,_,_,p));
            cp_async_fence();
        }

        // Inner N loop
        CUTE_NO_UNROLL
        for (int t = 0; t < tiles_per_group; ++t) {
            int t_next = t + PIPE_DC_B - 1;
            if (t_next < tiles_per_group) {
                copy(copy_dc, tDCg(_,_,_,base_tile + t_next), tDCs(_,_,_,pipe_w));
                copy(copy_b,  tBg(_,_,_,base_tile + t_next),  tBs(_,_,_,pipe_w));
            }
            cp_async_fence();
            cp_async_wait<decltype(PIPE_DC_B)::value - 1>();
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_prod));

            copy(s2r_dC{}, tXsdC(_,_,_,pipe_r), tXrdC);
            copy(s2r_B{},  tXsB(_,_,_,pipe_r),  tXrB);
            gemm(mma, rdC, rB, rDH);

            pipe_w = pipe_r;
            if (++pipe_r == PIPE_DC_B) pipe_r = 0;
        }

        cp_async_wait<0>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_prod));

        // Wait for consumer to free sdH slot, write, signal READY
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dh_pipe_w + BAR_CONSUMED_BASE), "n"(n_total));

        copy(r2s_atom{}, tXrDH_c, r2s_thr.partition_D(sdH(_,_,dh_pipe_w)));
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_prod));

        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dh_pipe_w + BAR_READY_BASE), "n"(n_total));
        if (++dh_pipe_w == bP_dh) dh_pipe_w = 0;
    }
}

// =============================================================================
// Consumer: dA accumulation + dR from sdH
// =============================================================================
template <class TensorGA, class SmemA,  class TiledCopyA,
          class TensorGR, class SmemR,  class TiledCopyR,
          class SmemDH,
          class TensorGDA, class TensorGDR, class ReconnectSize,
          class WarpLayoutProd, class WarpLayoutCons>
__device__ static inline
void dAdR_consumer(
    TensorGA const &gA, SmemA &sA, TiledCopyA copy_a,
    TensorGR const &gR, SmemR &sR, TiledCopyR copy_r,
    SmemDH const &sdH,
    TensorGDA &gdA, TensorGDR &gdR_partial,
    ReconnectSize reconn_sz,
    int thread_idx,
    WarpLayoutProd wl_prod, WarpLayoutCons wl_cons)
{
    constexpr int rs = decltype(reconn_sz)::value;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;
    constexpr int n_cw = size(WarpLayoutCons{});
    constexpr uint32_t n_cons = n_cw * 32;
    constexpr uint32_t n_total = size(WarpLayoutProd{}) * 32 + n_cons;

    constexpr int BAR_READY_BASE = 0;
    constexpr int BAR_CONSUMED_BASE = bP_dh;

    auto lane_idx = thread_idx % 32;
    auto warp_idx = thread_idx / 32;
    constexpr int _BLK_M = decltype(size<0>(sA))::value;
    constexpr int _BLK_K = decltype(size<1>(sA))::value;
    constexpr int WARP_K = _BLK_K / n_cw;
    constexpr int n_reconn = WARP_K / rs;
    int n_groups = size<2>(gR);

    // -- Tiled sub-views for reconn block access --
    auto sdH_rb = logical_divide(sdH, make_tile(_, make_layout(Int<rs>{})));
    auto sA_rb  = logical_divide(sA,  make_tile(_, make_layout(Int<rs>{})));
    auto sR_rb  = logical_divide(sR,  make_tile(_, make_layout(Int<rs>{})));

    // -- dA MMA: (BLK_M, rs) += (BLK_M, rs) @ (rs, rs)  [F32 accum, single warp] --
    using dA_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    using dA_atom_f16 = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;

    auto dA_mma = make_tiled_mma(dA_atom{}, Layout<Shape<_1,_1>>{},
                                  Tile<Int<_BLK_M>, Int<rs>>{});
    auto dA_thr = dA_mma.get_slice(lane_idx);

    auto dA_C_dummy = make_tensor(static_cast<half_t*>(nullptr),
        make_layout(make_shape(Int<_BLK_M>{}, Int<rs>{}), LayoutRight{}));
    auto tCdA_id = dA_thr.partition_C(make_identity_tensor(
        make_shape(Int<_BLK_M>{}, Int<rs>{})));

    using frag_c = decltype(dA_thr.make_fragment_C(dA_thr.partition_C(dA_C_dummy)));
    frag_c rDA[n_reconn];
    #pragma unroll
    for (int b = 0; b < n_reconn; ++b) clear(rDA[b]);

    // LDSM for dA: A-op from sdH, B-op from sR
    constexpr int dA_K = (rs < 16) ? 8 : 16;
    constexpr int dA_a_u32 = (_BLK_M * dA_K) / 64;
    using dA_s2r_A_atom = std::conditional_t<(dA_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(dA_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    constexpr int dA_b_u32 = (rs * dA_K) / 64;
    using dA_s2r_B_atom = std::conditional_t<(dA_b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(dA_b_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;

    auto dA_s2r_a = make_tiled_copy_A(dA_s2r_A_atom{}, dA_mma);
    auto dA_s2r_a_thr = dA_s2r_a.get_slice(lane_idx);
    auto dA_s2r_b = make_tiled_copy_B(dA_s2r_B_atom{}, dA_mma);
    auto dA_s2r_b_thr = dA_s2r_b.get_slice(lane_idx);

    // -- dR MMA: (dR_M, rs) reduced over BLK_M via atom-level gemm --
    // For rs >= 16: dR_M = rs, no padding, construct_operand_A works directly.
    // For rs < 16:  dR_M = 16, stride-0 padding on the K-mode of the smem view
    //               makes LDSM produce a (BLK_M, 16) fragment; extra rows discarded.
    constexpr int dR_M = (rs < 16) ? 16 : rs;
    constexpr int dR_pad = dR_M / rs;  // 1 when rs >= 16, 2 when rs = 8

    // Use a helper MMA for loading (BLK_M, dR_M) and for C-fragment / identity
    using dR_load_atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using dR_load_atom_f16 = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    auto dR_load_mma = make_tiled_mma(dR_load_atom{}, Layout<Shape<_1,_1>>{},
                                       Tile<Int<_BLK_M>, Int<dR_M>>{});
    auto dR_load_thr = dR_load_mma.get_slice(lane_idx);

    // LDSM for dR loading: A-operand from padded (BLK_M, dR_M) smem views
    constexpr int dR_a_u32 = (_BLK_M * 16) / 64;  // K_atom = 16 for SM80_16x8x16
    using dR_s2r_A_atom = std::conditional_t<(dR_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(dR_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    auto dR_s2r_a = make_tiled_copy_A(dR_s2r_A_atom{}, dR_load_mma);
    auto dR_s2r_a_thr = dR_s2r_a.get_slice(lane_idx);

    // dR identity for output writeback (created once, reused per block)
    auto dR_id_mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                                     Layout<Shape<_1,_1>>{}, Tile<Int<dR_M>, Int<rs>>{});
    auto dR_C_id = dR_id_mma.get_slice(lane_idx).partition_C(
        make_identity_tensor(make_shape(Int<dR_M>{}, Int<rs>{})));

    // -- Load sA once via cp.async --
    {
        ThrCopy thr_cp_a = copy_a.get_slice(thread_idx);
        Tensor tAg = thr_cp_a.partition_S(gA);
        Tensor tAs = thr_cp_a.partition_D(sA);
        copy(copy_a, tAg, tAs);
        cp_async_fence();
        cp_async_wait<0>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_cons));
    }

    // Pre-signal CONSUMED for all sdH slots
    for (int bid = BAR_CONSUMED_BASE; bid < BAR_CONSUMED_BASE + bP_dh; ++bid)
        asm volatile("bar.arrive %0, %1;\n" : : "r"(bid), "n"(n_total));

    // -- R pipeline: prefill --
    ThrCopy thr_cp_r = copy_r.get_slice(thread_idx);
    Tensor tRg = thr_cp_r.partition_S(gR);  // (CPY,...,n_groups)
    Tensor tRs = thr_cp_r.partition_D(sR);  // (CPY,...,PIPE_R)
    if (n_groups > 0) {
        copy(copy_r, tRg(_,_,_,0), tRs(_,_,_,_0{}));
        cp_async_fence();
    }

    int r_pipe_w = 1, r_pipe_r = 0, dh_pipe_r = 0;

    for (int g = 0; g < n_groups; ++g) {
        // Load next R
        if (g + 1 < n_groups) {
            copy(copy_r, tRg(_,_,_,g + 1), tRs(_,_,_,r_pipe_w));
        }
        cp_async_fence();
        cp_async_wait<0>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_cons));

        // Wait for producer's dH
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dh_pipe_r + BAR_READY_BASE), "n"(n_total));

        // Process each reconn block in this warp's K-range
        #pragma unroll
        for (int b = 0; b < n_reconn; ++b) {
            int rb_idx = warp_idx * (WARP_K / rs) + b;
            int blk_k_off = rb_idx * rs;

            // -- sdH sub-view (BLK_M, rs) and sR sub-view (rs, rs) for this block --
            Tensor sdH_blk = sdH_rb(_, make_coord(_, rb_idx), dh_pipe_r);
            Tensor sR_blk  = sR_rb(_, make_coord(_, rb_idx), r_pipe_r);

            // LDSM: load dH block → rDH_frag (A-operand of dA)
            auto tXsDH = dA_s2r_a_thr.partition_S(sdH_blk);
            auto rDH_frag = dA_thr.make_fragment_A(dA_thr.partition_A(sdH_blk));
            auto tXrDH = dA_s2r_a_thr.retile_D(rDH_frag);
            copy(dA_s2r_A_atom{}, tXsDH, tXrDH);

            // LDSM: load R block → rR_nat, then transpose for dA = dH @ R
            auto tXsR = dA_s2r_b_thr.partition_S(sR_blk);
            auto rR_nat = dA_thr.make_fragment_B(dA_thr.partition_B(sR_blk));
            auto tXrR = dA_s2r_b_thr.retile_D(rR_nat);
            copy(dA_s2r_B_atom{}, tXsR, tXrR);

            auto blocked_R = blocken_operand_B<dA_atom_f16>(rR_nat);
            auto transposed_R = inplace_transpose(blocked_R);
            auto rR_T = construct_operand_B<dA_atom_f16>(transposed_R);

            // dA += dH @ R  (TN GEMM with R transposed ⇒ computes dH @ R)
            gemm(dA_mma, rDH_frag, rR_T, rDA[b]);

            // -- dR: dH^T(dR_M, BLK_M) @ A^T(rs, BLK_M) via stride-0 pad + transpose + atom gemm --
            {
                // Stride-0 pad sdH block from (BLK_M, rs) to (BLK_M, dR_M), then LDSM load
                // When dR_pad == 1 (rs >= 16), pad is a no-op; when dR_pad == 2 (rs = 8),
                // K-mode becomes (rs, 2) with stride-0, then group_modes flattens to dR_M.
                auto sdH_padded = pad_mode_stride0<1, dR_pad>(sdH_blk);
                auto sdH_dr = group_modes<1, rank(sdH_padded.layout())>(sdH_padded);
                auto tXsDH_dr = dR_s2r_a_thr.partition_S(sdH_dr);
                auto rDH_dr = dR_load_thr.make_fragment_A(dR_load_thr.partition_A(sdH_dr));
                auto tXrDH_dr = dR_s2r_a_thr.retile_D(rDH_dr);
                copy(dR_s2r_A_atom{}, tXsDH_dr, tXrDH_dr);

                // Load sA block WITHOUT padding: (BLK_M, rs) via dA MMA's LDSM
                // B-operand of dR needs (rs, BLK_M) which has outer_N = rs/8 — no padding needed.
                Tensor sA_blk = sA_rb(_, make_coord(_, rb_idx));
                auto tXsA_dr = dA_s2r_a_thr.partition_S(sA_blk);
                auto rA_dr = dA_thr.make_fragment_A(dA_thr.partition_A(sA_blk));
                auto tXrA_dr = dA_s2r_a_thr.retile_D(rA_dr);
                copy(dA_s2r_A_atom{}, tXsA_dr, tXrA_dr);

                // Transpose dH: (BLK_M, dR_M) → (dR_M, BLK_M) → construct A-operand
                auto blocked_dH_dr = blocken_operand_A<dR_load_atom_f16>(rDH_dr);
                auto transposed_dH_dr = inplace_transpose(blocked_dH_dr);
                auto rA_dR = construct_operand_A<dR_load_atom_f16>(transposed_dH_dr);

                // Transpose A: (BLK_M, rs) → (rs, BLK_M) → construct B-operand
                // Use dR atom (SM80_16x8x16) for construct so outer_K matches A-operand
                auto blocked_A_dr = blocken_operand_A<dA_atom_f16>(rA_dr);
                auto transposed_A_dr = inplace_transpose(blocked_A_dr);
                auto rB_dR = construct_operand_B<dR_load_atom_f16>(transposed_A_dr);

                // dR MMA: dR(dR_M, rs) += dH^T(dR_M, BLK_M) @ A^T(rs, BLK_M)
                // Uses CuTe's gemm with register reuse optimization for K-outer iteration
                auto dR_mma = make_tiled_mma(
                    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
                    Layout<Shape<_1,_1>>{}, Tile<Int<dR_M>, Int<rs>>{});
                auto dR_thr_mma = dR_mma.get_slice(lane_idx);
                auto rDR = dR_thr_mma.make_fragment_C(dR_thr_mma.partition_C(
                    make_tensor(static_cast<half_t*>(nullptr),
                        make_layout(make_shape(Int<dR_M>{}, Int<rs>{}), LayoutRight{}))));
                clear(rDR);
                gemm(dR_mma, rA_dR, rB_dR, rDR);

                // atomicAdd valid dR elements (first rs rows) to gmem
                for (int f = 0; f < size(rDR); ++f) {
                    auto coord = dR_C_id(f);
                    int ii = get<0>(coord), j = get<1>(coord);
                    if (ii < rs && j < rs && rDR(f) != 0.0f) {
                        atomicAdd(&gdR_partial(g * rs + ii, blk_k_off + j), rDR(f));
                    }
                }
            }
        }

        // Signal CONSUMED
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dh_pipe_r + BAR_CONSUMED_BASE), "n"(n_total));
        if (++dh_pipe_r == bP_dh) dh_pipe_r = 0;
        r_pipe_w = r_pipe_r;
        r_pipe_r = 1 - r_pipe_r;
    }

    // Write dA to gmem
    for (int b = 0; b < n_reconn; ++b) {
        int blk_k_off = (warp_idx * (WARP_K / rs) + b) * rs;
        for (int f = 0; f < size(rDA[b]); ++f) {
            auto coord = tCdA_id(f);
            int mi = get<0>(coord);
            int ki = get<1>(coord) + blk_k_off;
            if (mi < size<0>(gdA) && ki < size<1>(gdA))
                gdA(mi, ki) = half_t(rDA[b](f));
        }
    }
}

// =============================================================================
// Kernel entry point
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
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;

    using wlp_t = typename cute::BwdDAdRParams::warp_layout_ar;   // producer
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_arb;  // consumer
    constexpr int n_cons = size(wlc_t{}) * 32;
    constexpr int n_prod = size(wlp_t{}) * 32;

    // Z-curve grid mapping
    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start = get<0>(grid_coord) * BLK_K;
    int m_start = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;

    // -- Shared memory --
    auto sdC_layout = tile_to_shape(get_smem_atom(Int<BLK_N>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_N>{}, Int<bP_dc_b>{}));

    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}), make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom,
        make_shape(Int<BLK_K>{}, Int<BLK_N>{}, Int<bP_dc_b>{}));

    auto sdH_layout = tile_to_shape(get_smem_atom(Int<rs>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_dh>{}));

    auto sA_layout = tile_to_shape(get_smem_atom(Int<rs>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}));

    auto sR_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}),
        make_shape(Int<rs>{}, Int<BLK_K>{}, _2{}));

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC   = make_tensor(make_smem_ptr(p), sdC_layout);   p += cosize(sdC_layout);
    auto sB    = make_tensor(make_smem_ptr(p), sB_layout);    p += cosize(sB_layout);
    auto sdH   = make_tensor(make_smem_ptr(p), sdH_layout);   p += cosize(sdH_layout);
    auto sA    = make_tensor(make_smem_ptr(p), sA_layout);    p += cosize(sA_layout);
    auto sR    = make_tensor(make_smem_ptr(p), sR_layout);

    __syncthreads();

    // -- Gmem tensors --
    int n_N_tiles = N / BLK_N;
    constexpr int tiles_per_group = gs / BLK_N;

    // Producer gmem: dC(BLK_M, BLK_N, n_N_tiles), B(BLK_K, BLK_N, n_N_tiles)
    Tensor gdC = make_tensor(make_gmem_ptr(dC_ptr + m_start * ldDC),
        make_layout(make_shape(Int<BLK_M>{}, Int<BLK_N>{}, n_N_tiles),
                    make_stride(ldDC, Int<1>{}, Int<BLK_N>{})));
    Tensor gB = make_tensor(make_gmem_ptr(B_ptr + k_start),
        make_layout(make_shape(Int<BLK_K>{}, Int<BLK_N>{}, n_N_tiles),
                    make_stride(Int<1>{}, ldB, Int<BLK_N>{} * ldB)));

    // Consumer gmem: A(BLK_M, BLK_K), R(rs, BLK_K, n_groups), dA(BLK_M, BLK_K)
    Tensor gA = make_tensor(make_gmem_ptr(A_ptr + m_start * ldA + k_start),
        make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                    make_stride(ldA, Int<1>{})));
    Tensor gR = make_tensor(make_gmem_ptr(R_ptr + k_start),
        make_layout(make_shape(Int<rs>{}, Int<BLK_K>{}, n_groups),
                    make_stride(ldR, Int<1>{}, Int<rs>{} * ldR)));
    Tensor gdA = make_tensor(make_gmem_ptr(dA_ptr + m_start * ldDA + k_start),
        make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                    make_stride(ldDA, Int<1>{})));

    // dR partial buffer: pre-slice to this CTA's buf_slot and k_start
    // Full shape is (n_buf_slots, n_groups * rs, K). We pass (n_groups * rs, BLK_K).
    int dR_rows = n_groups * rs;
    Tensor gdR_partial = make_tensor(
        make_gmem_ptr(dR_partial + buf_slot * dR_rows * K + k_start),
        make_layout(make_shape(dR_rows, Int<BLK_K>{}),
                    make_stride(K, Int<1>{})));

    // -- TiledCopy for cp.async --
    auto copy_dc = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_N>{}, Int<n_prod>{});
    constexpr int bt0 = BLK_K / 8;
    constexpr int bt1 = n_prod / bt0;
    constexpr int bv1 = BLK_N / bt1;
    static_assert(bt0 > 0 && bt1 > 0 && bv1 > 0);
    auto copy_b = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto copy_a = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_K>{}, Int<n_cons>{});
    auto copy_r = cp_layout<uint128_t, half_t>(Int<rs>{}, Int<BLK_K>{}, Int<n_cons>{});

    if (threadIdx.x >= n_cons) {
        dH_producer(
            gdC, sdC, copy_dc,
            gB,  sB,  copy_b,
            sdH, Int<tiles_per_group>{}, Int<rs>{},
            threadIdx.x - n_cons,
            wlp_t{}, wlc_t{});
    } else {
        dAdR_consumer(
            gA, sA, copy_a,
            gR, sR, copy_r,
            sdH,
            gdA, gdR_partial,
            Int<rs>{},
            threadIdx.x,
            wlp_t{}, wlc_t{});
    }
}

// =============================================================================
// dR reduction kernel
// =============================================================================
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

// =============================================================================
// Host launcher
// =============================================================================
void prism_backward_dA_dR_launch(
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
    constexpr int BLK_N = cute::BwdDAdRParams::bN;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;
    constexpr int n_buf_slots_param = cute::BwdDAdRParams::n_buf_slots;
    using wlp_t = typename cute::BwdDAdRParams::warp_layout_ar;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_arb;
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

    // Smem size
    auto smem_dC = get_smem_atom(cute::Int<BLK_N>{});
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(cute::Int<BLK_K>{}, _8{}),
                    make_stride(_1{}, cute::Int<BLK_K>{})));
    auto smem_rs = get_smem_atom(cute::Int<rs>{});
    auto smem_bk = get_smem_atom(cute::Int<BLK_K>{});

    int smem = (cosize(tile_to_shape(smem_dC,
                    make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_N>{}, cute::Int<bP_dc_b>{})))
              + cosize(tile_to_shape(sB_atom,
                    make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_N>{}, cute::Int<bP_dc_b>{})))
              + cosize(tile_to_shape(smem_rs,
                    make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{}, cute::Int<bP_dh>{})))
              + cosize(tile_to_shape(smem_rs,
                    make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{})))
              + cosize(tile_to_shape(smem_bk,
                    make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{}, _2{})))
              ) * sizeof(half_t) + 256;

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
