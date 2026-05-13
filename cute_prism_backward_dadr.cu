#include "cute_prism_backward_dadr.hpp"
#include "cute_prism_coop_pc.hpp"
#include <prism_config.hpp>
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
    // dH = dC @ B^T producer. For fp16: F16-accum (matches existing).
    // For bf16: F32-accum (only option) — rDH fragment is float, converted
    // to bf16 at the r2s write to sdH.
    using mma_atom_t = prism_ar_atom<_BLK_N>;

    auto mma = make_tiled_mma(mma_atom_t{}, wl_prod, Tile<Int<_BLK_M>, Int<_BLK_K>>{});
    auto thr_mma = mma.get_slice(thread_idx);

    auto rDH = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<prism_cute*>(nullptr),
                    make_layout(make_shape(Int<_BLK_M>{}, Int<_BLK_K>{}), LayoutRight{}))));

    // LDSM s2r for dC (A-operand, LDSM_N) and B (B-operand, LDSM_T)
    constexpr int K_atom = (_BLK_N < 16) ? 8 : 16;
    constexpr int a_u32 = (_BLK_M / size(WarpLayoutProd{}) * K_atom) / 64;
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>,
        std::conditional_t<(a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, prism_cute>,
            Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>>;

    constexpr int b_u16 = (_BLK_K * K_atom) / 32;
    using s2r_B = std::conditional_t<(b_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, prism_cute>,
        std::conditional_t<(b_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, prism_cute>,
            Copy_Atom<SM75_U16x2_LDSM_T, prism_cute>>>;

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
    using r2s_atom = Copy_Atom<UniversalCopy<uint32_t>, prism_cute>;
    auto r2s_c = make_tiled_copy_C(r2s_atom{}, mma);
    auto r2s_thr = r2s_c.get_slice(thread_idx);
#if PRISM_DTYPE == 0
    // fp16: rDH is half_t — direct retile.
    auto tXrDH_c = r2s_thr.retile_S(rDH);
#endif

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

#if PRISM_DTYPE == 0
        copy(r2s_atom{}, tXrDH_c, r2s_thr.partition_D(sdH(_,_,dh_pipe_w)));
#else
        // bf16: rDH is float; convert to bf16 fragment, then r2s.
        {
            constexpr int frag_sz = decltype(size(rDH))::value;
            prism_cute rDH_bf16_storage[frag_sz];
            #pragma unroll
            for (int i = 0; i < frag_sz; ++i) {
                rDH_bf16_storage[i] = prism_cute(float(rDH(i)));
            }
            auto rDH_bf16 = make_tensor(make_rmem_ptr(rDH_bf16_storage), rDH.layout());
            auto tXrDH_bf16 = r2s_thr.retile_S(rDH_bf16);
            copy(r2s_atom{}, tXrDH_bf16, r2s_thr.partition_D(sdH(_,_,dh_pipe_w)));
        }
#endif
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_prod));

        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dh_pipe_w + BAR_READY_BASE), "n"(n_total));
        if (++dh_pipe_w == bP_dh) dh_pipe_w = 0;
    }
}

// =============================================================================
// Consumer: dA accumulation + dR from sdH
// =============================================================================
template <bool GATED,
          class TensorGA, class SmemA,  class TiledCopyA,
          class TensorGR, class SmemR,  class TiledCopyR,
          class SmemDH,
          class TensorGDA, class TensorGDR, class ReconnectSize,
          class WarpLayoutProd, class WarpLayoutCons>
__device__ static inline
void dAdR_consumer(
    TensorGA const &gA, SmemA &sA, TiledCopyA copy_a,
    TensorGR const &gR, SmemR &sR, TiledCopyR copy_r,
    SmemDH &sdH,  // NOTE: writable for the gated path (dS overwrite)
    TensorGDA &gdA, TensorGDR &gdR_partial,
    ReconnectSize reconn_sz,
    int thread_idx,
    WarpLayoutProd wl_prod, WarpLayoutCons wl_cons,
    prism_cute const* bias_ptr,  // (n_groups, K) row-major; nullptr unless PRISM_INTERNAL_BIAS=1
    int K_full, int k_start,
    float* dIB_partial,  // (n_buf_slots, n_groups, K); nullptr unless PRISM_INTERNAL_BIAS=1
    int buf_slot,
    // input_shuffle: per-group A and per-group dA strides (in elements). 0 → off
    // (use gA / gdA as-is, accumulate dA across groups). Non-zero → per-group
    // sA reload from A_base + g*strideA_per_group, per-group dA_perm write to
    // dA_base + g*strideDA_per_group.
    prism_cute const* A_base, int strideA_per_group, int ldA,
    prism_cute* dA_base, int strideDA_per_group, int ldDA,
    int m_start,
    // Dropout (only used when PRISM_DROPOUT=1)
    int64_t const* dropout_seeds, float dropout_p, float inv_keep)
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
    using dA_atom = prism_dA_atom_f32<rs>;
    // Helper atom for blocken/transpose/construct paths — needs to share the
    // operand layout with the dA/AR atoms. For fp16 → F16-accum atom; for bf16
    // we reuse the F32-accum atom (same operand layouts as F16-accum since
    // SM80 16x8x{8,16} thread/val layouts are identical across accum types).
    using dA_atom_f16 = prism_ar_atom<rs>;

    auto dA_mma = make_tiled_mma(dA_atom{}, Layout<Shape<_1,_1>>{},
                                  Tile<Int<_BLK_M>, Int<rs>>{});
    auto dA_thr = dA_mma.get_slice(lane_idx);

    auto dA_C_dummy = make_tensor(static_cast<prism_cute*>(nullptr),
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
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>,
        std::conditional_t<(dA_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, prism_cute>,
            Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>>;
    constexpr int dA_b_u32 = (rs * dA_K) / 64;
    using dA_s2r_B_atom = std::conditional_t<(dA_b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>,
        std::conditional_t<(dA_b_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, prism_cute>,
            Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>>;

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
    // dR's MMA is always 16x8x16 F32-accum. For bf16 we need the bf16 variant
    // since operand types must match (dH and A in sdH/sA are bf16 there).
    using dR_load_atom = prism_dA_atom_f32<16>;
    using dR_load_atom_f16 = prism_ar_atom<16>;
    auto dR_load_mma = make_tiled_mma(dR_load_atom{}, Layout<Shape<_1,_1>>{},
                                       Tile<Int<_BLK_M>, Int<dR_M>>{});
    auto dR_load_thr = dR_load_mma.get_slice(lane_idx);

    // LDSM for dR loading: A-operand from padded (BLK_M, dR_M) smem views
    constexpr int dR_a_u32 = (_BLK_M * 16) / 64;  // K_atom = 16 for SM80_16x8x16
    using dR_s2r_A_atom = std::conditional_t<(dR_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>,
        std::conditional_t<(dR_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, prism_cute>,
            Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>>;
    auto dR_s2r_a = make_tiled_copy_A(dR_s2r_A_atom{}, dR_load_mma);
    auto dR_s2r_a_thr = dR_s2r_a.get_slice(lane_idx);

    // dR identity for output writeback (created once, reused per block)
    auto dR_id_mma = make_tiled_mma(prism_dA_atom_f32<16>{},
                                     Layout<Shape<_1,_1>>{}, Tile<Int<dR_M>, Int<rs>>{});
    auto dR_C_id = dR_id_mma.get_slice(lane_idx).partition_C(
        make_identity_tensor(make_shape(Int<dR_M>{}, Int<rs>{})));

    const bool shuffle_mode = (strideA_per_group != 0);

    // -- Load sA once via cp.async (no-shuffle path) --
    if (!shuffle_mode) {
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
        // shuffle_mode: reload sA from A_base + g*strideA_per_group (per-group A_perm).
        // Also reset rDA[b] so each group's dA contribution is written separately.
        if (shuffle_mode) {
            ThrCopy thr_cp_a = copy_a.get_slice(thread_idx);
            prism_cute const* A_for_g = A_base + (long long)g * strideA_per_group
                                    + (long long)m_start * ldA + k_start;
            Tensor gA_g = make_tensor(make_gmem_ptr(A_for_g),
                make_layout(make_shape(Int<_BLK_M>{}, Int<_BLK_K>{}),
                            make_stride(ldA, Int<1>{})));
            Tensor tAg_g = thr_cp_a.partition_S(gA_g);
            Tensor tAs_g = thr_cp_a.partition_D(sA);
            copy(copy_a, tAg_g, tAs_g);
            #pragma unroll
            for (int b = 0; b < n_reconn; ++b) clear(rDA[b]);
        }

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

            // LDSM: load R block → rR_nat (natural B-operand layout)
            auto tXsR = dA_s2r_b_thr.partition_S(sR_blk);
            auto rR_nat = dA_thr.make_fragment_B(dA_thr.partition_B(sR_blk));
            auto tXrR = dA_s2r_b_thr.retile_D(rR_nat);
            copy(dA_s2r_B_atom{}, tXsR, tXrR);

            if constexpr (GATED) {
                // -- Gated path: recompute S = AR + bias, gate, dS = dH * A * silu'(S),
                //    add dA_gate = dH * silu(S) to rDA[b], overwrite rDH_frag with dS,
                //    write dS back to sdH_blk so the dR step picks it up.
                Tensor sA_blk = sA_rb(_, make_coord(_, rb_idx));
                auto tXsA = dA_s2r_a_thr.partition_S(sA_blk);
                auto rA_frag = dA_thr.make_fragment_A(dA_thr.partition_A(sA_blk));
                auto tXrA = dA_s2r_a_thr.retile_D(rA_frag);
                copy(dA_s2r_A_atom{}, tXsA, tXrA);

                // f16-accum AR mma so we can run silu_h2 directly on the result.
                auto AR_mma = make_tiled_mma(dA_atom_f16{}, Layout<Shape<_1,_1>>{},
                                              Tile<Int<_BLK_M>, Int<rs>>{});
                auto AR_thr = AR_mma.get_slice(lane_idx);
                auto rAR = AR_thr.make_fragment_C(AR_thr.partition_C(
                    make_tensor(static_cast<prism_cute*>(nullptr),
                        make_layout(make_shape(Int<_BLK_M>{}, Int<rs>{}), LayoutRight{}))));
                clear(rAR);
                gemm(AR_mma, rA_frag, rR_nat, rAR);  // AR = A @ R^T (no R transpose)

                // Per (mma_n, mma_m) atom: 2 packed half2 (top row, bottom row) sharing
                // the same (col_lo, col_hi) cols. For rs=16, A-atom holds 8 vals per
                // thread; pick the right K-half via a_off = (mn % (rs/8)) * 4.
                constexpr int RS_v = rs;
                constexpr int A_OFFSET_STEP = 4;
                constexpr int N_PER_A_ATOM = RS_v / 8;
                const int col_lo = (lane_idx & 0x3) * 2;
                const auto MMA_M_v = size<1>(rAR);
                const auto MMA_N_v = size<2>(rAR);
#if PRISM_DROPOUT
                // Per-group seed for THIS group g; apply mask to dH on replay
                // (matches forward's mask exactly via the same hash).
                const uint64_t seed_g = (uint64_t)dropout_seeds[g];
#endif
                #pragma unroll
                for (int mn = 0; mn < int(MMA_N_v); ++mn) {
                    const int ri_lo = mn * 8 + col_lo;
                    const int ri_hi = ri_lo + 1;
                    const int a_off = (mn % N_PER_A_ATOM) * A_OFFSET_STEP;
#if PRISM_DTYPE == 0
                    // -- fp16 path: packed half2 silu --
                    __half2 bias2 = __floats2half2_rn(0.0f, 0.0f);
#if PRISM_INTERNAL_BIAS
                    bias2 = load_half2(
                        bias_ptr[g * K_full + k_start + blk_k_off + ri_lo],
                        bias_ptr[g * K_full + k_start + blk_k_off + ri_hi]);
#endif
                    #pragma unroll
                    for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                        auto& ar2_lo = reinterpret_cast<__half2&>(rAR(0, mm, mn));
                        auto& ar2_hi = reinterpret_cast<__half2&>(rAR(2, mm, mn));
                        auto& dh2_lo = reinterpret_cast<__half2&>(rDH_frag(0 + a_off, mm, 0));
                        auto& dh2_hi = reinterpret_cast<__half2&>(rDH_frag(2 + a_off, mm, 0));
                        auto& a2_lo  = reinterpret_cast<const __half2&>(rA_frag(0 + a_off, mm, 0));
                        auto& a2_hi  = reinterpret_cast<const __half2&>(rA_frag(2 + a_off, mm, 0));
#if PRISM_INTERNAL_BIAS
                        ar2_lo = __hadd2(ar2_lo, bias2);
                        ar2_hi = __hadd2(ar2_hi, bias2);
#else
                        (void)bias2;
#endif
#if PRISM_DROPOUT
                        // Apply the same mask the forward used to dH BEFORE
                        // the gate-derivative math, so dS and dA_gate inherit
                        // it. dadr's MMA is single-warp over (_BLK_M, rs), so
                        // m_local = mm*16 + lane/4 maps directly to global M.
                        const int m_lo_dr = m_start + mm * 16 + int(lane_idx >> 2);
                        const int m_hi_dr = m_lo_dr + 8;
                        const float u00 = prism_uniform_from_hash(seed_g, (int64_t)m_lo_dr * K_full + k_start + blk_k_off + ri_lo);
                        const float u01 = prism_uniform_from_hash(seed_g, (int64_t)m_lo_dr * K_full + k_start + blk_k_off + ri_hi);
                        const float u10 = prism_uniform_from_hash(seed_g, (int64_t)m_hi_dr * K_full + k_start + blk_k_off + ri_lo);
                        const float u11 = prism_uniform_from_hash(seed_g, (int64_t)m_hi_dr * K_full + k_start + blk_k_off + ri_hi);
                        const __half2 mask_lo = __floats2half2_rn(
                            (u00 >= dropout_p) ? inv_keep : 0.0f,
                            (u01 >= dropout_p) ? inv_keep : 0.0f);
                        const __half2 mask_hi = __floats2half2_rn(
                            (u10 >= dropout_p) ? inv_keep : 0.0f,
                            (u11 >= dropout_p) ? inv_keep : 0.0f);
                        dh2_lo = __hmul2(dh2_lo, mask_lo);
                        dh2_hi = __hmul2(dh2_hi, mask_hi);
#endif
                        const __half2 sig_lo = sigmoid_h2(ar2_lo);
                        const __half2 sig_hi = sigmoid_h2(ar2_hi);
                        const __half2 gate_lo = __hmul2(ar2_lo, sig_lo);
                        const __half2 gate_hi = __hmul2(ar2_hi, sig_hi);
                        const __half2 gprime_lo = silu_prime_h2(ar2_lo, sig_lo);
                        const __half2 gprime_hi = silu_prime_h2(ar2_hi, sig_hi);
                        const __half2 dA_gate_lo = __hmul2(dh2_lo, gate_lo);
                        const __half2 dA_gate_hi = __hmul2(dh2_hi, gate_hi);
                        rDA[b](0, mm, mn) += __half2float(__low2half(dA_gate_lo));
                        rDA[b](1, mm, mn) += __half2float(__high2half(dA_gate_lo));
                        rDA[b](2, mm, mn) += __half2float(__low2half(dA_gate_hi));
                        rDA[b](3, mm, mn) += __half2float(__high2half(dA_gate_hi));
                        dh2_lo = __hmul2(__hmul2(dh2_lo, a2_lo), gprime_lo);
                        dh2_hi = __hmul2(__hmul2(dh2_hi, a2_hi), gprime_hi);
                    }

#if PRISM_INTERNAL_BIAS
                    // d_internal_bias[g, k] = sum_m dS[m, k] (fp16 path).
                    {
                        float ps_lo = 0.0f, ps_hi = 0.0f;
                        #pragma unroll
                        for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                            const __half2& dh2_lo_v = reinterpret_cast<const __half2&>(rDH_frag(0 + a_off, mm, 0));
                            const __half2& dh2_hi_v = reinterpret_cast<const __half2&>(rDH_frag(2 + a_off, mm, 0));
                            ps_lo += __half2float(__low2half(dh2_lo_v));
                            ps_hi += __half2float(__high2half(dh2_lo_v));
                            ps_lo += __half2float(__low2half(dh2_hi_v));
                            ps_hi += __half2float(__high2half(dh2_hi_v));
                        }
                        ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 4);
                        ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 8);
                        ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 16);
                        ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 4);
                        ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 8);
                        ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 16);
                        if ((lane_idx >> 2) == 0) {
                            const int gK = g * K_full;
                            const int slot_off = buf_slot * (n_groups * K_full);
                            atomicAdd(&dIB_partial[slot_off + gK + k_start + blk_k_off + ri_lo], ps_lo);
                            atomicAdd(&dIB_partial[slot_off + gK + k_start + blk_k_off + ri_hi], ps_hi);
                        }
                    }
#endif
#else
                    // -- bf16 path: rAR is float, rA/rDH/rR are bf16; silu in float --
#if PRISM_INTERNAL_BIAS
                    const float bias_lo_f = prism_to_float(reinterpret_cast<const prism_native&>(bias_ptr[g * K_full + k_start + blk_k_off + ri_lo]));
                    const float bias_hi_f = prism_to_float(reinterpret_cast<const prism_native&>(bias_ptr[g * K_full + k_start + blk_k_off + ri_hi]));
#endif
                    float ps_lo = 0.0f, ps_hi = 0.0f;  // d_internal_bias accumulators
                    #pragma unroll
                    for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                        float ar0 = rAR(0, mm, mn);
                        float ar1 = rAR(1, mm, mn);
                        float ar2 = rAR(2, mm, mn);
                        float ar3 = rAR(3, mm, mn);
                        float dh0 = float(rDH_frag(0 + a_off, mm, 0));
                        float dh1 = float(rDH_frag(1 + a_off, mm, 0));
                        float dh2 = float(rDH_frag(2 + a_off, mm, 0));
                        float dh3 = float(rDH_frag(3 + a_off, mm, 0));
                        float a0 = float(rA_frag(0 + a_off, mm, 0));
                        float a1 = float(rA_frag(1 + a_off, mm, 0));
                        float a2 = float(rA_frag(2 + a_off, mm, 0));
                        float a3 = float(rA_frag(3 + a_off, mm, 0));
#if PRISM_INTERNAL_BIAS
                        ar0 += bias_lo_f; ar1 += bias_hi_f;
                        ar2 += bias_lo_f; ar3 += bias_hi_f;
#endif
#if PRISM_DROPOUT
                        // Replay the forward's mask on dH (same hash, same idx).
                        const int m_lo_dr = m_start + mm * 16 + int(lane_idx >> 2);
                        const int m_hi_dr = m_lo_dr + 8;
                        const float u00 = prism_uniform_from_hash(seed_g, (int64_t)m_lo_dr * K_full + k_start + blk_k_off + ri_lo);
                        const float u01 = prism_uniform_from_hash(seed_g, (int64_t)m_lo_dr * K_full + k_start + blk_k_off + ri_hi);
                        const float u10 = prism_uniform_from_hash(seed_g, (int64_t)m_hi_dr * K_full + k_start + blk_k_off + ri_lo);
                        const float u11 = prism_uniform_from_hash(seed_g, (int64_t)m_hi_dr * K_full + k_start + blk_k_off + ri_hi);
                        dh0 *= (u00 >= dropout_p) ? inv_keep : 0.0f;
                        dh1 *= (u01 >= dropout_p) ? inv_keep : 0.0f;
                        dh2 *= (u10 >= dropout_p) ? inv_keep : 0.0f;
                        dh3 *= (u11 >= dropout_p) ? inv_keep : 0.0f;
#endif
                        const float s0 = 0.5f * (1.0f + tanhf(0.5f * ar0));
                        const float s1 = 0.5f * (1.0f + tanhf(0.5f * ar1));
                        const float s2 = 0.5f * (1.0f + tanhf(0.5f * ar2));
                        const float s3 = 0.5f * (1.0f + tanhf(0.5f * ar3));
                        const float gate0 = ar0 * s0;  // silu(S)
                        const float gate1 = ar1 * s1;
                        const float gate2 = ar2 * s2;
                        const float gate3 = ar3 * s3;
                        // silu'(x) = sigma(x) * (1 + x*(1-sigma(x)))
                        const float gp0 = s0 * (1.0f + ar0 * (1.0f - s0));
                        const float gp1 = s1 * (1.0f + ar1 * (1.0f - s1));
                        const float gp2 = s2 * (1.0f + ar2 * (1.0f - s2));
                        const float gp3 = s3 * (1.0f + ar3 * (1.0f - s3));
                        // dA_gate = dH * gate
                        rDA[b](0, mm, mn) += dh0 * gate0;
                        rDA[b](1, mm, mn) += dh1 * gate1;
                        rDA[b](2, mm, mn) += dh2 * gate2;
                        rDA[b](3, mm, mn) += dh3 * gate3;
                        // dS = dH * A * gate' — write back as bf16 (lossy).
                        const float dS0 = dh0 * a0 * gp0;
                        const float dS1 = dh1 * a1 * gp1;
                        const float dS2 = dh2 * a2 * gp2;
                        const float dS3 = dh3 * a3 * gp3;
                        rDH_frag(0 + a_off, mm, 0) = prism_cute(dS0);
                        rDH_frag(1 + a_off, mm, 0) = prism_cute(dS1);
                        rDH_frag(2 + a_off, mm, 0) = prism_cute(dS2);
                        rDH_frag(3 + a_off, mm, 0) = prism_cute(dS3);
                        // d_internal_bias accumulators (sum over m, both row groups)
                        ps_lo += dS0 + dS2;
                        ps_hi += dS1 + dS3;
                    }
#if PRISM_INTERNAL_BIAS
                    ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 4);
                    ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 8);
                    ps_lo += __shfl_xor_sync(0xffffffff, ps_lo, 16);
                    ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 4);
                    ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 8);
                    ps_hi += __shfl_xor_sync(0xffffffff, ps_hi, 16);
                    if ((lane_idx >> 2) == 0) {
                        const int gK = g * K_full;
                        const int slot_off = buf_slot * (n_groups * K_full);
                        atomicAdd(&dIB_partial[slot_off + gK + k_start + blk_k_off + ri_lo], ps_lo);
                        atomicAdd(&dIB_partial[slot_off + gK + k_start + blk_k_off + ri_hi], ps_hi);
                    }
#endif
#endif
                }

                // Write dS back to sdH_blk (overwriting dH for this block). The
                // existing dR LDSM read will pick up dS via the padded view.
                //
                // FUTURE-OPT: this register→smem→register round-trip exists
                // because dA consumes dS as B-operand and dR consumes dS^T as
                // A-operand — different register layouts. A direct
                //   blocken_operand_B<dA_atom_f16>(rDH_frag)
                //     -> inplace_transpose
                //     -> construct_operand_A<dR_load_atom_f16>
                // chain would skip the smem trip but needs the inner-K shapes
                // to line up: dA's B-operand has MMA_K = rs, dR's A-operand
                // has MMA_K = BLK_M. The transpose maps M→K, so structurally
                // the data flow works; the open question is whether the
                // outer-block layouts produced by both atoms compose without
                // recoordering. Estimated 5-10% on dadr critical path; not
                // attempted yet — see prior chat's optimization #4.
                {
                    auto sdH_id = make_identity_tensor(make_shape(Int<_BLK_M>{}, Int<rs>{}));
                    auto rDS_coords = dA_thr.partition_A(sdH_id);
                    CUTE_UNROLL
                    for (int i = 0; i < size(rDH_frag); ++i) {
                        auto coord = rDS_coords(i);
                        sdH_blk(get<0>(coord), get<1>(coord)) = rDH_frag(i);
                    }
                    __syncwarp();
                }
            }

            auto blocked_R = blocken_operand_B<dA_atom_f16>(rR_nat);
            auto transposed_R = inplace_transpose(blocked_R);
            auto rR_T = construct_operand_B<dA_atom_f16>(transposed_R);

            // dA += rDH_frag @ R  (rDH_frag holds dS for gated path, dH for non-gated)
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
                    prism_dA_atom_f32<16>{},
                    Layout<Shape<_1,_1>>{}, Tile<Int<dR_M>, Int<rs>>{});
                auto dR_thr_mma = dR_mma.get_slice(lane_idx);
                auto rDR = dR_thr_mma.make_fragment_C(dR_thr_mma.partition_C(
                    make_tensor(static_cast<prism_cute*>(nullptr),
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

        // shuffle_mode: write THIS group's rDA to dA_perm[g] now (not at end).
        if (shuffle_mode) {
            prism_cute* dA_for_g = dA_base + (long long)g * strideDA_per_group
                               + (long long)m_start * ldDA + k_start;
            #pragma unroll
            for (int b = 0; b < n_reconn; ++b) {
                int blk_k_off = (warp_idx * (WARP_K / rs) + b) * rs;
                for (int f = 0; f < size(rDA[b]); ++f) {
                    auto coord = tCdA_id(f);
                    int mi = get<0>(coord);
                    int ki = get<1>(coord) + blk_k_off;
                    if (mi < _BLK_M && ki < _BLK_K) {
                        dA_for_g[mi * ldDA + ki] = prism_cute(rDA[b](f));
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

    // Write accumulated rDA to gdA — only when NOT in shuffle_mode (the
    // shuffle path already wrote per-group above and zeroed rDA each group).
    if (!shuffle_mode) {
        for (int b = 0; b < n_reconn; ++b) {
            int blk_k_off = (warp_idx * (WARP_K / rs) + b) * rs;
            for (int f = 0; f < size(rDA[b]); ++f) {
                auto coord = tCdA_id(f);
                int mi = get<0>(coord);
                int ki = get<1>(coord) + blk_k_off;
                if (mi < size<0>(gdA) && ki < size<1>(gdA))
                    gdA(mi, ki) = prism_cute(rDA[b](f));
            }
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
    const prism_cute* __restrict__ dC_ptr,  int ldDC,
    const prism_cute* __restrict__ A_ptr,   int ldA,
    const prism_cute* __restrict__ B_ptr,   int ldB,
    const prism_cute* __restrict__ R_ptr,   int ldR,
    const prism_cute* __restrict__ bias_ptr,
    prism_cute* __restrict__ dA_ptr,        int ldDA,
    float* __restrict__ dR_partial,
    float* __restrict__ dIB_partial,
    int M, int N, int K,
    int n_groups, int n_buf_slots,
    int strideA_per_group, int strideDA_per_group,
    int64_t const* dropout_seeds, float dropout_p, float inv_keep)
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

    extern __shared__ prism_cute smem_raw[];
    prism_cute* p = smem_raw;
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
    auto copy_dc = cp_layout<uint128_t, prism_cute>(Int<BLK_M>{}, Int<BLK_N>{}, Int<n_prod>{});
    constexpr int bt0 = BLK_K / 8;
    constexpr int bt1 = n_prod / bt0;
    constexpr int bv1 = BLK_N / bt1;
    static_assert(bt0 > 0 && bt1 > 0 && bv1 > 0);
    auto copy_b = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, prism_cute>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto copy_a = cp_layout<uint128_t, prism_cute>(Int<BLK_M>{}, Int<BLK_K>{}, Int<n_cons>{});
    auto copy_r = cp_layout<uint128_t, prism_cute>(Int<rs>{}, Int<BLK_K>{}, Int<n_cons>{});

    if (threadIdx.x >= n_cons) {
        dH_producer(
            gdC, sdC, copy_dc,
            gB,  sB,  copy_b,
            sdH, Int<tiles_per_group>{}, Int<rs>{},
            threadIdx.x - n_cons,
            wlp_t{}, wlc_t{});
    } else {
        dAdR_consumer<GATED>(
            gA, sA, copy_a,
            gR, sR, copy_r,
            sdH,
            gdA, gdR_partial,
            Int<rs>{},
            threadIdx.x,
            wlp_t{}, wlc_t{},
            bias_ptr, K, k_start,
            dIB_partial, buf_slot,
            A_ptr, strideA_per_group, ldA,
            dA_ptr, strideDA_per_group, ldDA,
            m_start,
            dropout_seeds, dropout_p, inv_keep);
    }
}

// =============================================================================
// dR reduction kernel
// =============================================================================
__global__ void dR_reduce_kernel(
    const float* __restrict__ dR_partial, prism_native* __restrict__ dR,
    int dR_elements, int n_buf_slots)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dR_elements) return;
    float sum = 0.0f;
    for (int s = 0; s < n_buf_slots; ++s)
        sum += dR_partial[s * dR_elements + idx];
    dR[idx] = prism_from_float(sum);
}

// =============================================================================
// Host launcher
// =============================================================================
void prism_backward_dA_dR_launch(
    int m, int n, int k,
    prism_native const* dC, int ldDC,
    prism_native const* A,  int ldA,
    prism_native const* B,  int ldB,
    prism_native const* R,  int ldR,
    prism_native const* bias,
    prism_native* dA, int ldDA,
    prism_native* dR, int ldDR,
    prism_native* dInternalBias,
    int strideA_per_group,
    int strideDA_per_group,
    int64_t const* dropout_seeds, float dropout_p, float inv_keep,
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
    int dIB_elements = n_groups * k;

    float* dR_partial_buf = nullptr;
    cudaMalloc(&dR_partial_buf, (int64_t)n_buf_slots * dR_elements * sizeof(float));
    cudaMemsetAsync(dR_partial_buf, 0, (int64_t)n_buf_slots * dR_elements * sizeof(float), stream);

    float* dIB_partial_buf = nullptr;
#if PRISM_INTERNAL_BIAS
    cudaMalloc(&dIB_partial_buf, (int64_t)n_buf_slots * dIB_elements * sizeof(float));
    cudaMemsetAsync(dIB_partial_buf, 0, (int64_t)n_buf_slots * dIB_elements * sizeof(float), stream);
#endif

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
              ) * sizeof(prism_cute) + 256;

    dim3 grid(n_k_tiles * n_m_tiles);
    dim3 block(n_threads);

#if PRISM_GATED
    auto kernel = bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, true>;
#else
    auto kernel = bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, false>;
#endif
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<prism_cute const*>(dC), ldDC,
        reinterpret_cast<prism_cute const*>(A), ldA,
        reinterpret_cast<prism_cute const*>(B), ldB,
        reinterpret_cast<prism_cute const*>(R), ldR,
        reinterpret_cast<prism_cute const*>(bias),
        reinterpret_cast<prism_cute*>(dA), ldDA,
        dR_partial_buf, dIB_partial_buf,
        m, n, k, n_groups, n_buf_slots,
        strideA_per_group, strideDA_per_group,
        dropout_seeds, dropout_p, inv_keep);

    {
        int threads = 256;
        int blocks_r = (dR_elements + threads - 1) / threads;
        dR_reduce_kernel<<<blocks_r, threads, 0, stream>>>(
            dR_partial_buf, dR, dR_elements, n_buf_slots);
    }

#if PRISM_INTERNAL_BIAS
    {
        int threads = 256;
        int blocks_ib = (dIB_elements + threads - 1) / threads;
        // Reuse dR_reduce_kernel — it sums per-slot float partials into a half output.
        dR_reduce_kernel<<<blocks_ib, threads, 0, stream>>>(
            dIB_partial_buf, dInternalBias, dIB_elements, n_buf_slots);
    }
#endif

    cudaStreamSynchronize(stream);
    cudaFree(dR_partial_buf);
#if PRISM_INTERNAL_BIAS
    cudaFree(dIB_partial_buf);
#endif
}
