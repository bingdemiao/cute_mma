#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// Helper to avoid ambiguous ternary with half_t
__device__ __forceinline__
half_t load_or_zero(const half_t* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : half_t(0);
}

// =============================================================================
// Kernel 1: Fused dA + dR — Producer-Consumer
// =============================================================================

// Producer: compute dAR(BLK_M, rs) = dC_g(BLK_M, gs) @ B_g(gs, rs) via MMA
template <int BLK_M, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemDC, class SmemBt, class SmemDAR,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dAdR_producer(
    const half_t* __restrict__ dC_ptr, int ldDC,
    const half_t* __restrict__ B_ptr,  int ldB,
    SmemDC& sdC, SmemBt& sBt, SmemDAR& sdAR,
    int M, int N, int K, int b, int m_start,
    int n_groups,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dar = cute::BwdDAdRParams::bP_dar;

    // Offset barrier IDs by 1 to avoid conflict with __syncthreads() barrier 0
    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_dar;

    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;
    constexpr int n_producer_warps = size(WarpLayoutProducer{});
    constexpr int WARP_M = BLK_M / n_producer_warps;

    // -- MMA: dAR(WARP_M, rs) = dC(WARP_M, gs) @ B^T(rs, gs) [TN, K=gs] --
    using mma_atom_t = std::conditional_t<(gs < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;

    auto mma = make_tiled_mma(
        mma_atom_t{},
        Layout<Shape<_1, _1>>{},
        Tile<Int<WARP_M>, Int<rs>>{}
    );
    auto thr_mma = mma.get_slice(lane_idx);

    // -- Swizzle-preserving sub-views via logical_divide --
    // Flat WARP_M tiler for contiguous warp regions

    // sdC: (BLK_M, gs, bP_dc_b) -> warp sub-view (WARP_M, gs, bP_dc_b)
    Tensor sdC_warp = logical_divide(sdC,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _, _); // (WARP_M, gs, bP_dc_b)

    // sBt: (rs, gs, bP_dc_b) — no M-tiling needed
    // sdAR: (BLK_M, rs, bP_dar) -> warp sub-view (WARP_M, rs, bP_dar)
    Tensor sdAR_warp = logical_divide(sdAR,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _, _); // (WARP_M, rs, bP_dar)

    auto tCsDAR = thr_mma.partition_C(sdAR_warp);
    auto rDAR = thr_mma.make_fragment_C(tCsDAR(_,_,_,_0{}));

    using r2s_atom = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s = make_tiled_copy_C(r2s_atom{}, mma);
    auto r2s_thr = r2s.get_slice(lane_idx);
    auto tXrDAR = r2s_thr.retile_S(rDAR);
    auto tXsDAR = r2s_thr.partition_D(sdAR_warp); // (CPY, CPY_M, CPY_N, bP_dar)

    // -- s2r atoms: LDSM for smem→reg (producer smem uses K_atom-based swizzle) --
    // Per-thread u32 count is based on MMA operand size per step (K_atom),
    // not the full reduction dimension (gs).
    constexpr int K_atom_p = (gs < 16) ? 8 : 16;
    constexpr int a_u32 = (WARP_M * K_atom_p) / 64;
    using s2r_atom_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    constexpr int b_u32 = (rs * K_atom_p) / 64;
    using s2r_atom_Bt = std::conditional_t<(b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

    // Partition s2r once before loop
    auto s2r_a = make_tiled_copy_A(s2r_atom_dC{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx);
    auto tXsdC = s2r_a_thr.partition_S(sdC_warp); // (CPY, CPY_M, CPY_K, bP_dc_b)
    auto rDC = thr_mma.make_fragment_A(thr_mma.partition_A(sdC_warp(_,_,_0{})));
    auto tXrDC = s2r_a_thr.retile_D(rDC);

    auto s2r_b = make_tiled_copy_B(s2r_atom_Bt{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(lane_idx);
    auto tXsBt = s2r_b_thr.partition_S(sBt); // (CPY, CPY_N, CPY_K, bP_dc_b)
    auto rBt = thr_mma.make_fragment_B(thr_mma.partition_B(sBt(_,_,_0{})));
    auto tXrBt = s2r_b_thr.retile_D(rBt);

    // -- Pipeline state --
    int dc_b_pipe_write = bP_dc_b - 1;
    int dc_b_pipe_read = 0;
    int dar_pipe_write = 0;

    // -- Prefill dC+B pipeline --
    for (int p = 0; p < bP_dc_b - 1 && p < n_groups; ++p) {
        for (int i = thread_idx; i < BLK_M * gs; i += n_producer_threads) {
            int r = i / gs, c = i % gs;
            int m_idx = m_start + r;
            sdC(r, c, p) = load_or_zero(dC_ptr, m_idx * ldDC + p * gs + c, m_idx < M);
        }
        for (int i = thread_idx; i < gs * rs; i += n_producer_threads) {
            int src_row = i / rs;
            int src_col = i % rs;
            sBt(src_col, src_row, p) = *(B_ptr + (p * gs + src_row) * ldB + b * rs + src_col);
        }
        cp_async_fence();
    }

    // -- Main group loop --
    for (int g = 0; g < n_groups; ++g) {
        int g_next = g + bP_dc_b - 1;
        if (g_next < n_groups) {
            for (int i = thread_idx; i < BLK_M * gs; i += n_producer_threads) {
                int r = i / gs, c = i % gs;
                int m_idx = m_start + r;
                sdC(r, c, dc_b_pipe_write) = load_or_zero(dC_ptr, m_idx * ldDC + g_next * gs + c, m_idx < M);
            }
            for (int i = thread_idx; i < gs * rs; i += n_producer_threads) {
                int src_row = i / rs;
                int src_col = i % rs;
                sBt(src_col, src_row, dc_b_pipe_write) = *(B_ptr + (g_next * gs + src_row) * ldB + b * rs + src_col);
            }
        }
        cp_async_fence();
        cp_async_wait<bP_dc_b - 1>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        // LDSM s2r loads indexed by pipe
        copy(s2r_atom_dC{}, tXsdC(_,_,_,dc_b_pipe_read), tXrDC);
        copy(s2r_atom_Bt{}, tXsBt(_,_,_,dc_b_pipe_read), tXrBt);

        // MMA: dAR = dC @ B^T
        clear(rDAR);
        gemm(mma, rDC, rBt, rDAR);

        // Wait for consumer to release dAR buffer
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dar_pipe_write + BAR_CONSUMED_BASE), "n"(n_total_threads));

        // Write dAR to pipeline buffer (warp portion)
        copy(r2s_atom{}, tXrDAR, tXsDAR(_,_,_,dar_pipe_write));
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        // Signal consumer: dAR ready
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dar_pipe_write + BAR_READY_BASE), "n"(n_total_threads));

        ++dar_pipe_write;
        if (dar_pipe_write == bP_dar) dar_pipe_write = 0;

        dc_b_pipe_write = dc_b_pipe_read;
        ++dc_b_pipe_read;
        if (dc_b_pipe_read == bP_dc_b) dc_b_pipe_read = 0;
    }
}


// Consumer: loads A+R, uses dAR from producer for dA/dR computation (MMA-accelerated)
template <int BLK_M, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemA, class SmemR, class SmemRt, class SmemARtemp, class SmemDAR,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dAdR_consumer(
    const half_t* __restrict__ A_ptr, int ldA,
    const half_t* __restrict__ R_ptr, int ldR,
    half_t* __restrict__ dA_ptr, int ldDA,
    float* __restrict__ dR_partial,
    SmemA& sA, SmemR& sR_buf, SmemRt& sRt_buf, SmemARtemp& sAR_temp, SmemDAR& sdAR,
    int M, int K, int b, int m_start,
    int n_groups, int n_buf_slots, int buf_slot,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_dar = cute::BwdDAdRParams::bP_dar;
    constexpr int bP_a_r = cute::BwdDAdRParams::bP_a_r;

    // Offset barrier IDs by 1 to avoid conflict with __syncthreads() barrier 0
    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_dar;

    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    constexpr int n_consumer_warps = size(WarpLayoutConsumer{});
    constexpr int WARP_M = BLK_M / n_consumer_warps;
    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;

    int m_valid = min(m_start + BLK_M, M) - m_start;

    // -- Signal producer that all dAR pipe slots are initially available --
    for (int bid = BAR_CONSUMED_BASE; bid < BAR_CONSUMED_BASE + bP_dar; ++bid) {
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(bid), "n"(n_total_threads));
    }

    // -- Swizzle-preserving sub-views via logical_divide --
    // Flat WARP_M tiler for contiguous warp regions

    // sdAR: (BLK_M, rs, bP_dar) -> warp sub-view (WARP_M, rs, bP_dar)
    Tensor sdAR_warp = logical_divide(sdAR,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _, _); // (WARP_M, rs, bP_dar)

    // sA: (BLK_M, rs) -> warp sub-view (WARP_M, rs)
    Tensor sA_warp = logical_divide(sA,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _); // (WARP_M, rs)

    // sRt_buf: (rs, rs, bP_a_r) — no M-tiling needed
    // sR_buf: (rs, rs, bP_a_r) — no M-tiling needed (gated only)

    // sAR_temp: (BLK_M, rs) -> warp sub-view (WARP_M, rs) (gated only)
    Tensor sAR_temp_warp = logical_divide(sAR_temp,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _); // (WARP_M, rs)

    // -- dA MMA setup: F32 accumulator persists across group iterations --
    using mma_atom_da = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto mma_da = make_tiled_mma(
        mma_atom_da{},
        Layout<Shape<_1, _1>>{},
        Tile<Int<WARP_M>, Int<rs>>{}
    );
    auto thr_mma_da = mma_da.get_slice(lane_idx);

    // F32 accumulator fragment for dA
    auto dA_shape = make_shape(Int<WARP_M>{}, Int<rs>{});
    auto sdA_layout = make_layout(dA_shape, LayoutRight{});
    auto tMcDA_dummy = thr_mma_da.partition_C(make_tensor(
        static_cast<half_t*>(nullptr), sdA_layout));
    auto rDA = thr_mma_da.make_fragment_C(tMcDA_dummy);
    clear(rDA);

    // -- LDSM atoms for s2r (swizzled smem) --
    constexpr int a_u32_s3 = (WARP_M * rs) / 64;
    using s2r_atom_dAR = std::conditional_t<(a_u32_s3 % 4 == 0),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    constexpr int b_u32_s3 = (rs * rs) / 64;
    using s2r_atom_Rt = std::conditional_t<(b_u32_s3 % 4 == 0),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

    // Partition s2r once before loop for dA MMA (dAR @ Rt)
    auto s2r_dar = make_tiled_copy_A(s2r_atom_dAR{}, mma_da);
    auto s2r_dar_thr = s2r_dar.get_slice(lane_idx);
    auto tXsdAR = s2r_dar_thr.partition_S(sdAR_warp); // (CPY, CPY_M, CPY_K, bP_dar)
    auto rdAR_frag = thr_mma_da.make_fragment_A(thr_mma_da.partition_A(sdAR_warp(_,_,_0{})));
    auto tXrdAR = s2r_dar_thr.retile_D(rdAR_frag);

    auto s2r_rt = make_tiled_copy_B(s2r_atom_Rt{}, mma_da);
    auto s2r_rt_thr = s2r_rt.get_slice(lane_idx);
    auto tXsRt = s2r_rt_thr.partition_S(sRt_buf); // (CPY, CPY_N, CPY_K, bP_a_r)
    auto rRt_frag = thr_mma_da.make_fragment_B(thr_mma_da.partition_B(sRt_buf(_,_,_0{})));
    auto tXrRt = s2r_rt_thr.retile_D(rRt_frag);

    // -- Load A_b(BLK_M, rs) to smem --
    for (int i = thread_idx; i < BLK_M * rs; i += n_consumer_threads) {
        int r = i / rs, c = i % rs;
        int m_idx = m_start + r;
        sA(r, c) = load_or_zero(A_ptr, m_idx * ldA + b * rs + c, m_idx < M);
    }

    // -- Prefill R pipeline: load R^T (always) and R (gated only) --
    for (int p = 0; p < bP_a_r - 1 && p < n_groups; ++p) {
        for (int i = thread_idx; i < rs * rs; i += n_consumer_threads) {
            int r = i / rs, c = i % rs;
            half_t val = *(R_ptr + (p * rs + r) * ldR + b * rs + c);
            sRt_buf(c, r, p) = val;
            if constexpr (GATED) {
                sR_buf(r, c, p) = val;
            }
        }
        cp_async_fence();
    }

    int r_pipe_write = bP_a_r - 1;
    int r_pipe_read = 0;
    int dar_pipe_read = 0;

    // -- Main group loop --
    for (int g = 0; g < n_groups; ++g) {
        // Load next R^T (and R for gated)
        int g_next = g + bP_a_r - 1;
        if (g_next < n_groups) {
            for (int i = thread_idx; i < rs * rs; i += n_consumer_threads) {
                int r = i / rs, c = i % rs;
                half_t val = *(R_ptr + (g_next * rs + r) * ldR + b * rs + c);
                sRt_buf(c, r, r_pipe_write) = val;
                if constexpr (GATED) {
                    sR_buf(r, c, r_pipe_write) = val;
                }
            }
        }
        cp_async_fence();
        cp_async_wait<bP_a_r - 1>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

        // Wait for producer's dAR
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dar_pipe_read + BAR_READY_BASE), "n"(n_total_threads));

        // -- dA computation via MMA --
        if constexpr (GATED) {
            // Phase A: AR recompute via MMA — rAR = A_warp @ R_warp^T (natural TN fit)
            using mma_atom_ar = std::conditional_t<(rs < 16),
                MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
                MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;
            auto mma_ar = make_tiled_mma(
                mma_atom_ar{},
                Layout<Shape<_1, _1>>{},
                Tile<Int<WARP_M>, Int<rs>>{}
            );
            auto thr_mma_ar = mma_ar.get_slice(lane_idx);

            // LDSM s2r for AR recompute (same register counts as dA MMA)
            using s2r_atom_A_ar = std::conditional_t<(a_u32_s3 % 4 == 0),
                Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
                Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
            using s2r_atom_R_ar = std::conditional_t<(b_u32_s3 % 4 == 0),
                Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
                Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

            auto s2r_a_ar = make_tiled_copy_A(s2r_atom_A_ar{}, mma_ar);
            auto s2r_a_ar_thr = s2r_a_ar.get_slice(lane_idx);
            auto rA_ar = thr_mma_ar.make_fragment_A(thr_mma_ar.partition_A(sA_warp));
            copy(s2r_atom_A_ar{}, s2r_a_ar_thr.partition_S(sA_warp), s2r_a_ar_thr.retile_D(rA_ar));

            auto s2r_b_ar = make_tiled_copy_B(s2r_atom_R_ar{}, mma_ar);
            auto s2r_b_ar_thr = s2r_b_ar.get_slice(lane_idx);
            auto tXsR_ar = s2r_b_ar_thr.partition_S(sR_buf); // (CPY, CPY_N, CPY_K, bP_a_r)
            auto rR_ar = thr_mma_ar.make_fragment_B(thr_mma_ar.partition_B(sR_buf(_,_,_0{})));
            copy(s2r_atom_R_ar{}, tXsR_ar(_,_,_,r_pipe_read), s2r_b_ar_thr.retile_D(rR_ar));

            // AR MMA output -> sAR_temp
            auto tCsAR = thr_mma_ar.partition_C(sAR_temp_warp);
            auto rAR = thr_mma_ar.make_fragment_C(tCsAR);
            clear(rAR);
            gemm(mma_ar, rA_ar, rR_ar, rAR);

            // r2s: write rAR -> sAR_temp
            using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
            auto r2s_ar = make_tiled_copy_C(r2s_atom_AR{}, mma_ar);
            auto r2s_ar_thr = r2s_ar.get_slice(lane_idx);
            copy(r2s_atom_AR{}, r2s_ar_thr.retile_S(rAR), r2s_ar_thr.partition_D(sAR_temp_warp));
            asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

            // Phase B: Element-wise SiLU + dS (scalar — swizzle handled by CuTe layout)
            for (int idx = thread_idx; idx < m_valid * rs; idx += n_consumer_threads) {
                int mi = idx / rs, ri = idx % rs;
                float ar = float(sAR_temp(mi, ri));
                float dH = float(sdAR(mi, ri, dar_pipe_read));
                float a_val = float(sA(mi, ri));
                float sigma = 1.0f / (1.0f + __expf(-ar));
                float silu_ar = ar * sigma;
                float silu_prime = sigma * (1.0f + ar * (1.0f - sigma));
                float dS = dH * a_val * silu_prime;
                sdAR(mi, ri, dar_pipe_read) = half_t(dS);
                sAR_temp(mi, ri) = half_t(dH * silu_ar);
            }
            asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

            // Phase C: Load SiLU contribution (dH*SiLU(AR)) into rDA
            {
                auto tMcDA_id = thr_mma_da.partition_C(make_identity_tensor(dA_shape));
                for (int i = 0; i < size(rDA); ++i) {
                    auto coord = tMcDA_id(i);
                    int mi = get<0>(coord) + warp_idx * WARP_M;
                    if (mi < m_valid)
                        rDA(i) += float(sAR_temp(mi, get<1>(coord)));
                }
            }

            // Phase D: dA += dS @ R via MMA (sdAR now contains dS)
            {
                copy(s2r_atom_dAR{}, tXsdAR(_,_,_,dar_pipe_read), tXrdAR);
                copy(s2r_atom_Rt{}, tXsRt(_,_,_,r_pipe_read), tXrRt);
                gemm(mma_da, rdAR_frag, rRt_frag, rDA);
            }
        } else {
            // Non-gated: dA += dAR @ R via MMA
            copy(s2r_atom_dAR{}, tXsdAR(_,_,_,dar_pipe_read), tXrdAR);
            copy(s2r_atom_Rt{}, tXsRt(_,_,_,r_pipe_read), tXrRt);
            gemm(mma_da, rdAR_frag, rRt_frag, rDA);
        }

        asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

        // dR partial: dR_g_b(i,j) = sum_m dAR(m,i) * A(m,j)  (scalar, kept as-is)
        {
            for (int idx = thread_idx; idx < rs * rs; idx += n_consumer_threads) {
                int i = idx / rs, j = idx % rs;
                float val = 0.0f;
                for (int mi = 0; mi < m_valid; ++mi)
                    val += float(sdAR(mi, i, dar_pipe_read)) * float(sA(mi, j));
                if (val != 0.0f) {
                    int dr_row = g * rs + i;
                    int dr_col = b * rs + j;
                    int dR_rows = n_groups * rs;
                    atomicAdd(&dR_partial[buf_slot * dR_rows * K + dr_row * K + dr_col], val);
                }
            }
        }

        // Signal producer: dAR consumed
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dar_pipe_read + BAR_CONSUMED_BASE), "n"(n_total_threads));

        ++dar_pipe_read;
        if (dar_pipe_read == bP_dar) dar_pipe_read = 0;

        r_pipe_write = r_pipe_read;
        ++r_pipe_read;
        if (r_pipe_read == bP_a_r) r_pipe_read = 0;
    }

    // Write dA to global memory via MMA coordinate mapping
    {
        auto tMcDA = thr_mma_da.partition_C(make_identity_tensor(dA_shape));
        for (int i = 0; i < size(rDA); ++i) {
            auto coord = tMcDA(i);
            int mi = get<0>(coord) + warp_idx * WARP_M;
            int m_idx = m_start + mi;
            if (m_idx < M)
                *(dA_ptr + m_idx * ldDA + b * rs + get<1>(coord)) = half_t(rDA(i));
        }
    }
}


template <int BLK_M, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(
    (size(typename cute::BwdDAdRParams::warp_layout_arb{}) +
     size(typename cute::BwdDAdRParams::warp_layout_ar{})) * 32)
fused_dA_dR_kernel(
    const half_t* __restrict__ dC_ptr,  int ldDC,
    const half_t* __restrict__ A_ptr,   int ldA,
    const half_t* __restrict__ B_ptr,   int ldB,
    const half_t* __restrict__ R_ptr,   int ldR,
    half_t* __restrict__ dA_ptr,        int ldDA,
    float* __restrict__ dR_partial,
    int M, int N, int K,
    int n_groups, int n_blocks,
    int n_buf_slots)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dar = cute::BwdDAdRParams::bP_dar;
    constexpr int bP_a_r = cute::BwdDAdRParams::bP_a_r;

    using wlp_t = typename cute::BwdDAdRParams::warp_layout_arb;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_ar;
    wlp_t warp_layout_producer;
    wlc_t warp_layout_consumer;

    constexpr int n_producer_threads = size(wlp_t{}) * 32;
    constexpr int n_consumer_threads = size(wlc_t{}) * 32;

    // Grid with z-curve ordering
    auto grid_shape = make_shape(n_blocks, (M + BLK_M - 1) / BLK_M);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int b = get<0>(grid_coord);
    int m_tile = get<1>(grid_coord);

    int buf_slot = m_tile % n_buf_slots;
    int m_start = m_tile * BLK_M;

    // -- Shared memory layouts (SWIZZLED for LDSM_N s2r) --
    constexpr int K_atom = (gs < 16) ? 8 : 16;
    auto smem_producer = get_smem_atom(Int<K_atom>{});
    auto smem_rs = get_smem_atom(Int<rs>{});

    // Producer: sdC(BLK_M, gs, bP_dc_b)
    auto sdC_layout = tile_to_shape(smem_producer,
        make_shape(Int<BLK_M>{}, Int<gs>{}, Int<bP_dc_b>{}));
    // Producer: sBt(rs, gs, bP_dc_b)
    auto sBt_layout = tile_to_shape(smem_producer,
        make_shape(Int<rs>{}, Int<gs>{}, Int<bP_dc_b>{}));
    // Pipeline: sdAR(BLK_M, rs, bP_dar)
    auto sdAR_layout = tile_to_shape(smem_rs,
        make_shape(Int<BLK_M>{}, Int<rs>{}, Int<bP_dar>{}));
    // Consumer: sA(BLK_M, rs)
    auto sA_layout = tile_to_shape(smem_rs,
        make_shape(Int<BLK_M>{}, Int<rs>{}));
    // Consumer: sR(rs, rs, bP_a_r) — used only for gated AR recompute
    auto sR_layout = tile_to_shape(smem_rs,
        make_shape(Int<rs>{}, Int<rs>{}, Int<bP_a_r>{}));
    // Consumer: sRt(rs, rs, bP_a_r) — R transposed, for dA MMA
    auto sRt_layout = tile_to_shape(smem_rs,
        make_shape(Int<rs>{}, Int<rs>{}, Int<bP_a_r>{}));
    // Consumer: sAR_temp(BLK_M, rs) — temp for gated AR recompute MMA output + SiLU
    auto sAR_temp_layout = tile_to_shape(smem_rs,
        make_shape(Int<BLK_M>{}, Int<rs>{}));

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC      = make_tensor(make_smem_ptr(p), sdC_layout);      p += cosize(sdC_layout);
    auto sBt      = make_tensor(make_smem_ptr(p), sBt_layout);      p += cosize(sBt_layout);
    auto sdAR     = make_tensor(make_smem_ptr(p), sdAR_layout);     p += cosize(sdAR_layout);
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR_buf   = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sRt_buf  = make_tensor(make_smem_ptr(p), sRt_layout);      p += cosize(sRt_layout);
    auto sAR_temp = make_tensor(make_smem_ptr(p), sAR_temp_layout);

    __syncthreads();

    if (threadIdx.x >= n_consumer_threads) {
        dAdR_producer<BLK_M, gs, rs, GATED>(
            dC_ptr, ldDC, B_ptr, ldB,
            sdC, sBt, sdAR,
            M, N, K, b, m_start, n_groups,
            threadIdx.x - n_consumer_threads,
            warp_layout_producer, warp_layout_consumer);
    } else {
        dAdR_consumer<BLK_M, gs, rs, GATED>(
            A_ptr, ldA, R_ptr, ldR, dA_ptr, ldDA, dR_partial,
            sA, sR_buf, sRt_buf, sAR_temp, sdAR,
            M, K, b, m_start, n_groups, n_buf_slots, buf_slot,
            threadIdx.x,
            warp_layout_producer, warp_layout_consumer);
    }
}


// =============================================================================
// dR reduction kernel
// =============================================================================
__global__ void dR_reduce_kernel(
    const float* __restrict__ dR_partial,
    half* __restrict__ dR,
    int dR_elements,
    int n_buf_slots)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dR_elements) return;
    float sum = 0.0f;
    for (int s = 0; s < n_buf_slots; ++s)
        sum += dR_partial[s * dR_elements + idx];
    dR[idx] = __float2half(sum);
}


// =============================================================================
// Host launchers (split into dA+dR and dB for independent invocation)
// =============================================================================
void oft_backward_dA_dR_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* B,  int ldB,
    half const* R,  int ldR,
    half* dA, int ldDA,
    half* dR, int ldDR,
    bool gated,
    cudaStream_t stream)
{
    constexpr int gs = CurrKernelParams::group_size;
    constexpr int rs = CurrKernelParams::reconn_sz;

    constexpr int BLK_M_dAdR = cute::BwdDAdRParams::bM;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dar = cute::BwdDAdRParams::bP_dar;
    constexpr int bP_a_r = cute::BwdDAdRParams::bP_a_r;
    constexpr int n_buf_slots_param = cute::BwdDAdRParams::n_buf_slots;
    using wlp_dAdR = typename cute::BwdDAdRParams::warp_layout_arb;
    using wlc_dAdR = typename cute::BwdDAdRParams::warp_layout_ar;
    constexpr int n_threads_dAdR = (size(wlp_dAdR{}) + size(wlc_dAdR{})) * 32;

    int n_groups = n / gs;
    int n_blocks = k / rs;

    auto dC_h = reinterpret_cast<half_t const*>(dC);
    auto A_h  = reinterpret_cast<half_t const*>(A);
    auto B_h  = reinterpret_cast<half_t const*>(B);
    auto R_h  = reinterpret_cast<half_t const*>(R);
    auto dA_h = reinterpret_cast<half_t*>(dA);

    int n_m_tiles = (m + BLK_M_dAdR - 1) / BLK_M_dAdR;
    int n_buf_slots = min(n_buf_slots_param, n_m_tiles);
    if (n_buf_slots < 1) n_buf_slots = 1;
    int dR_elements = n_groups * rs * k;

    float* dR_partial_buf = nullptr;
    cudaMalloc(&dR_partial_buf, (int64_t)n_buf_slots * dR_elements * sizeof(float));
    cudaMemsetAsync(dR_partial_buf, 0, (int64_t)n_buf_slots * dR_elements * sizeof(float), stream);

    dim3 grid(n_blocks * n_m_tiles);
    dim3 block(n_threads_dAdR);

    constexpr int K_atom = (gs < 16) ? 8 : 16;
    auto smem_producer = get_smem_atom(cute::Int<K_atom>{});
    auto smem_rs = get_smem_atom(cute::Int<rs>{});
    int smem = (cosize(tile_to_shape(smem_producer, make_shape(cute::Int<BLK_M_dAdR>{}, cute::Int<gs>{}, cute::Int<bP_dc_b>{})))
                + cosize(tile_to_shape(smem_producer, make_shape(cute::Int<rs>{}, cute::Int<gs>{}, cute::Int<bP_dc_b>{})))
                + cosize(tile_to_shape(smem_rs, make_shape(cute::Int<BLK_M_dAdR>{}, cute::Int<rs>{}, cute::Int<bP_dar>{})))
                + cosize(tile_to_shape(smem_rs, make_shape(cute::Int<BLK_M_dAdR>{}, cute::Int<rs>{})))
                + cosize(tile_to_shape(smem_rs, make_shape(cute::Int<rs>{}, cute::Int<rs>{}, cute::Int<bP_a_r>{})))
                + cosize(tile_to_shape(smem_rs, make_shape(cute::Int<rs>{}, cute::Int<rs>{}, cute::Int<bP_a_r>{})))
                + cosize(tile_to_shape(smem_rs, make_shape(cute::Int<BLK_M_dAdR>{}, cute::Int<rs>{}))))
               * sizeof(half_t) + 256;

    auto kernel = gated ? fused_dA_dR_kernel<BLK_M_dAdR, gs, rs, true>
                        : fused_dA_dR_kernel<BLK_M_dAdR, gs, rs, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        dC_h, ldDC, A_h, ldA, B_h, ldB, R_h, ldR, dA_h, ldDA,
        dR_partial_buf, m, n, k, n_groups, n_blocks, n_buf_slots);

    // Reduce dR
    {
        int threads = 256;
        int blocks_r = (dR_elements + threads - 1) / threads;
        dR_reduce_kernel<<<blocks_r, threads, 0, stream>>>(
            dR_partial_buf, dR, dR_elements, n_buf_slots);
    }

    cudaStreamSynchronize(stream);
    cudaFree(dR_partial_buf);
}
