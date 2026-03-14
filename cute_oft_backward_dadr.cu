#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// Restructured Kernel: Fused dA + dR — Producer-Consumer
//
// Grid: (K/BLK_K) × (M/BLK_M) with z-curve ordering.
// Producer: dH(BLK_M, BLK_K) = dC_g(BLK_M, gs) @ B_g(gs, BLK_K), F32 output via sdH
// Consumer: dA accumulation + dR partial sums from sdH (F32) + sA + sR
// =============================================================================

__device__ __forceinline__
half_t load_or_zero(const half_t* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : half_t(0);
}

// =============================================================================
// Producer: dH GEMM with F32 accumulation, writing F32 sdH
// =============================================================================
template <int BLK_M, int BLK_K, int BLK_K_INNER, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemDC, class SmemB,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dAdR_producer_v2(
    const half_t* __restrict__ dC_ptr, int ldDC,
    const half_t* __restrict__ B_ptr,  int ldB,
    SmemDC& sdC, SmemB& sB,
    half_t* __restrict__ sdH,  // F16 shared memory, pipelined (BLK_M, BLK_K, bP_dh)
    int M, int N, int K, int m_start, int k_start,
    int n_groups,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;

    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_dh;

    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;
    constexpr int n_inner_tiles = gs / BLK_K_INNER;

    // -- MMA: F32 accumulator TN GEMM --
    using mma_atom_t = std::conditional_t<(BLK_K_INNER < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;

    auto mma = make_tiled_mma(mma_atom_t{}, WarpLayoutProducer{},
                               Tile<Int<BLK_M>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(lane_idx + warp_idx * 32);

    // F32 MMA accumulator for dH
    auto rDH = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}))));

    // Identity tensor for F32→F16 writeback to pipelined sdH
    auto tCdH_id = thr_mma.partition_C(make_identity_tensor(
        make_shape(Int<BLK_M>{}, Int<BLK_K>{})));

    // -- LDSM s2r atoms --
    // dC (A-operand): LDSM_N (K_INNER contiguous in smem)
    constexpr int K_atom = (BLK_K_INNER < 16) ? 8 : 16;
    constexpr int n_pw = size(WarpLayoutProducer{});
    constexpr int a_u32 = (BLK_M / n_pw * K_atom) / 64;
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

    // B (B-operand): LDSM_T (N-contiguous smem, transpose during s2r)
    // LDSM_T transposes N-major smem → K-major registers for TN MMA
    constexpr int b_u16 = (BLK_K * K_atom) / 32;
    using s2r_B = std::conditional_t<(b_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, half_t>,
        std::conditional_t<(b_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, half_t>,
            Copy_Atom<SM75_U16x2_LDSM_T, half_t>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dC{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx + warp_idx * 32);
    auto tXsdC = s2r_a_thr.partition_S(sdC);
    auto rdC = thr_mma.make_fragment_A(thr_mma.partition_A(sdC(_,_,_0{})));
    auto tXrdC = s2r_a_thr.retile_D(rdC);

    // sB has shape (BLK_K, BLK_K_INNER, pipe) — directly in MMA B-operand shape
    // (N=BLK_K first, BLK_K contiguous due to Swizzle<3,3,3> atom)
    auto s2r_b = make_tiled_copy_B(s2r_B{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(lane_idx + warp_idx * 32);
    auto tXsB = s2r_b_thr.partition_S(sB);
    auto rB = thr_mma.make_fragment_B(thr_mma.partition_B(sB(_,_,_0{})));
    auto tXrB = s2r_b_thr.retile_D(rB);

    // -- cp.async for dC and B (vectorized 128-bit) --
    auto copy_dC = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_K_INNER>{},
                                                  Int<n_producer_threads>{});
    auto thr_copy_dC = copy_dC.get_slice(thread_idx);
    auto tCsdC_cp = thr_copy_dC.partition_D(sdC);

    // B cp.async: (BLK_K, BLK_K_INNER) with BLK_K contiguous (dim 0)
    // Vectorize 8 halfs along dim 0 (BLK_K). Thread layout covers (BLK_K/8, BLK_K_INNER).
    constexpr int b_threads_dim0 = BLK_K / 8;  // 8 threads along BLK_K (each loads 8 halfs)
    constexpr int b_threads_dim1 = n_producer_threads / b_threads_dim0;
    constexpr int b_values_dim1 = BLK_K_INNER / b_threads_dim1;
    auto copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<b_threads_dim0>{}, Int<b_threads_dim1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<b_values_dim1>{})));
    auto thr_copy_B = copy_B.get_slice(thread_idx);
    auto tCsB_cp = thr_copy_B.partition_D(sB);

    int dh_pipe_write = 0;

    for (int g = 0; g < n_groups; ++g) {
        // sdH is NOT aliased with sdC+sB — producer can start GEMM immediately
        // CONSUMED barrier moved to before sdH write (below)
        clear(rDH);
        int pipe_w = bP_dc_b - 1, pipe_r = 0;

        // Helpers: load dC and B tiles via cp.async (vectorized 128-bit)
        auto load_dC_async = [&](int g, int ki_offset, int pipe) {
            auto gdC = make_tensor(
                make_gmem_ptr(dC_ptr + m_start * ldDC + g * gs + ki_offset),
                make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K_INNER>{}),
                            make_stride(ldDC, Int<1>{})));
            copy(copy_dC, thr_copy_dC.partition_S(gdC), tCsdC_cp(_,_,_,pipe));
        };

        // B async load helper
        auto load_B_async = [&](int g, int ki_offset, int pipe) {
            // B(N, K) row-major. We load (BLK_K cols, BLK_K_INNER rows) with BLK_K contiguous.
            // gmem tensor: (BLK_K, BLK_K_INNER) with stride (1, ldB) matching smem shape
            auto gB = make_tensor(
                make_gmem_ptr(B_ptr + (g * gs + ki_offset) * ldB + k_start),
                make_layout(make_shape(Int<BLK_K>{}, Int<BLK_K_INNER>{}),
                            make_stride(Int<1>{}, ldB)));
            copy(copy_B, thr_copy_B.partition_S(gB), tCsB_cp(_,_,_,pipe));
        };

        // Prefill
        for (int p = 0; p < bP_dc_b - 1 && p < n_inner_tiles; ++p) {
            int off = p * BLK_K_INNER;
            load_dC_async(g, off, p);
            load_B_async(g, off, p);
            cp_async_fence();
        }

        // Inner K loop
        for (int ki = 0; ki < n_inner_tiles; ++ki) {
            int ki_next = ki + bP_dc_b - 1;
            if (ki_next < n_inner_tiles) {
                int off = ki_next * BLK_K_INNER;
                load_dC_async(g, off, pipe_w);
                load_B_async(g, off, pipe_w);
            }
            cp_async_fence();
            cp_async_wait<bP_dc_b - 1>();
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

            copy(s2r_dC{}, tXsdC(_,_,_,pipe_r), tXrdC);
            copy(s2r_B{}, tXsB(_,_,_,pipe_r), tXrB);
            gemm(mma, rdC, rB, rDH);

            pipe_w = pipe_r;
            if (++pipe_r == bP_dc_b) pipe_r = 0;
        }

        // Ensure ALL outstanding async copies are complete
        cp_async_wait<0>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        // Wait for consumer to finish reading the sdH pipe slot we're about to overwrite
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dh_pipe_write + BAR_CONSUMED_BASE), "n"(n_total_threads));

        // Write F32 rDH → F16 sdH using identity coordinate mapping
        // sdH is F16 pipelined: sdH[pipe * BLK_M * BLK_K + mi * BLK_K + ki]
        {
            constexpr int sdH_stride = BLK_M * BLK_K;
            half_t* sdH_slot = reinterpret_cast<half_t*>(sdH) + dh_pipe_write * sdH_stride;
            for (int i = 0; i < size(rDH); ++i) {
                auto coord = tCdH_id(i);
                int mi = get<0>(coord), ki = get<1>(coord);
                sdH_slot[mi * BLK_K + ki] = half_t(rDH(i));  // F32 → F16
            }
        }
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dh_pipe_write + BAR_READY_BASE), "n"(n_total_threads));
        if (++dh_pipe_write == bP_dh) dh_pipe_write = 0;
    }
}

// =============================================================================
// Consumer: dA accumulation + dR from F32 sdH
// =============================================================================
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemA, class SmemR, class SmemARtemp,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dAdR_consumer_v2(
    const half_t* __restrict__ A_ptr, int ldA,
    const half_t* __restrict__ R_ptr, int ldR,
    half_t* __restrict__ dA_ptr, int ldDA,
    float* __restrict__ dR_partial,
    SmemA& sA, SmemR& sR, const half_t* __restrict__ sdH, SmemARtemp& sAR_temp,
    int M, int K, int m_start, int k_start,
    int n_groups, int n_buf_slots, int buf_slot,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int rs = RECONN_SZ;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;
    constexpr int bP_r = cute::BwdDAdRParams::bP_r;
    constexpr int n_reconn_blocks = BLK_K / rs;

    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_dh;

    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    constexpr int n_consumer_warps = size(WarpLayoutConsumer{});
    constexpr int WARP_M = BLK_M / n_consumer_warps;
    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;
    int m_valid = min(m_start + BLK_M, M) - m_start;

    // -- Consumer MMA setup for reconn dA: (WARP_M, rs) += (WARP_M, rs) @ (rs, rs)^T --
    using cons_mma_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto cons_mma = make_tiled_mma(cons_mma_atom{},
        Layout<Shape<_1, _1>>{},  // single warp
        Tile<Int<WARP_M>, Int<rs>>{});
    auto cons_thr = cons_mma.get_slice(lane_idx);

    // Per-reconn-block F32 C-accumulators (persist across all groups)
    auto dA_shape = make_shape(Int<WARP_M>{}, Int<rs>{});
    auto dA_layout = make_layout(dA_shape, LayoutRight{});
    auto tCdA_dummy = cons_thr.partition_C(make_tensor(static_cast<half_t*>(nullptr), dA_layout));
    // Array of n_reconn_blocks MMA C-accumulators (non-gated path)
    using frag_c_type = decltype(cons_thr.make_fragment_C(tCdA_dummy));
    frag_c_type rDA_blk[n_reconn_blocks];
    #pragma unroll
    for (int b = 0; b < n_reconn_blocks; ++b) clear(rDA_blk[b]);

    // (flat rDA array removed — both gated and non-gated use per-block MMA accumulators)

    // Identity tensor for consumer MMA C → (mi, ki) writeback
    auto tCdA_id = cons_thr.partition_C(make_identity_tensor(dA_shape));

    // sAR_temp warp sub-view for consumer MMA A-operand (dH slice)
    // sAR_temp: (BLK_M, rs) row-major. Each warp processes WARP_M rows.
    Tensor sAR_warp = make_tensor(
        &sAR_temp(warp_idx * WARP_M, 0),
        make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), LayoutRight{}));

    // LDSM s2r for consumer A-operand (dH from sAR_temp)
    constexpr int cons_a_u32 = (WARP_M * rs) / 64;
    using cons_s2r_A = std::conditional_t<(cons_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    auto cons_s2r_a = make_tiled_copy_A(cons_s2r_A{}, cons_mma);
    auto cons_s2r_a_thr = cons_s2r_a.get_slice(lane_idx);
    auto tXsAR = cons_s2r_a_thr.partition_S(sAR_warp);
    auto rDH_frag = cons_thr.make_fragment_A(cons_thr.partition_A(sAR_warp));
    auto tXrDH_frag = cons_s2r_a_thr.retile_D(rDH_frag);

    // Signal producer: all dH slots available
    for (int bid = BAR_CONSUMED_BASE; bid < BAR_CONSUMED_BASE + bP_dh; ++bid)
        asm volatile("bar.arrive %0, %1;\n" : : "r"(bid), "n"(n_total_threads));

    // Load A once
    for (int i = thread_idx; i < BLK_M * BLK_K; i += n_consumer_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sA(r, c) = (m_start + r < M && k_start + c < K) ? *(A_ptr + (m_start + r) * ldA + k_start + c) : half_t(0);
    }

    // Prefill R pipeline
    for (int p = 0; p < bP_r - 1 && p < n_groups; ++p) {
        for (int i = thread_idx; i < rs * BLK_K; i += n_consumer_threads) {
            int r = i / BLK_K, c = i % BLK_K;
            sR(r, c, p) = (k_start + c < K) ? *(R_ptr + (p * rs + r) * ldR + k_start + c) : half_t(0);
        }
        cp_async_fence();
    }

    int r_pipe_w = bP_r - 1, r_pipe_r = 0, dh_pipe_r = 0;

    for (int g = 0; g < n_groups; ++g) {
        // Load next R
        int g_next = g + bP_r - 1;
        if (g_next < n_groups) {
            for (int i = thread_idx; i < rs * BLK_K; i += n_consumer_threads) {
                int r = i / BLK_K, c = i % BLK_K;
                sR(r, c, r_pipe_w) = (k_start + c < K) ? *(R_ptr + (g_next * rs + r) * ldR + k_start + c) : half_t(0);
            }
        }
        cp_async_fence();
        cp_async_wait<bP_r - 1>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

        // Wait for producer's dH (F16 pipelined)
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dh_pipe_r + BAR_READY_BASE), "n"(n_total_threads));

        // Pointer to current sdH pipe slot (F16, row-major within slot)
        const half_t* sdH_cur = sdH + dh_pipe_r * BLK_M * BLK_K;

        if constexpr (GATED) {
            // Gated path: per-reconn-block AR recompute, SiLU, dA accumulation via MMA
            // Then write all dS to sAR_temp ONCE for dR computation
            auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                make_tensor(static_cast<half_t*>(nullptr),
                            make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
            auto tBid = cons_thr.partition_B(make_identity_tensor(
                make_shape(Int<rs>{}, Int<rs>{})));

            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;

                // AR recompute via MMA: rAR = A @ R^T
                // First write A slice to sAR_temp, sync, then LDSM
                for (int idx = thread_idx; idx < BLK_M * rs; idx += n_consumer_threads) {
                    int mi = idx / rs, ri = idx % rs;
                    sAR_temp(mi, ri) = (mi < m_valid) ? sA(mi, k_off + ri) : half_t(0);
                }
                asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);  // load A slice
                for (int i = 0; i < size(rR_frag); ++i) {
                    auto coord = tBid(i);
                    rR_frag(i) = sR(get<0>(coord), k_off + get<1>(coord), r_pipe_r);  // R natural for A@R^T
                }
                frag_c_type rAR;
                clear(rAR);
                gemm(cons_mma, rDH_frag, rR_frag, rAR);

                // Elementwise: dA += dH * SiLU(AR), compute dS, write dS to sAR_temp
                for (int i = 0; i < size(rAR); ++i) {
                    auto coord = tCdA_id(i);
                    int mi = get<0>(coord) + warp_idx * WARP_M;
                    int ri = get<1>(coord);
                    if (mi < m_valid) {
                        float ar_f = rAR(i);
                        // Fast sigmoid via tanh.approx: sigmoid(x) = 0.5 + 0.5*tanh(0.5*x)
                        half_t ar_h = half_t(ar_f);
                        half_t half_ar_h;
                        asm("mul.f16 %0, %1, %2;" : "=h"(*(uint16_t*)&half_ar_h)
                            : "h"(*(uint16_t*)&ar_h), "h"(uint16_t(0x3800)));  // 0.5 in f16
                        half_t tanh_h;
                        asm("tanh.approx.f16 %0, %1;" : "=h"(*(uint16_t*)&tanh_h)
                            : "h"(*(uint16_t*)&half_ar_h));
                        float sigma = 0.5f + 0.5f * float(tanh_h);

                        float dH_val = float(sdH_cur[mi * BLK_K + k_off + ri]);
                        float a_val = float(sA(mi, k_off + ri));

                        rDA_blk[b](i) += dH_val * ar_f * sigma;  // dH * SiLU(AR)
                        float silu_prime = sigma * (1.0f + ar_f * (1.0f - sigma));
                        sAR_temp(mi, ri) = half_t(dH_val * a_val * silu_prime);  // dS
                    }
                }
                asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

                // dA += dS @ R via MMA
                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);
                for (int i = 0; i < size(rR_frag); ++i) {
                    auto coord = tBid(i);
                    rR_frag(i) = sR(get<1>(coord), k_off + get<0>(coord), r_pipe_r);  // R for dA@R
                }
                gemm(cons_mma, rDH_frag, rR_frag, rDA_blk[b]);

                // dR partial sums — parallelize mi reduction
                {
                    constexpr int threads_per_elem = n_consumer_threads / (rs * rs);
                    for (int idx = thread_idx; idx < rs * rs * threads_per_elem; idx += n_consumer_threads) {
                        int elem = idx / threads_per_elem;
                        int chunk = idx % threads_per_elem;
                        int i = elem / rs, j = elem % rs;
                        int mi_start = chunk * (m_valid / threads_per_elem);
                        int mi_end = (chunk + 1) * (m_valid / threads_per_elem);
                        if (chunk == threads_per_elem - 1) mi_end = m_valid;
                        float val = 0.0f;
                        for (int mi = mi_start; mi < mi_end; ++mi)
                            val += float(sAR_temp(mi, i)) * float(sA(mi, k_off + j));
                        if (val != 0.0f) {
                            int dR_rows = n_groups * rs;
                            atomicAdd(&dR_partial[buf_slot * dR_rows * K + (g * rs + i) * K + k_start + k_off + j], val);
                        }
                    }
                }
            }
        } else {
            // Non-gated: dA via MMA per reconn block
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;

                // Step 1: Copy F32 sdH slice → F16 sAR_temp (all consumer threads cooperate)
                for (int idx = thread_idx; idx < BLK_M * rs; idx += n_consumer_threads) {
                    int mi = idx / rs, ri = idx % rs;
                    sAR_temp(mi, ri) = (mi < m_valid) ? sdH_cur[mi * BLK_K + k_off + ri] : half_t(0);
                }
                asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

                // Step 2: MMA dA_b += dH_b(WARP_M, rs) @ R_b(rs, rs)
                // Load A-operand from sAR_temp (warp's portion) via LDSM
                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);

                // Load B-operand (R sub-block) via scalar into MMA B-fragment
                // TN B-operand: (N=rs, K=rs) with K contiguous
                auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                    make_tensor(static_cast<half_t*>(nullptr),
                                make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                // Load using partition_B identity to find register positions
                {
                    auto tBid = cons_thr.partition_B(make_identity_tensor(
                        make_shape(Int<rs>{}, Int<rs>{})));
                    for (int i = 0; i < size(rR_frag); ++i) {
                        auto coord = tBid(i);
                        int col = get<0>(coord);  // N dimension
                        int ri = get<1>(coord);   // K dimension
                        rR_frag(i) = sR(ri, k_off + col, r_pipe_r);
                    }
                }

                // MMA accumulate
                gemm(cons_mma, rDH_frag, rR_frag, rDA_blk[b]);
            }

            // dR partial sums — parallelize mi reduction across all threads
            // Each (i, j, mi_chunk) is handled by one thread
            // rs*rs=64 elements, BLK_M=64 mi positions → 4096 work items for 128 threads = 32 each
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;
                constexpr int mi_chunks = (BLK_M + n_consumer_threads / (rs * rs) - 1) / (n_consumer_threads / (rs * rs));
                constexpr int threads_per_elem = n_consumer_threads / (rs * rs);  // 128/64 = 2
                for (int idx = thread_idx; idx < rs * rs * threads_per_elem; idx += n_consumer_threads) {
                    int elem = idx / threads_per_elem;
                    int chunk = idx % threads_per_elem;
                    int i = elem / rs, j = elem % rs;
                    int mi_start = chunk * (m_valid / threads_per_elem);
                    int mi_end = (chunk + 1) * (m_valid / threads_per_elem);
                    if (chunk == threads_per_elem - 1) mi_end = m_valid;  // handle remainder
                    float val = 0.0f;
                    for (int mi = mi_start; mi < mi_end; ++mi)
                        val += float(sdH_cur[mi * BLK_K + k_off + i]) * float(sA(mi, k_off + j));
                    if (val != 0.0f) {
                        int dR_rows = n_groups * rs;
                        atomicAdd(&dR_partial[buf_slot * dR_rows * K + (g * rs + i) * K + k_start + k_off + j], val);
                    }
                }
            }
        }

        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(dh_pipe_r + BAR_CONSUMED_BASE), "n"(n_total_threads));
        if (++dh_pipe_r == bP_dh) dh_pipe_r = 0;
        r_pipe_w = r_pipe_r;
        if (++r_pipe_r == bP_r) r_pipe_r = 0;
    }

    // Write dA to global memory
    // Both gated and non-gated use per-reconn-block MMA accumulators
    {
    // Non-gated: write from per-reconn-block MMA fragments
    for (int b = 0; b < n_reconn_blocks; ++b) {
        int k_off = b * rs;
        for (int i = 0; i < size(rDA_blk[b]); ++i) {
            auto coord = tCdA_id(i);
            int mi = get<0>(coord) + warp_idx * WARP_M;
            int ki = get<1>(coord) + k_off;
            int m_idx = m_start + mi;
            int k_idx = k_start + ki;
            if (m_idx < M && k_idx < K)
                *(dA_ptr + m_idx * ldDA + k_idx) = half_t(rDA_blk[b](i));
        }
    }
    } // end non-gated writeback
}

// =============================================================================
// Kernel entry point
// =============================================================================
template <int BLK_M, int BLK_K, int BLK_K_INNER, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(
    (size(typename cute::BwdDAdRParams::warp_layout_arb{}) +
     size(typename cute::BwdDAdRParams::warp_layout_ar{})) * 32)
fused_dA_dR_kernel_v2(
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
    constexpr int bP_r = cute::BwdDAdRParams::bP_r;

    using wlp_t = typename cute::BwdDAdRParams::warp_layout_arb;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_ar;

    constexpr int n_consumer_threads = size(wlc_t{}) * 32;

    // Z-curve grid mapping
    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start = get<0>(grid_coord) * BLK_K;
    int m_start = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;

    // Shared memory layouts
    auto smem_ki = get_smem_atom(Int<BLK_K_INNER>{});
    auto smem_bk = get_smem_atom(Int<BLK_K>{});

    auto sdC_layout = tile_to_shape(smem_ki, make_shape(Int<BLK_M>{}, Int<BLK_K_INNER>{}, Int<bP_dc_b>{}));

    // sB: (BLK_K, BLK_K_INNER, pipe) — MMA B-operand shape (N=BLK_K, K=BLK_K_INNER)
    // N-contiguous (BLK_K contiguous) for LDSM_T compatibility
    // Swizzle<3,3,3> with column-major atom following CUTLASS proven pattern
    auto sB_atom_ldsmt = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}),
                    make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom_ldsmt,
        make_shape(Int<BLK_K>{}, Int<BLK_K_INNER>{}, Int<bP_dc_b>{}));
    auto sA_layout = tile_to_shape(smem_bk, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto sR_layout = tile_to_shape(smem_bk, make_shape(Int<rs>{}, Int<BLK_K>{}, Int<bP_r>{}));
    auto sAR_temp_layout = make_layout(make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});

    // sdH as F16 with pipeline (bP_dh stages) — SEPARATE from sdC+sB for producer-consumer overlap
    auto sdH_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_dh>{}),
        make_stride(Int<BLK_K * bP_dh>{}, Int<bP_dh>{}, Int<1>{}));

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC = make_tensor(make_smem_ptr(p), sdC_layout); p += cosize(sdC_layout);
    auto sB  = make_tensor(make_smem_ptr(p), sB_layout);  p += cosize(sB_layout);
    // sdH is SEPARATE (not aliased) — enables producer-consumer overlap with bP_dh=2
    auto sdH_tensor = make_tensor(make_smem_ptr(p), sdH_layout); p += cosize(sdH_layout);
    auto sA  = make_tensor(make_smem_ptr(p), sA_layout);  p += cosize(sA_layout);
    auto sR  = make_tensor(make_smem_ptr(p), sR_layout);  p += cosize(sR_layout);
    auto sAR_temp = make_tensor(make_smem_ptr(p), sAR_temp_layout);

    __syncthreads();

    // Raw pointer to sdH F16 pipelined buffer
    half_t* sdH = &sdH_tensor(0, 0, 0);
    // Also as float* for producer's F32→F16 conversion
    // (producer writes F16 directly via identity mapping)

    if (threadIdx.x >= n_consumer_threads) {
        dAdR_producer_v2<BLK_M, BLK_K, BLK_K_INNER, gs, rs, GATED>(
            dC_ptr, ldDC, B_ptr, ldB,
            sdC, sB, sdH,
            M, N, K, m_start, k_start, n_groups,
            threadIdx.x - n_consumer_threads,
            wlp_t{}, wlc_t{});
    } else {
        dAdR_consumer_v2<BLK_M, BLK_K, gs, rs, GATED>(
            A_ptr, ldA, R_ptr, ldR, dA_ptr, ldDA, dR_partial,
            sA, sR, sdH, sAR_temp,
            M, K, m_start, k_start,
            n_groups, n_buf_slots, buf_slot,
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
    constexpr int BLK_M = cute::BwdDAdRParams::bM;
    constexpr int BLK_K = cute::BwdDAdRParams::bK;
    constexpr int BLK_K_INNER = cute::BwdDAdRParams::bK_inner;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
    constexpr int bP_dh = cute::BwdDAdRParams::bP_dh;
    constexpr int bP_r = cute::BwdDAdRParams::bP_r;
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

    auto smem_ki = get_smem_atom(cute::Int<BLK_K_INNER>{});
    auto smem_bk = get_smem_atom(cute::Int<BLK_K>{});
    // sdC + sB + sdH (F16, separate, pipelined) + sA + sR + sAR_temp
    int smem_halfs = cosize(tile_to_shape(smem_ki, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K_INNER>{}, cute::Int<bP_dc_b>{})))
                   + cosize(tile_to_shape(
                       composition(Swizzle<3,3,3>{},
                                   make_layout(make_shape(cute::Int<BLK_K>{}, _8{}),
                                               make_stride(_1{}, cute::Int<BLK_K>{}))),
                       make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_K_INNER>{}, cute::Int<bP_dc_b>{})))
                   + BLK_M * BLK_K * bP_dh  // F16 sdH (non-swizzled, pipelined)
                   + cosize(tile_to_shape(smem_bk, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{})))
                   + cosize(tile_to_shape(smem_bk, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{}, cute::Int<bP_r>{})))
                   + BLK_M * rs;
    int smem = smem_halfs * sizeof(half_t) + 256;

    dim3 grid(n_k_tiles * n_m_tiles);
    dim3 block(n_threads);

    auto kernel = gated ? fused_dA_dR_kernel_v2<BLK_M, BLK_K, BLK_K_INNER, gs, rs, true>
                        : fused_dA_dR_kernel_v2<BLK_M, BLK_K, BLK_K_INNER, gs, rs, false>;
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
