#include "cute_prism_backward_db.hpp"
#include "cute_prism_coop_pc.hpp"
#include <prism_config.hpp>
#include "cute_prism_util.hpp"
#include "z_curve.hpp"

using namespace cute;

__device__ __forceinline__
prism_cute load_or_zero(const prism_cute* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : prism_cute(0);
}

// =============================================================================
// Producer: computes AR, applies SiLU gating, writes to sAR_pipe (no transpose)
// =============================================================================
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemA, class SmemR, class SmemARpipe,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dB_producer(
    const prism_cute* __restrict__ A_ptr, int ldA,
    const prism_cute* __restrict__ bias_ptr,  // (n_groups, K) row-major; ignored unless PRISM_INTERNAL_BIAS=1
    SmemA& sA, SmemR& sR, SmemARpipe& sAR_pipe,
    int M, int K, int g, int k_start,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer,
    // Dropout (only used when PRISM_DROPOUT=1)
    int64_t const* dropout_seeds, float dropout_p, float inv_keep)
{
    constexpr int rs = RECONN_SZ;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_ar;
    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    constexpr int n_blocks_k = BLK_K / rs;
    constexpr int n_producer_warps = size(WarpLayoutProducer{});
    constexpr int WARP_M = BLK_M / n_producer_warps;

    using mma_atom_ar = prism_ar_atom<rs>;
    auto mma_ar = make_tiled_mma(mma_atom_ar{}, Layout<Shape<_1, _1>>{},
                                  Tile<Int<WARP_M>, Int<rs>>{});
    auto thr_mma_ar = mma_ar.get_slice(lane_idx);

    // Divide sA and sR for per-warp, per-reconn-block access
    Tensor sA_warp = logical_divide(sA,
        make_tile(make_layout(Int<WARP_M>{}), make_layout(Int<rs>{}))
    )(make_coord(_, warp_idx), make_coord(_, _), _);
    Tensor sR_divided = logical_divide(sR,
        make_tile(_, make_layout(Int<rs>{})))(_, make_coord(_, _));

    // Divide sAR_pipe for per-reconn-block, per-warp, per-pipe access
    // sAR_pipe shape: (BLK_K, BLK_M, bP_ar) — transposed orientation
    Tensor sAR_pipe_divided = logical_divide(sAR_pipe,
        make_tile(make_layout(Int<rs>{}), make_layout(Int<WARP_M>{}), _));
    // Shape: ((rs, n_blocks_k), (WARP_M, n_warps), bP_ar)

    // Consumer's MMA atom (f16 variant) for R2S via partition_B
    using mma_atom_cons = prism_ar_atom<BLK_M>;
    using mma_op_cons = prism_ar_op<BLK_M>;
    auto r2s_mma = make_tiled_mma(mma_op_cons{}, Layout<Shape<_1, _1>>{},
                                   Tile<Int<GROUP_SIZE>, Int<rs>>{});
    auto r2s_thr_mma = r2s_mma.get_slice(lane_idx);

    // LDSM for A and R operands
    constexpr int a_u32 = (WARP_M * rs) / 64;
    using s2r_A = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>, Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>;
    constexpr int b_u32 = (rs * rs) / 64;
    using s2r_R = std::conditional_t<(b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>, Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>>;

    auto s2r_a = make_tiled_copy_A(s2r_A{}, mma_ar);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx);
    auto tXsA = s2r_a_thr.partition_S(sA_warp);
    auto rA_frag = thr_mma_ar.make_fragment_A(thr_mma_ar.partition_A(sA_warp(_,_,_0{},_0{})));
    auto tXrA = s2r_a_thr.retile_D(rA_frag);
    auto s2r_r = make_tiled_copy_B(s2r_R{}, mma_ar);
    auto s2r_r_thr = s2r_r.get_slice(lane_idx);
    auto tXsR = s2r_r_thr.partition_S(sR_divided);
    auto rR_frag = thr_mma_ar.make_fragment_B(thr_mma_ar.partition_B(sR_divided(_,_,_0{})));
    auto tXrR = s2r_r_thr.retile_D(rR_frag);

    // cp.async tiled copy for A loading (vectorized 128-bit loads)
    auto copy_a = cp_layout<uint128_t, prism_cute>(Int<BLK_M>{}, Int<BLK_K>{}, Int<n_producer_threads>{});
    auto thr_copy_a = copy_a.get_slice(thread_idx);
    auto tCsA_cp = thr_copy_a.partition_D(sA);

    auto load_A_async = [&](int m_start, int pipe) {
        if (m_start + BLK_M <= M) {
            auto gA = make_tensor(
                make_gmem_ptr(A_ptr + m_start * ldA + k_start),
                make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}),
                            make_stride(ldA, Int<1>{})));
            copy(copy_a, thr_copy_a.partition_S(gA), tCsA_cp(_,_,_,pipe));
        } else {
            for (int i = thread_idx; i < BLK_M * BLK_K; i += n_producer_threads) {
                int r = i / BLK_K, c = i % BLK_K;
                sA(r, c, pipe) = load_or_zero(A_ptr, (m_start + r) * ldA + k_start + c, m_start + r < M);
            }
        }
    };

    int smem_pipe_write = bP_a - 1, smem_pipe_read = 0;
    int ar_pipe_write = 0, m_tile_next = 0, m_tiles_remaining = n_m_tiles;

    // Prefill A pipeline
    for (int p = 0; p < bP_a - 1 && m_tiles_remaining > 0; ++p) {
        load_A_async(m_tile_next * BLK_M, p);
        cp_async_fence(); --m_tiles_remaining; ++m_tile_next;
    }

    for (int mt = 0; mt < n_m_tiles; ++mt) {
        if (m_tiles_remaining > 0) {
            load_A_async(m_tile_next * BLK_M, smem_pipe_write);
            --m_tiles_remaining; ++m_tile_next;
        }
        cp_async_fence(); cp_async_wait<bP_a - 1>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        // Wait for consumer to signal buffer is free
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(BAR_CONSUMED_BASE), "n"(n_total_threads));

        for (int kb = 0; kb < n_blocks_k; ++kb) {
            // LDSM load A and R
            copy(s2r_A{}, tXsA(_,_,_,kb,smem_pipe_read), tXrA);
            copy(s2r_R{}, tXsR(_,_,_,kb), tXrR);

            // MMA: AR[WARP_M, rs] = A @ R^T
            auto rAR = thr_mma_ar.make_fragment_C(thr_mma_ar.partition_C(
                make_tensor(static_cast<prism_cute*>(nullptr),
                    make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), LayoutRight{}))));
            clear(rAR);
            gemm(mma_ar, rA_frag, rR_frag, rAR);

            if constexpr (GATED) {
                constexpr int RS_v = rs;
                constexpr int A_OFFSET_STEP = 4;
                constexpr int N_PER_A_ATOM = RS_v / 8;
                const int col_lo = (lane_idx & 0x3) * 2;
                const int K_offset = k_start + kb * RS_v;
                const auto MMA_M_v = size<1>(rAR);
                const auto MMA_N_v = size<2>(rAR);
#if PRISM_DROPOUT
                // dB processes one group g per CTA; one seed for the whole site.
                const uint64_t seed_g = (uint64_t)dropout_seeds[g];
                // Each warp owns rows [w*WARP_M, w*WARP_M+WARP_M) of the
                // current m-tile; m_global = mt*BLK_M + warp*WARP_M + row.
                const int warp_m_base = mt * BLK_M + warp_idx * WARP_M;
#endif
                #pragma unroll
                for (int mn = 0; mn < int(MMA_N_v); ++mn) {
                    // gn_lo/gn_hi < rs (single group at the dB site), so
                    // ri_lo == gn_lo and ri_hi == gn_hi.
                    const int ri_lo = mn * 8 + col_lo;
                    const int ri_hi = ri_lo + 1;
                    const int a_off = (mn % N_PER_A_ATOM) * A_OFFSET_STEP;
#if PRISM_DTYPE == 0
                    // -- fp16 path: packed half2 silu --
#if PRISM_INTERNAL_BIAS
                    const __half2 bias2 = load_half2(
                        bias_ptr[g * K + K_offset + ri_lo],
                        bias_ptr[g * K + K_offset + ri_hi]);
#endif
                    #pragma unroll
                    for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                        auto& ar2_lo = reinterpret_cast<__half2&>(rAR(0, mm, mn));
                        auto& a2_lo  = reinterpret_cast<const __half2&>(rA_frag(0 + a_off, mm, 0));
                        auto& ar2_hi = reinterpret_cast<__half2&>(rAR(2, mm, mn));
                        auto& a2_hi  = reinterpret_cast<const __half2&>(rA_frag(2 + a_off, mm, 0));
#if PRISM_INTERNAL_BIAS
                        ar2_lo = __hadd2(ar2_lo, bias2);
                        ar2_hi = __hadd2(ar2_hi, bias2);
#endif
                        ar2_lo = __hmul2(a2_lo, silu_h2(ar2_lo));
                        ar2_hi = __hmul2(a2_hi, silu_h2(ar2_hi));
#if PRISM_DROPOUT
                        // mask layout matches fwd: idx = m_global * K + k_global.
                        const int m_lo = warp_m_base + mm * 16 + int(lane_idx >> 2);
                        const int m_hi = m_lo + 8;
                        const float u00 = prism_uniform_from_hash(seed_g, (int64_t)m_lo * K + K_offset + ri_lo);
                        const float u01 = prism_uniform_from_hash(seed_g, (int64_t)m_lo * K + K_offset + ri_hi);
                        const float u10 = prism_uniform_from_hash(seed_g, (int64_t)m_hi * K + K_offset + ri_lo);
                        const float u11 = prism_uniform_from_hash(seed_g, (int64_t)m_hi * K + K_offset + ri_hi);
                        const __half2 mask_lo = __floats2half2_rn(
                            (u00 >= dropout_p) ? inv_keep : 0.0f,
                            (u01 >= dropout_p) ? inv_keep : 0.0f);
                        const __half2 mask_hi = __floats2half2_rn(
                            (u10 >= dropout_p) ? inv_keep : 0.0f,
                            (u11 >= dropout_p) ? inv_keep : 0.0f);
                        ar2_lo = __hmul2(ar2_lo, mask_lo);
                        ar2_hi = __hmul2(ar2_hi, mask_hi);
#endif
                    }
#else
                    // -- bf16 path: rAR is float (F32-accum); silu in float --
#if PRISM_INTERNAL_BIAS
                    const float bias_lo_f = prism_to_float(reinterpret_cast<const prism_native&>(bias_ptr[g * K + K_offset + ri_lo]));
                    const float bias_hi_f = prism_to_float(reinterpret_cast<const prism_native&>(bias_ptr[g * K + K_offset + ri_hi]));
#endif
                    #pragma unroll
                    for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                        float ar0 = rAR(0, mm, mn);
                        float ar1 = rAR(1, mm, mn);
                        float ar2 = rAR(2, mm, mn);
                        float ar3 = rAR(3, mm, mn);
                        float a0 = float(rA_frag(0 + a_off, mm, 0));
                        float a1 = float(rA_frag(1 + a_off, mm, 0));
                        float a2 = float(rA_frag(2 + a_off, mm, 0));
                        float a3 = float(rA_frag(3 + a_off, mm, 0));
#if PRISM_INTERNAL_BIAS
                        ar0 += bias_lo_f; ar1 += bias_hi_f;
                        ar2 += bias_lo_f; ar3 += bias_hi_f;
#endif
                        const float s0 = 0.5f * (1.0f + tanhf(0.5f * ar0));
                        const float s1 = 0.5f * (1.0f + tanhf(0.5f * ar1));
                        const float s2 = 0.5f * (1.0f + tanhf(0.5f * ar2));
                        const float s3 = 0.5f * (1.0f + tanhf(0.5f * ar3));
                        float h0 = a0 * (ar0 * s0);
                        float h1 = a1 * (ar1 * s1);
                        float h2 = a2 * (ar2 * s2);
                        float h3 = a3 * (ar3 * s3);
#if PRISM_DROPOUT
                        const int m_lo = warp_m_base + mm * 16 + int(lane_idx >> 2);
                        const int m_hi = m_lo + 8;
                        const float u00 = prism_uniform_from_hash(seed_g, (int64_t)m_lo * K + K_offset + ri_lo);
                        const float u01 = prism_uniform_from_hash(seed_g, (int64_t)m_lo * K + K_offset + ri_hi);
                        const float u10 = prism_uniform_from_hash(seed_g, (int64_t)m_hi * K + K_offset + ri_lo);
                        const float u11 = prism_uniform_from_hash(seed_g, (int64_t)m_hi * K + K_offset + ri_hi);
                        h0 *= (u00 >= dropout_p) ? inv_keep : 0.0f;
                        h1 *= (u01 >= dropout_p) ? inv_keep : 0.0f;
                        h2 *= (u10 >= dropout_p) ? inv_keep : 0.0f;
                        h3 *= (u11 >= dropout_p) ? inv_keep : 0.0f;
#endif
                        rAR(0, mm, mn) = h0;
                        rAR(1, mm, mn) = h1;
                        rAR(2, mm, mn) = h2;
                        rAR(3, mm, mn) = h3;
                    }
#endif
                }
            }

            // Transpose AR (or gated AR) and write to sAR_pipe.
            //
            // fp16 path uses the canonical CuTe transpose chain:
            //   blocken_C → inplace_transpose (movmatrix) → construct_B → copy
            //
            // bf16 path uses scalar transposed stores. Counterintuitively, the
            // transpose chain is ~4% SLOWER for bf16 even though it issues
            // fewer smem stores. Why: bf16 requires a float→bf16 conversion in
            // registers first, AND inplace_transpose adds a movmatrix per 8x8
            // sub-tile. The bf16 dB kernel's critical path is LDSM_T on sdCt
            // and the consumer MMA — the producer r2s has slack, so reducing
            // store count doesn't speed anything up while the conversion +
            // movmatrix overhead does. The scalar loop is also auto-unrolled
            // and likely vectorized by the compiler, so smem store rate is
            // close to peak for this access pattern. **Do not revert to the
            // transpose chain without first profiling and confirming the
            // bottleneck has shifted to producer r2s.**
#if PRISM_DTYPE == 0
            {
                auto blocked = blocken_operand_C<mma_atom_ar>(rAR);
                auto transposed = inplace_transpose(blocked);
                auto rB = construct_operand_B<mma_atom_cons>(transposed);
                auto sAR_sub = local_tile(
                    sAR_pipe(_, _, ar_pipe_write),
                    make_tile(Int<rs>{}, Int<WARP_M>{}),
                    make_coord(kb, warp_idx));
                auto thr_sB = r2s_thr_mma.partition_B(sAR_sub);
                copy(rB, thr_sB);
            }
#else
            // bf16 scalar transposed stores with float→bf16 conversion.
            // sAR_sub is (rs, WARP_M); rAR is (M, K) per warp; we write
            // rAR(m, k) → sAR_sub(k, m).
            {
                auto sAR_sub = local_tile(
                    sAR_pipe(_, _, ar_pipe_write),
                    make_tile(Int<rs>{}, Int<WARP_M>{}),
                    make_coord(kb, warp_idx));
                auto cAR_id = make_identity_tensor(make_shape(Int<WARP_M>{}, Int<rs>{}));
                auto tCcAR = thr_mma_ar.partition_C(cAR_id);
                constexpr int frag_sz_f = decltype(size(rAR))::value;
                #pragma unroll
                for (int i = 0; i < frag_sz_f; ++i) {
                    auto coord = tCcAR(i);
                    int m_pos = get<0>(coord);
                    int k_pos = get<1>(coord);
                    if (m_pos < WARP_M && k_pos < rs) {
                        sAR_sub(k_pos, m_pos) = prism_cute(float(rAR(i)));
                    }
                }
            }
#endif
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));
        }

        // Signal data ready — non-blocking (producer overlap only)
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(BAR_READY_BASE), "n"(n_total_threads));
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read; if (smem_pipe_read == bP_a) smem_pipe_read = 0;
    }
}

// =============================================================================
// Consumer: multi-warp MMA for dB, loads from sAR_pipe with LDSM
// =============================================================================
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemDCt, class SmemARpipe,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dB_consumer(
    const prism_cute* __restrict__ dC_ptr, int ldDC,
    prism_cute* __restrict__ dB_ptr, int ldDB,
    SmemDCt& sdCt, SmemARpipe& sAR_pipe,
    int M, int N, int K, int g, int k_start,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;
    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_ar;
    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    int lane_idx = thread_idx % 32;
    int warp_idx = thread_idx / 32;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;

    // Multi-warp MMA: dB(gs, BLK_K) += dCt(gs, BLK_M) @ ARt(BLK_K, BLK_M)^T
    using mma_atom_t = prism_dA_atom_f32<BLK_M>;
    auto mma = make_tiled_mma(mma_atom_t{}, WarpLayoutConsumer{},
                               Tile<Int<gs>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(lane_idx + warp_idx * 32);

    auto dB_shape = make_shape(Int<gs>{}, Int<BLK_K>{});
    auto rDB = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<prism_cute*>(nullptr),
                    make_layout(dB_shape, LayoutRight{}))));
    clear(rDB);

    // dCt A-operand: LDSM_T
    constexpr int a_u16 = (gs * BLK_M) / 32;
    using s2r_dCt = std::conditional_t<(a_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, prism_cute>,
        std::conditional_t<(a_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, prism_cute>,
            Copy_Atom<SM75_U16x2_LDSM_T, prism_cute>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dCt{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx + warp_idx * 32);
    auto tXsdCt = s2r_a_thr.partition_S(sdCt);
    auto rDCt = thr_mma.make_fragment_A(thr_mma.partition_A(sdCt(_,_,_0{})));
    auto tXrdCt = s2r_a_thr.retile_D(rDCt);

    // B-operand from sAR_pipe: (BLK_K, BLK_M, bP_ar) with smem_m swizzle
    using s2r_atom_b = Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>;
    auto s2r_b = make_tiled_copy_B(s2r_atom_b{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(lane_idx + warp_idx * 32);
    auto tXsARt = s2r_b_thr.partition_S(sAR_pipe);
    auto rARt = thr_mma.make_fragment_B(thr_mma.partition_B(sAR_pipe(_,_,_0{})));
    auto tXrARt = s2r_b_thr.retile_D(rARt);

    // cp.async for dCt
    constexpr int dc_threads_dim0 = gs / 8;
    constexpr int dc_threads_dim1 = n_consumer_threads / dc_threads_dim0;
    constexpr int dc_values_dim1 = BLK_M / dc_threads_dim1;
    auto copy_dC = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, prism_cute>{},
        make_layout(make_shape(Int<dc_threads_dim0>{}, Int<dc_threads_dim1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<dc_values_dim1>{})));
    auto thr_copy_dC = copy_dC.get_slice(thread_idx);
    auto tCsdCt_cp = thr_copy_dC.partition_D(sdCt);

    auto load_dCt_async = [&](int m_start, int pipe) {
        auto gdC = make_tensor(
            make_gmem_ptr(dC_ptr + m_start * ldDC + g * gs),
            make_layout(make_shape(Int<gs>{}, Int<BLK_M>{}),
                        make_stride(Int<1>{}, ldDC)));
        copy(copy_dC, thr_copy_dC.partition_S(gdC), tCsdCt_cp(_,_,_,pipe));
    };

    // Initial arrive: signal producer that buffer is free
    asm volatile("bar.arrive %0, %1;\n"
        : : "r"(BAR_CONSUMED_BASE), "n"(n_total_threads));

    int dc_pipe_write = bP_dc - 1, dc_pipe_read = 0;
    int m_tile_next = 0, m_tiles_remaining = n_m_tiles;

    // Prefill dCt pipeline
    for (int p = 0; p < bP_dc - 1 && m_tiles_remaining > 0; ++p) {
        load_dCt_async(m_tile_next * BLK_M, p);
        cp_async_fence(); --m_tiles_remaining; ++m_tile_next;
    }

    for (int mt = 0; mt < n_m_tiles; ++mt) {
        if (m_tiles_remaining > 0) {
            load_dCt_async(m_tile_next * BLK_M, dc_pipe_write);
            --m_tiles_remaining; ++m_tile_next;
        }
        cp_async_fence(); cp_async_wait<bP_dc - 1>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

        // Wait for AR data from producer
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(BAR_READY_BASE), "n"(n_total_threads));

        // LDSM load dCt and ARt
        copy(s2r_dCt{}, tXsdCt(_, _, _, dc_pipe_read), tXrdCt);
        copy(s2r_atom_b{}, tXsARt(_, _, _, 0), tXrARt);
        gemm(mma, rDCt, rARt, rDB);

        if (mt < n_m_tiles - 1) {
            // Signal consumed — sync with producer
            asm volatile("bar.sync %0, %1;\n"
                : : "r"(BAR_CONSUMED_BASE), "n"(n_total_threads));
        } else {
            // Last iteration — just arrive, producer won't sync again
            asm volatile("bar.arrive %0, %1;\n"
                : : "r"(BAR_CONSUMED_BASE), "n"(n_total_threads));
        }
        dc_pipe_write = dc_pipe_read;
        ++dc_pipe_read; if (dc_pipe_read == bP_dc) dc_pipe_read = 0;
    }

    auto tMcDB = thr_mma.partition_C(make_identity_tensor(dB_shape));
    for (int i = 0; i < size(rDB); ++i) {
        auto coord = tMcDB(i);
        int n_idx = g * gs + get<0>(coord);
        int k_idx = k_start + get<1>(coord);
        if (n_idx < N && k_idx < K)
            *(dB_ptr + n_idx * ldDB + k_idx) = prism_cute(rDB(i));
    }
}


template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(
    (size(typename cute::BwdDBParams::warp_layout_ar{}) +
     size(typename cute::BwdDBParams::warp_layout_arb{})) * 32)
dB_pc_kernel(
    const prism_cute* __restrict__ dC_ptr, int ldDC,
    const prism_cute* __restrict__ A_ptr,  int ldA,
    const prism_cute* __restrict__ R_ptr,  int ldR,
    const prism_cute* __restrict__ bias_ptr,
    prism_cute* __restrict__ dB_ptr,       int ldDB,
    int M, int N, int K,
    int n_groups, int n_blocks,
    int strideA_per_group,
    int64_t const* dropout_seeds, float dropout_p, float inv_keep)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;
    using wlp_t = typename cute::BwdDBParams::warp_layout_ar;
    using wlc_t = typename cute::BwdDBParams::warp_layout_arb;
    constexpr int n_consumer_threads = size(wlc_t{}) * 32;

    int k_tile = blockIdx.x;
    int g = blockIdx.y;
    int k_start = k_tile * BLK_K;

    auto smem_k = get_smem_atom(Int<BLK_K>{});

    auto sA_layout = tile_to_shape(smem_k, make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_a>{}));
    auto sR_layout = tile_to_shape(smem_k, make_shape(Int<rs>{}, Int<BLK_K>{}));
    // sAR_pipe: (BLK_K, BLK_M, bP_ar) — no swizzle.
    // Atom width = WARP_M (not BLK_M) so per-warp sub-views are atom-aligned
    // for both producer R2S and consumer LDSM.
    constexpr int n_prod_warps = size(wlp_t{});
    constexpr int WARP_M_k = BLK_M / n_prod_warps;
    auto sAR_pipe_layout = tile_to_shape(get_smem_atom<false>(Int<WARP_M_k>{}),
        make_shape(Int<BLK_K>{}, Int<BLK_M>{}, Int<bP_ar>{}));
    // sdCt: column-major layout for transposed dC. Atom width = gs.
    auto sdCt_atom = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<gs>{}, _8{}),
                    make_stride(Int<1>{}, Int<gs>{})));
    auto sdCt_layout = tile_to_shape(sdCt_atom, make_shape(Int<gs>{}, Int<BLK_M>{}, Int<bP_dc>{}));

    extern __shared__ prism_cute smem_raw[];
    prism_cute* p = smem_raw;
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR       = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sAR_pipe = make_tensor(make_smem_ptr(p), sAR_pipe_layout); p += cosize(sAR_pipe_layout);
    auto sdCt     = make_tensor(make_smem_ptr(p), sdCt_layout);

    constexpr int n_total = size(wlp_t{}) * 32 + n_consumer_threads;
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_total) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c) = load_or_zero(R_ptr, (g * rs + r) * K + k_start + c, k_start + c < K);
    }
    __syncthreads();

    // Per-group A pointer for the input_shuffle path; no-op when stride==0.
    const prism_cute* A_for_g = A_ptr + (long long)g * strideA_per_group;

    if (threadIdx.x >= n_consumer_threads) {
        dB_producer<BLK_M, BLK_K, gs, rs, GATED>(
            A_for_g, ldA, bias_ptr, sA, sR, sAR_pipe,
            M, K, g, k_start,
            threadIdx.x - n_consumer_threads, wlp_t{}, wlc_t{},
            dropout_seeds, dropout_p, inv_keep);
    } else {
        dB_consumer<BLK_M, BLK_K, gs, rs, GATED>(
            dC_ptr, ldDC, dB_ptr, ldDB,
            sdCt, sAR_pipe,
            M, N, K, g, k_start,
            threadIdx.x, wlp_t{}, wlc_t{});
    }
}


void prism_backward_dB_launch(
    int m, int n, int k,
    prism_native const* dC, int ldDC,
    prism_native const* A,  int ldA,
    prism_native const* R,  int ldR,
    prism_native const* bias,
    prism_native* dB, int ldDB,
    int strideA_per_group,
    int64_t const* dropout_seeds, float dropout_p, float inv_keep,
    cudaStream_t stream)
{
    constexpr int gs = CurrKernelParams::group_size;
    constexpr int rs = CurrKernelParams::reconn_sz;
    constexpr int BLK_M = cute::BwdDBParams::bM;
    constexpr int BLK_K = cute::BwdDBParams::bK;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;
    using wlp_t = typename cute::BwdDBParams::warp_layout_ar;
    using wlc_t = typename cute::BwdDBParams::warp_layout_arb;
    constexpr int n_threads = (size(wlp_t{}) + size(wlc_t{})) * 32;

    int n_groups = n / gs;
    int n_k_tiles = (k + BLK_K - 1) / BLK_K;
    dim3 grid(n_k_tiles, n_groups);
    dim3 block(n_threads);

    auto smem_k = get_smem_atom(cute::Int<BLK_K>{});
    int smem = (cosize(tile_to_shape(smem_k, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{}, cute::Int<bP_a>{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{})))
                + BLK_M * BLK_K * bP_ar      // sAR_pipe (replaces sAR_temp + sARt)
                + cosize(tile_to_shape(
                    composition(Swizzle<3,3,3>{},
                                make_layout(make_shape(cute::Int<gs>{}, _8{}),
                                            make_stride(cute::Int<1>{}, cute::Int<gs>{}))),
                    make_shape(cute::Int<gs>{}, cute::Int<BLK_M>{}, cute::Int<bP_dc>{}))))
               * sizeof(prism_cute) + 256;

#if PRISM_GATED
    auto kernel = dB_pc_kernel<BLK_M, BLK_K, gs, rs, true>;
#else
    auto kernel = dB_pc_kernel<BLK_M, BLK_K, gs, rs, false>;
#endif
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<prism_cute const*>(dC), ldDC,
        reinterpret_cast<prism_cute const*>(A), ldA,
        reinterpret_cast<prism_cute const*>(R), ldR,
        reinterpret_cast<prism_cute const*>(bias),
        reinterpret_cast<prism_cute*>(dB), ldDB,
        m, n, k, n_groups, k / rs,
        strideA_per_group,
        dropout_seeds, dropout_p, inv_keep);
}
