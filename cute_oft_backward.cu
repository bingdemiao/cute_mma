#include "cute_oft_backward.hpp"
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
// OFT Backward — Optimized with cp.async, pipeline, MMA, producer-consumer
// =============================================================================


// =============================================================================
// Kernel 2: dB — Producer-Consumer
// =============================================================================

// Producer: loads A, computes AR = A @_bd R^T via MMA per rs-block,
// transposes AR into sARt pipeline buffer for the consumer.
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemA, class SmemR, class SmemARtemp, class SmemARt,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dB_producer(
    const half_t* __restrict__ A_ptr, int ldA,
    SmemA& sA, SmemR& sR, SmemARtemp& sAR_temp, SmemARt& sARt,
    int M, int K, int g, int k_start,
    int thread_idx,  // local thread index within producer warps
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int rs = RECONN_SZ;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;

    // Offset barrier IDs by 1 to avoid conflict with __syncthreads() barrier 0
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

    // -- MMA setup for AR: (WARP_M, rs) = (WARP_M, rs) @ (rs, rs)^T --
    using mma_atom_ar = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;

    auto mma_ar = make_tiled_mma(
        mma_atom_ar{},
        Layout<Shape<_1, _1>>{},
        Tile<Int<WARP_M>, Int<rs>>{}
    );
    auto thr_mma_ar = mma_ar.get_slice(lane_idx);

    // Create a representative sub-view for fragment sizing
    // sAR_temp is (BLK_M, rs) LayoutRight — take warp slice
    auto sAR_temp_warp = make_tensor(
        &sAR_temp(warp_idx * WARP_M, 0),
        make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
    );
    auto tCsAR = thr_mma_ar.partition_C(sAR_temp_warp);
    auto rAR = thr_mma_ar.make_fragment_C(tCsAR);

    // r2s copy for MMA output -> sAR_temp
    using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s_ar = make_tiled_copy_C(r2s_atom_AR{}, mma_ar);
    auto r2s_ar_thr = r2s_ar.get_slice(lane_idx);
    auto tXrAR = r2s_ar_thr.retile_S(rAR);
    auto tXsAR_w = r2s_ar_thr.partition_D(sAR_temp_warp);

    // s2r copy using UniversalCopy (non-swizzled smem, no LDSM_N)
    using s2r_atom_u16 = Copy_Atom<UniversalCopy<uint16_t>, half_t>;
    using s2r_atom_u32 = Copy_Atom<UniversalCopy<uint32_t>, half_t>;

    // -- Pipeline state --
    int smem_pipe_write = bP_a - 1;
    int smem_pipe_read = 0;
    int ar_pipe_write = 0;
    int m_tile_next = 0;
    int m_tiles_remaining = n_m_tiles;

    // -- Prefill A pipeline --
    for (int p = 0; p < bP_a - 1 && m_tiles_remaining > 0; ++p) {
        int m_s = m_tile_next * BLK_M;
        for (int i = thread_idx; i < BLK_M * BLK_K; i += n_producer_threads) {
            int r = i / BLK_K, c = i % BLK_K;
            sA(r, c, p) = load_or_zero(A_ptr, (m_s + r) * ldA + k_start + c, m_s + r < M);
        }
        cp_async_fence();
        --m_tiles_remaining;
        ++m_tile_next;
    }

    // -- Main M-tile loop --
    for (int mt = 0; mt < n_m_tiles; ++mt) {
        // Issue load for next M-tile
        if (m_tiles_remaining > 0) {
            int m_s_next = m_tile_next * BLK_M;
            for (int i = thread_idx; i < BLK_M * BLK_K; i += n_producer_threads) {
                int r = i / BLK_K, c = i % BLK_K;
                sA(r, c, smem_pipe_write) = load_or_zero(A_ptr, (m_s_next + r) * ldA + k_start + c, m_s_next + r < M);
            }
            --m_tiles_remaining;
            ++m_tile_next;
        }
        cp_async_fence();
        cp_async_wait<bP_a - 1>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        // Wait for consumer to finish with this AR pipe slot (once per M-tile)
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(ar_pipe_write + BAR_CONSUMED_BASE), "n"(n_total_threads));

        // For each rs-block in BLK_K, compute AR via MMA
        for (int kb = 0; kb < n_blocks_k; ++kb) {
            int k_offset = kb * rs;

            // Create sub-views for this (warp, rs-block, pipe)
            auto sA_sub = make_tensor(
                &sA(warp_idx * WARP_M, k_offset, smem_pipe_read),
                make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<BLK_K>{}, Int<1>{}))
            );
            auto sR_sub = make_tensor(
                &sR(0, k_offset),
                make_layout(make_shape(Int<rs>{}, Int<rs>{}), make_stride(Int<BLK_K>{}, Int<1>{}))
            );

            // Partition for MMA
            auto tCsA_sub = thr_mma_ar.partition_A(sA_sub);
            auto tCsR_sub = thr_mma_ar.partition_B(sR_sub);
            auto rA = thr_mma_ar.make_fragment_A(tCsA_sub);
            auto rR = thr_mma_ar.make_fragment_B(tCsR_sub);

            // s2r copies (UniversalCopy for non-swizzled smem)
            auto s2r_a = make_tiled_copy_A(s2r_atom_u32{}, mma_ar);
            auto s2r_a_thr = s2r_a.get_slice(lane_idx);
            copy(s2r_atom_u32{}, s2r_a_thr.partition_S(sA_sub), s2r_a_thr.retile_D(rA));

            auto s2r_r = make_tiled_copy_B(s2r_atom_u16{}, mma_ar);
            auto s2r_r_thr = s2r_r.get_slice(lane_idx);
            copy(s2r_atom_u16{}, s2r_r_thr.partition_S(sR_sub), s2r_r_thr.retile_D(rR));

            // MMA: AR(WARP_M, rs) = A(WARP_M, rs) @ R(rs, rs)^T
            clear(rAR);
            gemm(mma_ar, rA, rR, rAR);

            // Write MMA output to sAR_temp via r2s
            copy(r2s_atom_AR{}, tXrAR, tXsAR_w);
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

            if constexpr (GATED) {
                // Apply SiLU gating in smem: H(m,k) = A(m,k) * SiLU(AR(m,k))
                for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                    int mi = idx / rs, ri = idx % rs;
                    int gmi = warp_idx * WARP_M + mi;
                    float ar = float(sAR_temp(gmi, ri));
                    float a = float(sA(gmi, k_offset + ri, smem_pipe_read));
                    float silu_ar = ar / (1.0f + __expf(-ar));
                    sAR_temp(gmi, ri) = half_t(a * silu_ar);
                }
                asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));
            }

            // Transpose sAR_temp warp region -> sARt slice
            for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                int mi = idx / rs, ri = idx % rs;
                sARt(k_offset + ri, warp_idx * WARP_M + mi, ar_pipe_write) =
                    sAR_temp(warp_idx * WARP_M + mi, ri);
            }
        }

        // All rs-blocks done — signal consumer that AR is ready
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(ar_pipe_write + BAR_READY_BASE), "n"(n_total_threads));
        ++ar_pipe_write;
        if (ar_pipe_write == bP_ar) ar_pipe_write = 0;

        // Advance A pipeline
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        if (smem_pipe_read == bP_a) smem_pipe_read = 0;
    }
}


// Consumer: loads dC (transposed), waits for AR, computes dB via MMA
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemDCt, class SmemARt,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dB_consumer(
    const half_t* __restrict__ dC_ptr, int ldDC,
    half_t* __restrict__ dB_ptr, int ldDB,
    SmemDCt& sdCt, SmemARt& sARt,
    int M, int N, int K, int g, int k_start,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;

    // Offset barrier IDs by 1 to avoid conflict with __syncthreads() barrier 0
    constexpr int BAR_READY_BASE = 1;
    constexpr int BAR_CONSUMED_BASE = BAR_READY_BASE + bP_ar;

    constexpr uint32_t n_producer_threads = size(WarpLayoutProducer{}) * 32;
    constexpr uint32_t n_consumer_threads = size(WarpLayoutConsumer{}) * 32;
    constexpr uint32_t n_total_threads = n_producer_threads + n_consumer_threads;

    int lane_idx = thread_idx % 32;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;

    // -- MMA setup: dB(gs, BLK_K) = dCt(gs, BLK_M) @ ARt(BLK_K, BLK_M)^T --
    // TN: A-op(gs, BLK_M) BLK_M-contiguous, B-op(BLK_K, BLK_M) BLK_M-contiguous
    // Use F32 accumulator to avoid precision loss over many M-tile iterations
    using mma_atom_t = std::conditional_t<(BLK_M < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;

    auto mma = make_tiled_mma(
        mma_atom_t{},
        Layout<Shape<_1, _1>>{},
        Tile<Int<gs>, Int<BLK_K>>{}
    );
    auto thr_mma = mma.get_slice(lane_idx);

    auto tMsdCt = thr_mma.partition_A(sdCt);
    auto tMsARt = thr_mma.partition_B(sARt);

    auto rDCt = thr_mma.make_fragment_A(tMsdCt(_, _, _, _0{}));
    auto rARt = thr_mma.make_fragment_B(tMsARt(_, _, _, _0{}));

    auto dB_shape = make_shape(Int<gs>{}, Int<BLK_K>{});
    auto sdB_layout = make_layout(dB_shape, LayoutRight{});
    auto tMcDB_dummy = thr_mma.partition_C(make_tensor(
        static_cast<half_t*>(nullptr), sdB_layout));
    // Float32 accumulator for precision across M-tile iterations
    auto rDB = thr_mma.make_fragment_C(tMcDB_dummy);
    clear(rDB);

    using s2r_atom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    auto s2r_a = make_tiled_copy_A(s2r_atom{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx);
    auto tXsdCt = s2r_a_thr.partition_S(sdCt);
    auto tXrdCt = s2r_a_thr.retile_D(rDCt);

    auto s2r_b = make_tiled_copy_B(s2r_atom{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(lane_idx);
    auto tXsARt = s2r_b_thr.partition_S(sARt);
    auto tXrARt = s2r_b_thr.retile_D(rARt);

    // -- Signal producer that all AR pipe slots are initially available --
    for (int bid = BAR_CONSUMED_BASE; bid < BAR_CONSUMED_BASE + bP_ar; ++bid) {
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(bid), "n"(n_total_threads));
    }

    // -- Pipeline state --
    int dc_pipe_write = bP_dc - 1;
    int dc_pipe_read = 0;
    int ar_pipe_read = 0;
    int m_tile_next = 0;
    int m_tiles_remaining = n_m_tiles;

    // -- Prefill dC pipeline (scalar with transpose) --
    for (int p = 0; p < bP_dc - 1 && m_tiles_remaining > 0; ++p) {
        int m_s = m_tile_next * BLK_M;
        for (int i = thread_idx; i < BLK_M * gs; i += n_consumer_threads) {
            int m_local = i / gs;
            int gs_col = i % gs;
            int m_idx = m_s + m_local;
            sdCt(gs_col, m_local, p) = load_or_zero(dC_ptr, m_idx * ldDC + g * gs + gs_col, m_idx < M);
        }
        cp_async_fence();
        --m_tiles_remaining;
        ++m_tile_next;
    }

    // -- Main M-tile loop --
    for (int mt = 0; mt < n_m_tiles; ++mt) {
        if (m_tiles_remaining > 0) {
            int m_s_next = m_tile_next * BLK_M;
            for (int i = thread_idx; i < BLK_M * gs; i += n_consumer_threads) {
                int m_local = i / gs;
                int gs_col = i % gs;
                int m_idx = m_s_next + m_local;
                sdCt(gs_col, m_local, dc_pipe_write) = load_or_zero(dC_ptr, m_idx * ldDC + g * gs + gs_col, m_idx < M);
            }
            --m_tiles_remaining;
            ++m_tile_next;
        }
        cp_async_fence();
        cp_async_wait<bP_dc - 1>();
        asm volatile("bar.sync 15, %0;\n" : : "n"(n_consumer_threads));

        // Wait for producer's AR data
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(ar_pipe_read + BAR_READY_BASE), "n"(n_total_threads));

        // LDSM load and MMA
        copy(s2r_atom{}, tXsdCt(_, _, _, dc_pipe_read), tXrdCt);
        copy(s2r_atom{}, tXsARt(_, _, _, ar_pipe_read), tXrARt);
        gemm(mma, rDCt, rARt, rDB);

        // Signal producer that AR buffer is consumed
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(ar_pipe_read + BAR_CONSUMED_BASE), "n"(n_total_threads));

        ++ar_pipe_read;
        if (ar_pipe_read == bP_ar) ar_pipe_read = 0;

        dc_pipe_write = dc_pipe_read;
        ++dc_pipe_read;
        if (dc_pipe_read == bP_dc) dc_pipe_read = 0;
    }

    // Write dB to global memory (convert float32 accumulator → half)
    auto tMcDB = thr_mma.partition_C(make_identity_tensor(dB_shape));
    for (int i = 0; i < size(rDB); ++i) {
        auto coord = tMcDB(i);
        int row = get<0>(coord);
        int col = get<1>(coord);
        int n_idx = g * gs + row;
        int k_idx = k_start + col;
        if (n_idx < N && k_idx < K)
            *(dB_ptr + n_idx * ldDB + k_idx) = half_t(rDB(i));
    }
}


template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(
    (size(typename cute::BwdDBParams::warp_layout_producer{}) +
     size(typename cute::BwdDBParams::warp_layout_consumer{})) * 32)
dB_pc_kernel(
    const half_t* __restrict__ dC_ptr, int ldDC,
    const half_t* __restrict__ A_ptr,  int ldA,
    const half_t* __restrict__ R_ptr,  int ldR,
    half_t* __restrict__ dB_ptr,       int ldDB,
    int M, int N, int K,
    int n_groups, int n_blocks)
{
    constexpr int gs = GROUP_SIZE;
    constexpr int rs = RECONN_SZ;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;

    using warp_layout_producer_t = typename cute::BwdDBParams::warp_layout_producer;
    using warp_layout_consumer_t = typename cute::BwdDBParams::warp_layout_consumer;
    warp_layout_producer_t warp_layout_producer;
    warp_layout_consumer_t warp_layout_consumer;

    constexpr int n_producer_threads = size(warp_layout_producer_t{}) * 32;
    constexpr int n_consumer_threads = size(warp_layout_consumer_t{}) * 32;
    constexpr int n_total_threads = n_producer_threads + n_consumer_threads;

    int k_tile = blockIdx.x;
    int g = blockIdx.y;
    int k_start = k_tile * BLK_K;

    // -- Shared memory layouts --
    // sA: NON-swizzled (BLK_M, BLK_K, bP_a) — pipe outermost so each slot is contiguous
    auto sA_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_a>{}),
        make_stride(Int<BLK_K>{}, Int<1>{}, Int<BLK_M * BLK_K>{}));

    // sR: NON-swizzled (rs, BLK_K) — persists
    auto sR_layout = make_layout(
        make_shape(Int<rs>{}, Int<BLK_K>{}), LayoutRight{});

    // sAR_temp: NON-swizzled (BLK_M, rs) — temp for r2s before transpose
    auto sAR_temp_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});

    // sARt: SWIZZLED (BLK_K, BLK_M, bP_ar) — consumer reads via LDSM_N
    auto smem_m = get_smem_atom(Int<BLK_M>{});
    auto sARt_layout = tile_to_shape(smem_m,
        make_shape(Int<BLK_K>{}, Int<BLK_M>{}, Int<bP_ar>{}));

    // sdCt: SWIZZLED (gs, BLK_M, bP_dc) — consumer reads via LDSM_N
    auto sdCt_layout = tile_to_shape(smem_m,
        make_shape(Int<gs>{}, Int<BLK_M>{}, Int<bP_dc>{}));

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR       = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sAR_temp = make_tensor(make_smem_ptr(p), sAR_temp_layout); p += cosize(sAR_temp_layout);
    auto sARt     = make_tensor(make_smem_ptr(p), sARt_layout);     p += cosize(sARt_layout);
    auto sdCt     = make_tensor(make_smem_ptr(p), sdCt_layout);

    // -- Load R to smem (all threads) --
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_total_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c) = load_or_zero(R_ptr, (g * rs + r) * ldR + k_start + c, k_start + c < K);
    }
    __syncthreads();

    // -- Dispatch producer vs consumer --
    if (threadIdx.x >= n_consumer_threads) {
        dB_producer<BLK_M, BLK_K, gs, rs, GATED>(
            A_ptr, ldA,
            sA, sR, sAR_temp, sARt,
            M, K, g, k_start,
            threadIdx.x - n_consumer_threads,
            warp_layout_producer, warp_layout_consumer);
    } else {
        dB_consumer<BLK_M, BLK_K, gs, rs, GATED>(
            dC_ptr, ldDC, dB_ptr, ldDB,
            sdCt, sARt,
            M, N, K, g, k_start,
            threadIdx.x,
            warp_layout_producer, warp_layout_consumer);
    }
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

    // Representative sub-view for fragment sizing
    auto sdAR_warp = make_tensor(
        &sdAR(warp_idx * WARP_M, 0, 0),
        make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
    );
    auto tCsDAR = thr_mma.partition_C(sdAR_warp);
    auto rDAR = thr_mma.make_fragment_C(tCsDAR);

    using r2s_atom = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s = make_tiled_copy_C(r2s_atom{}, mma);
    auto r2s_thr = r2s.get_slice(lane_idx);
    auto tXrDAR = r2s_thr.retile_S(rDAR);

    using s2r_atom_u32 = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    using s2r_atom_u16 = Copy_Atom<UniversalCopy<uint16_t>, half_t>;

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

        // Create sub-views for current pipe slot and warp
        auto sdC_sub = make_tensor(
            &sdC(warp_idx * WARP_M, 0, dc_b_pipe_read),
            make_layout(make_shape(Int<WARP_M>{}, Int<gs>{}), make_stride(Int<gs>{}, Int<1>{}))
        );
        auto sBt_sub = make_tensor(
            &sBt(0, 0, dc_b_pipe_read),
            make_layout(make_shape(Int<rs>{}, Int<gs>{}), make_stride(Int<gs>{}, Int<1>{}))
        );

        auto tCsdC = thr_mma.partition_A(sdC_sub);
        auto tCsBt = thr_mma.partition_B(sBt_sub);
        auto rDC = thr_mma.make_fragment_A(tCsdC);
        auto rBt = thr_mma.make_fragment_B(tCsBt);

        auto s2r_a = make_tiled_copy_A(s2r_atom_u32{}, mma);
        auto s2r_a_thr = s2r_a.get_slice(lane_idx);
        copy(s2r_atom_u32{}, s2r_a_thr.partition_S(sdC_sub), s2r_a_thr.retile_D(rDC));

        auto s2r_b = make_tiled_copy_B(s2r_atom_u16{}, mma);
        auto s2r_b_thr = s2r_b.get_slice(lane_idx);
        copy(s2r_atom_u16{}, s2r_b_thr.partition_S(sBt_sub), s2r_b_thr.retile_D(rBt));

        // MMA: dAR = dC @ B^T
        clear(rDAR);
        gemm(mma, rDC, rBt, rDAR);

        // Wait for consumer to release dAR buffer
        asm volatile("bar.sync %0, %1;\n"
            : : "r"(dar_pipe_write + BAR_CONSUMED_BASE), "n"(n_total_threads));

        // Write dAR to pipeline buffer (warp portion)
        auto sdAR_warp_p = make_tensor(
            &sdAR(warp_idx * WARP_M, 0, dar_pipe_write),
            make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
        );
        auto tXsDAR_p = r2s_thr.partition_D(sdAR_warp_p);
        copy(r2s_atom{}, tXrDAR, tXsDAR_p);
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

    // s2r copies: UniversalCopy (non-swizzled smem)
    using s2r_atom_u32 = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    using s2r_atom_u16 = Copy_Atom<UniversalCopy<uint16_t>, half_t>;

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

            auto sA_sub = make_tensor(
                &sA(warp_idx * WARP_M, 0),
                make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
            );
            auto sR_sub = make_tensor(
                &sR_buf(0, 0, r_pipe_read),
                make_layout(make_shape(Int<rs>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
            );

            auto tCsA_ar = thr_mma_ar.partition_A(sA_sub);
            auto tCsR_ar = thr_mma_ar.partition_B(sR_sub);
            auto rA_ar = thr_mma_ar.make_fragment_A(tCsA_ar);
            auto rR_ar = thr_mma_ar.make_fragment_B(tCsR_ar);

            auto s2r_a_ar = make_tiled_copy_A(s2r_atom_u32{}, mma_ar);
            auto s2r_a_ar_thr = s2r_a_ar.get_slice(lane_idx);
            copy(s2r_atom_u32{}, s2r_a_ar_thr.partition_S(sA_sub), s2r_a_ar_thr.retile_D(rA_ar));

            auto s2r_b_ar = make_tiled_copy_B(s2r_atom_u16{}, mma_ar);
            auto s2r_b_ar_thr = s2r_b_ar.get_slice(lane_idx);
            copy(s2r_atom_u16{}, s2r_b_ar_thr.partition_S(sR_sub), s2r_b_ar_thr.retile_D(rR_ar));

            // AR MMA output -> sAR_temp
            auto sAR_temp_warp = make_tensor(
                &sAR_temp(warp_idx * WARP_M, 0),
                make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
            );
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

            // Phase B: Element-wise SiLU + dS (scalar)
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
                auto sdAR_sub = make_tensor(
                    &sdAR(warp_idx * WARP_M, 0, dar_pipe_read),
                    make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
                );
                auto sRt_sub = make_tensor(
                    &sRt_buf(0, 0, r_pipe_read),
                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
                );

                auto tCsdAR_d = thr_mma_da.partition_A(sdAR_sub);
                auto tCsRt_d = thr_mma_da.partition_B(sRt_sub);
                auto rdAR_d = thr_mma_da.make_fragment_A(tCsdAR_d);
                auto rRt_d = thr_mma_da.make_fragment_B(tCsRt_d);

                auto s2r_dar = make_tiled_copy_A(s2r_atom_u32{}, mma_da);
                auto s2r_dar_thr = s2r_dar.get_slice(lane_idx);
                copy(s2r_atom_u32{}, s2r_dar_thr.partition_S(sdAR_sub), s2r_dar_thr.retile_D(rdAR_d));

                auto s2r_rt = make_tiled_copy_B(s2r_atom_u16{}, mma_da);
                auto s2r_rt_thr = s2r_rt.get_slice(lane_idx);
                copy(s2r_atom_u16{}, s2r_rt_thr.partition_S(sRt_sub), s2r_rt_thr.retile_D(rRt_d));

                gemm(mma_da, rdAR_d, rRt_d, rDA);
            }
        } else {
            // Non-gated: dA += dAR @ R via MMA
            // TN MMA with R^T as B-operand gives: dAR @ (R^T)^T = dAR @ R
            auto sdAR_sub = make_tensor(
                &sdAR(warp_idx * WARP_M, 0, dar_pipe_read),
                make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
            );
            auto sRt_sub = make_tensor(
                &sRt_buf(0, 0, r_pipe_read),
                make_layout(make_shape(Int<rs>{}, Int<rs>{}), make_stride(Int<rs>{}, Int<1>{}))
            );

            auto tCsdAR = thr_mma_da.partition_A(sdAR_sub);
            auto tCsRt = thr_mma_da.partition_B(sRt_sub);
            auto rdAR = thr_mma_da.make_fragment_A(tCsdAR);
            auto rRt = thr_mma_da.make_fragment_B(tCsRt);

            auto s2r_dar = make_tiled_copy_A(s2r_atom_u32{}, mma_da);
            auto s2r_dar_thr = s2r_dar.get_slice(lane_idx);
            copy(s2r_atom_u32{}, s2r_dar_thr.partition_S(sdAR_sub), s2r_dar_thr.retile_D(rdAR));

            auto s2r_rt = make_tiled_copy_B(s2r_atom_u16{}, mma_da);
            auto s2r_rt_thr = s2r_rt.get_slice(lane_idx);
            copy(s2r_atom_u16{}, s2r_rt_thr.partition_S(sRt_sub), s2r_rt_thr.retile_D(rRt));

            gemm(mma_da, rdAR, rRt, rDA);
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
    (size(typename cute::BwdDAdRParams::warp_layout_producer{}) +
     size(typename cute::BwdDAdRParams::warp_layout_consumer{})) * 32)
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

    using wlp_t = typename cute::BwdDAdRParams::warp_layout_producer;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_consumer;
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

    // -- Shared memory layouts --
    // Producer: sdC(BLK_M, gs, bP_dc_b) — pipe outermost
    auto sdC_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<gs>{}, Int<bP_dc_b>{}),
        make_stride(Int<gs>{}, Int<1>{}, Int<BLK_M * gs>{}));
    // Producer: sBt(rs, gs, bP_dc_b) — pipe outermost
    auto sBt_layout = make_layout(
        make_shape(Int<rs>{}, Int<gs>{}, Int<bP_dc_b>{}),
        make_stride(Int<gs>{}, Int<1>{}, Int<rs * gs>{}));
    // Pipeline: sdAR(BLK_M, rs, bP_dar) — pipe outermost
    auto sdAR_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<rs>{}, Int<bP_dar>{}),
        make_stride(Int<rs>{}, Int<1>{}, Int<BLK_M * rs>{}));
    // Consumer: sA(BLK_M, rs) non-swizzled
    auto sA_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});
    // Consumer: sR(rs, rs, bP_a_r) — pipe outermost (used only for gated AR recompute)
    auto sR_layout = make_layout(
        make_shape(Int<rs>{}, Int<rs>{}, Int<bP_a_r>{}),
        make_stride(Int<rs>{}, Int<1>{}, Int<rs * rs>{}));
    // Consumer: sRt(rs, rs, bP_a_r) — R transposed, for dA MMA
    auto sRt_layout = make_layout(
        make_shape(Int<rs>{}, Int<rs>{}, Int<bP_a_r>{}),
        make_stride(Int<rs>{}, Int<1>{}, Int<rs * rs>{}));
    // Consumer: sAR_temp(BLK_M, rs) — temp for gated AR recompute MMA output + SiLU
    auto sAR_temp_layout = make_layout(
        make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});

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
    using wlp_dAdR = typename cute::BwdDAdRParams::warp_layout_producer;
    using wlc_dAdR = typename cute::BwdDAdRParams::warp_layout_consumer;
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

    int smem = (BLK_M_dAdR * gs * bP_dc_b + rs * gs * bP_dc_b
                + BLK_M_dAdR * rs * bP_dar + BLK_M_dAdR * rs + rs * rs * bP_a_r
                + rs * rs * bP_a_r + BLK_M_dAdR * rs)
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


void oft_backward_dB_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* R,  int ldR,
    half* dB, int ldDB,
    bool gated,
    cudaStream_t stream)
{
    constexpr int gs = CurrKernelParams::group_size;
    constexpr int rs = CurrKernelParams::reconn_sz;

    constexpr int BLK_M_dB = cute::BwdDBParams::bM;
    constexpr int BLK_K_dB = cute::BwdDBParams::bK;
    constexpr int bP_a = cute::BwdDBParams::bP_a;
    constexpr int bP_ar = cute::BwdDBParams::bP_ar;
    constexpr int bP_dc = cute::BwdDBParams::bP_dc;
    using wlp_dB = typename cute::BwdDBParams::warp_layout_producer;
    using wlc_dB = typename cute::BwdDBParams::warp_layout_consumer;
    constexpr int n_threads_dB = (size(wlp_dB{}) + size(wlc_dB{})) * 32;

    int n_groups = n / gs;
    int n_blocks = k / rs;

    auto dC_h = reinterpret_cast<half_t const*>(dC);
    auto A_h  = reinterpret_cast<half_t const*>(A);
    auto R_h  = reinterpret_cast<half_t const*>(R);
    auto dB_h = reinterpret_cast<half_t*>(dB);

    int n_k_tiles = (k + BLK_K_dB - 1) / BLK_K_dB;
    dim3 grid(n_k_tiles, n_groups);
    dim3 block(n_threads_dB);

    auto smem_m = get_smem_atom(cute::Int<BLK_M_dB>{});
    int smem_sA = BLK_M_dB * BLK_K_dB * bP_a;
    int smem_sR = rs * BLK_K_dB;
    int smem_sAR_temp = BLK_M_dB * rs;
    int smem_sARt = cosize(tile_to_shape(smem_m,
        make_shape(cute::Int<BLK_K_dB>{}, cute::Int<BLK_M_dB>{}, cute::Int<bP_ar>{})));
    int smem_sdCt = cosize(tile_to_shape(smem_m,
        make_shape(cute::Int<gs>{}, cute::Int<BLK_M_dB>{}, cute::Int<bP_dc>{})));
    int smem = (smem_sA + smem_sR + smem_sAR_temp + smem_sARt + smem_sdCt)
               * sizeof(half_t) + 256;

    auto kernel = gated ? dB_pc_kernel<BLK_M_dB, BLK_K_dB, gs, rs, true>
                        : dB_pc_kernel<BLK_M_dB, BLK_K_dB, gs, rs, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        dC_h, ldDC, A_h, ldA, R_h, ldR, dB_h, ldDB,
        m, n, k, n_groups, n_blocks);
}
