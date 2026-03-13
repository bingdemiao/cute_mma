#include "cute_oft_backward_db.hpp"
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

    // -- Swizzle-preserving sub-views via logical_divide --
    // Use flat WARP_M tiler for contiguous warp regions (unlike forward kernel's
    // interleaved (n_warps, 8) pattern, backward needs contiguous rows for writeback)

    // sA: (BLK_M, BLK_K, bP_a) -> warp sub-view (WARP_M, rs, n_blocks_k, bP_a)
    // logical_divide produces (inner, outer) for each mode
    Tensor sA_warp = logical_divide(sA,
        make_tile(make_layout(Int<WARP_M>{}), make_layout(Int<rs>{}))
    )(
        make_coord(_, warp_idx),     // keep WARP_M (inner), select warp (outer)
        make_coord(_, _), _          // keep (rs, n_blocks_k), pipe
    ); // (WARP_M, rs, n_blocks_k, bP_a)

    // sR: (rs, BLK_K) -> (rs, rs, n_blocks_k)
    Tensor sR_divided = logical_divide(sR,
        make_tile(_, make_layout(Int<rs>{}))
    )(_, make_coord(_, _)); // (rs, rs, n_blocks_k)

    // sAR_temp warp sub-view via logical_divide
    Tensor sAR_temp_warp = logical_divide(sAR_temp,
        make_tile(make_layout(Int<WARP_M>{}), _)
    )(make_coord(_, warp_idx), _); // (WARP_M, rs)

    auto tCsAR = thr_mma_ar.partition_C(sAR_temp_warp);
    auto rAR = thr_mma_ar.make_fragment_C(tCsAR);

    // r2s copy for MMA output -> sAR_temp
    using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s_ar = make_tiled_copy_C(r2s_atom_AR{}, mma_ar);
    auto r2s_ar_thr = r2s_ar.get_slice(lane_idx);
    auto tXrAR = r2s_ar_thr.retile_S(rAR);
    auto tXsAR_w = r2s_ar_thr.partition_D(sAR_temp_warp);

    // -- LDSM atoms for s2r (swizzled smem) --
    // Select atom size based on per-thread register count: u32_regs = (M * K) / 64
    constexpr int a_u32_s1 = (WARP_M * rs) / 64;
    using s2r_atom_A = std::conditional_t<(a_u32_s1 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    constexpr int b_u32_s1 = (rs * rs) / 64;
    using s2r_atom_R = std::conditional_t<(b_u32_s1 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

    // Partition s2r once before loop
    auto s2r_a = make_tiled_copy_A(s2r_atom_A{}, mma_ar);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx);
    auto tXsA = s2r_a_thr.partition_S(sA_warp);   // (CPY, CPY_M, CPY_K, n_blocks_k, bP_a)
    auto rA = thr_mma_ar.make_fragment_A(thr_mma_ar.partition_A(sA_warp(_,_,_0{},_0{})));
    auto tXrA = s2r_a_thr.retile_D(rA);

    auto s2r_r = make_tiled_copy_B(s2r_atom_R{}, mma_ar);
    auto s2r_r_thr = s2r_r.get_slice(lane_idx);
    auto tXsR = s2r_r_thr.partition_S(sR_divided); // (CPY, CPY_N, CPY_K, n_blocks_k)
    auto rR = thr_mma_ar.make_fragment_B(thr_mma_ar.partition_B(sR_divided(_,_,_0{})));
    auto tXrR = s2r_r_thr.retile_D(rR);

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

            // LDSM s2r loads indexed by (kb, pipe)
            copy(s2r_atom_A{}, tXsA(_,_,_,kb,smem_pipe_read), tXrA);
            copy(s2r_atom_R{}, tXsR(_,_,_,kb), tXrR);

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
    (size(typename cute::BwdDBParams::warp_layout_ar{}) +
     size(typename cute::BwdDBParams::warp_layout_arb{})) * 32)
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

    using warp_layout_producer_t = typename cute::BwdDBParams::warp_layout_ar;
    using warp_layout_consumer_t = typename cute::BwdDBParams::warp_layout_arb;
    warp_layout_producer_t warp_layout_producer;
    warp_layout_consumer_t warp_layout_consumer;

    constexpr int n_producer_threads = size(warp_layout_producer_t{}) * 32;
    constexpr int n_consumer_threads = size(warp_layout_consumer_t{}) * 32;
    constexpr int n_total_threads = n_producer_threads + n_consumer_threads;

    int k_tile = blockIdx.x;
    int g = blockIdx.y;
    int k_start = k_tile * BLK_K;

    // -- Shared memory layouts --
    // sA: SWIZZLED (BLK_M, BLK_K, bP_a) — enables LDSM_N for s2r
    auto smem_k = get_smem_atom(Int<BLK_K>{});
    auto sA_layout = tile_to_shape(smem_k,
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_a>{}));

    // sR: SWIZZLED (rs, BLK_K) — enables LDSM_N for s2r
    auto sR_layout = tile_to_shape(smem_k,
        make_shape(Int<rs>{}, Int<BLK_K>{}));

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
    using wlp_dB = typename cute::BwdDBParams::warp_layout_ar;
    using wlc_dB = typename cute::BwdDBParams::warp_layout_arb;
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
    auto smem_k = get_smem_atom(cute::Int<BLK_K_dB>{});
    int smem_sA = cosize(tile_to_shape(smem_k,
        make_shape(cute::Int<BLK_M_dB>{}, cute::Int<BLK_K_dB>{}, cute::Int<bP_a>{})));
    int smem_sR = cosize(tile_to_shape(smem_k,
        make_shape(cute::Int<rs>{}, cute::Int<BLK_K_dB>{})));
    int smem_sAR_temp = BLK_M_dB * rs;  // non-swizzled temp buffer
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
