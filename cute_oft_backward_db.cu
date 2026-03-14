#include "cute_oft_backward_db.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

__device__ __forceinline__
half_t load_or_zero(const half_t* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : half_t(0);
}

// =============================================================================
// Producer: computes AR, applies SiLU gating, transposes to sARt
// =============================================================================
template <int BLK_M, int BLK_K, int GROUP_SIZE, int RECONN_SZ, bool GATED,
          class SmemA, class SmemR, class SmemARtemp, class SmemARt,
          class WarpLayoutProducer, class WarpLayoutConsumer>
__device__ static inline
void dB_producer(
    const half_t* __restrict__ A_ptr, int ldA,
    SmemA& sA, SmemR& sR, SmemARtemp& sAR_temp, SmemARt& sARt,
    int M, int K, int g, int k_start,
    int thread_idx,
    WarpLayoutProducer, WarpLayoutConsumer)
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

    using mma_atom_ar = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;
    auto mma_ar = make_tiled_mma(mma_atom_ar{}, Layout<Shape<_1, _1>>{},
                                  Tile<Int<WARP_M>, Int<rs>>{});
    auto thr_mma_ar = mma_ar.get_slice(lane_idx);

    Tensor sA_warp = logical_divide(sA,
        make_tile(make_layout(Int<WARP_M>{}), make_layout(Int<rs>{}))
    )(make_coord(_, warp_idx), make_coord(_, _), _);
    Tensor sR_divided = logical_divide(sR,
        make_tile(_, make_layout(Int<rs>{})))(_, make_coord(_, _));
    Tensor sAR_temp_warp = logical_divide(sAR_temp,
        make_tile(make_layout(Int<WARP_M>{}), _))(make_coord(_, warp_idx), _);

    auto tCsAR = thr_mma_ar.partition_C(sAR_temp_warp);
    auto rAR = thr_mma_ar.make_fragment_C(tCsAR);
    using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
    auto r2s_ar = make_tiled_copy_C(r2s_atom_AR{}, mma_ar);
    auto r2s_ar_thr = r2s_ar.get_slice(lane_idx);
    auto tXrAR = r2s_ar_thr.retile_S(rAR);
    auto tXsAR_w = r2s_ar_thr.partition_D(sAR_temp_warp);

    constexpr int a_u32 = (WARP_M * rs) / 64;
    using s2r_A = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    constexpr int b_u32 = (rs * rs) / 64;
    using s2r_R = std::conditional_t<(b_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;

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

    int smem_pipe_write = bP_a - 1, smem_pipe_read = 0;
    int ar_pipe_write = 0, m_tile_next = 0, m_tiles_remaining = n_m_tiles;

    for (int p = 0; p < bP_a - 1 && m_tiles_remaining > 0; ++p) {
        int m_s = m_tile_next * BLK_M;
        for (int i = thread_idx; i < BLK_M * BLK_K; i += n_producer_threads) {
            int r = i / BLK_K, c = i % BLK_K;
            sA(r, c, p) = load_or_zero(A_ptr, (m_s + r) * ldA + k_start + c, m_s + r < M);
        }
        cp_async_fence(); --m_tiles_remaining; ++m_tile_next;
    }

    for (int mt = 0; mt < n_m_tiles; ++mt) {
        if (m_tiles_remaining > 0) {
            int m_s = m_tile_next * BLK_M;
            for (int i = thread_idx; i < BLK_M * BLK_K; i += n_producer_threads) {
                int r = i / BLK_K, c = i % BLK_K;
                sA(r, c, smem_pipe_write) = load_or_zero(A_ptr, (m_s + r) * ldA + k_start + c, m_s + r < M);
            }
            --m_tiles_remaining; ++m_tile_next;
        }
        cp_async_fence(); cp_async_wait<bP_a - 1>();
        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

        asm volatile("bar.sync %0, %1;\n"
            : : "r"(ar_pipe_write + BAR_CONSUMED_BASE), "n"(n_total_threads));

        for (int kb = 0; kb < n_blocks_k; ++kb) {
            copy(s2r_A{}, tXsA(_,_,_,kb,smem_pipe_read), tXrA);
            copy(s2r_R{}, tXsR(_,_,_,kb), tXrR);
            clear(rAR);
            gemm(mma_ar, rA_frag, rR_frag, rAR);
            copy(r2s_atom_AR{}, tXrAR, tXsAR_w);
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));

            if constexpr (GATED) {
                for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                    int mi = idx / rs, ri = idx % rs;
                    int gmi = warp_idx * WARP_M + mi;
                    float ar = float(sAR_temp(gmi, ri));
                    float a = float(sA(gmi, kb * rs + ri, smem_pipe_read));
                    float silu_ar = ar / (1.0f + __expf(-ar));
                    sAR_temp(gmi, ri) = half_t(a * silu_ar);
                }
                asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));
            }

            // Transpose AR to sARt: (BLK_K, BLK_M) with BLK_M contiguous
            for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                int mi = idx / rs, ri = idx % rs;
                sARt(kb * rs + ri, warp_idx * WARP_M + mi, ar_pipe_write) =
                    sAR_temp(warp_idx * WARP_M + mi, ri);
            }
        }

        asm volatile("bar.sync 14, %0;\n" : : "n"(n_producer_threads));
        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(ar_pipe_write + BAR_READY_BASE), "n"(n_total_threads));
        ++ar_pipe_write; if (ar_pipe_write == bP_ar) ar_pipe_write = 0;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read; if (smem_pipe_read == bP_a) smem_pipe_read = 0;
    }
}

// =============================================================================
// Consumer: multi-warp MMA for dB, scalar dCt loads (coalesced)
// =============================================================================
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
    using mma_atom_t = std::conditional_t<(BLK_M < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto mma = make_tiled_mma(mma_atom_t{}, WarpLayoutConsumer{},
                               Tile<Int<gs>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(lane_idx + warp_idx * 32);

    auto dB_shape = make_shape(Int<gs>{}, Int<BLK_K>{});
    auto rDB = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(dB_shape, LayoutRight{}))));
    clear(rDB);

    // dCt A-operand: LDSM_T (gs-contiguous smem → BLK_M-contiguous registers for TN MMA)
    constexpr int a_u16 = (gs * BLK_M) / 32;
    using s2r_dCt = std::conditional_t<(a_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, half_t>,
        std::conditional_t<(a_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, half_t>,
            Copy_Atom<SM75_U16x2_LDSM_T, half_t>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dCt{}, mma);
    auto s2r_a_thr = s2r_a.get_slice(lane_idx + warp_idx * 32);
    auto tXsdCt = s2r_a_thr.partition_S(sdCt);
    auto rDCt = thr_mma.make_fragment_A(thr_mma.partition_A(sdCt(_,_,_0{})));
    auto tXrdCt = s2r_a_thr.retile_D(rDCt);

    // B-operand (sARt): LDSM_N, BLK_M contiguous (K for TN MMA)
    using s2r_atom_b = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

    auto s2r_b = make_tiled_copy_B(s2r_atom_b{}, mma);
    auto s2r_b_thr = s2r_b.get_slice(lane_idx + warp_idx * 32);
    auto tXsARt = s2r_b_thr.partition_S(sARt);
    auto rARt = thr_mma.make_fragment_B(thr_mma.partition_B(sARt(_,_,_0{})));
    auto tXrARt = s2r_b_thr.retile_D(rARt);

    // cp.async for dCt loading: gmem (gs, BLK_M) with gs contiguous → smem (gs, BLK_M) gs contiguous
    constexpr int dc_threads_dim0 = gs / 8;
    constexpr int dc_threads_dim1 = n_consumer_threads / dc_threads_dim0;
    constexpr int dc_values_dim1 = BLK_M / dc_threads_dim1;
    auto copy_dC = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
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

    for (int bid = BAR_CONSUMED_BASE; bid < BAR_CONSUMED_BASE + bP_ar; ++bid)
        asm volatile("bar.arrive %0, %1;\n" : : "r"(bid), "n"(n_total_threads));

    int dc_pipe_write = bP_dc - 1, dc_pipe_read = 0;
    int ar_pipe_read = 0, m_tile_next = 0, m_tiles_remaining = n_m_tiles;

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

        asm volatile("bar.sync %0, %1;\n"
            : : "r"(ar_pipe_read + BAR_READY_BASE), "n"(n_total_threads));

        copy(s2r_dCt{}, tXsdCt(_, _, _, dc_pipe_read), tXrdCt);
        copy(s2r_atom_b{}, tXsARt(_, _, _, ar_pipe_read), tXrARt);  // LDSM_T transposes M→K
        gemm(mma, rDCt, rARt, rDB);

        asm volatile("bar.arrive %0, %1;\n"
            : : "r"(ar_pipe_read + BAR_CONSUMED_BASE), "n"(n_total_threads));

        ++ar_pipe_read; if (ar_pipe_read == bP_ar) ar_pipe_read = 0;
        dc_pipe_write = dc_pipe_read;
        ++dc_pipe_read; if (dc_pipe_read == bP_dc) dc_pipe_read = 0;
    }

    auto tMcDB = thr_mma.partition_C(make_identity_tensor(dB_shape));
    for (int i = 0; i < size(rDB); ++i) {
        auto coord = tMcDB(i);
        int n_idx = g * gs + get<0>(coord);
        int k_idx = k_start + get<1>(coord);
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
    using wlp_t = typename cute::BwdDBParams::warp_layout_ar;
    using wlc_t = typename cute::BwdDBParams::warp_layout_arb;
    constexpr int n_consumer_threads = size(wlc_t{}) * 32;

    int k_tile = blockIdx.x;
    int g = blockIdx.y;
    int k_start = k_tile * BLK_K;

    auto smem_k = get_smem_atom(Int<BLK_K>{});
    auto smem_m = get_smem_atom(Int<BLK_M>{});

    auto sA_layout = tile_to_shape(smem_k, make_shape(Int<BLK_M>{}, Int<BLK_K>{}, Int<bP_a>{}));
    auto sR_layout = tile_to_shape(smem_k, make_shape(Int<rs>{}, Int<BLK_K>{}));
    auto sAR_temp_layout = make_layout(make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});
    // sARt: (BLK_K, BLK_M, bP_ar) — BLK_M contiguous for LDSM_N B-operand
    auto sARt_layout = tile_to_shape(smem_m,
        make_shape(Int<BLK_K>{}, Int<BLK_M>{}, Int<bP_ar>{}));
    // sdCt: (gs, BLK_M, bP_dc) with gs contiguous — for LDSM_T + cp.async
    // Swizzle<3,3,3> with column-major atom (64, 8) stride (1, 64) following CUTLASS pattern
    auto sdCt_atom = composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(_64{}, _8{}),
                    make_stride(_1{}, _64{})));
    auto sdCt_layout = tile_to_shape(sdCt_atom, make_shape(Int<gs>{}, Int<BLK_M>{}, Int<bP_dc>{}));

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR       = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sAR_temp = make_tensor(make_smem_ptr(p), sAR_temp_layout); p += cosize(sAR_temp_layout);
    auto sARt     = make_tensor(make_smem_ptr(p), sARt_layout);     p += cosize(sARt_layout);
    auto sdCt     = make_tensor(make_smem_ptr(p), sdCt_layout);

    constexpr int n_total = size(wlp_t{}) * 32 + n_consumer_threads;
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_total) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c) = load_or_zero(R_ptr, (g * rs + r) * K + k_start + c, k_start + c < K);
    }
    __syncthreads();

    if (threadIdx.x >= n_consumer_threads) {
        dB_producer<BLK_M, BLK_K, gs, rs, GATED>(
            A_ptr, ldA, sA, sR, sAR_temp, sARt,
            M, K, g, k_start,
            threadIdx.x - n_consumer_threads, wlp_t{}, wlc_t{});
    } else {
        dB_consumer<BLK_M, BLK_K, gs, rs, GATED>(
            dC_ptr, ldDC, dB_ptr, ldDB,
            sdCt, sARt,
            M, N, K, g, k_start,
            threadIdx.x, wlp_t{}, wlc_t{});
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

    auto smem_m = get_smem_atom(cute::Int<BLK_M>{});
    auto smem_k = get_smem_atom(cute::Int<BLK_K>{});
    int smem = (cosize(tile_to_shape(smem_k, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{}, cute::Int<bP_a>{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{})))
                + BLK_M * rs
                + cosize(tile_to_shape(smem_m, make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_M>{}, cute::Int<bP_ar>{})))
                + cosize(tile_to_shape(
                    composition(Swizzle<3,3,3>{},
                                make_layout(make_shape(_64{}, _8{}), make_stride(_1{}, _64{}))),
                    make_shape(cute::Int<gs>{}, cute::Int<BLK_M>{}, cute::Int<bP_dc>{}))))
               * sizeof(half_t) + 256;

    auto kernel = gated ? dB_pc_kernel<BLK_M, BLK_K, gs, rs, true>
                        : dB_pc_kernel<BLK_M, BLK_K, gs, rs, false>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, block, smem, stream>>>(
        reinterpret_cast<half_t const*>(dC), ldDC,
        reinterpret_cast<half_t const*>(A), ldA,
        reinterpret_cast<half_t const*>(R), ldR,
        reinterpret_cast<half_t*>(dB), ldDB,
        m, n, k, n_groups, k / rs);
}
