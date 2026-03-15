#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// Unified backward dA + dR kernel
//
// All warps cooperate on both the GEMM and the group-boundary epilogue.
// No producer-consumer split — avoids work imbalance and maximizes thread
// utilization for the dR reduction (the dominant bottleneck).
//
// GEMM: dH(BLK_M, BLK_K) += dC(BLK_M, BLK_N) @ B(BLK_K, BLK_N)^T
//   Reduces over entire N dimension. Uses LDSM_N for dC, LDSM_T for B,
//   F16 accumulator, cp.async pipelined.
//
// Group boundary epilogue (every gs/BLK_N tiles):
//   1. Load R for next group (pipelined)
//   2. For each reconn block: write dH slice → sDH_temp, reconn MMA, dR reduction
//   3. Flush batched sDR_acc to global via atomicAdd
//   4. Clear dH accumulator
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

    // GEMM warp layout: distribute all warps across (M, K) output
    constexpr int n_warps_K = 2;
    constexpr int n_warps_M = n_warps / n_warps_K;
    static_assert(n_warps_M * n_warps_K == n_warps, "total warps must be even");
    static_assert(BLK_M % n_warps_M == 0 && BLK_K % n_warps_K == 0);
    using GemmWarpLayout = Layout<Shape<Int<n_warps_M>, Int<n_warps_K>>>;

    int lane_idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

    // Grid with z-curve
    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start  = get<0>(grid_coord) * BLK_K;
    int m_start  = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;
    int m_valid  = min(m_start + BLK_M, M) - m_start;

    // === Shared memory layouts ===
    auto sdC_layout = tile_to_shape(get_smem_atom(Int<BLK_N>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_N>{}, _2{}));
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}), make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom,
        make_shape(Int<BLK_K>{}, Int<BLK_N>{}, _2{}));
    auto sA_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}),
        make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto sR_layout = tile_to_shape(get_smem_atom(Int<BLK_K>{}),
        make_shape(Int<rs>{}, Int<BLK_K>{}, _2{}));
    auto sDH_temp_layout = make_layout(make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});
    constexpr int sDR_acc_elems = rs * BLK_K;

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC      = make_tensor(make_smem_ptr(p), sdC_layout);      p += cosize(sdC_layout);
    auto sB       = make_tensor(make_smem_ptr(p), sB_layout);       p += cosize(sB_layout);
    auto sA       = make_tensor(make_smem_ptr(p), sA_layout);       p += cosize(sA_layout);
    auto sR       = make_tensor(make_smem_ptr(p), sR_layout);       p += cosize(sR_layout);
    auto sDH_temp = make_tensor(make_smem_ptr(p), sDH_temp_layout); p += cosize(sDH_temp_layout);
    // Align sDR_acc to float boundary
    p += (reinterpret_cast<uintptr_t>(p) % 4) ? (4 - reinterpret_cast<uintptr_t>(p) % 4) / sizeof(half_t) : 0;
    float* sDR_acc = reinterpret_cast<float*>(p);

    // === GEMM MMA setup: dH(BLK_M, BLK_K) += dC(BLK_M, BLK_N) @ B(BLK_K, BLK_N)^T [TN] ===
    using mma_atom_t = std::conditional_t<(BLK_N < 16),
        MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>,
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;
    auto mma = make_tiled_mma(mma_atom_t{}, GemmWarpLayout{},
                               Tile<Int<BLK_M>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    auto rDH = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}))));
    clear(rDH);
    auto tCid = thr_mma.partition_C(
        make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{})));

    // LDSM for dC (A-operand, LDSM_N) and B (B-operand, LDSM_T)
    constexpr int K_atom = (BLK_N < 16) ? 8 : 16;
    constexpr int a_u32 = (BLK_M * K_atom) / (n_warps_M * 64);
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    constexpr int b_u16 = (BLK_K * K_atom) / 32;
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

    // cp.async for dC
    auto copy_dC = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_N>{}, Int<n_threads>{});
    auto thr_copy_dC = copy_dC.get_slice(threadIdx.x);
    auto tCsdC_cp = thr_copy_dC.partition_D(sdC);

    // cp.async for B
    constexpr int bt0 = BLK_K / 8;
    constexpr int b_copy_threads = (n_threads < bt0 * (BLK_N / 1)) ? n_threads : bt0 * (BLK_N / 1);
    constexpr int bt1 = b_copy_threads / bt0;
    constexpr int bv1 = BLK_N / bt1;
    static_assert(bt1 > 0 && bv1 > 0, "B copy layout invalid");
    auto copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto thr_copy_B = copy_B.get_slice(threadIdx.x);
    auto tCsB_cp = thr_copy_B.partition_D(sB);

    // === Reconn MMA setup: per-warp, dA(WARP_M, rs) += dH(WARP_M, rs) @ R(rs, rs) ===
    constexpr int WARP_M_reconn = BLK_M / n_warps;
    using cons_mma_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto cons_mma = make_tiled_mma(cons_mma_atom{}, Layout<Shape<_1, _1>>{},
                                    Tile<Int<WARP_M_reconn>, Int<rs>>{});
    auto cons_thr = cons_mma.get_slice(lane_idx);
    auto tCdA_id = cons_thr.partition_C(
        make_identity_tensor(make_shape(Int<WARP_M_reconn>{}, Int<rs>{})));

    // Per-reconn-block dA accumulators (persist across all groups)
    using frag_c_type = decltype(cons_thr.make_fragment_C(cons_thr.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<WARP_M_reconn>{}, Int<rs>{}), LayoutRight{})))));
    frag_c_type rDA_blk[n_reconn_blocks];
    #pragma unroll
    for (int b = 0; b < n_reconn_blocks; ++b) clear(rDA_blk[b]);

    // sDH_temp warp sub-view for reconn MMA
    Tensor sDH_warp = make_tensor(&sDH_temp(warp_idx * WARP_M_reconn, 0),
        make_layout(make_shape(Int<WARP_M_reconn>{}, Int<rs>{}), LayoutRight{}));
    constexpr int cons_a_u32 = (WARP_M_reconn * rs) / 64;
    using cons_s2r_A = std::conditional_t<(cons_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>,
        std::conditional_t<(cons_a_u32 >= 2),
            Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
            Copy_Atom<SM75_U32x1_LDSM_N, half_t>>>;
    auto cons_s2r_a = make_tiled_copy_A(cons_s2r_A{}, cons_mma);
    auto cons_s2r_a_thr = cons_s2r_a.get_slice(lane_idx);
    auto tXsDH_w = cons_s2r_a_thr.partition_S(sDH_warp);
    auto rDH_reconn = cons_thr.make_fragment_A(cons_thr.partition_A(sDH_warp));
    auto tXrDH_reconn = cons_s2r_a_thr.retile_D(rDH_reconn);

    // (AR MMA + packed f16x2 SiLU for gated path: TODO, see CLAUDE.md rules)

    // === Load sA once + sR for group 0 ===
    for (int i = threadIdx.x; i < BLK_M * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sA(r, c) = (m_start + r < M && k_start + c < K) ?
            *(A_ptr + (m_start + r) * ldA + k_start + c) : half_t(0);
    }
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c, _0{}) = (k_start + c < K) ?
            *(R_ptr + r * ldR + k_start + c) : half_t(0);
    }
    __syncthreads();

    int r_pipe_w = 1, r_pipe_r = 0;
    int n_total_tiles = N / BLK_N;

    // Prefill dC + B pipeline
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

    // === Main N-reduction loop ===
    for (int nt = 0; nt < n_total_tiles; ++nt) {
        int tile_in_group = nt % tiles_per_group;

        // Load next dC + B tile (pipelined)
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

        // MMA: rDH += dC_tile @ B_tile^T
        copy(s2r_dC{}, tXsdC(_,_,_,dc_b_pipe_r), tXrdC);
        copy(s2r_B{}, tXsB(_,_,_,dc_b_pipe_r), tXrB);
        gemm(mma, rdC_frag, rB_frag, rDH);

        dc_b_pipe_w = dc_b_pipe_r;
        dc_b_pipe_r = 1 - dc_b_pipe_r;

        // === Group boundary epilogue ===
        if (tile_in_group == tiles_per_group - 1) {
            int g = nt / tiles_per_group;

            // Load R for next group
            if (g + 1 < n_groups) {
                for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
                    int r = i / BLK_K, c = i % BLK_K;
                    sR(r, c, r_pipe_w) = (k_start + c < K) ?
                        *(R_ptr + ((g + 1) * rs + r) * ldR + k_start + c) : half_t(0);
                }
            }

            // Clear sDR_acc for this group
            for (int i = threadIdx.x; i < sDR_acc_elems; i += n_threads)
                sDR_acc[i] = 0.0f;
            __syncthreads();

            // Process each reconn block
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;

                if constexpr (GATED) {
                    // --- AR via MMA (compute ONCE, reuse sigma for Steps A & B) ---
                    // Copy sA_slice → sDH_temp for LDSM
                    for (int i = lane_idx; i < WARP_M_reconn * rs; i += 32) {
                        int mi = i / rs, ri = i % rs;
                        sDH_temp(warp_idx * WARP_M_reconn + mi, ri) =
                            sA(warp_idx * WARP_M_reconn + mi, k_off + ri);
                    }
                    __syncthreads();

                    // LDSM sA + load R^T + MMA: AR = sA @ R^T (F32 accum)
                    // NOTE: AR uses R^T in B-frag: B(n,k) = R(n, k_off+k) = sR(get<0>, k_off+get<1>)
                    // This is SWAPPED from the reconn dA loading which uses R: B(j,n) = R(n, k_off+j)
                    copy(cons_s2r_A{}, tXsDH_w, tXrDH_reconn);
                    auto rR_frag_ar = cons_thr.make_fragment_B(cons_thr.partition_B(
                        make_tensor(static_cast<half_t*>(nullptr),
                                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                    {
                        auto tBid = cons_thr.partition_B(
                            make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                        for (int i = 0; i < size(rR_frag_ar); ++i) {
                            auto coord2 = tBid(i);
                            // Swapped: get<0>=N_out=ri, get<1>=K_red=k → sR(ri, k_off+k)
                            rR_frag_ar(i) = sR(get<0>(coord2), k_off + get<1>(coord2), r_pipe_r);
                        }
                    }
                    frag_c_type rAR;
                    clear(rAR);
                    gemm(cons_mma, rDH_reconn, rR_frag_ar, rAR);

                    // CRITICAL: sync before scatter — GEMM warps cover different rows than
                    // reconn warps, so one warp's scatter can clobber another's LDSM source
                    __syncthreads();

                    // Scatter dH → sDH_temp (overwriting sA)
                    for (int i = 0; i < size(rDH); ++i) {
                        auto coord = tCid(i);
                        int mi = get<0>(coord);
                        int ki = get<1>(coord);
                        if (ki >= k_off && ki < k_off + rs)
                            sDH_temp(mi, ki - k_off) = rDH(i);
                    }
                    __syncthreads();

                    // Compute sigma ONCE per element pair (packed f16x2, reused in A & B)
                    constexpr int n_frag = decltype(size(rAR))::value;
                    constexpr int n_pairs = n_frag / 2;
                    __half2 rSigma[n_pairs], rAR_h2[n_pairs];
                    for (int p = 0; p < n_pairs; ++p) {
                        rAR_h2[p] = __floats2half2_rn(rAR(p * 2), rAR(p * 2 + 1));
                        __half2 half_ar = __hmul2(rAR_h2[p], __float2half2_rn(0.5f));
                        __half2 tanh_val;
                        asm("tanh.approx.f16x2 %0, %1;"
                            : "=r"(reinterpret_cast<uint32_t&>(tanh_val))
                            : "r"(reinterpret_cast<const uint32_t&>(half_ar)));
                        rSigma[p] = __hmul2(__float2half2_rn(0.5f),
                                            __hadd2(__float2half2_rn(1.0f), tanh_val));
                    }

                    // --- Step A: dA += dH * AR * sigma (packed f16x2) ---
                    for (int p = 0; p < n_pairs; ++p) {
                        int i = p * 2;
                        auto c0 = tCdA_id(i), c1 = tCdA_id(i + 1);
                        int mi0 = get<0>(c0) + warp_idx * WARP_M_reconn;
                        int mi1 = get<0>(c1) + warp_idx * WARP_M_reconn;
                        if (mi0 >= m_valid) continue;
                        __half2 dH2 = __halves2half2(
                            reinterpret_cast<const __half&>(sDH_temp(mi0, get<1>(c0))),
                            reinterpret_cast<const __half&>(sDH_temp(mi1, get<1>(c1))));
                        __half2 result = __hmul2(__hmul2(dH2, rAR_h2[p]), rSigma[p]);
                        rDA_blk[b](i)     += __half2float(__low2half(result));
                        rDA_blk[b](i + 1) += __half2float(__high2half(result));
                    }

                    // --- Step B: dS = dH * A * SiLU'(AR) → overwrite sDH_temp (per-warp) ---
                    // SiLU'(x) = sigma * (1 + x * (1 - sigma)), reuse rSigma & rAR_h2
                    for (int p = 0; p < n_pairs; ++p) {
                        int i = p * 2;
                        auto c0 = tCdA_id(i), c1 = tCdA_id(i + 1);
                        int mi0 = get<0>(c0) + warp_idx * WARP_M_reconn;
                        int mi1 = get<0>(c1) + warp_idx * WARP_M_reconn;
                        int ri0 = get<1>(c0), ri1 = get<1>(c1);
                        if (mi0 >= BLK_M) continue;
                        __half2 one_h2 = __float2half2_rn(1.0f);
                        __half2 silu_prime = __hmul2(rSigma[p],
                            __hfma2(rAR_h2[p], __hsub2(one_h2, rSigma[p]), one_h2));
                        __half2 dH2 = __halves2half2(
                            reinterpret_cast<const __half&>(sDH_temp(mi0, ri0)),
                            reinterpret_cast<const __half&>(sDH_temp(mi1, ri1)));
                        __half2 a2 = __halves2half2(
                            reinterpret_cast<const __half&>(sA(mi0, k_off + ri0)),
                            reinterpret_cast<const __half&>(sA(mi1, k_off + ri1)));
                        __half2 dS = __hmul2(__hmul2(dH2, a2), silu_prime);
                        sDH_temp(mi0, ri0) = half_t(__low2half(dS));
                        sDH_temp(mi1, ri1) = half_t(__high2half(dS));
                    }
                    __syncthreads();

                    // --- Step C: dA += dS @ R via MMA ---
                    // Load R with reconn convention (NOT swapped like AR)
                    copy(cons_s2r_A{}, tXsDH_w, tXrDH_reconn);
                    auto rR_frag_reconn = cons_thr.make_fragment_B(cons_thr.partition_B(
                        make_tensor(static_cast<half_t*>(nullptr),
                                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                    {
                        auto tBid = cons_thr.partition_B(
                            make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                        for (int i = 0; i < size(rR_frag_reconn); ++i) {
                            auto coord2 = tBid(i);
                            rR_frag_reconn(i) = sR(get<1>(coord2), k_off + get<0>(coord2), r_pipe_r);
                        }
                    }
                    gemm(cons_mma, rDH_reconn, rR_frag_reconn, rDA_blk[b]);
                } else {
                    // Non-gated: write dH slice to sDH_temp, then dA += dH @ R via MMA
                    for (int i = 0; i < size(rDH); ++i) {
                        auto coord = tCid(i);
                        int mi = get<0>(coord);
                        int ki = get<1>(coord);
                        if (ki >= k_off && ki < k_off + rs)
                            sDH_temp(mi, ki - k_off) = rDH(i);
                    }
                    __syncthreads();
                    copy(cons_s2r_A{}, tXsDH_w, tXrDH_reconn);
                    __syncwarp();
                    auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                        make_tensor(static_cast<half_t*>(nullptr),
                                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                    auto tBid = cons_thr.partition_B(
                        make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                    for (int i = 0; i < size(rR_frag); ++i) {
                        auto coord = tBid(i);
                        rR_frag(i) = sR(get<1>(coord), k_off + get<0>(coord), r_pipe_r);
                    }
                    gemm(cons_mma, rDH_reconn, rR_frag, rDA_blk[b]);
                }

                // dR: M-reduction → sDR_acc via warp shuffle + plain store
                __syncthreads();
                if constexpr (n_threads >= rs * rs) {
                    constexpr int tpe = n_threads / (rs * rs);
                    for (int idx = threadIdx.x; idx < rs * rs * tpe; idx += n_threads) {
                        int elem = idx / tpe, chunk = idx % tpe;
                        int i = elem / rs, j = elem % rs;
                        int mi_s = chunk * (m_valid / tpe);
                        int mi_e = (chunk + 1) * (m_valid / tpe);
                        if (chunk == tpe - 1) mi_e = m_valid;
                        float val = 0.0f;
                        #pragma unroll 4
                        for (int mi = mi_s; mi < mi_e; ++mi)
                            val += float(sDH_temp(mi, i)) * float(sA(mi, k_off + j));
                        // Warp shuffle reduction across tpe threads
                        if constexpr (tpe == 2) {
                            float partner = __shfl_xor_sync(0xffffffff, val, 1);
                            if (chunk == 0)
                                sDR_acc[i * BLK_K + k_off + j] = val + partner;
                        } else if constexpr (tpe == 4) {
                            val += __shfl_xor_sync(0xffffffff, val, 1);
                            val += __shfl_xor_sync(0xffffffff, val, 2);
                            if (chunk == 0)
                                sDR_acc[i * BLK_K + k_off + j] = val;
                        } else if constexpr (tpe == 8) {
                            val += __shfl_xor_sync(0xffffffff, val, 1);
                            val += __shfl_xor_sync(0xffffffff, val, 2);
                            val += __shfl_xor_sync(0xffffffff, val, 4);
                            if (chunk == 0)
                                sDR_acc[i * BLK_K + k_off + j] = val;
                        } else {
                            if (val != 0.0f)
                                atomicAdd(&sDR_acc[i * BLK_K + k_off + j], val);
                        }
                    }
                } else {
                    for (int idx = threadIdx.x; idx < rs * rs; idx += n_threads) {
                        int i = idx / rs, j = idx % rs;
                        float val = 0.0f;
                        #pragma unroll 4
                        for (int mi = 0; mi < m_valid; ++mi)
                            val += float(sDH_temp(mi, i)) * float(sA(mi, k_off + j));
                        sDR_acc[i * BLK_K + k_off + j] = val;
                    }
                }
                __syncthreads();
            }

            // Flush sDR_acc → global dR_partial (batched, one per group)
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

    // Write dA from per-reconn-block fragments to global
    for (int b = 0; b < n_reconn_blocks; ++b) {
        int k_off = b * rs;
        for (int i = 0; i < size(rDA_blk[b]); ++i) {
            auto coord = tCdA_id(i);
            int mi = get<0>(coord) + warp_idx * WARP_M_reconn;
            int ki = get<1>(coord) + k_off;
            if (m_start + mi < M && k_start + ki < K)
                *(dA_ptr + (m_start + mi) * ldDA + k_start + ki) = half_t(rDA_blk[b](i));
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
    bool gated,
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

    // Compute shared memory size
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
                + BLK_M * rs)                 // sDH_temp
               * sizeof(half_t)
               + rs * BLK_K * sizeof(float)   // sDR_acc
               + 256;

    dim3 grid(n_k_tiles * n_m_tiles);
    dim3 block(n_threads);

    auto kernel = gated ? bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, true>
                        : bwd_dadr_kernel<BLK_M, BLK_K, BLK_N, gs, rs, false>;
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
