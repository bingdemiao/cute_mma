#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// Unified Kernel: Fused dA + dR
//
// ALL warps cooperate on each phase:
//   Phase 1 (GEMM): All warps compute dH(BLK_M, BLK_K) = dC_g @ B_g → sdH
//   __syncthreads
//   Phase 2 (Reconn): Each warp independently reads its sdH portion,
//     does MMA reconn for dA, scalar reduction for dR
//   __syncthreads → next group
// =============================================================================

__device__ __forceinline__
half_t load_or_zero(const half_t* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : half_t(0);
}

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
    constexpr int bP_r = cute::BwdDAdRParams::bP_r;
    constexpr int n_reconn_blocks = BLK_K / rs;
    constexpr int n_inner_tiles = gs / BLK_K_INNER;

    using wlp_t = typename cute::BwdDAdRParams::warp_layout_arb;
    using wlc_t = typename cute::BwdDAdRParams::warp_layout_ar;
    constexpr uint32_t n_total_threads = (size(wlp_t{}) + size(wlc_t{})) * 32;
    constexpr int n_total_warps = n_total_threads / 32;
    constexpr int WARP_M = BLK_M / n_total_warps;  // each warp's M range for reconn

    int lane_idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

    // Grid
    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start = get<0>(grid_coord) * BLK_K;
    int m_start = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;
    int m_valid = min(m_start + BLK_M, M) - m_start;

    // === Shared memory ===
    auto smem_ki = get_smem_atom(Int<BLK_K_INNER>{});
    auto smem_bk = get_smem_atom(Int<BLK_K>{});

    auto sdC_layout = tile_to_shape(smem_ki, make_shape(Int<BLK_M>{}, Int<BLK_K_INNER>{}, Int<bP_dc_b>{}));
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}), make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom, make_shape(Int<BLK_K>{}, Int<BLK_K_INNER>{}, Int<bP_dc_b>{}));
    auto sdH_layout = make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{});
    auto sA_layout = tile_to_shape(smem_bk, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    auto sR_layout = tile_to_shape(smem_bk, make_shape(Int<rs>{}, Int<BLK_K>{}, Int<bP_r>{}));
    auto sAR_temp_layout = make_layout(make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC = make_tensor(make_smem_ptr(p), sdC_layout); p += cosize(sdC_layout);
    auto sB  = make_tensor(make_smem_ptr(p), sB_layout);  p += cosize(sB_layout);
    auto sdH = make_tensor(make_smem_ptr(p), sdH_layout);  p += BLK_M * BLK_K;
    auto sA  = make_tensor(make_smem_ptr(p), sA_layout);  p += cosize(sA_layout);
    auto sR  = make_tensor(make_smem_ptr(p), sR_layout);  p += cosize(sR_layout);
    auto sAR_temp = make_tensor(make_smem_ptr(p), sAR_temp_layout);

    // === Phase 1 MMA: ALL warps do GEMM (4,2) layout ===
    using mma_atom_t = std::conditional_t<(BLK_K_INNER < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto mma_gemm = make_tiled_mma(mma_atom_t{}, Layout<Shape<_4, _2>>{},
                                    Tile<Int<BLK_M>, Int<BLK_K>>{});
    auto thr_gemm = mma_gemm.get_slice(threadIdx.x);
    auto rDH = thr_gemm.make_fragment_C(thr_gemm.partition_C(sdH));
    auto tCdH_id = thr_gemm.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{})));

    // LDSM s2r for GEMM
    constexpr int K_atom = (BLK_K_INNER < 16) ? 8 : 16;
    constexpr int WARP_M_gemm = BLK_M / 4;
    constexpr int a_u32 = (WARP_M_gemm * K_atom) / 64;
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    constexpr int b_u16 = (BLK_K * K_atom) / 32;
    using s2r_B = std::conditional_t<(b_u16 >= 8),
        Copy_Atom<SM75_U16x8_LDSM_T, half_t>,
        std::conditional_t<(b_u16 >= 4),
            Copy_Atom<SM75_U16x4_LDSM_T, half_t>,
            Copy_Atom<SM75_U16x2_LDSM_T, half_t>>>;

    auto s2r_a = make_tiled_copy_A(s2r_dC{}, mma_gemm);
    auto s2r_a_thr = s2r_a.get_slice(threadIdx.x);
    auto tXsdC = s2r_a_thr.partition_S(sdC);
    auto rdC = thr_gemm.make_fragment_A(thr_gemm.partition_A(sdC(_,_,_0{})));
    auto tXrdC = s2r_a_thr.retile_D(rdC);

    auto s2r_b = make_tiled_copy_B(s2r_B{}, mma_gemm);
    auto s2r_b_thr = s2r_b.get_slice(threadIdx.x);
    auto tXsB = s2r_b_thr.partition_S(sB);
    auto rB = thr_gemm.make_fragment_B(thr_gemm.partition_B(sB(_,_,_0{})));
    auto tXrB = s2r_b_thr.retile_D(rB);

    // cp.async
    auto copy_dC = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_K_INNER>{}, Int<n_total_threads>{});
    auto thr_copy_dC = copy_dC.get_slice(threadIdx.x);
    auto tCsdC_cp = thr_copy_dC.partition_D(sdC);

    constexpr int bt0 = BLK_K / 8, bt1 = n_total_threads / bt0;
    constexpr int bv1 = BLK_K_INNER / bt1;
    auto copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto thr_copy_B = copy_B.get_slice(threadIdx.x);
    auto tCsB_cp = thr_copy_B.partition_D(sB);

    // === Phase 2: Per-warp reconn MMA ===
    using cons_mma_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto cons_mma = make_tiled_mma(cons_mma_atom{}, Layout<Shape<_1, _1>>{},
                                    Tile<Int<WARP_M>, Int<rs>>{});
    auto cons_thr = cons_mma.get_slice(lane_idx);

    auto dA_shape = make_shape(Int<WARP_M>{}, Int<rs>{});
    auto tCdA_dummy = cons_thr.partition_C(
        make_tensor(static_cast<half_t*>(nullptr), make_layout(dA_shape, LayoutRight{})));
    using frag_c_type = decltype(cons_thr.make_fragment_C(tCdA_dummy));
    frag_c_type rDA_blk[n_reconn_blocks];
    #pragma unroll
    for (int b = 0; b < n_reconn_blocks; ++b) clear(rDA_blk[b]);
    auto tCdA_id = cons_thr.partition_C(make_identity_tensor(dA_shape));

    // sAR_temp warp sub-view for LDSM
    Tensor sAR_warp = make_tensor(&sAR_temp(warp_idx * WARP_M, 0),
        make_layout(make_shape(Int<WARP_M>{}, Int<rs>{}), LayoutRight{}));

    constexpr int cons_a_u32 = (WARP_M * rs) / 64;
    using cons_s2r_A = std::conditional_t<(cons_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    auto cons_s2r_a = make_tiled_copy_A(cons_s2r_A{}, cons_mma);
    auto cons_s2r_a_thr = cons_s2r_a.get_slice(lane_idx);
    auto tXsAR = cons_s2r_a_thr.partition_S(sAR_warp);
    auto rDH_frag = cons_thr.make_fragment_A(cons_thr.partition_A(sAR_warp));
    auto tXrDH_frag = cons_s2r_a_thr.retile_D(rDH_frag);

    // === Load A once ===
    for (int i = threadIdx.x; i < BLK_M * BLK_K; i += n_total_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sA(r, c) = (m_start + r < M && k_start + c < K) ? *(A_ptr + (m_start + r) * ldA + k_start + c) : half_t(0);
    }

    // Prefill R
    for (int pp = 0; pp < bP_r - 1 && pp < n_groups; ++pp) {
        for (int i = threadIdx.x; i < rs * BLK_K; i += n_total_threads) {
            int r = i / BLK_K, c = i % BLK_K;
            sR(r, c, pp) = (k_start + c < K) ? *(R_ptr + (pp * rs + r) * ldR + k_start + c) : half_t(0);
        }
    }
    __syncthreads();

    int r_pipe_w = bP_r - 1, r_pipe_r = 0;

    // === Main group loop ===
    for (int g = 0; g < n_groups; ++g) {
        // Load next R
        int g_next = g + bP_r - 1;
        if (g_next < n_groups) {
            for (int i = threadIdx.x; i < rs * BLK_K; i += n_total_threads) {
                int r = i / BLK_K, c = i % BLK_K;
                sR(r, c, r_pipe_w) = (k_start + c < K) ? *(R_ptr + (g_next * rs + r) * ldR + k_start + c) : half_t(0);
            }
        }

        // ---- Phase 1: GEMM (ALL warps) ----
        clear(rDH);
        int pipe_w = bP_dc_b - 1, pipe_r = 0;

        for (int pp = 0; pp < bP_dc_b - 1 && pp < n_inner_tiles; ++pp) {
            int off = pp * BLK_K_INNER;
            auto gdC = make_tensor(make_gmem_ptr(dC_ptr + m_start * ldDC + g * gs + off),
                make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K_INNER>{}), make_stride(ldDC, Int<1>{})));
            copy(copy_dC, thr_copy_dC.partition_S(gdC), tCsdC_cp(_,_,_,pp));
            auto gB = make_tensor(make_gmem_ptr(B_ptr + (g * gs + off) * ldB + k_start),
                make_layout(make_shape(Int<BLK_K>{}, Int<BLK_K_INNER>{}), make_stride(Int<1>{}, ldB)));
            copy(copy_B, thr_copy_B.partition_S(gB), tCsB_cp(_,_,_,pp));
            cp_async_fence();
        }

        for (int ki = 0; ki < n_inner_tiles; ++ki) {
            int ki_next = ki + bP_dc_b - 1;
            if (ki_next < n_inner_tiles) {
                int off = ki_next * BLK_K_INNER;
                auto gdC = make_tensor(make_gmem_ptr(dC_ptr + m_start * ldDC + g * gs + off),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K_INNER>{}), make_stride(ldDC, Int<1>{})));
                copy(copy_dC, thr_copy_dC.partition_S(gdC), tCsdC_cp(_,_,_,pipe_w));
                auto gB = make_tensor(make_gmem_ptr(B_ptr + (g * gs + ki_next * BLK_K_INNER) * ldB + k_start),
                    make_layout(make_shape(Int<BLK_K>{}, Int<BLK_K_INNER>{}), make_stride(Int<1>{}, ldB)));
                copy(copy_B, thr_copy_B.partition_S(gB), tCsB_cp(_,_,_,pipe_w));
            }
            cp_async_fence();
            cp_async_wait<bP_dc_b - 1>();
            __syncthreads();

            copy(s2r_dC{}, tXsdC(_,_,_,pipe_r), tXrdC);
            copy(s2r_B{}, tXsB(_,_,_,pipe_r), tXrB);
            gemm(mma_gemm, rdC, rB, rDH);

            pipe_w = pipe_r;
            if (++pipe_r == bP_dc_b) pipe_r = 0;
            __syncthreads();
        }
        cp_async_wait<0>();

        // Write F32 rDH → F16 sdH
        for (int i = 0; i < size(rDH); ++i) {
            auto coord = tCdH_id(i);
            sdH(get<0>(coord), get<1>(coord)) = half_t(rDH(i));
        }
        __syncthreads();

        // ---- Phase 2: Reconn (each warp reads its sdH portion independently) ----
        if constexpr (GATED) {
            auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                make_tensor(static_cast<half_t*>(nullptr),
                            make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
            auto tBid = cons_thr.partition_B(make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));

            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;

                // Step 1: Copy A slice to sAR_temp (each warp handles WARP_M rows)
                for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                    int mi_local = idx / rs, ri = idx % rs;
                    sAR_temp(warp_idx * WARP_M + mi_local, ri) = sA(warp_idx * WARP_M + mi_local, k_off + ri);
                }
                __syncwarp();

                // Step 2: AR recompute via MMA: rAR = A_b @ R_b^T
                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);
                // R loaded in R^T direction for AR = A @ R^T
                for (int i = 0; i < size(rR_frag); ++i) {
                    auto coord = tBid(i);
                    rR_frag(i) = sR(get<0>(coord), k_off + get<1>(coord), r_pipe_r);  // R natural = R^T for TN
                }
                frag_c_type rAR;
                clear(rAR);
                gemm(cons_mma, rDH_frag, rR_frag, rAR);  // AR = A @ R^T

                // Step 3: Elementwise SiLU + dS → write dS to sAR_temp
                for (int i = 0; i < size(rAR); ++i) {
                    auto coord = tCdA_id(i);
                    int mi = get<0>(coord) + warp_idx * WARP_M;
                    int ri = get<1>(coord);
                    if (mi < m_valid) {
                        float ar_f = rAR(i);
                        half_t ar_h = half_t(ar_f);
                        half_t half_ar;
                        asm("mul.f16 %0, %1, %2;" : "=h"(*(uint16_t*)&half_ar)
                            : "h"(*(uint16_t*)&ar_h), "h"(uint16_t(0x3800)));
                        half_t tanh_h;
                        asm("tanh.approx.f16 %0, %1;" : "=h"(*(uint16_t*)&tanh_h)
                            : "h"(*(uint16_t*)&half_ar));
                        float sigma = 0.5f + 0.5f * float(tanh_h);

                        float dH_val = float(sdH(mi, k_off + ri));
                        float a_val = float(sA(mi, k_off + ri));
                        rDA_blk[b](i) += dH_val * ar_f * sigma;  // dH * SiLU(AR)
                        float silu_prime = sigma * (1.0f + ar_f * (1.0f - sigma));
                        sAR_temp(mi, ri) = half_t(dH_val * a_val * silu_prime);  // dS
                    }
                }
                __syncwarp();

                // Step 4: dA += dS @ R via MMA
                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);
                for (int i = 0; i < size(rR_frag); ++i) {
                    auto coord = tBid(i);
                    rR_frag(i) = sR(get<1>(coord), k_off + get<0>(coord), r_pipe_r);  // R for dA
                }
                gemm(cons_mma, rDH_frag, rR_frag, rDA_blk[b]);

                // Step 5: dR per reconn block (needs full-block sync for cross-warp sAR_temp)
                __syncthreads();
                {
                    constexpr int threads_per_elem = n_total_threads / (rs * rs);
                    for (int idx = threadIdx.x; idx < rs * rs * threads_per_elem; idx += n_total_threads) {
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
                __syncthreads();  // before next reconn block overwrites sAR_temp
            }  // end reconn block loop
        } else {
            // Non-gated: dA per warp via sdH reads + scalar
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;
                // Copy sdH slice → sAR_temp for this warp's rows
                for (int idx = lane_idx; idx < WARP_M * rs; idx += 32) {
                    int mi_local = idx / rs, ri = idx % rs;
                    int mi = warp_idx * WARP_M + mi_local;
                    sAR_temp(mi, ri) = (mi < m_valid) ? sdH(mi, k_off + ri) : half_t(0);
                }
                __syncwarp();  // ensure warp's sAR_temp writes visible for LDSM

                // dA += dH @ R via MMA
                copy(cons_s2r_A{}, tXsAR, tXrDH_frag);
                auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                    make_tensor(static_cast<half_t*>(nullptr),
                                make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                {
                    auto tBid = cons_thr.partition_B(make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                    for (int i = 0; i < size(rR_frag); ++i) {
                        auto coord = tBid(i);
                        rR_frag(i) = sR(get<1>(coord), k_off + get<0>(coord), r_pipe_r);
                    }
                }
                gemm(cons_mma, rDH_frag, rR_frag, rDA_blk[b]);
            }

            // dR reduction — all threads
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;
                constexpr int threads_per_elem = n_total_threads / (rs * rs);
                for (int idx = threadIdx.x; idx < rs * rs * threads_per_elem; idx += n_total_threads) {
                    int elem = idx / threads_per_elem;
                    int chunk = idx % threads_per_elem;
                    int i = elem / rs, j = elem % rs;
                    int mi_start = chunk * (m_valid / threads_per_elem);
                    int mi_end = (chunk + 1) * (m_valid / threads_per_elem);
                    if (chunk == threads_per_elem - 1) mi_end = m_valid;
                    float val = 0.0f;
                    for (int mi = mi_start; mi < mi_end; ++mi)
                        val += float(sdH(mi, k_off + i)) * float(sA(mi, k_off + j));
                    if (val != 0.0f) {
                        int dR_rows = n_groups * rs;
                        atomicAdd(&dR_partial[buf_slot * dR_rows * K + (g * rs + i) * K + k_start + k_off + j], val);
                    }
                }
            }
        }
        __syncthreads();

        r_pipe_w = r_pipe_r;
        if (++r_pipe_r == bP_r) r_pipe_r = 0;
    }

    // Write dA from per-reconn-block MMA fragments to global memory
    for (int b = 0; b < n_reconn_blocks; ++b) {
        int k_off = b * rs;
        for (int i = 0; i < size(rDA_blk[b]); ++i) {
            auto coord = tCdA_id(i);
            int mi = get<0>(coord) + warp_idx * WARP_M;
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
    constexpr int BLK_K_INNER = cute::BwdDAdRParams::bK_inner;
    constexpr int bP_dc_b = cute::BwdDAdRParams::bP_dc_b;
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
    int smem = (cosize(tile_to_shape(smem_ki, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K_INNER>{}, cute::Int<bP_dc_b>{})))
                + cosize(tile_to_shape(
                    composition(Swizzle<3,3,3>{},
                                make_layout(make_shape(cute::Int<BLK_K>{}, _8{}),
                                            make_stride(_1{}, cute::Int<BLK_K>{}))),
                    make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_K_INNER>{}, cute::Int<bP_dc_b>{})))
                + BLK_M * BLK_K  // sdH F16
                + cosize(tile_to_shape(smem_bk, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{})))
                + cosize(tile_to_shape(smem_bk, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{}, cute::Int<bP_r>{})))
                + BLK_M * rs)
               * sizeof(half_t) + 256;

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
