#include "cute_oft_backward_dadr.hpp"
#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
#include "z_curve.hpp"

using namespace cute;

// =============================================================================
// GEMM-style backward dA + dR kernel
//
// dA(M, K) = dC(M, N) @ B_rot(N, K)  where B_rot = B @ R (per-group rotation)
// dR_g(rs, rs) = sum_m dH_g(m, :)^T @ A(m, :) per reconn block
//
// Architecture: Standard GEMM reducing over N, with group-boundary epilogues
// for rotation (dA) and dR extraction. No producer-consumer split needed.
//
// Grid: (M/BLK_M, K/BLK_K) with z-curve ordering
// All warps cooperate on the GEMM. At group boundaries (every gs/BLK_N tiles):
//   1. Extract dR from the per-group partial sum rDH_g
//   2. Apply reconn rotation: rDA += rDH_g @ R_g per reconn block
//   3. Clear rDH_g for next group
// =============================================================================

__device__ __forceinline__
half_t load_or_zero(const half_t* ptr, int idx, bool valid) {
    return valid ? ptr[idx] : half_t(0);
}

template <int BLK_M, int BLK_K, int BLK_N, int GROUP_SIZE, int RECONN_SZ, bool GATED>
__global__ void __launch_bounds__(256)
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
    constexpr int n_reconn_blocks = BLK_K / rs;
    constexpr int tiles_per_group = gs / BLK_N;  // N-tiles per group
    constexpr int n_threads = 256;  // 8 warps

    int lane_idx = threadIdx.x % 32;
    int warp_idx = threadIdx.x / 32;

    // Grid with z-curve
    int n_k_tiles = (K + BLK_K - 1) / BLK_K;
    int n_m_tiles = (M + BLK_M - 1) / BLK_M;
    auto grid_shape = make_shape(n_k_tiles, n_m_tiles);
    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    int k_start = get<0>(grid_coord) * BLK_K;
    int m_start = get<1>(grid_coord) * BLK_M;
    int buf_slot = get<1>(grid_coord) % n_buf_slots;
    int m_valid = min(m_start + BLK_M, M) - m_start;

    // === Shared memory ===
    auto smem_n = get_smem_atom(Int<BLK_N>{});
    auto smem_k = get_smem_atom(Int<BLK_K>{});

    // sdC: (BLK_M, BLK_N, 2) for pipelined dC loading — BLK_N contiguous for TN A-operand
    auto sdC_layout = tile_to_shape(smem_n, make_shape(Int<BLK_M>{}, Int<BLK_N>{}, _2{}));
    // sB: (BLK_K, BLK_N, 2) for pipelined B loading — BLK_N contiguous for TN B-operand
    // Use LDSM_T: store BLK_K contiguous, LDSM_T transposes to BLK_N contiguous
    auto sB_atom = composition(Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<BLK_K>{}, _8{}), make_stride(_1{}, Int<BLK_K>{})));
    auto sB_layout = tile_to_shape(sB_atom, make_shape(Int<BLK_K>{}, Int<BLK_N>{}, _2{}));
    // sA: (BLK_M, BLK_K) for dR computation — loaded once
    auto sA_layout = tile_to_shape(smem_k, make_shape(Int<BLK_M>{}, Int<BLK_K>{}));
    // sR: (rs, BLK_K, 2) for R — loaded per group
    auto sR_layout = tile_to_shape(smem_k, make_shape(Int<rs>{}, Int<BLK_K>{}, _2{}));
    // sDH: (BLK_M, BLK_K) for dH → sAR_temp (reused for reconn MMA + dR)
    auto sDH_layout = make_layout(make_shape(Int<BLK_M>{}, Int<rs>{}), LayoutRight{});

    extern __shared__ half_t smem_raw[];
    half_t* p = smem_raw;
    auto sdC = make_tensor(make_smem_ptr(p), sdC_layout); p += cosize(sdC_layout);
    auto sB  = make_tensor(make_smem_ptr(p), sB_layout);  p += cosize(sB_layout);
    auto sA  = make_tensor(make_smem_ptr(p), sA_layout);  p += cosize(sA_layout);
    auto sR  = make_tensor(make_smem_ptr(p), sR_layout);  p += cosize(sR_layout);
    auto sDH_temp = make_tensor(make_smem_ptr(p), sDH_layout); // for dR + reconn MMA

    // === Main GEMM MMA: dH_g(BLK_M, BLK_K) += dC(BLK_M, BLK_N) @ B(BLK_K, BLK_N)^T ===
    // TN: both operands have BLK_N (=K_reduction) contiguous
    using mma_atom_t = std::conditional_t<(BLK_N < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto mma = make_tiled_mma(mma_atom_t{}, Layout<Shape<_4, _2>>{},
                               Tile<Int<BLK_M>, Int<BLK_K>>{});
    auto thr_mma = mma.get_slice(threadIdx.x);

    // Two F32 accumulators: rDH_g (per-group, cleared at group boundaries) and rDA (final output)
    auto rDH_g = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}))));
    auto rDA = thr_mma.make_fragment_C(thr_mma.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<BLK_M>{}, Int<BLK_K>{}), LayoutRight{}))));
    clear(rDH_g);
    clear(rDA);
    auto tCid = thr_mma.partition_C(make_identity_tensor(make_shape(Int<BLK_M>{}, Int<BLK_K>{})));

    // LDSM s2r for dC (A-operand) and B (B-operand via LDSM_T)
    constexpr int K_atom = (BLK_N < 16) ? 8 : 16;
    constexpr int WARP_M = BLK_M / 4;  // 4 warps along M in (4,2) layout
    constexpr int a_u32 = (WARP_M * K_atom) / 64;
    using s2r_dC = std::conditional_t<(a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
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

    // cp.async for dC and B
    auto copy_dC = cp_layout<uint128_t, half_t>(Int<BLK_M>{}, Int<BLK_N>{}, Int<n_threads>{});
    auto thr_copy_dC = copy_dC.get_slice(threadIdx.x);
    auto tCsdC_cp = thr_copy_dC.partition_D(sdC);

    constexpr int bt0 = BLK_K / 8, bt1 = n_threads / bt0;
    constexpr int bv1 = BLK_N / bt1;
    auto copy_B = make_tiled_copy(
        Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
        make_layout(make_shape(Int<bt0>{}, Int<bt1>{}), LayoutRight{}),
        make_layout(make_shape(_8{}, Int<bv1>{})));
    auto thr_copy_B = copy_B.get_slice(threadIdx.x);
    auto tCsB_cp = thr_copy_B.partition_D(sB);

    // === Reconn MMA for group epilogue: per-warp, per-reconn-block ===
    constexpr int WARP_M_reconn = BLK_M / 8;  // 8 warps each handle WARP_M rows
    using cons_mma_atom = std::conditional_t<(rs < 16),
        MMA_Atom<SM80_16x8x8_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>>;
    auto cons_mma = make_tiled_mma(cons_mma_atom{}, Layout<Shape<_1, _1>>{},
                                    Tile<Int<WARP_M_reconn>, Int<rs>>{});
    auto cons_thr = cons_mma.get_slice(lane_idx);
    auto tCdA_id = cons_thr.partition_C(make_identity_tensor(
        make_shape(Int<WARP_M_reconn>{}, Int<rs>{})));

    // sAR_temp warp sub-view for reconn MMA
    Tensor sDH_warp = make_tensor(&sDH_temp(warp_idx * WARP_M_reconn, 0),
        make_layout(make_shape(Int<WARP_M_reconn>{}, Int<rs>{}), LayoutRight{}));
    constexpr int cons_a_u32 = (WARP_M_reconn * rs) / 64;
    using cons_s2r_A = std::conditional_t<(cons_a_u32 >= 4),
        Copy_Atom<SM75_U32x4_LDSM_N, half_t>, Copy_Atom<SM75_U32x1_LDSM_N, half_t>>;
    auto cons_s2r_a = make_tiled_copy_A(cons_s2r_A{}, cons_mma);
    auto cons_s2r_a_thr = cons_s2r_a.get_slice(lane_idx);
    auto tXsDH = cons_s2r_a_thr.partition_S(sDH_warp);
    auto rDH_reconn = cons_thr.make_fragment_A(cons_thr.partition_A(sDH_warp));
    auto tXrDH_reconn = cons_s2r_a_thr.retile_D(rDH_reconn);

    // Per-reconn-block dA accumulators (for rotation epilogue)
    using frag_c_type = decltype(cons_thr.make_fragment_C(cons_thr.partition_C(
        make_tensor(static_cast<half_t*>(nullptr),
                    make_layout(make_shape(Int<WARP_M_reconn>{}, Int<rs>{}), LayoutRight{})))));
    frag_c_type rDA_blk[n_reconn_blocks];
    #pragma unroll
    for (int b = 0; b < n_reconn_blocks; ++b) clear(rDA_blk[b]);

    // === Load A once ===
    for (int i = threadIdx.x; i < BLK_M * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sA(r, c) = (m_start + r < M && k_start + c < K) ?
            *(A_ptr + (m_start + r) * ldA + k_start + c) : half_t(0);
    }

    // Prefill R for group 0
    for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
        int r = i / BLK_K, c = i % BLK_K;
        sR(r, c, _0{}) = (k_start + c < K) ? *(R_ptr + r * ldR + k_start + c) : half_t(0);
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

    int dc_b_pipe_w = 1, dc_b_pipe_r = 0;  // read from slot 0 (prefilled), write to slot 1

    // === Main N-reduction loop ===
    for (int nt = 0; nt < n_total_tiles; ++nt) {
        int g = nt / tiles_per_group;  // current group
        int tile_in_group = nt % tiles_per_group;
        int n_offset = nt * BLK_N;

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

        // MMA: rDH_g += dC_tile @ B_tile^T
        copy(s2r_dC{}, tXsdC(_,_,_,dc_b_pipe_r), tXrdC);
        copy(s2r_B{}, tXsB(_,_,_,dc_b_pipe_r), tXrB);
        gemm(mma, rdC_frag, rB_frag, rDH_g);
        __syncthreads();

        // Swap pipeline slots
        dc_b_pipe_w = dc_b_pipe_r;
        dc_b_pipe_r = 1 - dc_b_pipe_r;

        // === Group boundary epilogue ===
        if (tile_in_group == tiles_per_group - 1) {
            // Load R for next group (if exists)
            int g_next = g + 1;
            if (g_next < n_groups) {
                for (int i = threadIdx.x; i < rs * BLK_K; i += n_threads) {
                    int r = i / BLK_K, c = i % BLK_K;
                    sR(r, c, r_pipe_w) = (k_start + c < K) ?
                        *(R_ptr + (g_next * rs + r) * ldR + k_start + c) : half_t(0);
                }
            }

            // Write rDH_g to sDH_temp per reconn block for MMA reconn + dR
            // Each warp writes its WARP_M_reconn rows
            for (int b = 0; b < n_reconn_blocks; ++b) {
                int k_off = b * rs;

                // Write dH slice to sDH_temp (F32 → F16, per warp)
                for (int i = 0; i < size(rDH_g); ++i) {
                    auto coord = tCid(i);
                    int mi = get<0>(coord);
                    int ki = get<1>(coord);
                    if (ki >= k_off && ki < k_off + rs) {
                        sDH_temp(mi, ki - k_off) = half_t(rDH_g(i));
                    }
                }
                __syncthreads();

                if constexpr (GATED) {
                    // Step A: dA += dH * SiLU(AR) — read dH from sDH_temp BEFORE overwriting
                    // Each warp processes its own WARP_M_reconn rows
                    for (int i = 0; i < size(rDA_blk[b]); ++i) {
                        auto coord2 = tCdA_id(i);
                        int mi = get<0>(coord2) + warp_idx * WARP_M_reconn;
                        int ri = get<1>(coord2);
                        if (mi < m_valid) {
                            float ar = 0.0f;
                            for (int j = 0; j < rs; ++j)
                                ar += float(sA(mi, k_off + j)) * float(sR(ri, k_off + j, r_pipe_r));
                            float sigma = 1.0f / (1.0f + __expf(-ar));
                            float dH_val = float(sDH_temp(mi, ri));  // original dH
                            rDA_blk[b](i) += dH_val * ar * sigma;   // dH * SiLU(AR)
                        }
                    }

                    // Step B: compute dS → overwrite sDH_temp
                    for (int idx = threadIdx.x; idx < m_valid * rs; idx += n_threads) {
                        int mi = idx / rs, ri = idx % rs;
                        float ar = 0.0f;
                        for (int j = 0; j < rs; ++j)
                            ar += float(sA(mi, k_off + j)) * float(sR(ri, k_off + j, r_pipe_r));
                        float dH_val = float(sDH_temp(mi, ri));
                        float a_val = float(sA(mi, k_off + ri));
                        float sigma = 1.0f / (1.0f + __expf(-ar));
                        float silu_prime = sigma * (1.0f + ar * (1.0f - sigma));
                        sDH_temp(mi, ri) = half_t(dH_val * a_val * silu_prime);  // dS
                    }
                    __syncthreads();

                    // Step C: dA += dS @ R via MMA
                    copy(cons_s2r_A{}, tXsDH, tXrDH_reconn);
                    auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                        make_tensor(static_cast<half_t*>(nullptr),
                                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                    auto tBid = cons_thr.partition_B(make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                    for (int i = 0; i < size(rR_frag); ++i) {
                        auto coord2 = tBid(i);
                        rR_frag(i) = sR(get<1>(coord2), k_off + get<0>(coord2), r_pipe_r);
                    }
                    gemm(cons_mma, rDH_reconn, rR_frag, rDA_blk[b]);
                } else {
                    // Non-gated: dA += dH @ R per reconn block via MMA
                    copy(cons_s2r_A{}, tXsDH, tXrDH_reconn);
                    __syncwarp();
                    auto rR_frag = cons_thr.make_fragment_B(cons_thr.partition_B(
                        make_tensor(static_cast<half_t*>(nullptr),
                                    make_layout(make_shape(Int<rs>{}, Int<rs>{}), LayoutRight{}))));
                    auto tBid = cons_thr.partition_B(make_identity_tensor(make_shape(Int<rs>{}, Int<rs>{})));
                    for (int i = 0; i < size(rR_frag); ++i) {
                        auto coord = tBid(i);
                        rR_frag(i) = sR(get<1>(coord), k_off + get<0>(coord), r_pipe_r);
                    }
                    gemm(cons_mma, rDH_reconn, rR_frag, rDA_blk[b]);
                }

                // dR: all threads cooperate
                __syncthreads();
                {
                    constexpr int tpe = n_threads / (rs * rs);
                    for (int idx = threadIdx.x; idx < rs * rs * tpe; idx += n_threads) {
                        int elem = idx / tpe, chunk = idx % tpe;
                        int i = elem / rs, j = elem % rs;
                        int mi_s = chunk * (m_valid / tpe);
                        int mi_e = (chunk + 1) * (m_valid / tpe);
                        if (chunk == tpe - 1) mi_e = m_valid;
                        float val = 0.0f;
                        for (int mi = mi_s; mi < mi_e; ++mi)
                            val += float(sDH_temp(mi, i)) * float(sA(mi, k_off + j));
                        if (val != 0.0f) {
                            int dR_rows = n_groups * rs;
                            atomicAdd(&dR_partial[buf_slot * dR_rows * K + (g * rs + i) * K + k_start + k_off + j], val);
                        }
                    }
                }
                __syncthreads();
            }

            // Clear per-group accumulator
            clear(rDH_g);
            // Advance R pipe
            r_pipe_w = r_pipe_r;
            r_pipe_r = 1 - r_pipe_r;
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
    constexpr int BLK_N = cute::BwdDAdRParams::bK_inner;  // reuse bK_inner as BLK_N
    constexpr int n_buf_slots_param = cute::BwdDAdRParams::n_buf_slots;

    int n_groups = n / gs;
    int n_k_tiles = (k + BLK_K - 1) / BLK_K;
    int n_m_tiles = (m + BLK_M - 1) / BLK_M;
    int n_buf_slots = min(n_buf_slots_param, n_m_tiles);
    if (n_buf_slots < 1) n_buf_slots = 1;
    int dR_elements = n_groups * rs * k;

    float* dR_partial_buf = nullptr;
    cudaMalloc(&dR_partial_buf, (int64_t)n_buf_slots * dR_elements * sizeof(float));
    cudaMemsetAsync(dR_partial_buf, 0, (int64_t)n_buf_slots * dR_elements * sizeof(float), stream);

    auto smem_n = get_smem_atom(cute::Int<BLK_N>{});
    auto smem_k = get_smem_atom(cute::Int<BLK_K>{});
    int smem = (cosize(tile_to_shape(smem_n, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_N>{}, _2{})))
                + cosize(tile_to_shape(
                    composition(Swizzle<3,3,3>{},
                                make_layout(make_shape(cute::Int<BLK_K>{}, _8{}),
                                            make_stride(_1{}, cute::Int<BLK_K>{}))),
                    make_shape(cute::Int<BLK_K>{}, cute::Int<BLK_N>{}, _2{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<BLK_M>{}, cute::Int<BLK_K>{})))
                + cosize(tile_to_shape(smem_k, make_shape(cute::Int<rs>{}, cute::Int<BLK_K>{}, _2{})))
                + BLK_M * rs)
               * sizeof(half_t) + 256;

    dim3 grid(n_k_tiles * n_m_tiles);
    dim3 block(256);

    auto kernel = gated ? fused_dA_dR_kernel_v2<BLK_M, BLK_K, BLK_N, gs, rs, true>
                        : fused_dA_dR_kernel_v2<BLK_M, BLK_K, BLK_N, gs, rs, false>;
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
