#include "cute_prism_coop_pc.hpp"
#include <prism_config.hpp>
#include "cute_prism_util.hpp"
#include "z_curve.hpp"

template<class CTATiler, class GroupSize, class ReconnectSize, class ConsumptionWidth, class PipelineA_R, class PipelineAR, class PipelineB>
size_t get_smem_size(CTATiler cta, GroupSize group_size, ReconnectSize reconn_sz, ConsumptionWidth c_width, PipelineA_R pipeA_R, PipelineAR pipeAR, PipelineB pipeB)
{
    using namespace cute;
    auto size_A = size<0>(cta) * size<2>(cta) * pipeA_R; // BLK_M * BLK_K * PIPE2
    auto size_B = size<1>(cta) * size<2>(cta) * pipeB; // BLK_N * BLK_K * PIPE2
    auto n_groups = max(size<1>(cta) / group_size, _1{}); // Number of groups in N dimension
    auto size_R = n_groups * reconn_sz * size<2>(cta) * pipeA_R; // N_GROUPS * RECONN_SZ * BLK_K * PIPE2
    auto size_AR = n_groups * c_width * size<0>(cta) * pipeAR; // N_GROUPS * CONSUMPTION_WIDTH * BLK_M * PIPE1
    // Compute shared memory size based on the tiler and other parameters
    return (size_A + size_B + size_R + size_AR) * sizeof(prism_native);
}

template <class TensorGA, class TensorSA, class TiledCopyA,
          class TensorGR, class TensorSR, class TiledCopyR, class ReconnectSize, class ConsumptionBlocks,
          class TensorSAR,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device__ static inline
void prism_ar(TensorGA const &gA, TensorSA &sA, TiledCopyA copy_a,
            TensorGR const &gR, TensorSR &sR, TiledCopyR copy_r, ReconnectSize reconn_sz, ConsumptionBlocks c_blocks,
            TensorSAR &sAR, int thread_idx,
            WarpLayoutStage1 warps_stage1, WarpLayoutStage2 warps_stage2,
            prism_native const* bias_ptr, int cta_g_base, int K_full,
    // Dropout (only used when PRISM_DROPOUT=1, otherwise nullptr/0/1.0).
    // Per-group seed indexed by global group g; idx = m_global*K_full + k_global.
    int64_t const* dropout_seeds, float dropout_p, float inv_keep, int m_start)
{
    // This function should revice blocked corresponding tensors
    using namespace cute;
    ThrCopy thr_copy_a = copy_a.get_slice(thread_idx);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_r = copy_r.get_slice(thread_idx);
    Tensor tRgR = thr_copy_r.partition_S(gR); // (CPY,CPY_M,CPY_K,k)
    Tensor tRsR = thr_copy_r.partition_D(sR); // (CPY,CPY_M,CPY_K,PIPE)

    auto n_warps = size(warps_stage1);
    constexpr uint32_t n_threads1 = n_warps * 32;
    constexpr uint32_t n_threads_total = size(warps_stage2) * 32 + n_threads1;
    uint32_t warp_idx = thread_idx / 32;
    uint32_t lane_idx = thread_idx % 32;

    auto cta_atom_layout_m = make_layout(
        make_shape(n_warps, _8{}), LayoutRight{}
    ); // the layout design without breaking 8 contiguous rows

    Tensor sA_warp_atom = group_modes<0,2>(
        logical_divide(
            sA,
            make_tile(
                cta_atom_layout_m,
                make_layout(make_shape(reconn_sz, c_blocks))
            )
        )( // (((N_WARPS, 8), REST_M), ((RECONN_SZ, CONSUME_BLOCKS), BLOCKS_ALONG_K), PIPELINE)
            make_coord(make_coord(warp_idx, _), _),
            make_coord(make_coord(_,_),_), _
        ) // (8, REST_M, RECONN_SZ, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)
    ); // (WARP_M_REGION, RECONN_SZ, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)

    Tensor sR_warp_atom = logical_divide(
        sR,
        make_tile(
            _,
            make_layout(make_shape(reconn_sz, c_blocks))
        )
    )( // (RECONN_SZ * N_GROUPS, ((RECONN_SZ, CONSUME_BLOCKS), BLOCKS_ALONG_K), PIPELINE)
        _, make_coord(make_coord(_,_),_), _
    ); // (RECONN_SZ * N_GROUPS, RECONN_SZ, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)

    auto producer_block_size = size<1>(sAR) / c_blocks;
    auto n_groups = producer_block_size / reconn_sz;
    auto producer_layout_n = make_layout(
        make_shape(reconn_sz, c_blocks, n_groups)
    );
    auto layout_n_stage1 = select<0,2>(producer_layout_n);

    Tensor sAR_warp_atom = group_modes<0,2>(
        logical_divide(
            sAR,
            make_tile(cta_atom_layout_m, layout_n_stage1)
        )( // (((N_WARPS, 8), REST_M), (RECONN_SZ * N_GROUPS, CONSUME_BLOCKS), PIPELINE)
            make_coord(make_coord(warp_idx, _), _), make_coord(_, _), _
        ) // (8, REST_M, RECONN_SZ * N_GROUPS, CONSUME_BLOCKS, PIPELINE)
    ); // (WARP_M_REGION, RECONN_SZ * N_GROUPS, CONSUME_BLOCKS, PIPELINE)

    CUTE_STATIC_ASSERT_V(size<0>(sA_warp_atom) == size<0>(sAR_warp_atom));
    CUTE_STATIC_ASSERT_V(size<0>(sR_warp_atom) == size<1>(sAR_warp_atom));

    
    // For fp16: F16-accum AR mma. For bf16: F32-accum (only option for bf16).
    using mma_atom1 = prism_ar_atom<decltype(reconn_sz)::value>;

    TiledMMA single_warp_mma1 = make_tiled_mma(
        mma_atom1{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(sAR_warp_atom)), decltype(size<1>(sAR_warp_atom))>{}
    );

    ThrMMA thr_mma1 = single_warp_mma1.get_slice(lane_idx);
    Tensor tCsA = thr_mma1.partition_A(sA_warp_atom); // (MMA, MMA_M, MMA_K, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsR = thr_mma1.partition_B(sR_warp_atom); // (MMA, MMA_N, MMA_K, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCsAR = thr_mma1.partition_C(sAR_warp_atom); // (MMA, MMA_M, MMA_N, CONSUME_BLOCKS, PIPELINE)

    Tensor tCrA = thr_mma1.make_fragment_A(tCsA(_,_,_,_,_0{},_0{})); // (MMA, MMA_M, MMA_K, CONSUME_BLOCKS)
    Tensor tCrR = thr_mma1.make_fragment_B(tCsR(_,_,_,_,_0{},_0{})); // (MMA, MMA_N, MMA_K, CONSUME_BLOCKS)
    Tensor tCrAR = thr_mma1.make_fragment_C(tCsAR(_,_,_,_,_0{})); // (MMA, MMA_M, MMA_N, CONSUME_BLOCKS)

    using s2r_atom_A = Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>;
    using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, prism_cute>;
    // auto base_n_blocks = std::conditional_t<(reconn_sz < 16), _1, _4>{};
    // auto total_R_blocks = base_n_blocks * c_blocks * n_groups;
    // CUTE_STATIC_ASSERT(has_single_bit(total_R_blocks)); // Ensure total_R_blocks is a power of 2
    // using s2r_atom_R = std::conditional_t<
    //                         (total_R_blocks == 1),
    //                         Copy_Atom<SM75_U32x1_LDSM_N, half_t>,
    //                         std::conditional_t<
    //                             (total_R_blocks == 2),
    //                             Copy_Atom<SM75_U32x2_LDSM_N, half_t>,
    //                             Copy_Atom<SM75_U32x4_LDSM_N, half_t>
    //                         >
    //                     >;
    using s2r_atom_R = std::conditional_t<
                            (reconn_sz < 16),
                            Copy_Atom<SM75_U32x1_LDSM_N, prism_cute>,
                            Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>
                        >;

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_A{}, single_warp_mma1);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(lane_idx);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA_warp_atom);  // (CPY, CPY_M, CPY_K, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K, CONSUME_BLOCKS)

    TiledCopy s2r_copy_r = make_tiled_copy_B(s2r_atom_R{}, single_warp_mma1);
    ThrCopy s2r_thr_copy_r = s2r_copy_r.get_slice(lane_idx);
    Tensor tXsR = s2r_thr_copy_r.partition_S(sR_warp_atom); // (CPY, CPY_N, CPY_K, CONSUME_BLOCKS, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrR = s2r_thr_copy_r.retile_D(tCrR); // (CPY, CPY_N, CPY_K, CONSUME_BLOCKS)

    TiledCopy r2s_copy_ar = make_tiled_copy_C(r2s_atom_AR{}, single_warp_mma1);
    ThrCopy r2s_thr_copy_ar = r2s_copy_ar.get_slice(lane_idx);
#if PRISM_DTYPE == 0
    // fp16: tCrAR is half_t — directly retile for the prism_cute (= half_t) atom.
    Tensor tXrAR = r2s_thr_copy_ar.retile_S(tCrAR); // (CPY, CPY_M, CPY_N, CONSUME_BLOCKS)
#endif
    Tensor tXsAR = r2s_thr_copy_ar.partition_D(sAR_warp_atom); // (CPY, CPY_M, CPY_N, CONSUME_BLOCKS, PIPELINE)

    int smem_pipe_read = 0;
    auto K_PIPE_MAX = size<3>(tAsA);
    int k_tile_next = 0;
    int k_tile_count = size<3>(tAgA);
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;
    auto K_BLOCK_MAX = size<4>(tCsA);
    auto K_PIPE2_MAX = size<4>(tCsAR);
    int ar_pipe_write = 0;
    // Tracks the gmem K-stripe index being processed by the current
    // while-loop iteration. Producer iterates exactly K/BLK_K times after the
    // K_PIPE_MAX-1 prefetch decrement, so outer_iter ∈ [0, K/BLK_K) is valid
    // for every iteration that contributes to the output.
    int outer_iter = 0;

    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
        if (thread_idx < size(copy_r)) {
            copy(copy_r, tRgR(_,_,_,k_tile_next), tRsR(_,_,_,k_pipe));
        }
        __syncwarp();
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        if (k_tile_count > 0) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
            if (thread_idx < size(copy_r)) {
                // Only copy R if the threadIdx.x is within the range of copy_r
                copy(copy_r, tRgR(_,_,_,k_tile_next), tRsR(_,_,_,smem_pipe_write));
            }
            __syncwarp();
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX-1>();
        asm volatile("bar.sync 14, %0;\n"
                            :
                            : "n"(n_threads1));
        // wait for the data to be ready in the smem
        // slice the shared memory for the reading of this round
        Tensor tXsA_p = tXsA(_,_,_,_,_,smem_pipe_read);
        Tensor tXsR_p = tXsR(_,_,_,_,_,smem_pipe_read);
        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            copy(s2r_atom_A{}, tXsA_p(_,_,_,_,j), tXrA);
            copy(s2r_atom_R{}, tXsR_p(_,_,_,_,j), tXrR);
            clear(tCrAR);
            CUTE_UNROLL
            for (int b = 0; b < c_blocks; ++b) {
                gemm(single_warp_mma1, tCrA(_,_,_,b), tCrR(_,_,_,b), tCrAR(_,_,_,b));
            }
            asm volatile("bar.sync %0, %1;\n"
                                :
                                : "r"(ar_pipe_write + K_PIPE2_MAX), "n"(n_threads_total)); // wait for the previous data to be consumed

#if PRISM_GATED
            // Apply SiLU gating + optional internal_bias to tCrAR.
            //
            // For fp16 (PRISM_DTYPE=0): MMA is F16-accum → tCrAR is half_t.
            //   Use packed half2 PTX activations.
            // For bf16 (PRISM_DTYPE=1): MMA is F32-accum → tCrAR is float.
            //   Compute silu in float, convert at the smem write boundary.
            //
            // The (m, n) layout of the C-fragment is the same in both cases
            // (SM80_16x8_Row): frag[0,1] = (row_lo, col_lo, col_lo+1);
            //                  frag[2,3] = (row_hi, col_lo, col_lo+1).
            // A-atom holds 4 elements (rs=8) or 8 elements (rs=16) per thread;
            // pick the right K-half via a_off = (mn % (rs/8)) * 4.
            {
                constexpr int RS_v = decltype(reconn_sz)::value;
                constexpr int CB_v = decltype(c_blocks)::value;
                constexpr int A_OFFSET_STEP = 4;
                constexpr int N_PER_A_ATOM = RS_v / 8;
                const int col_lo = (lane_idx & 0x3) * 2;
                const int k_block_base =
                    outer_iter * (int(K_BLOCK_MAX) * CB_v) + j * CB_v;
                const auto MMA_M_v = size<1>(tCrAR);
                const auto MMA_N_v = size<2>(tCrAR);
                #pragma unroll
                for (int b_idx = 0; b_idx < CB_v; ++b_idx) {
                    // Global K offset for this rs-block. Used by both bias
                    // indexing and dropout idx computation.
                    const int K_offset = (k_block_base + b_idx) * RS_v;
                    #pragma unroll
                    for (int mn = 0; mn < int(MMA_N_v); ++mn) {
                        // Per-thread (m, k, group) decomposition for the four
                        // C-fragment elements. Used by bias indexing,
                        // dropout, and the per-element value math below.
                        const int gn_lo = mn * 8 + col_lo;
                        const int gn_hi = gn_lo + 1;
                        const int g_lo = gn_lo / RS_v;
                        const int ri_lo = gn_lo - g_lo * RS_v;
                        const int g_hi = gn_hi / RS_v;
                        const int ri_hi = gn_hi - g_hi * RS_v;
#if PRISM_INTERNAL_BIAS
                        const int idx_lo = (cta_g_base + g_lo) * K_full + K_offset + ri_lo;
                        const int idx_hi = (cta_g_base + g_hi) * K_full + K_offset + ri_hi;
#endif
                        const int a_off = (mn % N_PER_A_ATOM) * A_OFFSET_STEP;
#if PRISM_DROPOUT
                        // Per-group seeds (within this CTA's groups).
                        const uint64_t seed_lo_g = (uint64_t)dropout_seeds[cta_g_base + g_lo];
                        const uint64_t seed_hi_g = (uint64_t)dropout_seeds[cta_g_base + g_hi];
#endif
#if PRISM_DTYPE == 0
                        // -- fp16 path: packed half2 silu --
#if PRISM_INTERNAL_BIAS
                        const __half b_lo_h = bias_ptr[idx_lo];
                        const __half b_hi_h = bias_ptr[idx_hi];
                        const __half2 bias2 = __halves2half2(b_lo_h, b_hi_h);
#endif
                        #pragma unroll
                        for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                            auto& ar2_lo = reinterpret_cast<__half2&>(tCrAR(0, mm, mn, b_idx));
                            auto& a2_lo  = reinterpret_cast<const __half2&>(tCrA(0 + a_off, mm, 0, b_idx));
                            auto& ar2_hi = reinterpret_cast<__half2&>(tCrAR(2, mm, mn, b_idx));
                            auto& a2_hi  = reinterpret_cast<const __half2&>(tCrA(2 + a_off, mm, 0, b_idx));
#if PRISM_INTERNAL_BIAS
                            ar2_lo = __hadd2(ar2_lo, bias2);
                            ar2_hi = __hadd2(ar2_hi, bias2);
#endif
                            ar2_lo = __hmul2(a2_lo, silu_h2(ar2_lo));
                            ar2_hi = __hmul2(a2_hi, silu_h2(ar2_hi));
#if PRISM_DROPOUT
                            // Same hash-derived (seed, m*K+k) mask as cublas for parity.
                            // Layout (cta_atom_layout_m = (n_warps, 8) row-major):
                            //   warp w covers CTA-M rows [w*8, w*8+8) within each
                            //   atom_n, atoms stride n_warps*8 along M. SM80 16x8 atom
                            //   groups two atom_n indices: mm*16+0..7 → atom_n=2*mm,
                            //   mm*16+8..15 → atom_n=2*mm+1.
                            constexpr int N_WARPS_v = decltype(n_warps)::value;
                            const int m_lo_dr = m_start + 2 * mm * (N_WARPS_v * 8) + int(warp_idx) * 8 + int(lane_idx >> 2);
                            const int m_hi_dr = m_lo_dr + (N_WARPS_v * 8);
                            const float u00 = prism_uniform_from_hash(seed_lo_g, (int64_t)m_lo_dr * K_full + K_offset + ri_lo);
                            const float u01 = prism_uniform_from_hash(seed_hi_g, (int64_t)m_lo_dr * K_full + K_offset + ri_hi);
                            const float u10 = prism_uniform_from_hash(seed_lo_g, (int64_t)m_hi_dr * K_full + K_offset + ri_lo);
                            const float u11 = prism_uniform_from_hash(seed_hi_g, (int64_t)m_hi_dr * K_full + K_offset + ri_hi);
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
                        // -- bf16 path: F32-accum, do silu in float --
#if PRISM_INTERNAL_BIAS
                        const float bias_lo_f = prism_to_float(bias_ptr[idx_lo]);
                        const float bias_hi_f = prism_to_float(bias_ptr[idx_hi]);
#endif
                        #pragma unroll
                        for (int mm = 0; mm < int(MMA_M_v); ++mm) {
                            float ar0 = tCrAR(0, mm, mn, b_idx);
                            float ar1 = tCrAR(1, mm, mn, b_idx);
                            float ar2 = tCrAR(2, mm, mn, b_idx);
                            float ar3 = tCrAR(3, mm, mn, b_idx);
                            float a0 = float(tCrA(0 + a_off, mm, 0, b_idx));
                            float a1 = float(tCrA(1 + a_off, mm, 0, b_idx));
                            float a2 = float(tCrA(2 + a_off, mm, 0, b_idx));
                            float a3 = float(tCrA(3 + a_off, mm, 0, b_idx));
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
                            constexpr int N_WARPS_v = decltype(n_warps)::value;
                            const int m_lo_dr = m_start + 2 * mm * (N_WARPS_v * 8) + int(warp_idx) * 8 + int(lane_idx >> 2);
                            const int m_hi_dr = m_lo_dr + (N_WARPS_v * 8);
                            const float u00 = prism_uniform_from_hash(seed_lo_g, (int64_t)m_lo_dr * K_full + K_offset + ri_lo);
                            const float u01 = prism_uniform_from_hash(seed_hi_g, (int64_t)m_lo_dr * K_full + K_offset + ri_hi);
                            const float u10 = prism_uniform_from_hash(seed_lo_g, (int64_t)m_hi_dr * K_full + K_offset + ri_lo);
                            const float u11 = prism_uniform_from_hash(seed_hi_g, (int64_t)m_hi_dr * K_full + K_offset + ri_hi);
                            h0 *= (u00 >= dropout_p) ? inv_keep : 0.0f;
                            h1 *= (u01 >= dropout_p) ? inv_keep : 0.0f;
                            h2 *= (u10 >= dropout_p) ? inv_keep : 0.0f;
                            h3 *= (u11 >= dropout_p) ? inv_keep : 0.0f;
#endif
                            tCrAR(0, mm, mn, b_idx) = h0;
                            tCrAR(1, mm, mn, b_idx) = h1;
                            tCrAR(2, mm, mn, b_idx) = h2;
                            tCrAR(3, mm, mn, b_idx) = h3;
                        }
#endif
                    }
                }
            }
#endif
#if PRISM_DTYPE == 0
            copy(r2s_atom_AR{}, tXrAR, tXsAR(_,_,_,_,ar_pipe_write));
#else
            // bf16: tCrAR is float — convert to bf16 fragment in registers, then r2s.
            {
                constexpr int frag_sz = decltype(size(tCrAR))::value;
                prism_cute tCrAR_bf16_storage[frag_sz];
                #pragma unroll
                for (int i = 0; i < frag_sz; ++i) {
                    tCrAR_bf16_storage[i] = prism_cute(float(tCrAR(i)));
                }
                auto tCrAR_bf16 = make_tensor(make_rmem_ptr(tCrAR_bf16_storage), tCrAR.layout());
                Tensor tXrAR_bf16 = r2s_thr_copy_ar.retile_S(tCrAR_bf16);
                copy(r2s_atom_AR{}, tXrAR_bf16, tXsAR(_,_,_,_,ar_pipe_write));
            }
#endif
            asm volatile("bar.arrive %0, %1;\n"
                                :
                                : "r"(ar_pipe_write), "n"(n_threads_total)); // signal that the data is ready
            ++ar_pipe_write;
            ar_pipe_write = (ar_pipe_write == K_PIPE2_MAX) ? 0 : ar_pipe_write; // wrap around the pipe index
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
        ++outer_iter;
    }
}

template <class TensorGB, class TensorSB, class TiledCopyB,
          class TensorSAR, class GroupSize,  class ConsumptionWidth,
          class TensorGC,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device__ static inline
void prism_arb(TensorGB const &gB, TensorSB &sB, TiledCopyB copy_b,
             TensorSAR const &sAR, GroupSize group_sz, ConsumptionWidth c_width,
             TensorGC &gC, int thread_idx,
             WarpLayoutStage1 warps_stage1, WarpLayoutStage2 warp_layout_stage2)
{
    using namespace cute;
    ThrCopy thr_copy_b = copy_b.get_slice(thread_idx);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    int warp_idx = thread_idx / 32;
    int lane_idx = thread_idx % 32;
    auto warp_coord = warp_layout_stage2.get_hier_coord(warp_idx); // (WARP_M, WARP_N)
    uint32_t warp_m = get<0>(warp_coord);
    uint32_t warp_n = get<1>(warp_coord);


    auto cta_atom_layout_m = make_layout(
        make_shape(
            size<0>(warp_layout_stage2), _8{}
        ),
        LayoutRight{}
    );

    constexpr uint32_t n_threads2 = size(warp_layout_stage2) * 32;
    constexpr uint32_t n_threads_total = size(warps_stage1) * 32 + n_threads2;
    auto tile_size_n = size<0>(gB);
    auto n_groups = max(tile_size_n / group_sz, _1{});
    auto warp_per_group = size<1>(warp_layout_stage2) / n_groups;
    auto warp_tile_n = tile_size_n / size<1>(warp_layout_stage2);
    uint32_t warp_group_id = warp_n / warp_per_group; // The group id of the current warp
    CUTE_STATIC_ASSERT_V(tile_size_n % size<1>(warp_layout_stage2) == _0{}); // Ensure the tile size is divisible by the number of warps in N dimension
    CUTE_STATIC_ASSERT_V(warp_tile_n <= group_sz); // Each warp should handle at most one group
    CUTE_STATIC_ASSERT_V(warp_tile_n >= _8{}); // The size of the atom should be at least 8
    CUTE_STATIC_ASSERT_V(warp_per_group >= _1{}); // Each group should have at least one warp
    Tensor sAR_warp_atom = group_modes<0,2>(
        logical_divide(
            sAR,
            make_tile(
                cta_atom_layout_m,
                make_layout(c_width)
            )
        )( // (((M_WARPS, 8), REST_M), (CONSUME_WIDTH, N_GROUPS), PIPELINE)
            make_coord(make_coord(warp_m, _), _), make_coord(_, warp_group_id), _
        ) // (8, REST_M, CONSUME_WIDTH, PIPELINE)
    ); // (WARP_M_REGION, CONSUME_WIDTH, PIPELINE)

    Tensor sB_warp_atom = logical_divide(
        sB,
        make_tile(
            make_layout(warp_tile_n),
            make_layout(c_width)
        )
    )( // ((TILE_N, WARP_ALONG_N), (CONSUME_WIDTH, BLOCKS_ALONG_K), PIPELINE)
        make_coord(_, warp_n), make_coord(_, _), _
    ); // (TILE_N, CONSUME_WIDTH, BLOCKS_ALONG_K, PIPELINE)

    Tensor gC_warp = group_modes<0,2>(
        logical_divide(
            gC,
            make_tile(
                cta_atom_layout_m,
                make_layout(warp_tile_n)
            )
        )( // (((M_WARPS, 8), REST_M), (TILE_N, WARP_ALONG_N))
            make_coord(make_coord(warp_m, _), _), make_coord(_, warp_n)
        ) // (8, REST_M, TILE_N)
    );  // (WARP_M_REGION, TILE_N)

    // For fp16: F16-accum ARB mma. For bf16: F32-accum.
    using mma_atom2 = prism_ar_atom<decltype(c_width)::value>;
    TiledMMA single_warp_mma2 = make_tiled_mma(
        mma_atom2{},
        make_layout(make_shape(_1{}, _1{})),
        Tile<decltype(size<0>(gC_warp)), decltype(size<1>(gC_warp))>{}
    );

    ThrMMA thr_mma2 = single_warp_mma2.get_slice(lane_idx);
    Tensor tCsAR = thr_mma2.partition_A(sAR_warp_atom); // (MMA, MMA_M, MMA_K, PIPELINE)
    Tensor tCsB  = thr_mma2.partition_B(sB_warp_atom);  // (MMA, MMA_N, MMA_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tCgC  = thr_mma2.partition_C(gC_warp);       // (MMA, MMA_M, MMA_N)
    
    Tensor tCrAR = thr_mma2.make_fragment_A(tCsAR(_,_,_,_0{})); // (MMA, MMA_M, MMA_K)
    Tensor tCrB  = thr_mma2.make_fragment_B(tCsB(_, _, _, _0{}, _0{})); // (MMA, MMA_N, MMA_K)
    Tensor tCrC  = thr_mma2.make_fragment_C(tCgC); // (MMA, MMA_M, MMA_N)
    clear(tCrC); // Clear the accumulators

    using s2r_atom_AR = Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>;
    using s2r_atom_B = Copy_Atom<SM75_U32x4_LDSM_N, prism_cute>;
    using r2g_atom_C = Copy_Atom<UniversalCopy<uint32_t>, prism_cute>;

    TiledCopy s2r_copy_ar = make_tiled_copy_A(s2r_atom_AR{}, single_warp_mma2);
    ThrCopy s2r_thr_copy_ar = s2r_copy_ar.get_slice(lane_idx);
    Tensor tXsAR = s2r_thr_copy_ar.partition_S(sAR_warp_atom); // (CPY, CPY_M, CPY_K, PIPELINE)
    Tensor tXrAR = s2r_thr_copy_ar.retile_D(tCrAR); // (CPY, CPY_M, CPY_K)

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_B{}, single_warp_mma2);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(lane_idx);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB_warp_atom); // (CPY, CPY_N, CPY_K, BLOCKS_ALONG_K, PIPELINE)
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    TiledCopy r2g_copy_C = make_tiled_copy_C(r2g_atom_C{}, single_warp_mma2);
    ThrCopy r2g_thr_copy_C = r2g_copy_C.get_slice(lane_idx);
    Tensor tXrC = r2g_thr_copy_C.retile_S(tCrC); // (CPY, CPY_M, CPY_N)
    Tensor tXgC = r2g_thr_copy_C.partition_D(gC_warp); // (CPY, CPY_M, CPY_N)

    int smem_pipe_read = 0;
    auto K_PIPE_MAX = size<3>(tBsB);
    int k_tile_next = 0; // Current tile index in gmem to read from
    int k_tile_count = size<3>(tBgB);
    int smem_pipe_write = K_PIPE_MAX - 1;
    auto K_BLOCK_MAX = size<3>(tCsB);

    auto K_PIPE2_MAX = size<3>(tCsAR);
    int ar_pipe_read = 0;
    
    CUTE_UNROLL
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; ++k_pipe) {
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    CUTE_UNROLL
    for (int bid = K_PIPE2_MAX; bid < K_PIPE2_MAX * 2; ++bid) {
        asm volatile("bar.arrive %0, %1;\n"
                        :
                        : "r"(bid), "n"(n_threads_total)); // signal the producer that current threads are ready to consume data
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX - 1)) {
        // slice the shared memory for the reading of this round
        Tensor tXsB_p = tXsB(_,_,_,_,smem_pipe_read);
        if (k_tile_count > 0) {
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        }
        cp_async_fence();
        // wait for the data to be ready in the smem
        cp_async_wait<K_PIPE_MAX-1>();
        asm volatile("bar.sync 15, %0;\n"
                            :
                            : "n"(n_threads2));

        CUTE_UNROLL
        for (int j = 0; j < K_BLOCK_MAX; ++j) {
            copy(s2r_atom_B{}, tXsB_p(_,_,_,j), tXrB);
            // wait for producer's data
            asm volatile("bar.sync %0, %1;\n"
                                :
                                : "r"(ar_pipe_read), "n"(n_threads_total));
            copy(s2r_atom_AR{}, tXsAR(_,_,_,ar_pipe_read), tXrAR);
            asm volatile("bar.arrive %0, %1;\n"
                                :
                                : "r"(ar_pipe_read + K_PIPE2_MAX), "n"(n_threads_total)); // signal the producer
            gemm(single_warp_mma2, tCrAR, tCrB, tCrC);
            ++ar_pipe_read;
            ar_pipe_read = (ar_pipe_read == K_PIPE2_MAX) ? 0 : ar_pipe_read; // wrap around the pipe index
        }
        --k_tile_count;
        ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        ++smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX) ? 0 : smem_pipe_read;
    }
    copy(tCrC, tCgC); // Copy the final result to gC
    
    // copy(r2g_atom_C{}, tXrC, tXgC); // Copy the final result to gC
}

template <class GridShape, class CtaTiler,
          class ALayout, class TiledCopyA,
          class RLayout, class TiledCopyR, class GroupSize, class ReconnectSize, class ConsumptionWidth,
          class BLayout, class TiledCopyB,
          class CLayout, class WarpLayoutStage1, class WarpLayoutStage2,
          class PipelineA_R, class PipelineAR, class PipelineB>
__global__ static __launch_bounds__(decltype((size(WarpLayoutStage1{}) + size(WarpLayoutStage2{})) * cute::_32{})::value)
void prism_device(GridShape grid_shape, CtaTiler cta_tiler,
                prism_native const *A, ALayout layout_a, TiledCopyA copy_a,
                prism_native const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_sz, ReconnectSize reconn_sz, ConsumptionWidth c_width,
                prism_native const *B, BLayout layout_b, TiledCopyB copy_b,
                prism_native       *C, CLayout layout_c, WarpLayoutStage1 warp_layout_stage1, WarpLayoutStage2 warp_layout_stage2,
                PipelineA_R pipeline_a_r, PipelineAR pipeline_ar, PipelineB pipeline_b,
                prism_native const *bias, int K_full, int strideA_per_group,
                // Dropout (only used when PRISM_DROPOUT=1).
                int64_t const* dropout_seeds, float dropout_p, float inv_keep)
{
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(is_integral<decltype(size<1>(cta_tiler))>{});
    static_assert(is_static<CtaTiler>::value);
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N_GROUPS, BLK_K_BLOCKS)
    // CUTE_STATIC_ASSERT_V(reconn_sz == _8{}); // Assume the reconnection size is 8, which is the size of the atom

    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % group_sz == _0{} || group_sz % size<1>(cta_tiler) == _0{}); // Ensure the N dimension of the CTA tiler is divisible by group_sz
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % c_width == _0{}); // Ensure the K dimension of the CTA tiler is divisible by c_width
    auto cta_n_groups = max(size<1>(cta_tiler) / group_sz, _1{}); // Number of groups in N dimension

    auto smem_atom = get_smem_atom(size<2>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(smem_atom) == size<2>(cta_tiler)); // Ensure the shared memory atom size matches the K dimension of the CTA tiler
 
    auto sA_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<0>(cta_tiler), // BLK_M
            size<2>(cta_tiler), // BLK_K
            pipeline_a_r           // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_M, BLK_K, PIPE)

    auto n_consume_blocks = c_width / reconn_sz;
    CUTE_STATIC_ASSERT_V(n_consume_blocks >= _1{}); // Ensure the number of consume blocks is at least 1
    CUTE_STATIC_ASSERT_V(c_width % reconn_sz == _0{}); // Ensure the consumption width is divisible by the reconnection size
    auto ar_smem_atom = get_smem_atom<false>(c_width * cta_n_groups);
    // for storing the intermediate result of AR
    auto sAR_layout = coalesce(tile_to_shape(
        ar_smem_atom,
        make_shape(
            size<0>(cta_tiler), // BLK_M
            c_width * cta_n_groups,
            pipeline_ar           // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_M, CONSUME_WIDTH * N_GROUPS, PIPE)

    auto sB_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            size<1>(cta_tiler), // BLK_N
            size<2>(cta_tiler), // BLK_K
            pipeline_b           // PIPE
        )
    ), make_tuple(_1{}, _1{}, _1{})); // (BLK_N, BLK_K, PIPE)

    auto sR_layout = coalesce(tile_to_shape(
        smem_atom,
        make_shape(
            cta_n_groups * reconn_sz,
            size<2>(cta_tiler),
            pipeline_a_r
            )
        ), make_tuple(_1{}, _1{}, _1{})); // (N_GROUPS * R, BLK_K, PIPE)

    // Shared memory buffers
    extern __shared__ prism_native smem[];
    prism_native* smemA = smem;
    prism_native* smemR = smemA + cosize_v<decltype(sA_layout)>;
    prism_native* smemB = smemR + cosize_v<decltype(sR_layout)>;
    prism_native* smemAR = smemB + cosize_v<decltype(sB_layout)>;

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR = make_tensor(make_smem_ptr(smemR), sR_layout); // (GROUP * R, BLOCK * R, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K, PIPE)
    Tensor sAR = make_tensor(make_smem_ptr(smemAR), sAR_layout); // (BLK_M, RECONN_SZ)

    // Full and Tiled Tensors
    Tensor mR = make_tensor(make_gmem_ptr(R), layout_r); // (GROUP * R, BLOCK * R)
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_b); // (N,K)
    // C is reinterpreted as the cutlass-typed pointer so float→bf16 conversion
    // works at the final r2g write (cutlass::bfloat16_t has a constructor from
    // float; the raw __nv_bfloat16 type does not).
    Tensor mC = make_tensor(make_gmem_ptr(reinterpret_cast<prism_cute*>(C)), layout_c);

    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    auto cta_coord =  append<3>(grid_coord, _);

    // First group covered by this CTA tile, in mR's group-row units. Works for
    // both bN >= group_sz (cta covers multiple groups) and bN < group_sz
    // (multiple ctas share one group).
    int cta_g_base = (int(get<1>(cta_coord)) * int(size<1>(cta_tiler))) / int(group_sz);
    // CTA's M offset (start row in the global M dim). Used by dropout to
    // compute m_global per fragment element.
    int cta_m_base = int(get<0>(cta_coord)) * int(size<0>(cta_tiler));

    // When shuffle is on (strideA_per_group != 0), each group has its own
    // (M, K) A_perm slice; advance A to the slice for cta_g_base. When off,
    // strideA_per_group == 0 → all CTAs see the same A.
    prism_native const* A_for_cta = A + (long long)cta_g_base * strideA_per_group;
    Tensor mA = make_tensor(make_gmem_ptr(A_for_cta), layout_a); // (M,K)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});  // (BLK_M,BLK_N)
    Tensor gR = local_tile(mR,
        make_shape(
            cta_n_groups * reconn_sz,
            size<2>(cta_tiler)
        ), make_coord(get<1>(cta_coord) * ratio(size<1>(cta_tiler), max(size<1>(cta_tiler), group_sz)), _));

    int thread_idx = threadIdx.x;
    int stage2_threads = size(warp_layout_stage2) * 32;
    if (thread_idx < stage2_threads) {
        // Call the ARB kernel
        prism_arb(
            gB, sB, copy_b,
            sAR, group_sz, c_width,
            gC, thread_idx,
            warp_layout_stage1, warp_layout_stage2
        );
    } else {
        // Call the AR kernel
        prism_ar(
            gA, sA, copy_a,
            gR, sR, copy_r, reconn_sz, n_consume_blocks,
            sAR, thread_idx - stage2_threads,
            warp_layout_stage1, warp_layout_stage2,
            bias, cta_g_base, K_full,
            dropout_seeds, dropout_p, inv_keep, cta_m_base
        );
    }
}

// // Setup params for a TN GEMM, K-Major inputs
template <class KernelParams>
void prism_tn(int m, int n, int k,
        prism_native const* A, int ldA,
        prism_native const* B, int ldB,
        prism_native const* R, int ldR,
        prism_native      * C, int ldC,
        prism_native const* bias,
        int strideA_per_group,
        // dropout (only used when PRISM_DROPOUT=1; pass nullptr/0/1 otherwise)
        int64_t const* dropout_seeds,
        float dropout_p,
        float inv_keep,
        cudaStream_t stream)
{
    using namespace cute;
    using CompParams = CurrCompParams;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

    // Define CTA tile sizes (static)
    auto group_size = Int<KernelParams::group_size>{};
    auto reconn_sz = Int<KernelParams::reconn_sz>{};
    auto bM = Int<CompParams::bM>{};
    auto bN = Int<CompParams::bN>{};
    auto bK = Int<CompParams::bK>{};
    auto c_width = Int<CompParams::c_width>{};
    auto bN_group = max(bN / group_size, _1{});
    auto cta_tiler = make_shape(bM, bN, bK);                   // (CTA_M, CTA_N, CTA_K)
    auto bP_a_r = Int<CompParams::bP_a_r>{};  // Pipeline for A and R
    auto bP_ar = Int<CompParams::bP_ar>{};  // Pipeline for AR
    auto bP_b = Int<CompParams::bP_b>{};  // Pipeline for B
    int n_groups = N / group_size;
    auto warp_layout1 = typename CompParams::warp_layout_ar{};
    auto warp_layout2 = typename CompParams::warp_layout_arb{};

    // Define the gmem layouts
    auto A_layout = make_layout(
        make_shape(M, K),
        make_stride(ldA, Int<1>{})
    );

    auto B_layout = make_layout(
        make_shape(N, K),
        make_stride(ldB, Int<1>{})
    );

    auto R_layout = make_layout(
        make_shape(n_groups * reconn_sz, K),
        make_stride(ldR, Int<1>{})
    );

    auto C_layout = make_layout(
        make_shape(M, N),
        make_stride(ldC, Int<1>{})
    );

    TiledCopy copyA = cp_layout<uint128_t, prism_native>(bM, bK, size(warp_layout1) * _32{});
    TiledCopy copyR = cp_layout<uint128_t, prism_native>(bN_group * reconn_sz, bK, size(warp_layout1) * _32{});
    TiledCopy copyB = cp_layout<uint128_t, prism_native>(bN, bK, size(warp_layout2) * _32{});

    dim3 dimBlock((size(warp_layout1) + size(warp_layout2)) * _32{});
    auto grid_shape = make_shape(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    dim3 dimGrid(get<0>(grid_shape) * get<1>(grid_shape));

    uint32_t smem_size = get_smem_size(cta_tiler, group_size, reconn_sz, c_width, bP_a_r, bP_ar, bP_b);

    #ifdef DEBUG
    printf("dimGrid: (%d, %d), dimBlock: (%d, %d)\n",
            dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    #endif
    // Opt in to >48KB dynamic shared memory before launch. Required whenever
    // smem_size exceeds the 48KB default cap — e.g. group_size < bN packs
    // n_groups>1 per CTA, doubling the R/AR buffers. Without this the launch
    // fails with cudaErrorInvalidValue ("invalid argument"), which (since the
    // raw <<<>>> launch error is unchecked) only surfaces at the next CUDA op.
    // The backward kernels already do this (cute_prism_backward_dadr.cu:951,
    // cute_prism_backward_db.cu:572); the forward path had been missing it.
    auto prism_kernel = &prism_device<
        decltype(grid_shape), decltype(cta_tiler),
        decltype(A_layout), decltype(copyA),
        decltype(R_layout), decltype(copyR),
        decltype(group_size), decltype(reconn_sz), decltype(c_width),
        decltype(B_layout), decltype(copyB),
        decltype(C_layout), decltype(warp_layout1), decltype(warp_layout2),
        decltype(bP_a_r), decltype(bP_ar), decltype(bP_b)>;
    cudaFuncSetAttribute(prism_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    prism_kernel<<<dimGrid, dimBlock, smem_size, stream>>>(
        grid_shape, cta_tiler,
        A, A_layout, copyA,
        R, R_layout, copyR, group_size, reconn_sz, c_width,
        B, B_layout, copyB,
        C, C_layout, warp_layout1, warp_layout2,
        bP_a_r, bP_ar, bP_b,
        bias, K, strideA_per_group,
        dropout_seeds, dropout_p, inv_keep
    );
}

template void prism_tn<CurrKernelParams>(int m, int n, int k,
        prism_native const* A, int ldA,
        prism_native const* B, int ldB,
        prism_native const* R, int ldR,
        prism_native      * C, int ldC,
        prism_native const* bias,
        int strideA_per_group,
        int64_t const* dropout_seeds,
        float dropout_p,
        float inv_keep,
        cudaStream_t stream);