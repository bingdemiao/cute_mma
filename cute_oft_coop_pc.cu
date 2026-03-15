#include "cute_oft_coop_pc.hpp"
#include <oft_config.hpp>
#include "cute_oft_util.hpp"
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
    return (size_A + size_B + size_R + size_AR) * sizeof(half);
}

template <class TensorGA, class TensorSA, class TiledCopyA,
          class TensorGR, class TensorSR, class TiledCopyR, class ReconnectSize, class ConsumptionBlocks,
          class TensorSAR,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device__ static inline
void oft_ar(TensorGA const &gA, TensorSA &sA, TiledCopyA copy_a,
            TensorGR const &gR, TensorSR &sR, TiledCopyR copy_r, ReconnectSize reconn_sz, ConsumptionBlocks c_blocks,
            TensorSAR &sAR, int thread_idx,
            WarpLayoutStage1 warps_stage1, WarpLayoutStage2 warps_stage2)
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

    
    using mma_atom1 = std::conditional_t<(reconn_sz < 16), MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>, MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;

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

    using s2r_atom_A = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using r2s_atom_AR = Copy_Atom<UniversalCopy<uint32_t>, half_t>;
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
                            Copy_Atom<SM75_U32x1_LDSM_N, half_t>,
                            Copy_Atom<SM75_U32x4_LDSM_N, half_t>
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
    Tensor tXrAR = r2s_thr_copy_ar.retile_S(tCrAR); // (CPY, CPY_M, CPY_N, CONSUME_BLOCKS)
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

            copy(r2s_atom_AR{}, tXrAR, tXsAR(_,_,_,_,ar_pipe_write));

#if OFT_GATED
            // Apply SiLU gating in smem: sAR(m,c) = sA(m,a_col) * SiLU(sAR(m,c))
            // Uses packed f16x2 + tanh.approx.f16x2:
            //   sigma = 0.5*(1+tanh(ar*0.5)), silu = ar*sigma
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_threads1)); // ensure r2s complete
            {
                constexpr int n_grp = decltype(n_groups)::value;
                constexpr int n_cb = decltype(c_blocks)::value;
                constexpr int rs = decltype(reconn_sz)::value;
                constexpr int blk_m = decltype(size<0>(sAR))::value;
                constexpr int total = blk_m * rs * n_cb * n_grp;
                // sAR column c maps to (r, b, g) via producer_layout_n strides:
                //   g = c % n_grp, b = (c / n_grp) % n_cb, r = c / (n_cb * n_grp)
                // Corresponding sA column: j * c_width + b * rs + r
                for (int idx = 2 * thread_idx; idx < total; idx += 2 * n_threads1) {
                    int m0 = idx / (rs * n_cb * n_grp);
                    int c0 = idx % (rs * n_cb * n_grp);
                    int g0 = c0 % n_grp, b0 = (c0 / n_grp) % n_cb, r0 = c0 / (n_cb * n_grp);
                    int a_col0 = j * rs * n_cb + r0 * n_cb + b0;
                    int m1 = (idx + 1) / (rs * n_cb * n_grp);
                    int c1 = (idx + 1) % (rs * n_cb * n_grp);
                    int g1 = c1 % n_grp, b1 = (c1 / n_grp) % n_cb, r1 = c1 / (n_cb * n_grp);
                    int a_col1 = j * rs * n_cb + r1 * n_cb + b1;
                    __half2 ar2 = load_half2(sAR(m0, c0, ar_pipe_write),
                                             sAR(m1, c1, ar_pipe_write));
                    __half2 a2 = load_half2(sA(m0, a_col0, smem_pipe_read),
                                            sA(m1, a_col1, smem_pipe_read));
                    __half2 result = __hmul2(a2, silu_h2(ar2));
                    store_half2(sAR(m0, c0, ar_pipe_write),
                                sAR(m1, c1, ar_pipe_write), result);
                }
            }
            asm volatile("bar.sync 14, %0;\n" : : "n"(n_threads1)); // ensure gating complete
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
    }
}

template <class TensorGB, class TensorSB, class TiledCopyB,
          class TensorSAR, class GroupSize,  class ConsumptionWidth,
          class TensorGC,
          class WarpLayoutStage1, class WarpLayoutStage2>
__device__ static inline
void oft_arb(TensorGB const &gB, TensorSB &sB, TiledCopyB copy_b,
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

    using mma_atom2 = std::conditional_t<(c_width < 16), MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>, MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>>;
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

    using s2r_atom_AR = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using s2r_atom_B = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    using r2g_atom_C = Copy_Atom<UniversalCopy<uint32_t>, half_t>;

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
void oft_device(GridShape grid_shape, CtaTiler cta_tiler,
                half const *A, ALayout layout_a, TiledCopyA copy_a,
                half const *R, RLayout layout_r, TiledCopyR copy_r, GroupSize group_sz, ReconnectSize reconn_sz, ConsumptionWidth c_width,
                half const *B, BLayout layout_b, TiledCopyB copy_b,
                half       *C, CLayout layout_c, WarpLayoutStage1 warp_layout_stage1, WarpLayoutStage2 warp_layout_stage2,
                PipelineA_R pipeline_a_r, PipelineAR pipeline_ar, PipelineB pipeline_b)
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
    extern __shared__ half smem[]; // Shared memory buffer
    // Cast the shared memory buffer to half type
    half* smemA = smem;
    half* smemR = smemA + cosize_v<decltype(sA_layout)>;
    half* smemB = smemR + cosize_v<decltype(sR_layout)>;
    half* smemAR = smemB + cosize_v<decltype(sB_layout)>;

    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout); // (BLK_M, BLK_K, PIPE)
    Tensor sR = make_tensor(make_smem_ptr(smemR), sR_layout); // (GROUP * R, BLOCK * R, PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout); // (BLK_N, BLK_K, PIPE)
    Tensor sAR = make_tensor(make_smem_ptr(smemAR), sAR_layout); // (BLK_M, RECONN_SZ)

    // Full and Tiled Tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), layout_a); // (M,K)
    Tensor mR = make_tensor(make_gmem_ptr(R), layout_r); // (GROUP * R, BLOCK * R)
    Tensor mB = make_tensor(make_gmem_ptr(B), layout_b); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), layout_c); // (M,N)

    auto grid_coord = z_curve(grid_shape, blockIdx.x);
    // auto grid_coord = make_tuple(blockIdx.x / get<1>(grid_shape),
    //                              blockIdx.x % get<1>(grid_shape)); // (m,n)
    auto cta_coord =  append<3>(grid_coord, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1,X,_1>{});  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X,_1,_1>{});  // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1,X>{});  // (BLK_M,BLK_N)
    Tensor gR = local_tile(mR,
        make_shape(
            cta_n_groups * reconn_sz,
            size<2>(cta_tiler)
        ), make_coord(get<1>(cta_coord) * ratio(size<1>(cta_tiler), max(size<1>(cta_tiler), group_sz)), _)); // (N_GROUPS * R, BLK_K, k)， assuming one thread block would handle at least one group of R
    
    int thread_idx = threadIdx.x;
    int stage2_threads = size(warp_layout_stage2) * 32;
    if (thread_idx < stage2_threads) {
        // Call the ARB kernel
        oft_arb(
            gB, sB, copy_b,
            sAR, group_sz, c_width,
            gC, thread_idx,
            warp_layout_stage1, warp_layout_stage2
        );
    } else {
        // Call the AR kernel
        oft_ar(
            gA, sA, copy_a,
            gR, sR, copy_r, reconn_sz, n_consume_blocks,
            sAR, thread_idx - stage2_threads,
            warp_layout_stage1, warp_layout_stage2
        );
    }
}

// // Setup params for a TN GEMM, K-Major inputs
template <class KernelParams>
void oft_tn(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
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

    TiledCopy copyA = cp_layout<uint128_t, half>(bM, bK, size(warp_layout1) * _32{});
    TiledCopy copyR = cp_layout<uint128_t, half>(bN_group * reconn_sz, bK, size(warp_layout1) * _32{});
    TiledCopy copyB = cp_layout<uint128_t, half>(bN, bK, size(warp_layout2) * _32{});

    dim3 dimBlock((size(warp_layout1) + size(warp_layout2)) * _32{});
    auto grid_shape = make_shape(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    dim3 dimGrid(get<0>(grid_shape) * get<1>(grid_shape));

    uint32_t smem_size = get_smem_size(cta_tiler, group_size, reconn_sz, c_width, bP_a_r, bP_ar, bP_b);

    #ifdef DEBUG
    printf("dimGrid: (%d, %d), dimBlock: (%d, %d)\n",
            dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
    #endif
    oft_device<<<dimGrid, dimBlock, smem_size, stream>>>(
        grid_shape, cta_tiler,
        A, A_layout, copyA,
        R, R_layout, copyR, group_size, reconn_sz, c_width,
        B, B_layout, copyB,
        C, C_layout, warp_layout1, warp_layout2,
        bP_a_r, bP_ar, bP_b
    );
}

template void oft_tn<CurrKernelParams>(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
        cudaStream_t stream);