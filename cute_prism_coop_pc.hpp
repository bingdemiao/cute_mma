#pragma once
#include <cute/tensor.hpp>
#include "cute_prism_dtype.hpp"

#ifndef PRISM_GROUP_SIZE
#define PRISM_GROUP_SIZE 256
#endif

#ifndef PRISM_RECONN_SIZE
#define PRISM_RECONN_SIZE 8
#endif

#ifndef PRISM_GATED
#define PRISM_GATED 0
#endif

#ifndef PRISM_INTERNAL_BIAS
#define PRISM_INTERNAL_BIAS 0
#endif

struct CurrKernelParams {
    static constexpr unsigned int group_size = PRISM_GROUP_SIZE;
    static constexpr unsigned int reconn_sz = PRISM_RECONN_SIZE;
    static constexpr bool gated = PRISM_GATED;
    static constexpr bool internal_bias = PRISM_INTERNAL_BIAS;
};

template <class KernelParams>
void prism_tn(int m, int n, int k,
        prism_native const* A, int ldA,
        prism_native const* B, int ldB,
        prism_native const* R, int ldR,
        prism_native      * C, int ldC,
        prism_native const* bias,  // (n_groups, K) row-major, or nullptr when PRISM_INTERNAL_BIAS=0
        // Per-group A stride (in elements). When 0, all groups share the
        // same A. When non-zero (typically M*K), each group uses its own
        // (M, K) A_perm slice at offset cta_g_base * strideA_per_group.
        int strideA_per_group,
        // Dropout (only used when PRISM_DROPOUT=1; pass nullptr/0/1 otherwise).
        // dropout_seeds: (n_groups,) int64 — one uint64 seed per group.
        // dropout_p:    float — drop probability.
        // inv_keep:     1 / (1 - p).
        int64_t const* dropout_seeds,
        float dropout_p,
        float inv_keep,
        cudaStream_t stream = 0);