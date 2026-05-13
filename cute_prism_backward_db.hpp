#pragma once
#include <cuda_fp16.h>

// CuTe backward pass for Prism — dB kernel launcher.

// Kernel 2: dB from dC, A, R (recomputes AR; producer-consumer with MMA)
#include "cute_prism_dtype.hpp"

void prism_backward_dB_launch(
    int m, int n, int k,
    prism_native const* dC, int ldDC,
    prism_native const* A,  int ldA,
    prism_native const* R,  int ldR,
    prism_native const* bias,  // (n_groups, K) row-major; nullptr unless PRISM_INTERNAL_BIAS=1
    prism_native* dB, int ldDB,
    // Per-group A stride (in elements). 0 → all groups share A. M*K → each
    // group g uses A[g*M*K..] (input_shuffle path).
    int strideA_per_group,
    // Dropout (only used when PRISM_DROPOUT=1). Seeds must match the forward
    // call so the same hash-derived mask is replayed on the recomputed gate.
    int64_t const* dropout_seeds, float dropout_p, float inv_keep,
    cudaStream_t stream);
