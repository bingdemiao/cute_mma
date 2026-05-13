#pragma once
#include <cuda_fp16.h>

// CuTe backward pass for Prism — dA + dR kernel launcher.

// Kernel 1: dA + dR from dC, A, B, R (producer-consumer with MMA)
#include "cute_prism_dtype.hpp"

void prism_backward_dA_dR_launch(
    int m, int n, int k,
    prism_native const* dC, int ldDC,
    prism_native const* A,  int ldA,
    prism_native const* B,  int ldB,
    prism_native const* R,  int ldR,
    prism_native const* bias,  // (n_groups, K) row-major; nullptr unless PRISM_INTERNAL_BIAS=1
    prism_native* dA, int ldDA,
    prism_native* dR, int ldDR,
    prism_native* dInternalBias,  // (n_groups, K); nullptr unless PRISM_INTERNAL_BIAS=1
    // input_shuffle: per-group A and per-group dA strides. 0 → off.
    int strideA_per_group,
    int strideDA_per_group,
    // Dropout (only used when PRISM_DROPOUT=1).
    int64_t const* dropout_seeds, float dropout_p, float inv_keep,
    cudaStream_t stream);
