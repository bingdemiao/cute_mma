#pragma once
#include <cuda_fp16.h>

// CuTe backward pass for OFT — split into independent dA+dR and dB launchers.

// Kernel 1: dA + dR from dC, A, B, R (producer-consumer with MMA)
void oft_backward_dA_dR_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* B,  int ldB,
    half const* R,  int ldR,
    half* dA, int ldDA,
    half* dR, int ldDR,
    bool gated,
    cudaStream_t stream);

// Kernel 2: dB from dC, A, R (recomputes AR; producer-consumer with MMA)
void oft_backward_dB_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* R,  int ldR,
    half* dB, int ldDB,
    bool gated,
    cudaStream_t stream);
