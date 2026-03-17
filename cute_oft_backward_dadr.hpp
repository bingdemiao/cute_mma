#pragma once
#include <cuda_fp16.h>

// CuTe backward pass for OFT — dA + dR kernel launcher.

// Kernel 1: dA + dR from dC, A, B, R (producer-consumer with MMA)
void oft_backward_dA_dR_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* B,  int ldB,
    half const* R,  int ldR,
    half* dA, int ldDA,
    half* dR, int ldDR,
    cudaStream_t stream);
