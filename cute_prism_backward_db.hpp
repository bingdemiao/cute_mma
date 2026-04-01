#pragma once
#include <cuda_fp16.h>

// CuTe backward pass for OFT — dB kernel launcher.

// Kernel 2: dB from dC, A, R (recomputes AR; producer-consumer with MMA)
void prism_backward_dB_launch(
    int m, int n, int k,
    half const* dC, int ldDC,
    half const* A,  int ldA,
    half const* R,  int ldR,
    half* dB, int ldDB,
    cudaStream_t stream);
