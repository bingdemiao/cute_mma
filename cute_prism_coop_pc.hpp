#pragma once
#include <cute/tensor.hpp>

#ifndef PRISM_GROUP_SIZE
#define PRISM_GROUP_SIZE 256
#endif

#ifndef PRISM_RECONN_SIZE
#define PRISM_RECONN_SIZE 8
#endif

#ifndef PRISM_GATED
#define PRISM_GATED 0
#endif

struct CurrKernelParams {
    static constexpr unsigned int group_size = PRISM_GROUP_SIZE;
    static constexpr unsigned int reconn_sz = PRISM_RECONN_SIZE;
    static constexpr bool gated = PRISM_GATED;
};

template <class KernelParams>
void prism_tn(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
        cudaStream_t stream = 0);