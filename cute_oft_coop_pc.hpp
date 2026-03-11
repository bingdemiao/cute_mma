#pragma once
#include <cute/tensor.hpp>

#ifndef OFT_GROUP_SIZE
#define OFT_GROUP_SIZE 256
#endif

#ifndef OFT_RECONN_SIZE
#define OFT_RECONN_SIZE 8
#endif

struct CurrKernelParams {
    static constexpr unsigned int group_size = OFT_GROUP_SIZE;
    static constexpr unsigned int reconn_sz = OFT_RECONN_SIZE;
};

template <class KernelParams>
void oft_tn(int m, int n, int k,
        half const* A, int ldA,
        half const* B, int ldB,
        half const* R, int ldR,
        half      * C, int ldC,
        cudaStream_t stream = 0);