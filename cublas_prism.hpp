#pragma once

#include <cublas_v2.h>
#include "common.hpp"
#include <thrust/device_vector.h>

#define GEMM_CHECK_CUBLAS(call)                                                                   \
    do {                                                                                          \
        cublasStatus_t status = call;                                                             \
        if (GEMM_UNLIKELY(status != CUBLAS_STATUS_SUCCESS)) {                                     \
            throw std::runtime_error("CUBLAS call failed with status " + std::to_string(status)); \
        }                                                                                         \
    } while (0)
  
void getCublasTensorOpHandle(cublasHandle_t* handle) {
    GEMM_CHECK_CUBLAS(cublasCreate(handle));
    GEMM_CHECK_CUBLAS(cublasSetMathMode(*handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

void transform_weight(
  const thrust::device_vector<half> &d_B,
  const thrust::device_vector<half> &d_R,
        thrust::device_vector<half> &d_B_transformed,
  int group_sz, int k, int n_groups, int reconn_sz,
  cublasHandle_t* handle
) {
  half alpha = 1.0;
  half beta = 0.0;
  for (int i = 0; i < n_groups; ++i) {
    GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
      *handle, CUBLAS_OP_N, CUBLAS_OP_N,
      reconn_sz, group_sz, reconn_sz, &alpha,
      d_R.data().get() + i * reconn_sz * k, k,
      reconn_sz,
      d_B.data().get() + i * group_sz * k, k,
      reconn_sz,
      &beta,
      d_B_transformed.data().get() + i * group_sz * k, k,
      reconn_sz,
      k / reconn_sz
    ));
  }
}

void cublas_prism(
  const thrust::device_vector<half> &d_A,
  const thrust::device_vector<half> &d_R,
  const thrust::device_vector<half> &d_B,
        thrust::device_vector<half> &d_C,
  int m, int group_size, int n_groups, int k,
  int reconn_sz,
  cublasHandle_t* handle,
  bool RW_mode
) {
  half alpha = 1.0;
  half beta = 0.0;
  int n = group_size * n_groups;
  if (RW_mode) {
    // printf("RW mode\n");
    thrust::device_vector<half> d_Bp(n * k);
    transform_weight(d_B, d_R, d_Bp, group_size, k, n_groups, reconn_sz, handle);
    GEMM_CHECK_CUBLAS(cublasHgemm(
        *handle, CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k, &alpha, d_Bp.data().get(), k, d_A.data().get(), k, &beta, d_C.data().get(), n)
      );
  } else {
    // printf("AR mode\n");
    thrust::device_vector<half> d_AR(m * k);
    for (int i = 0; i < n_groups; ++i) {
      GEMM_CHECK_CUBLAS(cublasHgemmStridedBatched(
        *handle, CUBLAS_OP_T, CUBLAS_OP_N,
        reconn_sz, m, reconn_sz, &alpha,
        d_R.data().get() + i * reconn_sz * k, k,
        reconn_sz,
        d_A.data().get(), k,
        reconn_sz,
        &beta,
        d_AR.data().get(), k,
        reconn_sz,
        k / reconn_sz
      ));
      GEMM_CHECK_CUBLAS(cublasHgemm(
        *handle, CUBLAS_OP_T, CUBLAS_OP_N,
        group_size, m, k, &alpha, d_B.data().get() + i * group_size * k, k, d_AR.data().get(), k, &beta, d_C.data().get() + i * group_size, n)
      );
    }
  }
}