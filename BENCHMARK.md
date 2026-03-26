# OFT Kernel Performance

Benchmarked on NVIDIA RTX 3080 Ti (SM86), M=N=K=8192, group_size=256, reconn_sz=8, float16.

## Reference: Regular Linear Layer (cuBLAS GEMM)

| Operation | Time (ms) | TFLOP/s |
|-----------|-----------|-----------|
| Forward (C = A @ W^T) | 15.5 | 72.82 |
| Backward dA (dA = dC @ W) | 16.0 | 68.72 |
| Backward dW (dW = dC^T @ A) | 15.7 | 70.03 |
| **Total fwd+bwd** | **47.2** | **69.88** |

## OFT Kernel Performance

| Operation | cute (ms) | cublas (ms) | pytorch (ms) |
|-----------|-----------|-------------|--------------|
| Forward (non-gated) | 15.1 | 45.1 | 91.3 |
| Forward (gated) | 13.9 | 61.3 | 106.3 |
| Backward (non-gated, dA+dR) | 31.0 | 115.4 | 3335.8 |
| Backward (gated, dA+dR+dB) | 75.2 | 231.3 | 3238.7 |

### Backward Kernel Breakdown (cute)

| Kernel | non-gated (ms) | gated (ms) |
|--------|----------------|------------|
| dAdR | 31.0 | 35.7 |
| dB | — | 39.5 |

## Speedup vs Backends

| Operation | cute vs cublas | cute vs pytorch |
|-----------|---------------|-----------------|
| Forward (non-gated) | 3.0× | 6.0× |
| Forward (gated) | 4.4× | 7.7× |
| Backward (non-gated) | 3.7× | 108× |
| Backward (gated) | 3.1× | 43× |

## Overhead vs Regular Linear Layer

| | cute (ms) | Linear (ms) | Overhead |
|---|-----------|-------------|----------|
| Forward (non-gated) | 15.1 | 15.5 | 0.97× (faster) |
| Forward (gated) | 13.9 | 15.5 | 0.90× (faster) |
| Fwd+Bwd (non-gated) | 46.1 | 47.2 | 0.98× (faster) |
| Fwd+Bwd (gated) | 89.1 | 47.2 | 1.89× |

---

## Backward dA+dR (Producer-Consumer Kernel, reconn_sz=16)

**Setup**: M=N=K=8192, group_size=128, reconn_sz=16, float16, RTX 3080 Ti

### FLOP Breakdown

| Operation | Description | TFLOP |
|-----------|-------------|-------|
| dH GEMM | `dH = dC @ B` per group (reduces over N) | 1.10 |
| dA (reconn MMA) | `dA_b += dH_b @ R_b` per reconn block | 0.14 |
| AR recomputation | `AR_b = A_b @ R_b^T` (recomputed for dR) | 0.14 |
| dR (M-reduction) | `dR_b = dH_b^T @ A_b` per reconn block | 0.14 |
| **Total** | | **1.51** |

### Performance

Best autotuned config: `bM=64 bK=128 bN=16 bP_dc_b=2 bP_dh=1 n_buf_slots=8 warp_layout_ar=(2,2) warp_layout_arb=(2,)`

| Backend | Time (ms) | TFLOPS | Speedup vs pytorch |
|---------|-----------|--------|-------------------|
| **cute (PC, autotuned)** | **40.82** | **37.0** | **82.3x** |
| cublas | 148.65 | 10.2 | 22.6x |
| pytorch | 3359.64 | 0.4 | 1.0x |

### Efficiency Analysis

| Metric | Value |
|--------|-------|
| Peak FP16 tensor core (spec) | 136 TFLOPS |
| Achieved throughput | 37.0 TFLOPS |
| **Efficiency** | **27.2%** |

The 27% efficiency is expected: the dH GEMM (73% of FLOPs) is a non-standard GEMM with per-group reconn-block epilogues that limit occupancy. The producer-consumer architecture hides consumer gmem store latency (dR atomicAdd) behind the producer's next-group GEMM. The kernel is **3.6x faster than cuBLAS** for this structured operation.

### Correctness (autotuned config)

Verified against PyTorch reference:
- dA mean relative error: 0.52%
- dR mean relative error: 0.60%

Within FP16 precision bounds for atomic reductions.
