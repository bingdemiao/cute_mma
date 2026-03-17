# OFT Kernel Performance

Benchmarked on NVIDIA RTX 3080 Ti (SM86), M=N=K=8192, group_size=256, reconn_sz=8, float16.

## Reference: Regular Linear Layer (cuBLAS GEMM)

| Operation | Time (ms) |
|-----------|-----------|
| Forward (C = A @ W^T) | 16.1 |
| Backward dA (dA = dC @ W) | 16.0 |
| Backward dW (dW = dC^T @ A) | 15.7 |
| **Total fwd+bwd** | **47.8** |

## OFT Kernel Performance

| Operation | cute (ms) | cublas (ms) | pytorch (ms) |
|-----------|-----------|-------------|--------------|
| Forward (non-gated) | 14.8 | 45.1 | 91.3 |
| Forward (gated) | 13.0 | 61.3 | 106.3 |
| Backward (non-gated, dA+dR) | 30.8 | 115.4 | 3335.8 |
| Backward (gated, dA+dR+dB) | 72.0 | 231.3 | 3238.7 |

## Speedup vs Backends

| Operation | cute vs cublas | cute vs pytorch |
|-----------|---------------|-----------------|
| Forward (non-gated) | 3.0× | 6.2× |
| Forward (gated) | 4.7× | 8.2× |
| Backward (non-gated) | 3.7× | 108× |
| Backward (gated) | 3.2× | 45× |

## Overhead vs Regular Linear Layer

| | cute (ms) | Linear (ms) | Overhead |
|---|-----------|-------------|----------|
| Forward (non-gated) | 14.8 | 16.1 | 0.92× (faster) |
| Forward (gated) | 13.0 | 16.1 | 0.81× (faster) |
| Fwd+Bwd (non-gated) | 45.6 | 47.8 | 0.95× (faster) |
| Fwd+Bwd (gated) | 85.0 | 47.8 | 1.78× |
