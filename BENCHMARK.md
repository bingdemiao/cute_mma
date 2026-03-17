# OFT Kernel Performance

Benchmarked on NVIDIA RTX 3080 Ti (SM86), M=N=K=8192, group_size=256, reconn_sz=8, float16.

## Reference: Regular Linear Layer (cuBLAS GEMM)

| Operation | Time (ms) |
|-----------|-----------|
| Forward (C = A @ W^T) | 15.5 |
| Backward dA (dA = dC @ W) | 16.0 |
| Backward dW (dW = dC^T @ A) | 15.7 |
| **Total fwd+bwd** | **47.2** |

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
