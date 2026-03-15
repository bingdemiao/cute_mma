# cute-oft

High-performance Python library for **OFT (Orthogonal Fine-Tuning)** — a parameter-efficient fine-tuning technique for large language models. Provides JIT-compiled CUDA kernels that compute the OFT linear layer via a PyTorch interface, with automatic kernel autotuning and caching.

## What is OFT?

OFT applies a block-diagonal orthogonal transformation to activations before the weight multiply, enabling efficient fine-tuning with far fewer trainable parameters than full fine-tuning. The core operation is:

```
C = (A @ R^T) @ B^T
```

where **A** (M, K) is the input activation, **R** (n_groups \* reconn_sz, K) is a block-diagonal reconnection matrix containing small orthogonal blocks, and **B** (N, K) is the frozen weight matrix.

A gated variant is also supported:

```
C = (A * SiLU(A @ R^T)) @ B^T
```

This replaces the linear reconnection with a gated non-linearity, enabling richer adaptation while training both R and B. The gated version can also serve as a **width expansion** technique: because the reconnection R is block-diagonal with `n_groups` independent blocks, each group applies its own orthogonal transform to the activations. With 64 groups, this is effectively a 64x width expansion after the AR transformation — the network gains 64 independent pathways through the same layer. The SiLU non-linearity between AR and B is essential: without it, `(A @ R^T) @ B^T` would collapse into a single linear map `A @ (R^T @ B^T)`, making R redundant. The gating breaks this collapse and ensures R contributes a genuinely distinct transformation.

## Features

- **Three backends**: `cute` (JIT-compiled CuTe tensor core kernels, default), `cublas` (cuBLAS GEMM), `pytorch` (pure PyTorch reference)
- **Forward and backward**: Full differentiable support with separate, independently tunable kernels for forward, backward dA+dR, and backward dB passes
- **Automatic autotuning**: Pipelined compilation + benchmarking discovers the fastest kernel configuration for each problem shape, with results cached to disk
- **Gated activation**: Optional `silu_gate` mode for gated OFT (`A * SiLU(A @ R^T)`)
- **Two computation modes**: `ar` (activation reconnection) and `rw` (reweighting, cublas/pytorch only)
- **Safe fallback**: If a kernel configuration fails to compile, the library automatically retries with conservative defaults
- **Compile-failure caching**: Configurations that fail to compile are recorded globally and skipped in future autotuning runs, even for different input shapes

## Installation

Requires Python >= 3.10, PyTorch with CUDA support, and CUTLASS headers.

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick start

```python
import torch
import cute_oft

# Create test tensors (float16, CUDA)
M, N, K = 1024, 1024, 256
group_size, reconn_sz = 256, 8
n_groups = N // group_size

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")
R = torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device="cuda")

# Forward pass — JIT-compiles the kernel on first call
C = cute_oft.forward(A, B, R, group_size, reconn_sz)

# Backward pass
dC = torch.randn_like(C)
dA, dR, dB = cute_oft.backward(dC, A, B, R, group_size, reconn_sz, backend="cute")
```

## `OFTLinear` module

`OFTLinear` is a drop-in replacement for `torch.nn.Linear` that uses OFT structure under the hood. It manages the weight (B), reconnection matrix (R), and optional bias, and integrates with PyTorch's autograd.

### Fine-tuning a pretrained model

Replace existing linear layers with OFT layers, loading the pretrained weights. R is initialized as identity so the layer starts with the same behavior as the original, then only R is trained:

```python
import cute_oft

# Convert an existing nn.Linear layer
pretrained_linear = model.some_layer  # nn.Linear(256, 1024)
model.some_layer = cute_oft.OFTLinear.from_linear(
    pretrained_linear,
    group_size=256, reconn_sz=8,
    autotuning=True,
)

# Only R requires gradients — B (weight) is frozen
for name, p in model.some_layer.named_parameters():
    print(name, p.requires_grad)
    # weight False
    # reconn True
    # bias   True
```

### Width expansion with gated OFT

Use `activation="silu_gate"` to enable gated OFT, which acts as a width expansion. Both R and B become trainable:

```python
layer = cute_oft.OFTLinear(
    in_features=256, out_features=1024,
    group_size=256, reconn_sz=8,
    activation="silu_gate",
    autotuning=True,
)

# All parameters are trainable
for name, p in layer.named_parameters():
    print(name, p.requires_grad)
    # weight True
    # reconn True
    # bias   True

x = torch.randn(batch_size, 256, dtype=torch.float16, device="cuda")
output = layer(x)  # (batch_size, 1024)
output.sum().backward()  # gradients flow to weight, reconn, and bias
```

### Arbitrary batch dimensions

`OFTLinear` handles any number of leading batch dimensions, just like `nn.Linear`:

```python
layer = cute_oft.OFTLinear(256, 1024, group_size=256, reconn_sz=8).cuda().half()
x = torch.randn(2, 4, 8, 256, dtype=torch.float16, device="cuda")
output = layer(x)  # (2, 4, 8, 1024)
```

## Backends

| Backend | Forward | Backward | Modes | Notes |
|---------|---------|----------|-------|-------|
| `cute` | Yes | Yes | `ar` | JIT-compiled CuTe tensor core kernels. Fastest, requires compilation on first use. |
| `cublas` | Yes | Yes | `ar`, `rw` | Uses cuBLAS GEMM. Good baseline, supports both modes. |
| `pytorch` | Yes | Yes | `ar`, `rw` | Pure PyTorch. No compilation required, useful for testing and debugging. |

```python
# Switch backends
C = cute_oft.forward(A, B, R, group_size, reconn_sz, backend="cublas")
C = cute_oft.forward(A, B, R, group_size, reconn_sz, backend="cublas", mode="rw")
C = cute_oft.forward(A, B, R, group_size, reconn_sz, backend="pytorch")
```

## Gated activation

The `silu_gate` activation replaces the linear reconnection with a gated non-linearity. Combined with the group structure of R, this acts as a width expansion — each group provides an independent transformation pathway, and the non-linearity prevents R from collapsing into B. In this mode, both R and B are trainable, and the backward pass computes all three gradients (dA, dR, dB).

```python
# Forward with gated activation
C = cute_oft.forward(A, B, R, group_size, reconn_sz, activation="silu_gate")

# Backward — returns (dA, dR, dB) where dB is not None
dA, dR, dB = cute_oft.backward(
    dC, A, B, R, group_size, reconn_sz,
    backend="cute", activation="silu_gate",
)
```

Without `silu_gate` (standard OFT), B is frozen and `dB` is returned as `None`.

## Autotuning

The autotuner searches over tile sizes, pipeline depths, and warp layouts to find the fastest kernel configuration for each problem shape. Compilation and benchmarking are pipelined — benchmarking starts as soon as the first configuration finishes compiling.

Results are cached to disk, so subsequent calls with the same shape return instantly.

```python
# Easiest: enable autotuning directly in forward/backward
C = cute_oft.forward(A, B, R, group_size, reconn_sz, autotuning=True)
dA, dR, dB = cute_oft.backward(
    dC, A, B, R, group_size, reconn_sz,
    backend="cute", autotuning=True,
)

# Or autotune explicitly and reuse the result
best_params = cute_oft.autotune(M, N, K, group_size, reconn_sz)
C = cute_oft.forward(A, B, R, group_size, reconn_sz, comp_params=best_params)

# Autotune backward kernels independently
best_dadr = cute_oft.autotune_bwd_dadr(M, N, K, group_size, reconn_sz)
best_db = cute_oft.autotune_bwd_db(M, N, K, group_size, reconn_sz, gated=True)
```

### Multi-GPU benchmarking

Distribute benchmark work across multiple GPUs for faster autotuning:

```python
best = cute_oft.autotune(M, N, K, group_size, reconn_sz, device=[0, 1, 2, 3])
```

### Force re-benchmarking

If you suspect previous timings are unreliable (e.g., due to thermal throttling or background load), force a re-benchmark:

```python
best = cute_oft.autotune(M, N, K, group_size, reconn_sz, force_rebenchmark=True)

# Also available through forward/backward
C = cute_oft.forward(A, B, R, group_size, reconn_sz, autotuning=True, force_rebenchmark=True)
```

### Custom search spaces

Provide your own configurations with optional callbacks for adaptive strategies:

```python
from cute_oft import CompParams

my_configs = [
    CompParams(bM=128, bN=128, bK=16),
    CompParams(bM=256, bN=128, bK=32),
]

# With callbacks for progress tracking or early stopping
def on_result(config, time_ms):
    if time_ms is not None:
        print(f"{config.cache_key()}: {time_ms:.3f} ms")

search_space = [(c, on_result) for c in my_configs]
best = cute_oft.autotune(M, N, K, group_size, reconn_sz, search_space=search_space)

# Generators work too (for adaptive/early-stopping strategies)
def adaptive_search():
    for c in my_configs:
        yield (c, on_result)

best = cute_oft.autotune(M, N, K, group_size, reconn_sz, search_space=adaptive_search())
```

### Cache management

```python
# Inspect cached autotune results
cache = cute_oft.get_autotune_cache()
# Returns: {(M, N, K, group_size, reconn_sz): {(kernel_type, gated): {cache_key: time_ms}}}

# Clear a specific entry
cute_oft.clear_autotune_cache(
    M=1024, N=1024, K=256, group_size=256, reconn_sz=8,
    device=0, kernel_type="fwd", gated=False,
)

# Full wipe (keeps compiled .so files referenced as "best")
cute_oft.clear_autotune_cache()

# Full wipe including all compiled kernels
cute_oft.clear_autotune_cache(keep_best_kernels=False)

# Clear only compiled kernel builds (not autotune results)
cute_oft.clear_cache()
```

## `OFTLinear` reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_features` | (required) | Input dimension (K) |
| `out_features` | (required) | Output dimension (N) |
| `group_size` | `256` | Output channels per reconnection group |
| `reconn_sz` | `8` | Block size of orthogonal reconnection matrix |
| `bias` | `True` | Learnable output bias |
| `activation` | `None` | `None` (standard OFT) or `"silu_gate"` (gated/width expansion) |
| `backend` | `"cute"` | `"cute"`, `"cublas"`, or `"pytorch"` |
| `autotuning` | `False` | Enable automatic kernel autotuning |

Methods:
- `OFTLinear.from_linear(linear, ...)` — create from an existing `nn.Linear`, copying weights and bias
- `load_pretrained_weight(weight, bias=None)` — load pretrained weight/bias tensors

## Configuration classes

Each kernel type has its own frozen dataclass controlling tile sizes, pipeline depths, and warp layouts:

| Class | Kernel | Key parameters |
|-------|--------|----------------|
| `CompParams` | Forward | `bM`, `bN`, `bK`, `c_width`, `bP_a_r`, `bP_ar`, `bP_b`, `warp_layout_ar`, `warp_layout_arb` |
| `BwdDAdRCompParams` | Backward dA+dR | `bM`, `bK`, `bK_inner`, `n_buf_slots`, `warp_layout_arb`, `warp_layout_ar` |
| `BwdDBCompParams` | Backward dB | `bM`, `bK`, `bP_a`, `bP_ar`, `bP_dc`, `warp_layout_ar`, `warp_layout_arb` |

All three classes provide:
- `safe_defaults()` — conservative configuration that compiles for all valid shapes
- `to_dict()` / `from_dict()` — JSON-compatible serialization
- `cache_key()` — short hash identifying the configuration
- `to_header()` — generates the C++ struct definition used during compilation

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUTE_OFT_CACHE_DIR` | `~/.cache/cute_oft` | Directory for compiled kernels and autotune results |
| `CUTE_OFT_SOURCE_DIR` | auto-detected | Kernel source root directory |
| `CUTE_OFT_COMPILE_WORKERS` | `4` | Maximum parallel compilation workers |

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run autotuning tests (set workers to avoid OOM)
CUTE_OFT_COMPILE_WORKERS=2 uv run pytest tests/test_autotune_e2e.py -v
```

## Project structure

```
src/cute_oft/           Python package
  __init__.py             Public API (forward, backward, autotune, ...)
  _autotune.py            Autotuning engine with producer-consumer pipeline
  _compiler.py            JIT compilation and caching
  _config.py              CompParams, BwdDAdRCompParams, BwdDBCompParams
  _loader.py              Module loading from cache
  _validate.py            Tensor and shared memory validation
  _pytorch_backend.py     Pure PyTorch reference implementation
torch_ext/              C++/CUDA PyTorch extensions
  oft_fwd_torch.cu        Forward kernel torch binding
  oft_bwd_dadr_torch.cu   Backward dA+dR kernel torch binding
  oft_bwd_db_torch.cu     Backward dB kernel torch binding
  cublas_oft_torch.cu     cuBLAS backend torch binding
*.cu, *.hpp             CUDA kernel source (CuTe cooperative kernels, utilities)
cmake/                  CMake modules (CUTLASS detection)
tests/                  pytest test suite
```

## License

See repository for license information.
