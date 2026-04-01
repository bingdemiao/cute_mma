# cute-prism

High-performance Python library for the **Prism linear layer** — a block-diagonal reconnection layer with orthogonal structure. Provides JIT-compiled CUDA kernels that compute the Prism linear operation via a PyTorch interface, with automatic kernel autotuning and caching.

## What is Prism?

Prism applies a block-diagonal orthogonal transformation to activations before the weight multiply, enabling efficient fine-tuning with far fewer trainable parameters than full fine-tuning. The core operation is:

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
- **Gated activation**: Optional `silu_gate` mode for gated Prism (`A * SiLU(A @ R^T)`)
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
import cute_prism

# Create test tensors (float16, CUDA)
M, N, K = 1024, 1024, 256
group_size, reconn_sz = 256, 8
n_groups = N // group_size

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")
R = torch.randn(n_groups * reconn_sz, K, dtype=torch.float16, device="cuda")

# Forward pass — JIT-compiles the kernel on first call
C = cute_prism.forward(A, B, R, group_size, reconn_sz)

# Backward pass
dC = torch.randn_like(C)
dA, dR, dB = cute_prism.backward(dC, A, B, R, group_size, reconn_sz, backend="cute")
```

## `PrismLinear` module

`PrismLinear` is a drop-in replacement for `torch.nn.Linear` that uses Prism structure under the hood. It manages the weight (B), reconnection matrix (R), and optional bias, and integrates with PyTorch's autograd.

### Fine-tuning a pretrained model

Replace existing linear layers with Prism layers, loading the pretrained weights. R is initialized as identity so the layer starts with the same behavior as the original, then only R is trained:

```python
import cute_prism

# Convert an existing nn.Linear layer
pretrained_linear = model.some_layer  # nn.Linear(256, 1024)
model.some_layer = cute_prism.PrismLinear.from_linear(
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

### Width expansion with gated mode

Use `activation="silu_gate"` to enable gated mode, which acts as a width expansion. Both R and B become trainable:

```python
layer = cute_prism.PrismLinear(
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

`PrismLinear` handles any number of leading batch dimensions, just like `nn.Linear`:

```python
layer = cute_prism.PrismLinear(256, 1024, group_size=256, reconn_sz=8).cuda().half()
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
C = cute_prism.forward(A, B, R, group_size, reconn_sz, backend="cublas")
C = cute_prism.forward(A, B, R, group_size, reconn_sz, backend="cublas", mode="rw")
C = cute_prism.forward(A, B, R, group_size, reconn_sz, backend="pytorch")
```

## Gated activation

The `silu_gate` activation replaces the linear reconnection with a gated non-linearity. Combined with the group structure of R, this acts as a width expansion — each group provides an independent transformation pathway, and the non-linearity prevents R from collapsing into B. In this mode, both R and B are trainable, and the backward pass computes all three gradients (dA, dR, dB).

```python
# Forward with gated activation
C = cute_prism.forward(A, B, R, group_size, reconn_sz, activation="silu_gate")

# Backward — returns (dA, dR, dB) where dB is not None
dA, dR, dB = cute_prism.backward(
    dC, A, B, R, group_size, reconn_sz,
    backend="cute", activation="silu_gate",
)
```

Without `silu_gate` (standard mode), B is frozen and `dB` is returned as `None`.

## Autotuning

The autotuner searches over tile sizes, pipeline depths, and warp layouts to find the fastest kernel configuration for each problem shape. Compilation and benchmarking are pipelined — benchmarking starts as soon as the first configuration finishes compiling.

Results are cached to disk, so subsequent calls with the same shape return instantly.

```python
# Easiest: enable autotuning directly in forward/backward
C = cute_prism.forward(A, B, R, group_size, reconn_sz, autotuning=True)
dA, dR, dB = cute_prism.backward(
    dC, A, B, R, group_size, reconn_sz,
    backend="cute", autotuning=True,
)

# Or autotune explicitly and reuse the result
best_params = cute_prism.autotune(M, N, K, group_size, reconn_sz)
C = cute_prism.forward(A, B, R, group_size, reconn_sz, comp_params=best_params)

# Autotune backward kernels independently
best_dadr = cute_prism.autotune_bwd_dadr(M, N, K, group_size, reconn_sz)
best_db = cute_prism.autotune_bwd_db(M, N, K, group_size, reconn_sz, gated=True)
```

### Multi-GPU benchmarking

Distribute benchmark work across multiple GPUs for faster autotuning:

```python
best = cute_prism.autotune(M, N, K, group_size, reconn_sz, device=[0, 1, 2, 3])
```

### Force re-benchmarking

If you suspect previous timings are unreliable (e.g., due to thermal throttling or background load), force a re-benchmark:

```python
best = cute_prism.autotune(M, N, K, group_size, reconn_sz, force_rebenchmark=True)

# Also available through forward/backward
C = cute_prism.forward(A, B, R, group_size, reconn_sz, autotuning=True, force_rebenchmark=True)
```

### Custom search spaces

Provide your own configurations with optional callbacks for adaptive strategies:

```python
from cute_prism import CompParams

my_configs = [
    CompParams(bM=128, bN=128, bK=16),
    CompParams(bM=256, bN=128, bK=32),
]

# With callbacks for progress tracking or early stopping
def on_result(config, time_ms):
    if time_ms is not None:
        print(f"{config.cache_key()}: {time_ms:.3f} ms")

search_space = [(c, on_result) for c in my_configs]
best = cute_prism.autotune(M, N, K, group_size, reconn_sz, search_space=search_space)

# Generators work too (for adaptive/early-stopping strategies)
def adaptive_search():
    for c in my_configs:
        yield (c, on_result)

best = cute_prism.autotune(M, N, K, group_size, reconn_sz, search_space=adaptive_search())
```

### Cache management

```python
# Inspect cached autotune results
cache = cute_prism.get_autotune_cache()
# Returns: {(M, N, K, group_size, reconn_sz): {(kernel_type, gated): {cache_key: time_ms}}}

# Clear a specific entry
cute_prism.clear_autotune_cache(
    M=1024, N=1024, K=256, group_size=256, reconn_sz=8,
    device=0, kernel_type="fwd", gated=False,
)

# Full wipe (keeps compiled .so files referenced as "best")
cute_prism.clear_autotune_cache()

# Full wipe including all compiled kernels
cute_prism.clear_autotune_cache(keep_best_kernels=False)

# Clear only compiled kernel builds (not autotune results)
cute_prism.clear_cache()
```

## Input shuffle

When using multiple groups, each group's R blocks operate on the same partition of input features by default. **Input shuffle** assigns each group a different pairing of 8-element segments into R blocks, so different groups mix different feature pairs — increasing view diversity across groups.

```python
layer = cute_prism.PrismLinear(
    768, 768,
    group_size=64, reconn_sz=16,
    activation="silu_gate",
    backend="cublas",
    input_shuffle=True,    # enable per-group segment shuffling
    shuffle_blk_k=128,     # permutation locality (max 256, for dAdR kernel compat)
)
```

### How it works

With `reconn_sz=16` and `segment_sz=8`, each R block pairs exactly 2 segments. Pairings are generated using a **deterministic affine code** over round-robin tournament matchings:

- Within each `shuffle_blk_k`-sized chunk, up to `n_segments - 1` unique perfect matchings are available (15 for `blk_k=128`)
- Group `g`'s matching at chunk `c` is `(a * c + b) % n_matchings` where `a, b` are derived from `g`
- Same-`a` groups have **zero** chunk collisions; supports up to `n_matchings² + 1` groups without repetition

The segment pair assignments are stored as a buffer (`_seg_pairs`) and saved with model checkpoints.

### Constraints

| Constraint | Reason |
|---|---|
| `reconn_sz=16` | Each R block pairs exactly 2 segments of 8 elements |
| `shuffle_blk_k ≤ 256` | Permutation must be local within dAdR backward kernel's BLK_K tile |
| `backend ≠ "cute"` | CuTe fused kernel does not support shuffle (use `cublas` or `pytorch`) |

### Shuffle in finetuning mode

In non-gated (finetuning) mode, B is frozen and expects natural column order. When `input_shuffle=True`, `load_pretrained_weight()` automatically permutes each group's B columns to match the shuffled AR layout — a one-time cost at load time with zero runtime overhead.

```python
# Convert pretrained layer with shuffle for finetuning
prism = cute_prism.PrismLinear.from_linear(
    pretrained_linear,
    group_size=64, reconn_sz=16,
    input_shuffle=True,
    cayley_order=1,
)
# B columns are permuted per group — forward path needs no scatter
```

### Performance

The shuffle forward uses a two-stream pipeline:
1. **Producer stream**: gather A segments + block-diagonal GEMM + SiLU gate
2. **Consumer stream**: dense GEMM (`H_perm @ B^T`)

Producer for group `g+1` overlaps with consumer for group `g`. The same pipelining is applied to backward passes and to the non-shuffle cublas backend.

## Cayley orthogonal parameterization

For finetuning, R is parameterized via the Cayley transform to guarantee orthogonality:

```python
layer = cute_prism.PrismLinear(
    768, 768,
    group_size=64, reconn_sz=16,
    cayley_order=float("inf"),  # exact Cayley: R = (I+S)(I-S)^{-1}
)

# Or use k-th order approximation for speed:
layer = cute_prism.PrismLinear(
    768, 768,
    group_size=64, reconn_sz=16,
    cayley_order=2,  # R = I + 2S + 2S²  (linear cost in order)
)
```

The trainable parameter M is stored in `reconn`; at forward time, `S = M - M^T` (skew-symmetric) is formed and R is constructed. `reconn_diag_sq_sum()` provides a diagnostic for monitoring R's health during training.

## Normalization and stacking

When stacking PrismLinear layers, place an `nn.GroupNorm` (with `affine=False`) **before** each PrismLinear. This serves two purposes:

1. **Pre-normalization**: PrismLinear's internal gated SiLU (`h = a * SiLU(a @ R^T)`) and weight projection (`y = h @ B^T`) are calibrated for zero-mean, unit-variance input. GroupNorm ensures this.

2. **Non-linear break**: The spherical projection in GroupNorm is a genuine non-linearity ([Ni et al., 2024](https://arxiv.org/abs/2406.01255)) that prevents adjacent layers from collapsing into a single linear map. No external activation (ReLU, SiLU, etc.) is needed — and omitting them keeps the backward gradient zero-centered, which matters for the backward gain correction below.

Use `num_groups = in_features // group_size` to align groups with Prism's block-diagonal structure.

**Backward gain correction** (built-in): The gated SiLU's multiplicative structure (`h = a * SiLU(ar)`) amplifies backward gradients by ~1.437x. PrismLinear corrects this internally with a fixed scalar (`1/1.437`) on the output gradient — no user action needed.

```python
# Recommended stacking pattern
model = nn.Sequential(
    nn.GroupNorm(in_features // group_size, in_features, affine=False),
    PrismLinear(in_features, hidden_dim, group_size=gs, activation="silu_gate"),
    nn.Dropout(),
    nn.GroupNorm(hidden_dim // group_size, hidden_dim, affine=False),
    PrismLinear(hidden_dim, hidden_dim, group_size=gs, activation="silu_gate"),
    nn.Dropout(),
    nn.GroupNorm(hidden_dim // group_size, hidden_dim, affine=False),
    nn.Linear(hidden_dim, num_classes),
)
```

## muP compatibility

PrismLinear is designed to be fully compatible with the [muP](https://github.com/microsoft/mup) library (`MuAdamW`, `set_base_shapes`) and can be freely mixed with `nn.Linear` in the same model.

- **Weight B**: No runtime multiplier. Scaling is absorbed into initialization (`std = alpha/sqrt(K)` where `alpha ≈ 1.77` compensates for gated SiLU variance reduction with skew-symmetric R). muP treats B identically to `nn.Linear`.
- **Reconnection R**: Initialized as skew-symmetric at `std = 1/sqrt(reconn_sz)`. `mup_fix_prism_shapes()` sets R's K-dimension `base_dim = sqrt(base_K * K)` giving `width_mult = sqrt(K/base_K)`, because R's output propagates through B which adds extra `O(1/sqrt(K))` damping.

```python
from mup import MuAdamW, set_base_shapes
from cute_prism._module import mup_fix_prism_shapes

model = make_model(hidden_dim=1024)
base = make_model(hidden_dim=64)
set_base_shapes(model, base)
mup_fix_prism_shapes(model)  # fix R's infshape
optimizer = MuAdamW(model.parameters(), lr=0.01)
```

## `PrismLinear` reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_features` | (required) | Input dimension (K) |
| `out_features` | (required) | Output dimension (N) |
| `group_size` | `256` | Output channels per reconnection group |
| `reconn_sz` | `8` | Block size of orthogonal reconnection matrix |
| `bias` | `True` | Learnable output bias |
| `activation` | `None` | `None` (standard mode) or `"silu_gate"` (gated/width expansion) |
| `cayley_order` | `inf` | Cayley approximation order (`inf` = exact, `k` = k-th order) |
| `input_shuffle` | `False` | Per-group segment shuffling for cross-block feature mixing |
| `shuffle_blk_k` | `128` | Shuffle locality chunk size (max 256) |
| `backend` | `"cute"` | `"cute"`, `"cublas"`, or `"pytorch"` |
| `autotuning` | `False` | Enable automatic kernel autotuning |

Methods:
- `PrismLinear.from_linear(linear, ...)` — create from an existing `nn.Linear`, copying weights and bias
- `load_pretrained_weight(weight, bias=None)` — load pretrained weight/bias tensors (permutes B columns per group when `input_shuffle=True` in finetuning mode)
- `reconn_diag_sq_sum()` — diagnostic: sum of squared diagonal elements of R blocks
- `mup_fix_prism_shapes(model)` — fix PrismLinear infshapes for correct muP LR scaling

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
| `CUTE_PRISM_CACHE_DIR` | `~/.cache/cute_prism` | Directory for compiled kernels and autotune results |
| `CUTE_PRISM_SOURCE_DIR` | auto-detected | Kernel source root directory |
| `CUTE_PRISM_COMPILE_WORKERS` | `4` | Maximum parallel compilation workers |

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run autotuning tests (set workers to avoid OOM)
CUTE_PRISM_COMPILE_WORKERS=2 uv run pytest tests/test_autotune_e2e.py -v
```

## Project structure

```
src/cute_prism/           Python package
  __init__.py             Public API (forward, backward, autotune, ...)
  _autotune.py            Autotuning engine with producer-consumer pipeline
  _compiler.py            JIT compilation and caching
  _config.py              CompParams, BwdDAdRCompParams, BwdDBCompParams
  _loader.py              Module loading from cache
  _validate.py            Tensor and shared memory validation
  _pytorch_backend.py     Pure PyTorch reference implementation
torch_ext/              C++/CUDA PyTorch extensions
  prism_fwd_torch.cu        Forward kernel torch binding
  prism_bwd_dadr_torch.cu   Backward dA+dR kernel torch binding
  prism_bwd_db_torch.cu     Backward dB kernel torch binding
  cublas_prism_torch.cu     cuBLAS backend (pipelined two-stream)
  shuffle_prism_torch.cu    Shuffle backend (gather + cuBLAS + scatter-add)
*.cu, *.hpp             CUDA kernel source (CuTe cooperative kernels, utilities)
cmake/                  CMake modules (CUTLASS detection)
tests/                  pytest test suite
```

## License

See repository for license information.
