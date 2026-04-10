# cute-prism

High-performance Python library for the Prism linear layer — a block-diagonal reconnection layer with orthogonal structure. Provides JIT-compiled CUDA kernels that compute `C = A @ diag(R) @ B^T` via a PyTorch interface.

## Key concepts

- **Prism operation**: Multiplies activation matrix A by block-diagonal orthogonal matrix R and weight matrix B
- **Backends**: `cute` (JIT-compiled CuTe kernel, default), `cublas` (cuBLAS), `pytorch` (pure PyTorch reference)
- **Modes**: `ar` (activation reconnection: `(A @ R^T) @ B^T`), `rw` (reweighting: `A @ (B @ R)^T`, cublas/pytorch only)
- **Activations**: `None` (standard linear mode), `"silu_gate"` (gated: `A * SiLU(A @ R^T)` replaces `A @ R^T`, AR mode only)
- **Input shuffle** (`cublas`/`pytorch` only): per-group gather of A into a permuted layout before AR, controlled by `seg_pairs: (n_groups, n_blocks, 2)`. Unified with `activation`, `internal_bias`, and `dropout` — all 16 combinations run through a single code path. Shuffle backward uses a non-atomic unpermute-add (gather on the inverse permutation) in place of the old scatter-add fp32 accumulator.
- **CompParams / BwdDAdRCompParams / BwdDBCompParams**: Frozen dataclasses controlling tile sizes, pipeline depths, and warp layouts for each kernel type (forward, backward dA+dR, backward dB)
- **Autotuning**: Per-kernel autotuning via pipelined compilation + benchmarking (producer-consumer overlap), caches all tested results to disk. Supports generators, callbacks, multi-GPU benchmarking, and `force_rebenchmark`. All compiled kernels are kept on disk across autotune runs (keyed by `group_size/reconn_sz/comp_params`, not MNK). Configs that fail compilation are recorded in a global registry (`compile_failed.json`) and skipped across all MNK sizes.
- **Safe fallback**: When default `CompParams` fail to compile, `forward()`/`backward()` automatically retry with `safe_defaults()`
- **Separate compilation**: Forward, backward dAdR, and backward dB kernels compile independently for faster incremental builds

## Project structure

- `src/cute_prism/` — Python package (public API in `__init__.py`)
- `torch_ext/` — C++/CUDA PyTorch extensions (CuTe and cuBLAS backends)
- `*.cu`, `*.hpp` — CUDA kernel source files (CuTe cooperative kernel, utilities)
- `tests/` — pytest test suite
- `cmake/` — CMake modules (CUTLASS detection)

## Python Environment
The python environment of this project is managed through `uv`. The system-wide `python` or `python3` is not supposed to be used.

## Build & test

```bash
pip install -e .          # install in editable mode
pytest tests/             # run all tests
pytest tests/ --backend cute --mode ar   # filter by backend/mode
```

Dependencies: `torch`, `einops`, `cmake`, CUDA toolkit, CUTLASS headers.

## Environment variables

- `CUTE_PRISM_CACHE_DIR` — override build/autotune cache location (default: `~/.cache/cute_prism`)
- `CUTE_PRISM_SOURCE_DIR` — override kernel source root (auto-detected)
- `CUTE_PRISM_COMPILE_WORKERS` — cap parallel compilation workers (default: 4)

## Public API

- `cute_prism.forward(A, B, R, group_size, reconn_sz, backend, mode, comp_params, activation, autotuning, autotuning_search_space, force_rebenchmark)` — main entry point
  - `activation=None`: standard mode `C = (A @ R^T) @ B^T`
  - `activation="silu_gate"`: gated mode `C = (A * SiLU(A @ R^T)) @ B^T` (AR mode only)
  - `autotuning=True`: check cache first, autotune if miss, use best config
  - `autotuning_search_space`: custom `Iterable[CompParams | (CompParams, callback)]`
- `cute_prism.backward(dC, A, B, R, group_size, reconn_sz, backend, activation, autotuning, ...)` — backward pass
  - Returns `(dA, dR, dB)` gradients, all K-major matching input shapes
  - Recomputes AR (or gated intermediate) from inputs instead of storing
  - `autotuning=True`: autotunes dAdR and dB kernels independently
  - `bwd_dadr_params` / `bwd_db_params`: explicit per-kernel params override
  - `autotuning_search_space_dadr` / `autotuning_search_space_db`: custom search spaces
  - Supported backends: `pytorch`, `cublas`, `cute`
- `cute_prism.autotune(M, N, K, group_size, reconn_sz, device=0, ..., force_rebenchmark=False)` — find optimal forward CompParams
  - `device=0` (single GPU) or `device=[0,1,2,3]` (multi-GPU benchmarking)
  - Compilation and benchmarking are pipelined: benchmarking starts as soon as the first config compiles
  - `force_rebenchmark=True`: re-benchmark all configs even if cached results exist (replaces old timings)
- `cute_prism.autotune_bwd_dadr(M, N, K, group_size, reconn_sz, device=0, ..., force_rebenchmark=False)` — find optimal BwdDAdRCompParams
- `cute_prism.autotune_bwd_db(M, N, K, group_size, reconn_sz, device=0, ..., force_rebenchmark=False)` — find optimal BwdDBCompParams
- `cute_prism.clear_autotune_cache(M, N, K, group_size, reconn_sz, device, kernel_type, gated, keep_best_kernels)` — selectively clear autotune results
  - All params provided: remove that specific entry from autotune_results.json (keeps compiled kernels)
  - No params: full wipe. `keep_best_kernels=True` (default) keeps best .so files; `False` deletes everything
- `cute_prism.get_autotune_cache()` — inspect autotune results as nested dict `{(M,N,K,gs,rs): {(kernel_type,gated): {cache_key: time_ms}}}`
- `cute_prism.clear_cache()` — remove cached kernel builds
- `cute_prism.CompParams` — forward kernel configuration
- `cute_prism.BwdDAdRCompParams` — backward dA+dR kernel configuration
- `cute_prism.BwdDBCompParams` — backward dB kernel configuration
- All three `CompParams` classes have a `safe_defaults()` classmethod returning conservative configs that compile for all valid shapes
- `cute_prism.PrismLinear` — `nn.Module` drop-in replacement for `nn.Linear` with Prism structure
  - `PrismLinear.from_linear(linear, ...)` creates from an existing `nn.Linear`
  - Standard mode: weight frozen, reconn trainable. Gated mode (`activation="silu_gate"`): both trainable.

### Search space protocol
- Items: `CompParam` or `(CompParam, callback)` tuples
- `callback(config, time_ms)` called after benchmark (or with cached time for already-tested configs)
- `time_ms` is `None` for failed compilations/runs (also cached)
- Supports generators for adaptive/early-stopping strategies

## CUDA kernel coding rules

These are **mandatory** rules when writing or modifying CUDA kernels in this project. Violating them produces slow code that must be rewritten.

1. **Never use scalar loops for small matrix multiplies.** Any reduction of the form `sum_j A(m,j) * B(j,n)` or dot product over a tile dimension MUST use MMA tensor core instructions via CuTe's `gemm()`, not a scalar `for` loop. Scalar FMA loops are orders of magnitude slower than tensor cores.

2. **Never use scalar smem loads when LDSM is available.** Loading data from shared memory into MMA operand registers MUST use LDSM copy atoms via CuTe's `make_tiled_copy_A/B` + `copy()`. Never manually loop over smem elements with scalar loads to fill MMA fragments.
   - Ampere tensor cores require both operands to be **K-major** (this is the TN convention — BLAS default is column-major, so T=transposed-A and N=non-transposed-B both give K-contiguous data).
   - `SM75_U16x{2,4,8}_LDSM_T` (the f16\_T variants): loads a **K-major** operand from smem as-is. No transpose happens — the data is already in the layout tensor cores expect. Use this when smem data has K as the contiguous dimension.
   - `SM75_U32x{1,2,4}_LDSM_N` (the u32 variants): loads a **non-K-major** operand from smem and rearranges it into K-major register layout. Use this when smem data has M or N as the contiguous dimension.
   - For f16 operands that are non-K-major in smem: load with `LDSM_T`, then transpose in-register using the PTX instruction `movmatrix.sync.aligned.m8n8.trans.b16 d, a;` (no CuTe trait exists — use inline PTX).

3. **Use packed f16x2 instructions for activation functions.** Sigmoid, tanh, SiLU, and their derivatives MUST use:
   - `tanh.approx.f16x2` PTX instruction (not f32 `__expf` or `tanh.approx.f32`)
   - Packed `__half2` arithmetic: `__hmul2`, `__hadd2`, `__hsub2`, `__hfma2`
   - Compute `sigmoid(x) = 0.5 * (1 + tanh(x * 0.5))` to avoid reciprocal
   - Process 2 elements per instruction whenever possible

4. **Use `make_tiled_copy_C` for R2S writes from MMA fragments.** Writing MMA C-fragment data to shared memory should use CuTe's `make_tiled_copy_C` with `Copy_Atom<UniversalCopy<uint32_t>, half_t>`, not manual loops with `partition_C(identity_tensor)` coordinate lookups.

## Misc
- This file should be keep updated with the progress of the project.
- When running the kernels using the cute backend with autotuning enabled, make sure to set `CUTE_PRISM_COMPILE_WORKERS` environnment variable to 2. So that it will not trigger OOM error on my machine and kill my wsl
