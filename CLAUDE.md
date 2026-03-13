# cute-oft

High-performance Python library for OFT (Orthogonal Fine-Tuning) — a parameter-efficient fine-tuning technique. Provides JIT-compiled CUDA kernels that compute `C = A @ diag(R) @ B^T` via a PyTorch interface.

## Key concepts

- **OFT operation**: Multiplies activation matrix A by block-diagonal orthogonal matrix R and weight matrix B
- **Backends**: `cute` (JIT-compiled CuTe kernel, default), `cublas` (cuBLAS), `pytorch` (pure PyTorch reference)
- **Modes**: `ar` (activation reconnection: `(A @ R^T) @ B^T`), `rw` (reweighting: `A @ (B @ R)^T`, cublas/pytorch only)
- **Activations**: `None` (standard linear OFT), `"silu_gate"` (gated: `A * SiLU(A @ R^T)` replaces `A @ R^T`, AR mode only)
- **CompParams / BwdDAdRCompParams / BwdDBCompParams**: Frozen dataclasses controlling tile sizes, pipeline depths, and warp layouts for each kernel type (forward, backward dA+dR, backward dB)
- **Autotuning**: Per-kernel autotuning via pipelined compilation + benchmarking (producer-consumer overlap), caches all tested results to disk. Supports generators, callbacks, and multi-GPU benchmarking. All compiled kernels are kept on disk across autotune runs (keyed by `group_size/reconn_sz/comp_params`, not MNK).
- **Safe fallback**: When default `CompParams` fail to compile, `forward()`/`backward()` automatically retry with `safe_defaults()`
- **Separate compilation**: Forward, backward dAdR, and backward dB kernels compile independently for faster incremental builds

## Project structure

- `src/cute_oft/` — Python package (public API in `__init__.py`)
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

- `CUTE_OFT_CACHE_DIR` — override build/autotune cache location (default: `~/.cache/cute_oft`)
- `CUTE_OFT_SOURCE_DIR` — override kernel source root (auto-detected)
- `CUTE_OFT_COMPILE_WORKERS` — cap parallel compilation workers (default: 4)

## Public API

- `cute_oft.forward(A, B, R, group_size, reconn_sz, backend, mode, comp_params, activation, autotuning, autotuning_search_space)` — main entry point
  - `activation=None`: standard OFT `C = (A @ R^T) @ B^T`
  - `activation="silu_gate"`: gated OFT `C = (A * SiLU(A @ R^T)) @ B^T` (AR mode only)
  - `autotuning=True`: check cache first, autotune if miss, use best config
  - `autotuning_search_space`: custom `Iterable[CompParams | (CompParams, callback)]`
- `cute_oft.backward(dC, A, B, R, group_size, reconn_sz, backend, activation, autotuning, ...)` — backward pass
  - Returns `(dA, dR, dB)` gradients, all K-major matching input shapes
  - Recomputes AR (or gated intermediate) from inputs instead of storing
  - `autotuning=True`: autotunes dAdR and dB kernels independently
  - `bwd_dadr_params` / `bwd_db_params`: explicit per-kernel params override
  - `autotuning_search_space_dadr` / `autotuning_search_space_db`: custom search spaces
  - Supported backends: `pytorch`, `cublas`, `cute`
- `cute_oft.autotune(M, N, K, group_size, reconn_sz, device=0, ...)` — find optimal forward CompParams
  - `device=0` (single GPU) or `device=[0,1,2,3]` (multi-GPU benchmarking)
  - Compilation and benchmarking are pipelined: benchmarking starts as soon as the first config compiles
- `cute_oft.autotune_bwd_dadr(M, N, K, group_size, reconn_sz, device=0, ...)` — find optimal BwdDAdRCompParams
- `cute_oft.autotune_bwd_db(M, N, K, group_size, reconn_sz, device=0, ...)` — find optimal BwdDBCompParams
- `cute_oft.clear_autotune_cache(M, N, K, group_size, reconn_sz, device, kernel_type, gated, keep_best_kernels)` — selectively clear autotune results
  - All params provided: remove that specific entry from autotune_results.json (keeps compiled kernels)
  - No params: full wipe. `keep_best_kernels=True` (default) keeps best .so files; `False` deletes everything
- `cute_oft.get_autotune_cache()` — inspect autotune results as nested dict `{(M,N,K,gs,rs): {(kernel_type,gated): {cache_key: time_ms}}}`
- `cute_oft.clear_cache()` — remove cached kernel builds
- `cute_oft.CompParams` — forward kernel configuration
- `cute_oft.BwdDAdRCompParams` — backward dA+dR kernel configuration
- `cute_oft.BwdDBCompParams` — backward dB kernel configuration
- All three `CompParams` classes have a `safe_defaults()` classmethod returning conservative configs that compile for all valid shapes

### Search space protocol
- Items: `CompParam` or `(CompParam, callback)` tuples
- `callback(config, time_ms)` called after benchmark (or with cached time for already-tested configs)
- `time_ms` is `None` for failed compilations/runs (also cached)
- Supports generators for adaptive/early-stopping strategies

## Misc
- This file should be keep updated with the progress of the project.
- When running the kernels using the cute backend with autotuning enabled, make sure to set `CUTE_OFT_COMPILE_WORKERS` environnment variable to 2. So that it will not trigger OOM error on my machine and kill my wsl
