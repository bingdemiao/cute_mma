# cute-prism

High-performance Python library for the Prism linear layer — a block-diagonal reconnection layer with orthogonal structure. Provides JIT-compiled CUDA kernels that compute `C = A @ diag(R) @ B^T` via a PyTorch interface.

## Key concepts

- **Prism operation**: Multiplies activation matrix A by block-diagonal orthogonal matrix R and weight matrix B
- **Backends**: `cute` (JIT-compiled CuTe kernel, default), `cublas` (cuBLAS), `pytorch` (pure PyTorch reference)
- **Modes**: `ar` (activation reconnection: `(A @ R^T) @ B^T`), `rw` (reweighting: `A @ (B @ R)^T`, cublas/pytorch only)
- **Activations**: `None` (standard linear mode), `"silu_gate"` (gated: `A * SiLU(A @ R^T)` replaces `A @ R^T`, AR mode only, supported by all backends), `"sigmoid_gate"` (bounded gate: `A * 2*sigmoid(A @ R^T)`, AR mode only, **cublas and pytorch backends only** — no `cute` kernel). The sigmoid variant is useful when the linearly-growing `SiLU(S)` term in the dA gate contribution is undesirable; `2*sigmoid(S)` is capped in (0, 2) while `2*sigmoid(0) = 1` matches SiLU's zero slope. Kernel dispatch uses an integer `gate_kind` (0=none, 1=silu_gate, 2=sigmoid_gate) — the Python `_GATE_KIND` dict in `src/cute_prism/__init__.py` must stay in sync with the `GATE_*` enum in `torch_ext/cublas_prism_torch.cu`.
- **Input shuffle** (all backends, including `cute`): per-group butterfly shuffle of A via `__shfl_xor_sync` before AR, controlled by `shuffle_masks: (n_groups, n_chunks, n_rounds)`. Uses 2 rounds with XOR deltas [8, 16] and hash-based mask construction (seg_sz=2). Unified with `activation`, `internal_bias`, and `dropout` — all 16 combinations run through a single code path.
- **Dropout** (all backends, including `cute`): hash-derived per-element mask applied to `H = A * gate(S)` after the activation. Stateless — no mask buffer is materialized. The forward generates one `uint64` seed per group (returned as the `_seeds` extra of `forward()`), and the backward replays the same `(seed, m·K + k)` index through `splitmix64` to reproduce the mask in the dB recompute and on `dH`. Cute and cublas use bit-identical hashes, so passing the same seed list to both produces the same mask. Kernel binaries are split by `dropout` flag (suffixed `_drop` in the build cache); enabling dropout requires recompilation but does not affect the no-dropout binary. Forward returns `(C, [seeds])` when `training=True and dropout_p>0`; pass the seeds back to `backward(..., dropout_seeds=seeds, dropout_p=p)`.
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
  - `activation="sigmoid_gate"`: bounded gated mode `C = (A * 2*sigmoid(A @ R^T)) @ B^T` (AR mode only, cublas/pytorch backends only)
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
  - Standard mode: weight frozen, reconn trainable. Gated modes (`activation="silu_gate"` or `"sigmoid_gate"`): both trainable. `_GATED_SILU_BWD_GAIN` correction on the backward pass is only applied for `silu_gate`; `sigmoid_gate` gets `_bwd_scale=1.0`.

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

## Backward kernel performance — verified findings

These are **measured, not estimated**. They concern the cute backward path
(`cute_prism_backward_dadr.cu` + `cute_prism_backward_db.cu`). Each entry here
has been validated by re-running the regression suite (`tests/test_correctness.py`,
`test_internal_bias_bwd.py`, `test_input_shuffle_bwd.py` — 112 tests pass) and
by direct timing on GH200.

### Working baseline (M=N=K=4096, fp16, gated, gs=256, rs=8, GH200)
- Default `BwdDAdRCompParams` + `BwdDBCompParams`: **4.115 ms ± 0.024 ms**
  (mean ± stdev across 5 trials × 50 iters).
- Within bf16/fp16 numerical noise of cublas (`dA, dR, dB` rel err < 0.27%).
- bf16 backward at the same shape: **5.417 ms ± 0.067 ms**.

### Verified non-regressive features
- **Dropout plumbing** (this codebase, both fp16 and bf16). The no-dropout
  path is performance-neutral within trial noise: a few hoisted integer index
  calculations were moved out of `#if PRISM_INTERNAL_BIAS` blocks so the
  dropout site could share them. The compiler DCEs them when neither bias nor
  dropout is on. End-to-end backward at 4096^3 fp16 is unchanged within ±1.1%
  (≈ 2σ).

### Investigated and rejected (do NOT redo without new evidence)

These were tried and shown to be neutral or negative; the audit notes are kept
inline at the relevant code sites. Re-attempting them is a known dead end
unless the bottleneck has shifted.

- **bf16 dB producer's r2s via the canonical `blocken_C → inplace_transpose →
  construct_B → copy` chain.** Rejected — measured ~4% **slower** than the
  scalar transposed-store fallback at 4k³. The producer r2s is not on the
  critical path of bf16 dB; replacing slow scalar stores with `movmatrix`-driven
  faster stores adds `movmatrix` latency without saving anything because the
  consumer is the bottleneck. See the `do not revert` comment block in
  `cute_prism_backward_db.cu` next to the bf16 r2s site.

- **Public-API autotuning at 4k³** (`cute_prism.backward(autotuning=True)`).
  Rejected at this shape — produces a ~3× wall-clock regression
  (4.113 ms → 12.056 ms). Two compounding causes:
  1. Per-call autotune-cache lookup overhead inside `_autotune.py`. Every
     `backward()` call re-resolves the cache and prints, adding ~7 ms/iter.
  2. The dB autotuner's per-kernel timing methodology picks a config
     (`bM=16, bK=128`) that's fastest in isolation but slower than the
     default (`bK=32`) in the real sequential dadr→dB→dR_reduce pipeline.
     The default `BwdDBCompParams` is empirically close to optimal at this
     shape; the search-space best is a fluke of isolated timing.

  The individual autotune entries (`autotune_bwd_dadr`, `autotune_bwd_db`)
  remain useful for *exploration* at unfamiliar shapes — but feeding their
  outputs back through `backward(autotuning=True)` is currently a perf bug.
  Until the per-call lookup overhead is fixed, prefer explicit
  `bwd_dadr_params` / `bwd_db_params` arguments after a one-shot autotune
  exploration.

- **Kernel-fusion of dadr's S recompute with dB's S recompute** (#3 in the
  optimization audit). Rejected after re-doing the FLOP arithmetic: the AR
  recompute is ~3.1% of total backward FLOPs (`M·N·K · rs/gs` per side, both
  sides), not the ~30% originally hand-waved. Realistic upper-bound speedup
  ~2-4%, against ~1-2 weeks of careful kernel-merge engineering across two
  cooperative producer-consumer schedules with different reduction axes.
  Cost-benefit is poor.

### When to revisit
Open kernel-perf work again only if **profiling** (e.g. `ncu --set full` on a
real training step) shows the prism backward in the top 3 hotspots of total
step time. The "obvious" micro-optimizations have all been tried; the next
real win requires knowing precisely which section is saturated, not guessing.

## Misc
- This file should be keep updated with the progress of the project.
- When running the kernels using the cute backend with autotuning enabled, make sure to set `CUTE_PRISM_COMPILE_WORKERS` environnment variable to 2. So that it will not trigger OOM error on my machine and kill my wsl
