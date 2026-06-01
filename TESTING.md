# Testing the new cute-backend features

This folder is self-contained: move it to the A100/Ault cluster and run the
whole suite there. The features under test (all on the **cute** backend) are
`input_shuffle`, `internal_bias`, `dropout`, and the backward-kernel rewrites.
`cublas` and `pytorch` are the **trusted references** — the cute kernel is
checked for parity against them.

## TL;DR

```bash
# inside a CUDA-enabled env (see "Environment" below)
bash run_tests.sh                 # CPU glue tests + all GPU cute tests + integration
bash run_tests.sh --cpu-only      # only the backend-agnostic CPU tests
bash run_tests.sh --clear-cache   # wipe JIT kernel cache first (after editing .cu)
```

No manual recompile is needed: the cute kernels are JIT-compiled and the build
cache is keyed by a hash of the kernel sources, so edited/new feature code
rebuilds automatically (one-time cost on first run).

## What runs where

### CPU, backend-agnostic (no GPU, no cute build) — validates the shared glue
- **`tests/test_module_integration.py`** — the `PrismLinear` module wiring that
  every backend shares, exercised via the pytorch backend in fp64:
  - `gradcheck` of both autograd paths (`_PrismFn` and the custom op) →
    catches a mis-ordered gradient tuple (e.g. `d_internal_bias` on the wrong
    input);
  - `gradcheck` of the full module in plain mode (reshape + bias + Cayley R);
  - `eval()` disables dropout / `train()` randomizes;
  - `state_dict` round-trips the `_internal_bias` param and `_shuffle_masks`
    buffer;
  - `torch.compile` numerics (the fullgraph case is skipped on the aarch64
    inductor toolchain bug — it runs on the GPU env).

  This was run on the dev box and is green (6 passed, 1 skipped). Running it on
  Ault first means any GPU-only failure points at the **kernel**, not the glue.

### GPU, cute kernel parity vs cublas/pytorch
Existing (per-feature):
- `tests/test_internal_bias.py`, `…_bwd.py`, `…_grad.py`
- `tests/test_input_shuffle_fwd.py`, `…_bwd.py`
- `tests/test_dropout_bwd.py`

New (gaps that were missing):
- **`tests/test_cute_feature_cross.py`** — full `{shuffle × internal_bias ×
  dropout}` 8-combo cross-product, fwd+bwd (feature *interactions*, not just
  each alone).
- **`tests/test_cute_bf16.py`** — bf16 parity (training dtype; the other tests
  are fp16 only).
- **`tests/test_cute_dropout_fwd.py`** — forward dropout statistical
  correctness (inverted-dropout mean-preservation; cross-backend seed-matched
  forward parity is impossible because seeds are drawn internally per backend).
- **`tests/test_cute_shapes.py`** — non-square / non-pow2 / large-K /
  multi-CTA-per-group shapes.
- **`tests/test_cute_determinism.py`** — identical inputs (and seeds, for
  dropout) ⇒ bit-identical outputs.

### GPU, end-to-end (real model: correctness + speed)
- **`integration/compare_backends.py`** — the nanoGPT GPT (copied into
  `integration/model.py`) with the gated PrismLinear MLP, run cute vs cublas:
  1. step-0 parity (logits, loss, **every gradient**),
  2. overfit a fixed batch (no NaN, loss descends, curves reported),
  3. fwd / fwd+bwd timing with a cute-vs-cublas speedup ratio.

  Examples:
  ```bash
  python integration/compare_backends.py
  python integration/compare_backends.py --internal_bias --dropout 0.1
  python integration/compare_backends.py --input_shuffle --reconn_sz 16
  python integration/compare_backends.py --steps 300 --speed-iters 50
  ```

## Environment

The cute/cublas backends need **CUDA torch** + CUTLASS headers + CUDA toolkit.
On Clariden that is the `oft.sqfs` container (`~/.edf/nanogpt.toml`,
`/opt/oft/.venv`, `CUDA_HOME=/usr/local/cuda`); a bare login-node `uv` env is
CPU-only on aarch64 and runs only the `--cpu-only` tests.

On **Ault (A100)**: launch inside whatever CUDA-enabled torch env you have
there, `pip install -e .` (or set `PYTHONPATH=$PWD/src`), then `bash
run_tests.sh`. The cute compiler auto-detects the GPU arch
(`torch.cuda.get_device_capability`), so it builds for sm_80 (A100)
automatically — no flags needed. Re-autotune on the A100; timings and optimal
`CompParams` do not transfer from other GPUs.

Notes:
- Kernel *math* transfers across GPU archs (same MMA PTX), but A100's lower
  shared-memory ceiling (~164 KB vs ~227 KB on Hopper) can make a large-tile
  config that launched elsewhere fail `check_smem_limit` on A100. If a default
  config fails, the library retries with `safe_defaults()`.
- `CUTE_PRISM_COMPILE_WORKERS=2` (set by `run_tests.sh`) caps nvcc parallelism.

## Fixes applied alongside these tests
- The `forward()` docstring and `nanogpt/model.py` claimed `input_shuffle` was
  cublas/pytorch only; the cute forward actually wires `shuffle_masks` into the
  kernel. Docstring corrected; the nanogpt guard relaxed (validate the
  cute+shuffle path with `compare_backends.py --input_shuffle` before a long
  run). `integration/model.py` here already has the guard relaxed.
- The custom-op autograd returns a **zero `dB` in plain mode** (frozen-B
  finetuning convention). This is now documented in `_module.py` and pinned by
  `test_gradcheck_custom_op` — it is correct for `PrismLinear` (B is frozen
  there) but will silently not train a B that is trainable in plain mode; use a
  gated activation if B must learn.
