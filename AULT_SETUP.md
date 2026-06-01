# Running cute_prism on Ault (A100) — setup & commands

This folder is self-contained: it has the cute_prism library + kernels, the
test suite (`tests/`, `run_tests.sh`), and a full nanoGPT training stack
(`integration/`: `train.py`, `model.py`, configs, data prep). Below are the
concrete steps to provision Ault and run both the tests and training.

Assumptions: A100 node (sm_80), x86_64 Linux, a CUDA toolkit available, git,
and internet on a login/build node (for the torch wheel + CUTLASS). If your
A100 *compute* nodes are offline, do all downloads on the login node first.

--------------------------------------------------------------------------
## 1. Copy the folder to Ault

From Clariden (or wherever this folder lives):

```bash
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '*.so' \
    ~/cute_testing/cute_mma/  <user>@ault:~/cute_mma/
```

The small Shakespeare dataset bins (`integration/data/shakespeare_char/*.bin`)
are included, so training is runnable immediately. (`rsync` keeps them.)

--------------------------------------------------------------------------
## 2. CUDA toolkit (needed to JIT-compile the kernels)

```bash
module avail cuda                 # see what's there
module load cuda/12.4             # or whatever 12.x / 11.8+ Ault provides
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
nvcc --version                    # must print; CUDA >= 11.8 (A100 = sm_80)
```

If there is no module, point `CUDA_HOME` at a local CUDA install and put its
`bin` on `PATH`.

--------------------------------------------------------------------------
## 3. Python env with CUDA torch (A100 / x86_64)

Pick ONE. Verify CUDA at the end.

**Option A — conda/mamba (most robust):**
```bash
# install miniforge if you don't have conda:
#   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
#   bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge && source ~/miniforge/bin/activate
mamba create -y -n cuteprism python=3.12
mamba activate cuteprism
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match your CUDA (cu121/cu124)
pip install einops numpy "cmake>=3.24" ninja
pip install mup wandb            # optional: muP runs / logging
```

**Option B — venv + module python:**
```bash
module load python/3.12
python -m venv ~/venvs/cuteprism && source ~/venvs/cuteprism/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install einops numpy "cmake>=3.24" ninja
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.x.y+cu124  True  NVIDIA A100-...
```
`cuda.is_available()` MUST be True (run this on a GPU node, or one with the
driver visible). If False you got a CPU wheel — reinstall from the cu-index URL.

--------------------------------------------------------------------------
## 4. CUTLASS headers (the cute kernels are built against CuTe)

```bash
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
export CUTLASS_DIR=~/cutlass
# If the JIT later errors with CuTe API mismatches, pin a 3.x tag:
#   cd ~/cutlass && git checkout v3.5.1   (try v3.6.0 / v3.5.x if needed)
```

If you skip `CUTLASS_DIR`, the build auto-clones CUTLASS into its build dir —
but that needs internet at *build* time (often on the compute node), so setting
`CUTLASS_DIR` to a pre-cloned checkout is safer.

--------------------------------------------------------------------------
## 5. Install the package + JIT env vars

```bash
cd ~/cute_mma
pip install -e .                 # or: export PYTHONPATH=$PWD/src
export CUTE_PRISM_COMPILE_WORKERS=2          # cap nvcc parallelism (avoid OOM)
export CUTE_PRISM_CACHE_DIR=$HOME/.cache/cute_prism   # or a scratch path
mkdir -p "$CUTE_PRISM_CACHE_DIR"
```

Persist steps 2–5's `export`s in a small `env.sh` you `source` before each run.

--------------------------------------------------------------------------
## 6. Run the test suite (on a node WITH an A100)

Get onto a GPU node (interactive):
```bash
salloc -p <a100_partition> --gres=gpu:1 -t 02:00:00     # adjust to Ault
# then, inside the allocation:
source ~/cute_mma/env.sh          # your saved exports
cd ~/cute_mma
bash run_tests.sh                 # CPU glue + all GPU cute tests + integration
bash run_tests.sh --clear-cache   # after any .cu edit
```
The FIRST run compiles many kernels (minutes); later runs use the cache.

Run individual pieces:
```bash
python tests/test_cute_feature_cross.py -v      # 8-combo feature parity vs cublas
python tests/test_cute_bf16.py -v               # bf16 parity
python tests/test_cute_shapes.py -v             # shape sweep
python tests/test_cute_determinism.py -v
python tests/test_cute_dropout_fwd.py -v
# plus the pre-existing: test_internal_bias*.py, test_input_shuffle_*.py, test_dropout_bwd.py
```

Re-autotune for A100 if you care about speed (timings/configs don't transfer
from other GPUs):
```bash
python -c "import cute_prism; print(cute_prism.autotune(4096,4096,4096,256,16,device=0))"
```

--------------------------------------------------------------------------
## 7. End-to-end cute-vs-cublas validation + speed

```bash
cd ~/cute_mma
python integration/compare_backends.py                              # plain gated
python integration/compare_backends.py --internal_bias --dropout 0.1
python integration/compare_backends.py --input_shuffle --reconn_sz 16
python integration/compare_backends.py --steps 300 --speed-iters 50
```
Phase 1 = step-0 grad parity vs cublas, Phase 2 = overfit curve, Phase 3 = speed.

--------------------------------------------------------------------------
## 8. Train the model

Data: Shakespeare bins are already included. To (re)generate or add datasets:
```bash
python integration/data/shakespeare_char/prepare.py    # offline, instant
python integration/data/tinystories/prepare.py         # needs internet (HF datasets)
```

Single GPU, cute backend:
```bash
cd ~/cute_mma/integration
python train.py config/train_shakespeare_char.py \
    --prism_backend=cute --wandb_log=False --compile=False --out_dir=out-shake-cute
```

cublas baseline (same config) for an apples-to-apples comparison:
```bash
python train.py config/train_shakespeare_char.py \
    --prism_backend=cublas --wandb_log=False --out_dir=out-shake-cublas
```

Multi-GPU (DDP) on one node, e.g. 4×A100:
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_tinystories_prism_3x.py \
    --prism_backend=cute --wandb_log=False
```

Feature flags (all forwardable on the CLI): `--prism_internal_bias=True`,
`--prism_internal_dropout=0.1`, `--prism_input_shuffle=True --reconn_sz=16`,
`--use_mup=True --mup_base_width=128`.

Notes:
- The configs set `wandb_log=True`; pass `--wandb_log=False` unless you've set
  `WANDB_API_KEY` (and `wandb login`).
- bf16 end-to-end, no autocast (PrismLinear needs A/B/R same dtype).
- `input_shuffle` requires `reconn_sz=16`.

--------------------------------------------------------------------------
## 9. Slurm batch template

`integration/run_ault.sbatch` is a starting point — fill in `--account` and
`--partition` for Ault, then:
```bash
sbatch integration/run_ault.sbatch
```

--------------------------------------------------------------------------
## 10. Troubleshooting

- `cuda.is_available()` is False → you installed a CPU torch wheel; reinstall
  from the `--index-url https://download.pytorch.org/whl/cuXXX` matching your CUDA.
- CUTLASS / CuTe compile errors → pin a CUTLASS 3.x tag (step 4).
- `check_smem_limit` failure on a config → A100's shared-mem ceiling (~164 KB)
  is lower than Hopper's; the library auto-retries `safe_defaults()`. For
  autotuning, exclude oversized tiles.
- nvcc not found → `module load cuda` / fix `CUDA_HOME` + `PATH` (step 2).
- Slow first run → expected (JIT compiles per config). Cached afterwards in
  `CUTE_PRISM_CACHE_DIR`.
