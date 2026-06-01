#!/bin/bash
# Slurm template for cute_prism on Ault (A100). Fill in ACCOUNT / PARTITION and
# adjust paths, then: sbatch integration/run_ault.sbatch
#
# Runs the test suite, then a short cute-vs-cublas training comparison.
#SBATCH --job-name=cuteprism
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=CHANGE_ME            # <-- your Ault account
#SBATCH --partition=CHANGE_ME          # <-- the A100 partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                    # use gpu:4 for the DDP training line below
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                  # submit from the repo root (~/cute_mma)
mkdir -p logs

# --- environment (edit to match your Ault setup) -------------------------
module load cuda/12.4 || true          # or set CUDA_HOME/PATH yourself
# Activate your python env (conda OR venv):
#   source ~/miniforge/bin/activate cuteprism
#   source ~/venvs/cuteprism/bin/activate
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
export PATH="$CUDA_HOME/bin:$PATH"
export CUTLASS_DIR=${CUTLASS_DIR:-$HOME/cutlass}
export CUTE_PRISM_COMPILE_WORKERS=2
export CUTE_PRISM_CACHE_DIR=${CUTE_PRISM_CACHE_DIR:-$HOME/.cache/cute_prism}
mkdir -p "$CUTE_PRISM_CACHE_DIR"

python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# --- 1) test suite -------------------------------------------------------
bash run_tests.sh

# --- 2) end-to-end cute-vs-cublas (correctness + speed) ------------------
python integration/compare_backends.py --internal_bias --dropout 0.1
python integration/compare_backends.py --input_shuffle --reconn_sz 16

# --- 3) short training, cute backend (single GPU) ------------------------
cd integration
python train.py config/train_shakespeare_char.py \
    --prism_backend=cute --wandb_log=False --compile=False \
    --max_iters=500 --out_dir=out-shake-cute

# Multi-GPU example (request --gres=gpu:4 above, then):
# torchrun --standalone --nproc_per_node=4 train.py \
#     config/train_tinystories_prism_3x.py --prism_backend=cute --wandb_log=False
