#!/bin/bash
# Portable test runner for cute_prism — runs the CPU glue tests anywhere and the
# GPU cute-backend tests + integration harness when a CUDA device is present.
#
# Designed to be moved with this folder to the A100/Ault cluster and run inside
# a CUDA-enabled environment (container venv with torch+CUDA, CUTLASS headers,
# CUDA toolkit). On Clariden this is the oft.sqfs container (see
# ~/.edf/nanogpt.toml); on Ault use whatever CUDA torch env you have there.
#
# Usage:
#     bash run_tests.sh              # auto: CPU tests always, GPU tests if CUDA
#     bash run_tests.sh --cpu-only   # only the backend-agnostic CPU glue tests
#     bash run_tests.sh --clear-cache# wipe the JIT kernel cache first (after .cu edits)
#     PYTHON=python3.12 bash run_tests.sh
#
# Per repo CLAUDE.md: cap nvcc parallelism to avoid OOM on the dev box.
set -uo pipefail

cd "$(dirname "$0")"

PY="${PYTHON:-python}"
export CUTE_PRISM_COMPILE_WORKERS="${CUTE_PRISM_COMPILE_WORKERS:-2}"

CPU_ONLY=0
CLEAR=""
for arg in "$@"; do
  case "$arg" in
    --cpu-only) CPU_ONLY=1 ;;
    --clear-cache) CLEAR="--clear-cache" ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

pass=0; fail=0; skip=0
run() {  # run <label> <cmd...>
  local label="$1"; shift
  echo; echo "==================================================================="
  echo ">>> $label"
  echo "-------------------------------------------------------------------"
  if "$@"; then
    echo "<<< PASS: $label"; pass=$((pass+1))
  else
    echo "<<< FAIL: $label (exit $?)"; fail=$((fail+1))
  fi
}

HAS_CUDA=$("$PY" - <<'EOF'
import torch
print("1" if torch.cuda.is_available() else "0")
EOF
)

echo "python      : $($PY -c 'import sys;print(sys.executable)')"
echo "torch       : $($PY -c 'import torch;print(torch.__version__)')"
echo "cuda avail  : $HAS_CUDA"
echo "workers     : $CUTE_PRISM_COMPILE_WORKERS"

if [ -n "$CLEAR" ] && [ "$HAS_CUDA" = "1" ]; then
  "$PY" -c "import cute_prism; cute_prism.clear_cache(); print('cleared JIT cache')"
fi

# --- CPU, backend-agnostic glue (pytorch backend, fp64) — runs anywhere ---
run "glue: module integration (CPU)" "$PY" tests/test_module_integration.py

if [ "$CPU_ONLY" = "1" ] || [ "$HAS_CUDA" != "1" ]; then
  [ "$HAS_CUDA" != "1" ] && echo; echo "No CUDA device — skipping GPU tests."
  echo; echo "SUMMARY: $pass passed, $fail failed (CPU-only)"
  [ "$fail" -eq 0 ] && exit 0 || exit 1
fi

# --- GPU: existing cute parity tests (cute vs cublas/pytorch) ---
for t in test_internal_bias test_internal_bias_bwd test_internal_bias_grad \
         test_input_shuffle_fwd test_input_shuffle_bwd test_dropout_bwd; do
  run "cute parity: $t" "$PY" "tests/$t.py" $CLEAR
done

# --- GPU: new cute coverage ---
for t in test_cute_feature_cross test_cute_bf16 test_cute_dropout_fwd \
         test_cute_shapes test_cute_determinism; do
  run "cute new: $t" "$PY" "tests/$t.py" $CLEAR
done

# --- GPU: end-to-end nanoGPT cute-vs-cublas (correctness + speed) ---
run "integration: compare_backends (plain gated)" \
    "$PY" integration/compare_backends.py
run "integration: compare_backends (+internal_bias +dropout)" \
    "$PY" integration/compare_backends.py --internal_bias --dropout 0.1
run "integration: compare_backends (+input_shuffle)" \
    "$PY" integration/compare_backends.py --input_shuffle --reconn_sz 16

echo; echo "==================================================================="
echo "SUMMARY: $pass passed, $fail failed"
[ "$fail" -eq 0 ] && exit 0 || exit 1
