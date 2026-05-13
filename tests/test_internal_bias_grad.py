"""Parity test for d_internal_bias from the cute backend's gated backward.

Verifies that the cute backend's reduced d_internal_bias matches the cublas
and pytorch references across several shapes.

Usage:
    python tests/test_internal_bias_grad.py
    python tests/test_internal_bias_grad.py --clear-cache
    python tests/test_internal_bias_grad.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism


def _max_rel_err(val: torch.Tensor, ref: torch.Tensor) -> float:
    return ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()


def _run_one(M, N, K, gs, rs, seed, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

    _, _, _, d_ib_cute = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cute", activation="silu_gate", internal_bias=bias,
    )
    _, _, _, d_ib_cb = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cublas", activation="silu_gate", internal_bias=bias,
    )
    _, _, _, d_ib_pt = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="pytorch", activation="silu_gate", internal_bias=bias,
    )

    if d_ib_cute is None:
        print(f"  FAIL M={M} N={N} K={K} — cute backend returned None d_ib")
        return False

    rel_cb = _max_rel_err(d_ib_cute, d_ib_cb)
    rel_pt = _max_rel_err(d_ib_cute, d_ib_pt)
    rtol = 0.02
    ok = rel_cb < rtol and rel_pt < rtol

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} seed={seed} "
            f"|cute-cb|={rel_cb:.4f} |cute-pt|={rel_pt:.4f}"
        )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    cases = [
        (256,  256,  256, 256, 8),
        (512,  512,  512, 256, 8),
        (1024, 1024, 1024, 256, 8),
        (512,  256,  512, 256, 8),
        (512,  512,  512, 256, 16),
    ]

    passed = failed = 0
    for M, N, K, gs, rs in cases:
        for seed in range(args.seeds):
            ok = _run_one(M, N, K, gs, rs, seed * 100, args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1

    print(f"Total: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
