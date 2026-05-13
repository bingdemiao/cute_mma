"""Forward parity tests for the cute backend's internal_bias support.

Compares cute, cublas, and pytorch backends on the gated forward pass with
``internal_bias`` set, across a few representative shapes.

Usage:
    python tests/test_internal_bias.py
    python tests/test_internal_bias.py --clear-cache    # after .cu changes
    python tests/test_internal_bias.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

# Limit nvcc parallelism per repo CLAUDE.md guidance to avoid OOM kills.
os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism


def _max_rel_err(val: torch.Tensor, ref: torch.Tensor) -> float:
    return ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()


def _run_one(M, N, K, gs, rs, seed, verbose) -> tuple[bool, float, float]:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1

    C_cute, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="cute", activation="silu_gate", internal_bias=bias,
    )
    C_cublas, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="cublas", activation="silu_gate", internal_bias=bias,
    )
    C_pt, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="pytorch", activation="silu_gate", internal_bias=bias,
    )

    rel_cublas = _max_rel_err(C_cute, C_cublas)
    rel_pt = _max_rel_err(C_cute, C_pt)
    # rtol matches test_correctness.py's default for fwd.
    ok = rel_cublas < 0.005 and rel_pt < 0.005

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} seed={seed} "
            f"|cute-cublas|={rel_cublas:.5f} |cute-pytorch|={rel_pt:.5f}"
        )
    return ok, rel_cublas, rel_pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    # (M, N, K, group_size, reconn_sz)
    cases = [
        (256,  256, 256, 256, 8),
        (512,  512, 512, 256, 8),
        (1024, 1024, 1024, 256, 8),
        # group_size > bN (default bN=128, multiple ctas share one group)
        (512,  256, 512, 256, 8),
        # rs=16 path (uses SM80_16x8x16 atom in stage1)
        (512,  512, 512, 256, 16),
    ]

    passed = failed = 0
    for M, N, K, gs, rs in cases:
        for seed in range(args.seeds):
            ok, _, _ = _run_one(M, N, K, gs, rs, seed * 100, args.verbose)
            if ok:
                passed += 1
            else:
                failed += 1

    print(f"Total: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
