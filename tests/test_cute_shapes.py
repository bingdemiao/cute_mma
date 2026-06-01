"""Cute-backend shape coverage (fp16).

The existing cute parity tests use small, mostly-square shapes. Tile/edge bugs
often hide at non-square, non-power-of-two, large-K, or multi-CTA-per-group
shapes. This sweeps a wider set and checks gated forward + backward parity
(with internal_bias on, to exercise the bias kernel path too) against the
trusted cublas reference and the pytorch reference.

Requires a CUDA GPU and the cute backend.

Usage::

    CUTE_PRISM_COMPILE_WORKERS=2 python tests/test_cute_shapes.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism


def _max_rel_err(val, ref) -> float:
    return ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()


def _run_one(M, N, K, gs, rs, seed, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

    C_cute, _ = cute_prism.forward(A, B, R, gs, rs, backend="cute",
                                   activation="silu_gate", internal_bias=bias)
    C_cb, _ = cute_prism.forward(A, B, R, gs, rs, backend="cublas",
                                 activation="silu_gate", internal_bias=bias)
    C_pt, _ = cute_prism.forward(A, B, R, gs, rs, backend="pytorch",
                                 activation="silu_gate", internal_bias=bias)
    fwd = max(_max_rel_err(C_cute, C_cb), _max_rel_err(C_cute, C_pt))

    dA_c, dR_c, dB_c, dIB_c = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cute", activation="silu_gate", internal_bias=bias)
    dA_b, dR_b, dB_b, dIB_b = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cublas", activation="silu_gate", internal_bias=bias)

    dA = _max_rel_err(dA_c, dA_b)
    dR = _max_rel_err(dR_c, dR_b)
    dB = _max_rel_err(dB_c, dB_b)
    dIB = _max_rel_err(dIB_c, dIB_b)

    ok = fwd < 0.005 and max(dA, dR, dB, dIB) < 0.02
    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} M={M:>4} N={N:>4} K={K:>4} gs={gs:>3} rs={rs:>2} "
              f"fwd={fwd:.4f} dA={dA:.4f} dR={dR:.4f} dB={dB:.4f} dIB={dIB:.4f}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    # (M, N, K, group_size, reconn_sz)  — N % gs == 0, K % rs == 0, all mult of 8.
    cases = [
        (128,  256,  512,  64,  8),   # non-square
        (384,  384,  768, 128,  8),   # 384 = 3*128 (non-pow2)
        (200,  512, 1024, 256, 16),   # non-pow2 M, large-ish K, gs>bN(128)
        (512,  128,  512, 128, 16),   # N == gs (single group)
        (256,  512, 2048, 256,  8),   # large K
        (128,  256,  256, 256, 16),   # gs spans multiple CTAs (bN default 128)
        (1024, 1024, 1024, 64, 16),   # bigger, many groups
    ]

    passed = failed = 0
    for M, N, K, gs, rs in cases:
        ok = _run_one(M, N, K, gs, rs, seed=0, verbose=args.verbose)
        passed += ok
        failed += (not ok)

    print(f"Total: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
