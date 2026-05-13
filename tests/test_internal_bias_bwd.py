"""Backward parity tests for the cute backend's internal_bias support.

Compares cute, cublas, and pytorch backends on the gated backward pass with
``internal_bias`` set, checking dA, dR, and dB.

Note: this test does not yet check d_internal_bias — that comes in B.3.

Usage:
    python tests/test_internal_bias_bwd.py
    python tests/test_internal_bias_bwd.py --clear-cache    # after .cu changes
    python tests/test_internal_bias_bwd.py -v
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

    dA_cute, dR_cute, dB_cute, _ = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cute", activation="silu_gate", internal_bias=bias,
    )
    dA_cb, dR_cb, dB_cb, _ = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cublas", activation="silu_gate", internal_bias=bias,
    )
    dA_pt, dR_pt, dB_pt, _ = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="pytorch", activation="silu_gate", internal_bias=bias,
    )

    rel_dA_cb = _max_rel_err(dA_cute, dA_cb)
    rel_dR_cb = _max_rel_err(dR_cute, dR_cb)
    rel_dB_cb = _max_rel_err(dB_cute, dB_cb)
    rel_dA_pt = _max_rel_err(dA_cute, dA_pt)
    rel_dR_pt = _max_rel_err(dR_cute, dR_pt)
    rel_dB_pt = _max_rel_err(dB_cute, dB_pt)
    # Backward tolerance is looser than forward — the silu' gate path
    # composes several fp16 multiplications. Match test_correctness.py's
    # default rtol=0.005 for forward; bump to 0.02 for backward.
    rtol = 0.02
    ok = (rel_dA_cb < rtol and rel_dR_cb < rtol and rel_dB_cb < rtol
          and rel_dA_pt < rtol and rel_dR_pt < rtol and rel_dB_pt < rtol)

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(
            f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} seed={seed} "
            f"|dA-cb|={rel_dA_cb:.4f} |dR-cb|={rel_dR_cb:.4f} |dB-cb|={rel_dB_cb:.4f} "
            f"|dA-pt|={rel_dA_pt:.4f} |dR-pt|={rel_dR_pt:.4f} |dB-pt|={rel_dB_pt:.4f}"
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
