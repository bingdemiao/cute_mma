"""Cute-backend forward dropout: statistical correctness (fp16).

There is no existing forward-dropout test for the cute backend, and a direct
cute-vs-cublas forward parity is not possible: each backend generates its own
per-group seeds internally (cute via torch.randint, cublas internally), so they
mask different elements. Instead we validate the two things that define a
correct inverted-dropout forward, independent of the seed:

  1. Mean-preservation: with inverted dropout (H *= mask / (1 - p)), the
     expectation E[C_drop] equals the no-dropout output C0. Averaging many
     stochastic forwards must converge to C0.
  2. Activity: a single dropout forward must differ from C0 (the mask is
     actually being applied), and the output must be finite.

The backward dropout path is parity-checked against cublas elsewhere
(test_dropout_bwd.py and test_cute_feature_cross.py) using shared seeds.

Requires a CUDA GPU and the cute backend.

Usage::

    CUTE_PRISM_COMPILE_WORKERS=2 python tests/test_cute_dropout_fwd.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism


def _mean_rel_err(val, ref) -> float:
    return ((val.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-6)).mean().item()


def _run_one(M, N, K, gs, rs, dropout_p, n_samples, seed, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")

    # No-dropout reference (also serves as the determinism anchor).
    C0, _ = cute_prism.forward(A, B, R, gs, rs, backend="cute", activation="silu_gate")
    assert torch.isfinite(C0).all(), "no-dropout cute forward produced non-finite values"

    # One stochastic forward must differ from C0 and stay finite.
    C1, _ = cute_prism.forward(
        A, B, R, gs, rs, backend="cute", activation="silu_gate",
        dropout_p=dropout_p, training=True)
    finite = torch.isfinite(C1).all().item()
    active = not torch.allclose(C0, C1)

    # Mean over many stochastic forwards should converge to C0 (inverted dropout
    # is mean-preserving). Accumulate in fp32.
    acc = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for _ in range(n_samples):
        Ci, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="cute", activation="silu_gate",
            dropout_p=dropout_p, training=True)
        acc += Ci.float()
    mean = (acc / n_samples).half()
    rel = _mean_rel_err(mean, C0)

    # Tolerance shrinks with sample count (~ std / sqrt(n)); generous constant.
    tol = 0.05
    ok = finite and active and rel < tol

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} p={dropout_p} "
              f"finite={finite} active={active} mean_rel={rel:.4f} (n={n_samples})")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    cases = [
        (256, 256, 256, 256, 8),
        (512, 512, 512, 256, 16),
    ]
    dropout_ps = [0.1, 0.3]

    passed = failed = 0
    for M, N, K, gs, rs in cases:
        for p in dropout_ps:
            ok = _run_one(M, N, K, gs, rs, p, args.samples, seed=0, verbose=args.verbose)
            passed += ok
            failed += (not ok)

    print(f"Total: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
