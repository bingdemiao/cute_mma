"""Cute-backend determinism / reproducibility (fp16).

A correct kernel must be a pure function of its inputs: identical inputs (and,
for dropout, identical seeds) must give bit-identical outputs across calls. A
mismatch reveals a race or uninitialized read in the new feature code paths.

  * forward + backward, for plain-gated / internal_bias / input_shuffle: two
    calls with identical inputs must be exactly equal (torch.equal).
  * dropout backward: two calls with the SAME seed tensor must be exactly equal
    (the splitmix64 mask replay is deterministic). (forward dropout is *not*
    determinism-checked: each forward draws fresh seeds internally by design.)

Requires a CUDA GPU and the cute backend.

Usage::

    CUTE_PRISM_COMPILE_WORKERS=2 python tests/test_cute_determinism.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism
from cute_prism._module import _build_shuffle_masks


def _inputs(M, N, K, gs, rs, seed):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")
    return A, B, R, dC


def _check(name, fwd_pair, bwd_pair, verbose) -> bool:
    ok_fwd = fwd_pair is None or torch.equal(*fwd_pair)
    ok_bwd = all(torch.equal(a, b) for a, b in bwd_pair)
    ok = ok_fwd and ok_bwd
    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag} {name:18s} fwd_equal={ok_fwd} bwd_equal={ok_bwd}")
    return ok


def _run_feature(name, M, N, K, gs, rs, *, ib, shuffle, verbose) -> bool:
    A, B, R, dC = _inputs(M, N, K, gs, rs, seed=0)
    bias = torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1 if ib else None
    masks = None
    if shuffle:
        masks, _ = _build_shuffle_masks(K, N, gs, rs, shuffle_blk_k=64)
        masks = masks.to("cuda")

    C1, _ = cute_prism.forward(A, B, R, gs, rs, backend="cute",
                               activation="silu_gate", internal_bias=bias, shuffle_masks=masks)
    C2, _ = cute_prism.forward(A, B, R, gs, rs, backend="cute",
                               activation="silu_gate", internal_bias=bias, shuffle_masks=masks)

    g1 = cute_prism.backward(dC, A, B, R, gs, rs, backend="cute",
                             activation="silu_gate", internal_bias=bias, shuffle_masks=masks)
    g2 = cute_prism.backward(dC, A, B, R, gs, rs, backend="cute",
                             activation="silu_gate", internal_bias=bias, shuffle_masks=masks)
    pairs = [(a, b) for a, b in zip(g1, g2) if a is not None and b is not None]
    return _check(name, (C1, C2), pairs, verbose)


def _run_dropout_bwd(M, N, K, gs, rs, dropout_p, verbose) -> bool:
    A, B, R, dC = _inputs(M, N, K, gs, rs, seed=0)
    # Fixed seeds from one cute forward; replay through two backwards.
    _C, extra = cute_prism.forward(A, B, R, gs, rs, backend="cute",
                                   activation="silu_gate", dropout_p=dropout_p, training=True)
    seeds = extra[0]
    g1 = cute_prism.backward(dC, A, B, R, gs, rs, backend="cute", activation="silu_gate",
                             dropout_seeds=seeds, dropout_p=dropout_p)
    g2 = cute_prism.backward(dC, A, B, R, gs, rs, backend="cute", activation="silu_gate",
                             dropout_seeds=seeds, dropout_p=dropout_p)
    pairs = [(a, b) for a, b in zip(g1, g2) if a is not None and b is not None]
    return _check("dropout_bwd(seed)", None, pairs, verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    M, N, K, gs, rs = 256, 256, 512, 64, 16

    passed = failed = 0
    for name, ib, shuffle in [
        ("plain_gated", False, False),
        ("internal_bias", True, False),
        ("input_shuffle", False, True),
        ("shuffle+bias", True, True),
    ]:
        ok = _run_feature(name, M, N, K, gs, rs, ib=ib, shuffle=shuffle, verbose=args.verbose)
        passed += ok
        failed += (not ok)

    ok = _run_dropout_bwd(M, N, K, gs, rs, 0.2, args.verbose)
    passed += ok
    failed += (not ok)

    print(f"Total: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
