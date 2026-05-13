"""Forward parity test for the cute backend's input_shuffle support.

Compares cute, cublas, and pytorch backends on the gated forward pass with
non-trivial ``shuffle_masks`` set.

Usage:
    python tests/test_input_shuffle_fwd.py
    python tests/test_input_shuffle_fwd.py --clear-cache
    python tests/test_input_shuffle_fwd.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism
from cute_prism._module import _build_shuffle_masks


def _max_rel_err(val: torch.Tensor, ref: torch.Tensor) -> float:
    return ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()


def _run_one(M, N, K, gs, rs, seed, with_bias, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    bias = (torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1
            if with_bias else None)

    masks, _ = _build_shuffle_masks(
        in_features=K, out_features=N, group_size=gs,
        reconn_sz=rs, shuffle_blk_k=64,
    )
    masks = masks.to("cuda")

    C_cute, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="cute", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )
    C_cb, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="cublas", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )
    C_pt, _ = cute_prism.forward(
        A, B, R, gs, rs,
        backend="pytorch", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )

    rel_cb = _max_rel_err(C_cute, C_cb)
    rel_pt = _max_rel_err(C_cute, C_pt)
    ok = rel_cb < 0.005 and rel_pt < 0.005

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        bias_tag = "+bias" if with_bias else "     "
        print(
            f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} {bias_tag} seed={seed} "
            f"|cute-cb|={rel_cb:.5f} |cute-pt|={rel_pt:.5f}"
        )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    # input_shuffle requires reconn_sz=16 (per PrismLinear's ctor)
    cases = [
        # (M, N, K, gs, rs)
        (256, 256, 256,  256, 16),
        (512, 512, 512,  256, 16),
        (1024, 1024, 1024, 256, 16),
        (512, 256, 512,  256, 16),  # bN < gs (one group spans 2 CTAs)
    ]

    passed = failed = 0
    for M, N, K, gs, rs in cases:
        for with_bias in (False, True):
            for seed in range(args.seeds):
                ok = _run_one(M, N, K, gs, rs, seed * 100 + 1, with_bias, args.verbose)
                if ok:
                    passed += 1
                else:
                    failed += 1

    print(f"Total: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
