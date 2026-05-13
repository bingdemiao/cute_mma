"""Backward parity test for the cute backend's input_shuffle support.

Verifies dA, dR, dB, d_internal_bias parity vs cublas + pytorch with
shuffle_masks set, both with and without internal_bias.

Usage:
    python tests/test_input_shuffle_bwd.py
    python tests/test_input_shuffle_bwd.py --clear-cache
    python tests/test_input_shuffle_bwd.py -v
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
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

    masks, _ = _build_shuffle_masks(
        in_features=K, out_features=N, group_size=gs,
        reconn_sz=rs, shuffle_blk_k=64,
    )
    masks = masks.to("cuda")

    dA_cute, dR_cute, dB_cute, dIB_cute = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cute", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )
    dA_cb, dR_cb, dB_cb, dIB_cb = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cublas", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )
    dA_pt, dR_pt, dB_pt, dIB_pt = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="pytorch", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
    )

    rtol = 0.02
    rel_dA = max(_max_rel_err(dA_cute, dA_cb), _max_rel_err(dA_cute, dA_pt))
    rel_dR = max(_max_rel_err(dR_cute, dR_cb), _max_rel_err(dR_cute, dR_pt))
    rel_dB = max(_max_rel_err(dB_cute, dB_cb), _max_rel_err(dB_cute, dB_pt))
    rel_dIB = 0.0
    if with_bias:
        rel_dIB = max(_max_rel_err(dIB_cute, dIB_cb), _max_rel_err(dIB_cute, dIB_pt))
    ok = rel_dA < rtol and rel_dR < rtol and rel_dB < rtol and rel_dIB < rtol

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        bias_tag = "+bias" if with_bias else "     "
        print(
            f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} {bias_tag} seed={seed} "
            f"dA={rel_dA:.4f} dR={rel_dR:.4f} dB={rel_dB:.4f} dIB={rel_dIB:.4f}"
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

    cases = [
        (256, 256, 256,  256, 16),
        (512, 512, 512,  256, 16),
        (1024, 1024, 1024, 256, 16),
        (512, 256, 512,  256, 16),
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
