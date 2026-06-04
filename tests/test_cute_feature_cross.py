"""Cute-backend feature cross-product parity (fp16).

The existing cute parity tests cover features mostly in isolation
(internal_bias alone, shuffle alone, shuffle+bias, dropout alone). The pytorch
backend, by contrast, is autograd-checked across the full
{shuffle x internal_bias x dropout} cross-product (test_pytorch_backend.py).
This test mirrors that cross-product **for the cute kernel**, against the
trusted cublas reference (and pytorch where masks match), so feature
*interactions* in the cute kernel are exercised, not just each feature alone.

  * Forward parity (cute vs cublas vs pytorch) for the 4 dropout-off combos.
  * Backward parity (cute vs cublas) for all 8 combos; dropout-on combos share
    the cute-generated per-group seeds (bit-identical splitmix64 mask), so the
    recompute replays the same mask in both backends.

reconn_sz is fixed at 16 (required by input_shuffle; kept uniform across combos).
Requires a CUDA GPU and the cute backend (run on GH200 or the A100/Ault target).

Usage::

    CUTE_PRISM_COMPILE_WORKERS=2 python tests/test_cute_feature_cross.py -v
    python tests/test_cute_feature_cross.py --clear-cache
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism
from cute_prism._module import _build_shuffle_masks


def _max_rel_err(val: torch.Tensor, ref: torch.Tensor) -> float:
    return ((val - ref).abs() / (ref.abs().max() + 1e-6)).max().item()


def _run_one(M, N, K, gs, rs, shuffle, ib_on, drop_on, dropout_p, seed, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

    bias = None
    if ib_on:
        bias = torch.randn(N // gs, K, dtype=torch.float16, device="cuda") * 0.1

    masks = None
    if shuffle:
        masks, _ = _build_shuffle_masks(K, N, gs, rs, shuffle_blk_k=64)
        masks = masks.to("cuda")

    p = dropout_p if drop_on else 0.0
    ok = True
    rel = {}

    # ---- forward parity (only when dropout is off; dropout fwd is validated
    #      statistically in test_cute_dropout_fwd.py because the per-backend
    #      seed generation differs) ----
    if not drop_on:
        C_cute, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="cute", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks)
        C_cb, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="cublas", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks)
        C_pt, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="pytorch", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks)
        rel["fwd_cb"] = _max_rel_err(C_cute, C_cb)
        rel["fwd_pt"] = _max_rel_err(C_cute, C_pt)
        ok = ok and rel["fwd_cb"] < 0.005 and rel["fwd_pt"] < 0.005

    # ---- backward parity (cute vs cublas) for every combo ----
    seeds_t = None
    if drop_on:
        # Allocate seeds via the cute forward; replay the same mask in both
        # backends' backward.
        _C, seeds_extra = cute_prism.forward(
            A, B, R, gs, rs, backend="cute", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks,
            dropout_p=p, training=True)
        assert seeds_extra, "cute forward did not return dropout seeds"
        seeds_t = seeds_extra[0]

    dA_cute, dR_cute, dB_cute, dIB_cute = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cute", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
        dropout_seeds=seeds_t, dropout_p=p)
    dA_cb, dR_cb, dB_cb, dIB_cb = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cublas", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks,
        dropout_seeds=seeds_t, dropout_p=p)

    rtol = 0.02
    rel["dA"] = _max_rel_err(dA_cute, dA_cb)
    rel["dR"] = _max_rel_err(dR_cute, dR_cb)
    rel["dB"] = _max_rel_err(dB_cute, dB_cb)
    ok = ok and rel["dA"] < rtol and rel["dR"] < rtol and rel["dB"] < rtol
    if ib_on:
        rel["dIB"] = _max_rel_err(dIB_cute, dIB_cb)
        ok = ok and rel["dIB"] < rtol

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        feats = f"shuf={int(shuffle)} ib={int(ib_on)} drop={int(drop_on)}"
        relstr = " ".join(f"{k}={v:.4f}" for k, v in rel.items())
        print(f"  {tag} {feats}  {relstr}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--dropout-p", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    # gs=128 (>= forward bN) so the input_shuffle combos are valid on cute;
    # shuffle at gs<bN is a separate, guarded limitation (see KNOWN_ISSUES.md).
    M, N, K, gs, rs = 256, 256, 512, 128, 16

    passed = failed = 0
    for shuffle, ib_on, drop_on in itertools.product((False, True), repeat=3):
        for s in range(args.seeds):
            ok = _run_one(M, N, K, gs, rs, shuffle, ib_on, drop_on,
                          args.dropout_p, s * 100 + 1, args.verbose)
            passed += ok
            failed += (not ok)

    print(f"Total: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
