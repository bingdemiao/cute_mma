"""Cute-backend bf16 parity for the new features.

Every existing cute parity test runs in **fp16**, but gated PrismLinear is
trained in **bf16** (per TRAINING_GATED_PRISM.md, "cublas + bf16 is the
supported training configuration"). bf16 has a wider exponent but only ~8
mantissa bits, so its accumulation/rounding differs materially from fp16 — a
kernel can be fp16-correct yet have a bf16 bug (e.g. an intermediate that
relies on fp16 precision, or a packed-half2 path with no bf16 analogue).

This test re-runs the feature parity checks (internal_bias, input_shuffle,
dropout) in bf16, cute vs the trusted cublas reference.

Requires a CUDA GPU and the cute backend.

Usage::

    CUTE_PRISM_COMPILE_WORKERS=2 python tests/test_cute_bf16.py -v
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUTE_PRISM_COMPILE_WORKERS", "2")

import torch

import cute_prism
from cute_prism._module import _build_shuffle_masks

DT = torch.bfloat16
# bf16 has ~3 decimal digits; tolerances are looser than the fp16 suite.
RTOL_FWD = 0.03
RTOL_BWD = 0.06


def _max_rel_err(val, ref) -> float:
    return ((val.float() - ref.float()).abs() / (ref.float().abs().max() + 1e-6)).max().item()


def _run_feature(name, M, N, K, gs, rs, *, ib_on, shuffle, drop_p, seed, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=DT, device="cuda")
    B = torch.randn(N, K, dtype=DT, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=DT, device="cuda")
    dC = torch.randn(M, N, dtype=DT, device="cuda")

    bias = torch.randn(N // gs, K, dtype=DT, device="cuda") * 0.1 if ib_on else None
    masks = None
    if shuffle:
        masks, _ = _build_shuffle_masks(K, N, gs, rs, shuffle_blk_k=64)
        masks = masks.to("cuda")

    rel = {}
    ok = True

    if drop_p == 0.0:
        C_cute, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="cute", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks)
        C_cb, _ = cute_prism.forward(
            A, B, R, gs, rs, backend="cublas", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks)
        rel["fwd"] = _max_rel_err(C_cute, C_cb)
        ok = ok and rel["fwd"] < RTOL_FWD

    seeds_t = None
    if drop_p > 0.0:
        _C, extra = cute_prism.forward(
            A, B, R, gs, rs, backend="cute", activation="silu_gate",
            internal_bias=bias, shuffle_masks=masks, dropout_p=drop_p, training=True)
        seeds_t = extra[0]

    dA_c, dR_c, dB_c, dIB_c = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cute", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks, dropout_seeds=seeds_t, dropout_p=drop_p)
    dA_b, dR_b, dB_b, dIB_b = cute_prism.backward(
        dC, A, B, R, gs, rs, backend="cublas", activation="silu_gate",
        internal_bias=bias, shuffle_masks=masks, dropout_seeds=seeds_t, dropout_p=drop_p)

    rel["dA"] = _max_rel_err(dA_c, dA_b)
    rel["dR"] = _max_rel_err(dR_c, dR_b)
    rel["dB"] = _max_rel_err(dB_c, dB_b)
    ok = ok and all(rel[k] < RTOL_BWD for k in ("dA", "dR", "dB"))
    if ib_on:
        rel["dIB"] = _max_rel_err(dIB_c, dIB_b)
        ok = ok and rel["dIB"] < RTOL_BWD

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        relstr = " ".join(f"{k}={v:.4f}" for k, v in rel.items())
        print(f"  {tag} {name:18s} {relstr}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        cute_prism.clear_cache()

    M, N, K, gs, rs = 256, 256, 512, 256, 16

    # (name, ib_on, shuffle, dropout_p)
    features = [
        ("plain_gated", False, False, 0.0),
        ("internal_bias", True, False, 0.0),
        ("input_shuffle", False, True, 0.0),
        ("shuffle+bias", True, True, 0.0),
        ("dropout", False, False, 0.2),
        ("dropout+bias", True, False, 0.2),
    ]

    passed = failed = 0
    for name, ib_on, shuffle, dp in features:
        for s in range(args.seeds):
            ok = _run_feature(name, M, N, K, gs, rs, ib_on=ib_on, shuffle=shuffle,
                              drop_p=dp, seed=s * 100 + 1, verbose=args.verbose)
            passed += ok
            failed += (not ok)

    print(f"Total: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
