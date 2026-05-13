"""Backward parity tests for the cute backend's dropout support.

Compares the **cute** and **cublas** backends on the gated backward pass with
dropout active. These two share a bit-identical splitmix64 mask
implementation in their CUDA kernels (per CLAUDE.md), so given the same
per-group seeds and dropout_p they must recompute the same H during backward
and therefore produce the same dA, dR, dB up to fp16 noise.

The pytorch backend is excluded from the dropout-on parity check because
``_pytorch_backend.py`` uses ``torch.rand(generator=Generator(seed))`` to
build its mask — structurally correct, deterministic, but **not** bit-
identical to the kernels' splitmix64 hash. It is included for the
``dropout_p=0.0`` control case so the test still covers all three backends
on the dropout-off path.

Usage:
    python tests/test_dropout_bwd.py
    python tests/test_dropout_bwd.py --clear-cache    # after .cu changes
    python tests/test_dropout_bwd.py -v
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


def _seeds_to_tensor(seeds, device) -> torch.Tensor:
    if isinstance(seeds, torch.Tensor):
        return seeds.to(device=device, dtype=torch.int64)
    return torch.tensor(list(seeds), dtype=torch.int64, device=device)


def _seeds_to_list(seeds) -> list[int]:
    if isinstance(seeds, torch.Tensor):
        return seeds.tolist()
    return list(seeds)


def _run_one(M, N, K, gs, rs, seed, dropout_p, verbose) -> bool:
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(N, K, dtype=torch.float16, device="cuda")
    R = torch.randn(N // gs * rs, K, dtype=torch.float16, device="cuda")
    dC = torch.randn(M, N, dtype=torch.float16, device="cuda")

    if dropout_p > 0.0:
        # Use the cute forward to allocate per-group seeds; the same tensor is
        # then passed in to all three backends' backward so the recompute
        # replays the same mask.
        _C, seeds_extra = cute_prism.forward(
            A, B, R, gs, rs,
            backend="cute", activation="silu_gate",
            dropout_p=dropout_p, training=True,
        )
        assert seeds_extra and len(seeds_extra) > 0, "cute forward did not return dropout seeds"
        seeds_t = seeds_extra[0]  # int64 tensor on GPU
    else:
        seeds_t = None

    dA_cute, dR_cute, dB_cute, _ = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cute", activation="silu_gate",
        dropout_seeds=seeds_t, dropout_p=dropout_p,
    )
    dA_cb, dR_cb, dB_cb, _ = cute_prism.backward(
        dC, A, B, R, gs, rs,
        backend="cublas", activation="silu_gate",
        dropout_seeds=seeds_t, dropout_p=dropout_p,
    )
    rel_dA_cb = _max_rel_err(dA_cute, dA_cb)
    rel_dR_cb = _max_rel_err(dR_cute, dR_cb)
    rel_dB_cb = _max_rel_err(dB_cute, dB_cb)
    rtol = 0.02
    ok = (rel_dA_cb < rtol and rel_dR_cb < rtol and rel_dB_cb < rtol)

    # pytorch backend only on the dropout-off control: its mask uses
    # torch.Generator(seed) instead of splitmix64, so it deliberately differs
    # from the kernel masks under dropout.
    rel_dA_pt = rel_dR_pt = rel_dB_pt = float("nan")
    if dropout_p == 0.0:
        seeds_l = _seeds_to_list(seeds_t) if seeds_t is not None else None
        dA_pt, dR_pt, dB_pt, _ = cute_prism.backward(
            dC, A, B, R, gs, rs,
            backend="pytorch", activation="silu_gate",
            dropout_seeds=seeds_l, dropout_p=dropout_p,
        )
        rel_dA_pt = _max_rel_err(dA_cute, dA_pt)
        rel_dR_pt = _max_rel_err(dR_cute, dR_pt)
        rel_dB_pt = _max_rel_err(dB_cute, dB_pt)
        ok = ok and (rel_dA_pt < rtol and rel_dR_pt < rtol and rel_dB_pt < rtol)

    if verbose or not ok:
        tag = "PASS" if ok else "FAIL"
        if dropout_p == 0.0:
            print(
                f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} seed={seed} dp={dropout_p} "
                f"|dA-cb|={rel_dA_cb:.4f} |dR-cb|={rel_dR_cb:.4f} |dB-cb|={rel_dB_cb:.4f} "
                f"|dA-pt|={rel_dA_pt:.4f} |dR-pt|={rel_dR_pt:.4f} |dB-pt|={rel_dB_pt:.4f}"
            )
        else:
            print(
                f"  {tag} M={M} N={N} K={K} gs={gs} rs={rs} seed={seed} dp={dropout_p} "
                f"|dA-cb|={rel_dA_cb:.4f} |dR-cb|={rel_dR_cb:.4f} |dB-cb|={rel_dB_cb:.4f}"
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
    dropout_ps = [0.0, 0.1, 0.3]

    passed = failed = 0
    for dp in dropout_ps:
        if args.verbose:
            print(f"--- dropout_p = {dp} ---")
        for M, N, K, gs, rs in cases:
            for seed in range(args.seeds):
                ok = _run_one(M, N, K, gs, rs, seed * 100, dp, args.verbose)
                if ok:
                    passed += 1
                else:
                    failed += 1

    print(f"Total: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
