"""Wall-time benchmark: cublas backend vs pytorch backend vs torch autograd.

Measures forward + backward latency for the Prism operator at
M = N = K = 4096, group_sz = 64, reconn_sz = 16 across the valid
combinations of {shuffle, gated, internal_bias, dropout}.

The "autograd" baseline re-expresses the forward as pure torch ops (the
traceable reference used in tests/test_pytorch_backend.py) and runs backward
through torch.autograd. It represents what a naive implementation written
directly in PyTorch would achieve — this is the "no custom kernel" baseline
the cublas backend has to beat.

Usage:
    uv run python tests/bench_cublas.py
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
from einops import rearrange

import cute_prism
from cute_prism._pytorch_backend import (
    _dropout_mask_from_seed,
    _shuffle_masks_to_elem_perm,
)
from cute_prism._module import _build_shuffle_masks


# ---------------------------------------------------------------------------
# Autograd reference — pure traceable forward (same as test_pytorch_backend)
# ---------------------------------------------------------------------------

def autograd_reference(
    A, B, R, group_size, reconn_sz,
    activation=None, internal_bias=None,
    dropout_p=0.0, dropout_seeds=None,
    shuffle_masks=None,
):
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    R_blocks = rearrange(
        R, "(g rs) (b rs2) -> g b rs rs2",
        g=n_groups, rs=reconn_sz, b=n_blocks, rs2=reconn_sz,
    )

    gated = activation == "silu_gate"
    apply_dropout = (
        dropout_seeds is not None and len(dropout_seeds) > 0 and dropout_p > 0.0
    )

    C_parts = []
    for g in range(n_groups):
        if shuffle_masks is not None:
            perm = _shuffle_masks_to_elem_perm(shuffle_masks[g], K, 64, reconn_sz)
            A_g = A.index_select(1, perm)
        else:
            A_g = A

        A_blocks = rearrange(A_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
        AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
        S_g = rearrange(AR_blocks, "b m rs -> m (b rs)")

        if gated:
            if internal_bias is not None:
                S_g = S_g + internal_bias[g]
            H = A_g * F.silu(S_g)
            if apply_dropout:
                mask = _dropout_mask_from_seed(
                    dropout_seeds[g], H.shape, dropout_p, H.device, H.dtype)
                H = H * mask / (1.0 - dropout_p)
            mult = H
        else:
            mult = S_g

        B_g = B[g * group_size : (g + 1) * group_size, :]
        C_parts.append(mult @ B_g.T)

    return torch.cat(C_parts, dim=1)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def cuda_time(fn, warmup=1, iters=3):
    """Return median wall time in ms for one call of fn()."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


# Autograd + pytorch backend loop over 64 groups in Python with block-level
# inner loops, so each call at M=N=K=4096 is on the order of a second. Keep
# their iteration count very low so the overall bench stays tractable.
AUTOGRAD_ITERS = 1
AUTOGRAD_WARMUP = 0
PYTORCH_ITERS = 1
PYTORCH_WARMUP = 0
CUBLAS_ITERS = 5
CUBLAS_WARMUP = 2


# ---------------------------------------------------------------------------
# Per-config benchmark runners
# ---------------------------------------------------------------------------

def make_inputs(M, K, N, gs, rs, has_bias, shuffle, dtype, device):
    torch.manual_seed(0)
    n_groups = N // gs
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(N, K, device=device, dtype=dtype)
    R = torch.randn(n_groups * rs, K, device=device, dtype=dtype)
    internal_bias = None
    if has_bias:
        internal_bias = torch.randn(n_groups, K, device=device, dtype=dtype)
    shuffle_masks = None
    if shuffle:
        shuffle_masks, _deltas = _build_shuffle_masks(K, N, gs, rs, shuffle_blk_k=64)
        shuffle_masks = shuffle_masks.to(device)
    dC = torch.randn(M, N, device=device, dtype=dtype)
    return A, B, R, internal_bias, shuffle_masks, dC


def bench_backend(
    backend, M, K, N, gs, rs,
    activation, has_bias, dropout_p, shuffle,
    dtype, device, warmup, iters,
):
    A, B, R, internal_bias, shuffle_masks, dC = make_inputs(
        M, K, N, gs, rs, has_bias, shuffle, dtype, device)

    def fwd_only():
        _C, _seeds = cute_prism.forward(
            A, B, R, gs, rs,
            backend=backend,
            activation=activation,
            internal_bias=internal_bias,
            dropout_p=dropout_p,
            training=(dropout_p > 0.0),
            shuffle_masks=shuffle_masks,
        )

    # Forward returns seeds; capture once for the bwd-only timing.
    _C, seeds = cute_prism.forward(
        A, B, R, gs, rs,
        backend=backend,
        activation=activation,
        internal_bias=internal_bias,
        dropout_p=dropout_p,
        training=(dropout_p > 0.0),
        shuffle_masks=shuffle_masks,
    )

    def fwd_bwd():
        C, seeds_local = cute_prism.forward(
            A, B, R, gs, rs,
            backend=backend,
            activation=activation,
            internal_bias=internal_bias,
            dropout_p=dropout_p,
            training=(dropout_p > 0.0),
            shuffle_masks=shuffle_masks,
        )
        cute_prism.backward(
            dC, A, B, R, gs, rs,
            backend=backend,
            activation=activation,
            internal_bias=internal_bias,
            dropout_seeds=seeds_local,
            dropout_p=dropout_p,
            shuffle_masks=shuffle_masks,
        )

    fwd_ms = cuda_time(fwd_only, warmup=warmup, iters=iters)
    full_ms = cuda_time(fwd_bwd, warmup=warmup, iters=iters)
    return fwd_ms, full_ms


def bench_autograd(
    M, K, N, gs, rs,
    activation, has_bias, dropout_p, shuffle,
    dtype, device, warmup, iters,
):
    A, B, R, internal_bias, shuffle_masks, dC = make_inputs(
        M, K, N, gs, rs, has_bias, shuffle, dtype, device)

    # Matches PrismLinear semantics: B is frozen in non-gated (finetuning)
    # mode and only trainable in gated mode. We mirror that here so autograd
    # doesn't pay for computing dB when we wouldn't use it.
    gated = activation == "silu_gate"

    dropout_seeds = None
    if dropout_p > 0.0 and gated:
        n_groups = N // gs
        dropout_seeds = [int(i + 1) for i in range(n_groups)]

    def make_leaves():
        A_l = A.detach().clone().requires_grad_()
        R_l = R.detach().clone().requires_grad_()
        if gated:
            B_l = B.detach().clone().requires_grad_()
        else:
            B_l = B  # frozen, autograd sees it as a constant
        ib_l = None
        if internal_bias is not None:
            ib_l = internal_bias.detach().clone().requires_grad_()
        return A_l, B_l, R_l, ib_l

    def fwd_only():
        A_l, B_l, R_l, ib_l = make_leaves()
        _C = autograd_reference(
            A_l, B_l, R_l, gs, rs,
            activation=activation,
            internal_bias=ib_l,
            dropout_p=dropout_p,
            dropout_seeds=dropout_seeds,
            shuffle_masks=shuffle_masks,
        )

    def fwd_bwd():
        A_l, B_l, R_l, ib_l = make_leaves()
        C = autograd_reference(
            A_l, B_l, R_l, gs, rs,
            activation=activation,
            internal_bias=ib_l,
            dropout_p=dropout_p,
            dropout_seeds=dropout_seeds,
            shuffle_masks=shuffle_masks,
        )
        C.backward(dC)

    fwd_ms = cuda_time(fwd_only, warmup=warmup, iters=iters)
    full_ms = cuda_time(fwd_bwd, warmup=warmup, iters=iters)
    return fwd_ms, full_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    M = K = N = 4096
    gs, rs = 64, 16

    configs = []
    for shuffle in [False, True]:
        for activation in [None, "silu_gate"]:
            for has_bias in [False, True]:
                if has_bias and activation != "silu_gate":
                    continue
                for dropout_p in [0.0, 0.2]:
                    if dropout_p > 0.0 and activation != "silu_gate":
                        continue
                    configs.append((shuffle, activation, has_bias, dropout_p))

    print(
        f"Benchmark: M=N=K={M}, group_sz={gs}, reconn_sz={rs}, dtype={dtype}\n"
        f"Times are medians over 10 iters (ms). fwd = forward only, "
        f"f+b = forward + backward.\n"
    )

    header = (
        f"{'config':35s}  "
        f"{'autograd':>20s}  {'cublas':>20s}  {'speedup':>14s}"
    )
    print(header)
    print(
        f"{'':35s}  "
        f"{'fwd':>9s} {'f+b':>10s}  "
        f"{'fwd':>9s} {'f+b':>10s}  "
        f"{'vs autograd':>14s}"
    )
    print("-" * 100)

    for shuffle, activation, has_bias, dropout_p in configs:
        name = (
            f"{'shuf' if shuffle else 'nat ':4s} "
            f"{'gated' if activation else 'plain':5s} "
            f"{'bias' if has_bias else '    '} "
            f"{'drop' if dropout_p else '    '}"
        )
        print(f"[{name}] running…", flush=True)

        print("  autograd…", flush=True, end=" ")
        auto_f, auto_fb = bench_autograd(
            M, K, N, gs, rs, activation, has_bias, dropout_p, shuffle,
            dtype, device, warmup=AUTOGRAD_WARMUP, iters=AUTOGRAD_ITERS)
        print(f"fwd={auto_f:.0f}ms  f+b={auto_fb:.0f}ms", flush=True)

        print("  cublas…  ", flush=True, end=" ")
        cub_f, cub_fb = bench_backend(
            "cublas", M, K, N, gs, rs, activation, has_bias, dropout_p,
            shuffle, dtype, device,
            warmup=CUBLAS_WARMUP, iters=CUBLAS_ITERS)
        print(f"fwd={cub_f:.2f}ms  f+b={cub_fb:.2f}ms", flush=True)

        spd_vs_auto = auto_fb / cub_fb

        print(
            f"{name:35s}  "
            f"{auto_f:9.2f} {auto_fb:10.2f}  "
            f"{cub_f:9.2f} {cub_fb:10.2f}  "
            f"{spd_vs_auto:13.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
