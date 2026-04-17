"""Correctness test for the cute_prism cublas backend against torch autograd.

The cublas kernel uses a splitmix64-based hash for per-element dropout masks
(see `uniform_from_hash` in torch_ext/cublas_prism_torch.cu). To test the
manually-written cublas backward against autograd, we need a Python reference
that generates the *exact same* mask so the two backward paths are comparing
gradients of the same function.

For each combination of {shuffle, gated, internal_bias, dropout}, we:

  1. Run cute_prism.forward / cute_prism.backward (cublas backend) to get
     manually-computed gradients; the forward returns the per-group seeds it
     used.
  2. Re-express the forward as autograd-traceable torch ops, using a mask
     generated from the SAME (seed, idx) hash as the CUDA kernel.
  3. Compare every gradient (dA, dR, dB, d_internal_bias) against autograd.

Usage:
    uv run python tests/test_cublas_backend.py
"""

from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
from einops import rearrange

import cute_prism
from cute_prism._pytorch_backend import _shuffle_masks_to_elem_perm


# ---------------------------------------------------------------------------
# splitmix64-based dropout mask, matching torch_ext/cublas_prism_torch.cu
# ---------------------------------------------------------------------------

_U64_MASK = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9e3779b97f4a7c15) & _U64_MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & _U64_MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & _U64_MASK
    return (x ^ (x >> 31)) & _U64_MASK


def _uniform_from_hash(seed: int, idx: int) -> float:
    h1 = _splitmix64((seed + 0xdeadbeef) & _U64_MASK)
    h2 = _splitmix64(idx & _U64_MASK)
    h = _splitmix64(h1 ^ h2)
    return ((h >> 40) & 0xFFFFFF) * (1.0 / 16777216.0)


def build_cublas_dropout_mask(seed: int, shape, dropout_p: float, device, dtype):
    """Return a (M, K) mask tensor in {0, 1} matching the CUDA kernel.

    kernel: if (u >= dropout_p) keep*inv_keep else 0
    We return just the keep (0/1) so callers can multiply by inv_keep.
    Kernel index layout: idx = row*K + col (linear over numel = M*K).
    """
    M, K = shape
    numel = M * K
    seed_u = int(seed) & _U64_MASK
    vals = [1.0 if _uniform_from_hash(seed_u, i) >= dropout_p else 0.0
            for i in range(numel)]
    t = torch.tensor(vals, dtype=torch.float32).reshape(M, K)
    return t.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Autograd reference using the cublas mask generator
# ---------------------------------------------------------------------------

def _build_shuffle_masks(in_features, out_features, group_size, reconn_sz, shuffle_blk_k):
    from cute_prism._module import _build_shuffle_masks as _f
    masks, _deltas = _f(in_features, out_features, group_size, reconn_sz, shuffle_blk_k)
    return masks


def autograd_reference(
    A, B, R, group_size, reconn_sz,
    activation=None, internal_bias=None,
    dropout_p=0.0, dropout_seeds=None,
    shuffle_masks=None,
):
    """Pure autograd-traceable forward, using the splitmix64 dropout mask."""
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
        dropout_seeds is not None
        and len(dropout_seeds) > 0
        and dropout_p > 0.0
    )
    inv_keep = 1.0 / (1.0 - dropout_p) if apply_dropout else 1.0

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
                seed = int(dropout_seeds[g])
                mask = build_cublas_dropout_mask(
                    seed, H.shape, dropout_p, H.device, H.dtype)
                H = H * mask * inv_keep
            mult = H
        else:
            mult = S_g

        B_g = B[g * group_size : (g + 1) * group_size, :]
        C_parts.append(mult @ B_g.T)

    return torch.cat(C_parts, dim=1)


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def max_abs_err(x, y):
    return (x.float() - y.float()).abs().max().item()


def rel_err(x, y):
    denom = y.float().abs().max().item() + 1e-6
    return max_abs_err(x, y) / denom


def run_one(name, M, K, N, gs, rs, activation, has_bias, dropout_p, shuffle,
            dtype, rtol, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    n_groups = N // gs

    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
    R = torch.randn(n_groups * rs, K, device=device, dtype=dtype, requires_grad=True)

    internal_bias = None
    if has_bias:
        internal_bias = torch.randn(
            n_groups, K, device=device, dtype=dtype, requires_grad=True)

    shuffle_masks = None
    if shuffle:
        shuffle_masks = _build_shuffle_masks(K, N, gs, rs, shuffle_blk_k=64).to(device)

    # 1) Cublas forward.
    C_manual, dropout_seeds = cute_prism.forward(
        A, B, R, gs, rs,
        backend="cublas",
        activation=activation,
        internal_bias=internal_bias,
        dropout_p=dropout_p,
        training=(dropout_p > 0.0),
        shuffle_masks=shuffle_masks,
    )

    # 2) Cublas backward (with the saved seeds).
    dC = torch.randn_like(C_manual)
    dA_manual, dR_manual, dB_manual, d_ib_manual = cute_prism.backward(
        dC,
        A.detach(), B.detach(), R.detach(), gs, rs,
        backend="cublas",
        activation=activation,
        internal_bias=internal_bias.detach() if internal_bias is not None else None,
        dropout_seeds=dropout_seeds,
        dropout_p=dropout_p,
        shuffle_masks=shuffle_masks,
    )

    # 3) Autograd reference using the same hash-based mask.
    A2 = A.detach().clone().requires_grad_()
    B2 = B.detach().clone().requires_grad_()
    R2 = R.detach().clone().requires_grad_()
    ib2 = None
    if internal_bias is not None:
        ib2 = internal_bias.detach().clone().requires_grad_()
    C_auto = autograd_reference(
        A2, B2, R2, gs, rs,
        activation=activation,
        internal_bias=ib2,
        dropout_p=dropout_p,
        dropout_seeds=dropout_seeds,
        shuffle_masks=shuffle_masks,
    )

    fwd_err = rel_err(C_manual, C_auto)

    # Backprop the same dC.
    C_auto.backward(dC)
    dA_auto = A2.grad
    dB_auto = B2.grad
    dR_auto = R2.grad
    d_ib_auto = ib2.grad if ib2 is not None else None

    dA_e = rel_err(dA_manual, dA_auto)
    dR_e = rel_err(dR_manual, dR_auto)
    dB_e = 0.0
    if dB_manual is not None and dB_auto is not None:
        dB_e = rel_err(dB_manual, dB_auto)
    d_ib_e = 0.0
    if d_ib_manual is not None and d_ib_auto is not None:
        d_ib_e = rel_err(d_ib_manual, d_ib_auto)

    worst = max(fwd_err, dA_e, dR_e, dB_e, d_ib_e)
    ok = worst < rtol
    status = "PASS" if ok else "FAIL"
    print(
        f"  {status} {name:40s} "
        f"fwd={fwd_err:.2e} dA={dA_e:.2e} dR={dR_e:.2e} "
        f"dB={dB_e:.2e} d_ib={d_ib_e:.2e}"
    )
    return ok


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return

    # cublas only supports fp16 / bf16. fp16 has more mantissa so we use it
    # for tighter numerical agreement in the tests.
    dtype = torch.float16
    M, K, N = 32, 128, 128
    gs, rs = 32, 16
    # Loose tolerance because both sides round in fp16; we only care about
    # gradient-formula correctness, which would manifest as much larger errors.
    rtol = 2e-2

    configs = []
    for shuffle in [False, True]:
        for activation in [None, "silu_gate"]:
            for has_bias in [False, True]:
                if has_bias and activation != "silu_gate":
                    continue
                for dropout_p in [0.0, 0.3]:
                    if dropout_p > 0.0 and activation != "silu_gate":
                        continue
                    configs.append((shuffle, activation, has_bias, dropout_p))

    print(f"Testing {len(configs)} configs "
          f"(fp16, M={M} K={K} N={N} gs={gs} rs={rs}, rtol={rtol})")
    print("-" * 90)

    n_pass, n_fail = 0, 0
    for shuffle, activation, has_bias, dropout_p in configs:
        name = (
            f"{'shuf' if shuffle else 'nat ':4s} "
            f"{'gated' if activation else 'plain':5s} "
            f"{'bias' if has_bias else '    '} "
            f"{'drop' if dropout_p else '    '}"
        )
        ok = run_one(
            name, M, K, N, gs, rs,
            activation=activation,
            has_bias=has_bias,
            dropout_p=dropout_p,
            shuffle=shuffle,
            dtype=dtype,
            rtol=rtol,
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    print("-" * 90)
    print(f"Total: {n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
