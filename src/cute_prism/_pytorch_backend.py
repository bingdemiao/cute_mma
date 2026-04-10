"""Pure PyTorch implementation of Prism forward and backward passes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


SHUFFLE_SEGMENT_SZ = 8  # must match _module.SHUFFLE_SEGMENT_SZ


def _dropout_mask_from_seed(
    seed: int, shape: tuple, dropout_p: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a deterministic dropout mask from a seed."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (torch.rand(shape, generator=gen, device=device, dtype=dtype) >= dropout_p).to(dtype)


def _seg_pairs_to_elem_perm(
    seg_pairs_g: torch.Tensor, K: int, reconn_sz: int,
) -> torch.Tensor:
    """Convert seg_pairs[g] to an element-level permutation of K columns.

    seg_pairs_g has shape (n_blocks, 2). For block b, positions
    [b*rs .. b*rs+seg_sz) read from seg_pairs_g[b,0] and
    [b*rs+seg_sz .. b*rs+2*seg_sz) from seg_pairs_g[b,1].
    Returns a length-K int64 tensor ``perm`` such that
    ``A_perm[:, i] = A[:, perm[i]]``.
    """
    seg_sz = SHUFFLE_SEGMENT_SZ
    n_blocks = seg_pairs_g.shape[0]
    device = seg_pairs_g.device
    perm = torch.empty(K, dtype=torch.long, device=device)
    offsets = torch.arange(seg_sz, device=device)
    for b in range(n_blocks):
        sa = int(seg_pairs_g[b, 0].item())
        sb = int(seg_pairs_g[b, 1].item())
        dst = b * reconn_sz
        perm[dst:dst + seg_sz] = sa * seg_sz + offsets
        perm[dst + seg_sz:dst + 2 * seg_sz] = sb * seg_sz + offsets
    return perm


def pytorch_forward(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    rw_mode: bool = False,
    activation: str | None = None,
    internal_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    seg_pairs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Reference Prism forward using pure PyTorch operations.

    AR mode: C = (A @ R^T) @ B^T, per group.
    AR mode with activation="silu_gate": C = (A * SiLU(A @ R^T + b)) @ B^T, per group.
    RW mode: C = A @ (B @ R)^T, by transforming B first.

    When ``seg_pairs`` is provided, each group g gathers A into a permuted
    layout using seg_pairs[g] before the AR multiply. All downstream
    element-wise ops (bias/silu/dropout) live in permuted space.

    Returns (C, dropout_seeds) where dropout_seeds is a list of per-group RNG
    seeds (empty when dropout is inactive).
    """
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    # R: (n_groups * reconn_sz, K) -> (n_groups, reconn_sz, n_blocks, reconn_sz)
    R_blocks = rearrange(
        R, "(g rs) (b rs2) -> g b rs rs2",
        g=n_groups, rs=reconn_sz, b=n_blocks, rs2=reconn_sz,
    )

    if rw_mode:
        if seg_pairs is not None:
            raise ValueError("seg_pairs is not supported in rw_mode")
        # B: (N, K) -> (n_groups, group_size, n_blocks, reconn_sz)
        B_blocks = rearrange(
            B, "(g gs) (b rs) -> g b gs rs",
            g=n_groups, gs=group_size, b=n_blocks, rs=reconn_sz,
        )
        # B' = B @ R per block: (g, b, gs, rs) @ (g, b, rs, rs) -> (g, b, gs, rs)
        Bp_blocks = torch.einsum("gbir,gbrj->gbij", B_blocks, R_blocks)
        Bp = rearrange(Bp_blocks, "g b gs rs -> (g gs) (b rs)")
        return A @ Bp.T, []

    apply_dropout = training and dropout_p > 0.0
    dropout_seeds: list[int] = []

    C = torch.zeros(M, N, dtype=A.dtype, device=A.device)

    # A_blocks used when no shuffle — shared across groups.
    A_blocks_shared = None
    if seg_pairs is None:
        A_blocks_shared = rearrange(A, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)

    for g in range(n_groups):
        if seg_pairs is not None:
            perm = _seg_pairs_to_elem_perm(seg_pairs[g], K, reconn_sz)
            A_g = A.index_select(1, perm)
            A_blocks = rearrange(A_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
        else:
            A_g = A
            A_blocks = A_blocks_shared

        AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
        AR = rearrange(AR_blocks, "b m rs -> m (b rs)")

        if activation == "silu_gate":
            if internal_bias is not None:
                AR = AR + internal_bias[g]
            H = A_g * F.silu(AR)
            if apply_dropout:
                seed = torch.randint(0, 2**62, (1,)).item()
                mask = _dropout_mask_from_seed(seed, H.shape, dropout_p, H.device, H.dtype)
                H = H * mask / (1.0 - dropout_p)
                dropout_seeds.append(seed)
            AR = H

        B_group = B[g * group_size : (g + 1) * group_size, :]
        # Dense GEMM — permutation order on K is irrelevant for sum over K.
        C[:, g * group_size : (g + 1) * group_size] = AR @ B_group.T

    return C, dropout_seeds


def pytorch_backward(
    dC: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    activation: str | None = None,
    internal_bias: torch.Tensor | None = None,
    dropout_seeds: list[int] | None = None,
    dropout_p: float = 0.0,
    seg_pairs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Reference Prism backward using pure PyTorch operations.

    When ``seg_pairs`` is provided, gradients are computed in permuted space
    per group and then accumulated into the natural-layout dA via
    ``dA.index_add_(1, perm_g, dA_perm_g)``. No atomics — each scatter is a
    gather on the inverse permutation.

    Returns (dA, dR, dB, d_internal_bias) all in K-major layout matching input shapes.
    dB is None when activation is None.
    d_internal_bias is None when internal_bias is None.
    """
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    R_blocks = rearrange(
        R, "(g rs) (b rs2) -> g b rs rs2",
        g=n_groups, rs=reconn_sz, b=n_blocks, rs2=reconn_sz,
    )

    gated = activation == "silu_gate"
    apply_dropout = dropout_seeds is not None and len(dropout_seeds) > 0

    dA = torch.zeros_like(A)
    dR = torch.zeros_like(R)
    dB = torch.zeros_like(B) if gated else None
    d_internal_bias = torch.zeros_like(internal_bias) if internal_bias is not None else None

    # A_blocks shared when no shuffle
    A_blocks_shared = None
    if seg_pairs is None:
        A_blocks_shared = rearrange(A, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)

    for g in range(n_groups):
        dC_g = dC[:, g * group_size : (g + 1) * group_size]  # (M, group_size)
        B_g = B[g * group_size : (g + 1) * group_size, :]     # (group_size, K)

        if seg_pairs is not None:
            perm = _seg_pairs_to_elem_perm(seg_pairs[g], K, reconn_sz)
            A_g = A.index_select(1, perm)
            A_blocks = rearrange(A_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
        else:
            A_g = A
            A_blocks = A_blocks_shared

        # Recompute AR_g = A_g @ R_g^T (in permuted space if shuffled)
        AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
        S_g = rearrange(AR_blocks, "b m rs -> m (b rs)")  # (M, K)

        if gated:
            if internal_bias is not None:
                S_g = S_g + internal_bias[g]
            silu_S = F.silu(S_g)
            H_g = A_g * silu_S

            if apply_dropout:
                mask = _dropout_mask_from_seed(
                    dropout_seeds[g], H_g.shape, dropout_p, H_g.device, H_g.dtype)
                H_g = H_g * mask / (1.0 - dropout_p)

            # dB_g = dC_g^T @ H_g (column order irrelevant — dense sum over rows)
            dB[g * group_size : (g + 1) * group_size, :] = dC_g.T @ H_g

            # dH_g = dC_g @ B_g (in permuted space when shuffled — ok because
            # the AR ↔ H_g ↔ B_g product contracts over K in that same order)
            dH_g = dC_g @ B_g  # (M, K)
            if apply_dropout:
                dH_g = dH_g * mask / (1.0 - dropout_p)

            sigma_g = torch.sigmoid(S_g)
            silu_prime_g = sigma_g * (1.0 + S_g * (1.0 - sigma_g))
            dS_g = dH_g * A_g * silu_prime_g  # (M, K)

            if d_internal_bias is not None:
                d_internal_bias[g] = dS_g.sum(dim=0)

            # dA (permuted) contribution: gate + dS_g @ R_g (block diagonal)
            dA_perm = dH_g * silu_S
            dS_blocks = rearrange(dS_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
            dA_blocks = torch.bmm(dS_blocks, R_blocks[g])
            dA_perm = dA_perm + rearrange(dA_blocks, "b m rs -> m (b rs)")

            if seg_pairs is not None:
                dA.index_add_(1, perm, dA_perm)
            else:
                dA += dA_perm

            # dR_g[b] = dS_g[:, b*rs:(b+1)*rs]^T @ A_g[:, b*rs:(b+1)*rs]
            for b_idx in range(n_blocks):
                dR[g * reconn_sz : (g + 1) * reconn_sz,
                   b_idx * reconn_sz : (b_idx + 1) * reconn_sz] = (
                    dS_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz].T
                    @ A_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz]
                )
        else:
            # Non-gated
            dAR_g = dC_g @ B_g  # (M, K)

            # dA (permuted) contribution: dAR_g @ R_g (block diagonal)
            dAR_blocks = rearrange(dAR_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
            dA_blocks = torch.bmm(dAR_blocks, R_blocks[g])
            dA_perm = rearrange(dA_blocks, "b m rs -> m (b rs)")

            if seg_pairs is not None:
                dA.index_add_(1, perm, dA_perm)
            else:
                dA += dA_perm

            # dR_g[b] = dAR_g[:, b*rs:(b+1)*rs]^T @ A_g[:, b*rs:(b+1)*rs]
            for b_idx in range(n_blocks):
                dR[g * reconn_sz : (g + 1) * reconn_sz,
                   b_idx * reconn_sz : (b_idx + 1) * reconn_sz] = (
                    dAR_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz].T
                    @ A_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz]
                )

    return dA, dR, dB, d_internal_bias
