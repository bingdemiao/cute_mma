"""Pure PyTorch implementation of Prism forward and backward passes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def _dropout_mask_from_seed(
    seed: int, shape: tuple, dropout_p: float,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Generate a deterministic dropout mask from a seed."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return (torch.rand(shape, generator=gen, device=device, dtype=dtype) >= dropout_p).to(dtype)


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
) -> tuple[torch.Tensor, list[int]]:
    """Reference Prism forward using pure PyTorch operations.

    AR mode: C = (A @ R^T) @ B^T, per group.
    AR mode with activation="silu_gate": C = (A * SiLU(A @ R^T + b)) @ B^T, per group.
    RW mode: C = A @ (B @ R)^T, by transforming B first.

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
        # B: (N, K) -> (n_groups, group_size, n_blocks, reconn_sz)
        B_blocks = rearrange(
            B, "(g gs) (b rs) -> g b gs rs",
            g=n_groups, gs=group_size, b=n_blocks, rs=reconn_sz,
        )
        # B' = B @ R per block: (g, b, gs, rs) @ (g, b, rs, rs) -> (g, b, gs, rs)
        Bp_blocks = torch.einsum("gbir,gbrj->gbij", B_blocks, R_blocks)
        Bp = rearrange(Bp_blocks, "g b gs rs -> (g gs) (b rs)")
        return A @ Bp.T, []
    else:
        # A: (M, K) -> (1, n_blocks, M, reconn_sz) for batched matmul
        A_blocks = rearrange(A, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)

        apply_dropout = training and dropout_p > 0.0
        dropout_seeds: list[int] = []

        C = torch.zeros(M, N, dtype=A.dtype, device=A.device)
        for g in range(n_groups):
            # AR_blocks: (n_blocks, M, reconn_sz) @ (n_blocks, reconn_sz, reconn_sz)^T
            AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
            AR = rearrange(AR_blocks, "b m rs -> m (b rs)")
            if activation == "silu_gate":
                if internal_bias is not None:
                    AR = AR + internal_bias[g]
                H = A * F.silu(AR)
                if apply_dropout:
                    seed = torch.randint(0, 2**62, (1,)).item()
                    mask = _dropout_mask_from_seed(seed, H.shape, dropout_p, H.device, H.dtype)
                    H = H * mask / (1.0 - dropout_p)
                    dropout_seeds.append(seed)
                AR = H
            B_group = B[g * group_size : (g + 1) * group_size, :]
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Reference Prism backward using pure PyTorch operations.

    Recomputes AR (or A*SiLU(AR)) from A and R rather than storing it.

    Non-gated: C_g = (A @ R_g^T) @ B_g^T
      dB_g = dC_g^T @ AR_g
      dAR_g = dC_g @ B_g
      dA = sum_g dAR_g @ R_g
      dR_g[b] = dAR_g[:, b*rs:(b+1)*rs]^T @ A[:, b*rs:(b+1)*rs]

    Gated: C_g = (A * SiLU(A @ R_g^T + b_g)) @ B_g^T
      Let S_g = A @ R_g^T + b_g, H_g = A * SiLU(S_g)
      dB_g = dC_g^T @ H_g
      dH_g = dC_g @ B_g
      sigma_g = sigmoid(S_g)
      silu_prime_g = sigma_g * (1 + S_g * (1 - sigma_g))
      dS_g = dH_g * A * silu_prime_g
      dA = sum_g [ dH_g * SiLU(S_g) + dS_g @ R_g ]
      dR_g[b] = dS_g[:, b*rs:(b+1)*rs]^T @ A[:, b*rs:(b+1)*rs]
      d_internal_bias_g = dS_g.sum(dim=0)

    Returns:
        (dA, dR, dB, d_internal_bias) all in K-major layout matching input shapes.
        dB is None when activation is None.
        d_internal_bias is None when internal_bias is None.
    """
    M, K = A.shape
    N = B.shape[0]
    n_groups = N // group_size
    n_blocks = K // reconn_sz

    # R: (n_groups * reconn_sz, K) -> (n_groups, n_blocks, reconn_sz, reconn_sz)
    R_blocks = rearrange(
        R, "(g rs) (b rs2) -> g b rs rs2",
        g=n_groups, rs=reconn_sz, b=n_blocks, rs2=reconn_sz,
    )
    A_blocks = rearrange(A, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)

    gated = activation == "silu_gate"
    apply_dropout = dropout_seeds is not None and len(dropout_seeds) > 0

    dA = torch.zeros_like(A)
    dR = torch.zeros_like(R)
    dB = torch.zeros_like(B) if gated else None
    d_internal_bias = torch.zeros_like(internal_bias) if internal_bias is not None else None

    for g in range(n_groups):
        dC_g = dC[:, g * group_size : (g + 1) * group_size]  # (M, group_size)
        B_g = B[g * group_size : (g + 1) * group_size, :]     # (group_size, K)

        # Recompute AR_g = A @ R_g^T
        AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
        S_g = rearrange(AR_blocks, "b m rs -> m (b rs)")  # (M, K)

        if gated:
            # Gated forward: H_g = A * SiLU(S_g + bias_g)
            if internal_bias is not None:
                S_g = S_g + internal_bias[g]
            silu_S = F.silu(S_g)
            H_g = A * silu_S

            # Apply dropout on H_g (same mask as forward)
            if apply_dropout:
                mask = _dropout_mask_from_seed(
                    dropout_seeds[g], H_g.shape, dropout_p, H_g.device, H_g.dtype)
                H_g = H_g * mask / (1.0 - dropout_p)

            # dB_g = dC_g^T @ H_g (uses dropped-out H_g)
            dB[g * group_size : (g + 1) * group_size, :] = dC_g.T @ H_g

            # dH_g = dC_g @ B_g, then replay dropout
            dH_g = dC_g @ B_g  # (M, K)
            if apply_dropout:
                dH_g = dH_g * mask / (1.0 - dropout_p)

            # Gating derivative: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sigma_g = torch.sigmoid(S_g)
            silu_prime_g = sigma_g * (1.0 + S_g * (1.0 - sigma_g))
            dS_g = dH_g * A * silu_prime_g  # (M, K)

            # d_internal_bias
            if d_internal_bias is not None:
                d_internal_bias[g] = dS_g.sum(dim=0)

            # dA contribution from this group
            # dA += dH_g * SiLU(S_g) + dS_g @ R_g
            dA += dH_g * silu_S
            dS_blocks = rearrange(dS_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
            dA_blocks = torch.bmm(dS_blocks, R_blocks[g])
            dA += rearrange(dA_blocks, "b m rs -> m (b rs)")

            # dR_g[b] = dS_g[:, b*rs:(b+1)*rs]^T @ A[:, b*rs:(b+1)*rs]
            for b_idx in range(n_blocks):
                dR[g * reconn_sz : (g + 1) * reconn_sz,
                   b_idx * reconn_sz : (b_idx + 1) * reconn_sz] = (
                    dS_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz].T
                    @ A[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz]
                )
        else:
            # Non-gated: AR_g is just S_g
            AR_g = S_g

            # dAR_g = dC_g @ B_g
            dAR_g = dC_g @ B_g  # (M, K)

            # dA += dAR_g @ R_g (block-diagonal)
            dAR_blocks = rearrange(dAR_g, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)
            dA_blocks = torch.bmm(dAR_blocks, R_blocks[g])
            dA += rearrange(dA_blocks, "b m rs -> m (b rs)")

            # dR_g[b] = dAR_g[:, b*rs:(b+1)*rs]^T @ A[:, b*rs:(b+1)*rs]
            for b_idx in range(n_blocks):
                dR[g * reconn_sz : (g + 1) * reconn_sz,
                   b_idx * reconn_sz : (b_idx + 1) * reconn_sz] = (
                    dAR_g[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz].T
                    @ A[:, b_idx * reconn_sz : (b_idx + 1) * reconn_sz]
                )

    return dA, dR, dB, d_internal_bias
