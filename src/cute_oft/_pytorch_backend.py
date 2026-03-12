"""Pure PyTorch implementation of OFT forward pass."""

from __future__ import annotations

import torch
from einops import rearrange


def pytorch_forward(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    rw_mode: bool = False,
) -> torch.Tensor:
    """Reference OFT forward using pure PyTorch operations.

    AR mode: C = (A @ R^T) @ B^T, per group.
    RW mode: C = A @ (B @ R)^T, by transforming B first.
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
        return A @ Bp.T
    else:
        # A: (M, K) -> (1, n_blocks, M, reconn_sz) for batched matmul
        A_blocks = rearrange(A, "m (b rs) -> b m rs", b=n_blocks, rs=reconn_sz)

        C = torch.zeros(M, N, dtype=A.dtype, device=A.device)
        for g in range(n_groups):
            # AR_blocks: (n_blocks, M, reconn_sz) @ (n_blocks, reconn_sz, reconn_sz)^T
            AR_blocks = torch.bmm(A_blocks, R_blocks[g].transpose(-1, -2))
            AR = rearrange(AR_blocks, "b m rs -> m (b rs)")
            B_group = B[g * group_size : (g + 1) * group_size, :]
            C[:, g * group_size : (g + 1) * group_size] = AR @ B_group.T
        return C
