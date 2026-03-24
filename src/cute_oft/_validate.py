"""Parameter validation for OFT kernels.

Validates user-provided parameters before compilation or kernel launch,
providing clear error messages for invalid configurations.
"""

from __future__ import annotations

import torch


def validate_kernel_params(group_size: int, reconn_sz: int) -> None:
    """Validate kernel-level parameters (compile-time constants).

    These checks determine whether a valid kernel can be compiled.
    """
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    if not isinstance(reconn_sz, int) or reconn_sz <= 0:
        raise ValueError(f"reconn_sz must be a positive integer, got {reconn_sz}")
    if reconn_sz % 8 != 0:
        raise ValueError(
            f"reconn_sz must be a multiple of 8 (tensor core K-dimension requirement), "
            f"got {reconn_sz}"
        )
    if group_size % 8 != 0:
        raise ValueError(
            f"group_size must be a multiple of 8 (warp tiling requirement), "
            f"got {group_size}"
        )


def validate_tensor_params(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
) -> None:
    """Validate tensor shapes and properties for a specific forward call."""
    # Device checks
    for name, t in [("A", A), ("B", B), ("R", R)]:
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got device={t.device}")
        if t.dtype != torch.float16:
            raise ValueError(f"{name} must be float16, got dtype={t.dtype}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (row-major)")
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2D, got {t.dim()}D")

    if not (A.device == B.device == R.device):
        raise ValueError("All tensors must be on the same CUDA device")

    M, K = A.shape
    N = B.shape[0]

    if B.shape[1] != K:
        raise ValueError(f"K dimension mismatch: A has K={K} but B has K={B.shape[1]}")
    if R.shape[1] != K:
        raise ValueError(f"K dimension mismatch: A has K={K} but R has K={R.shape[1]}")
    if N % group_size != 0:
        raise ValueError(
            f"N ({N}) must be divisible by group_size ({group_size})"
        )
    if K % reconn_sz != 0:
        raise ValueError(
            f"K ({K}) must be divisible by reconn_sz ({reconn_sz})"
        )
    n_groups = N // group_size
    expected_r_rows = n_groups * reconn_sz
    if R.shape[0] != expected_r_rows:
        raise ValueError(
            f"R must have shape ({expected_r_rows}, {K}) = "
            f"(n_groups={n_groups} * reconn_sz={reconn_sz}, K={K}), "
            f"but got ({R.shape[0]}, {R.shape[1]})"
        )


def compute_smem_bytes(
    bM: int,
    bN: int,
    bK: int,
    group_size: int,
    reconn_sz: int,
    c_width: int,
    bP_a_r: int,
    bP_ar: int,
    bP_b: int,
    gated: bool = False,
) -> int:
    """Compute shared memory usage in bytes.

    Mirrors get_smem_size() in cute_oft_coop_pc.cu.
    Gated mode does in-place gating on sAR using sA data, so no extra smem needed.
    """
    n_groups = max(bN // group_size, 1)
    size_A = bM * bK * bP_a_r
    size_R = n_groups * reconn_sz * bK * bP_a_r
    size_B = bN * bK * bP_b
    size_AR = n_groups * c_width * bM * bP_ar
    return (size_A + size_R + size_B + size_AR) * 2  # sizeof(half) = 2


def compute_smem_bytes_bwd_dadr(
    bM: int,
    bK: int,
    bN: int,
    group_size: int,
    reconn_sz: int,
    bP_dc_b: int = 2,
    bP_dh: int = 1,
) -> int:
    """Compute shared memory usage for backward dA+dR kernel (producer-consumer).

    Layout: sdC(bM, bN, bP_dc_b) + sB(bK, bN, bP_dc_b)
          + sdH(bM, bK, bP_dh) + sA(bM, bK) + sR(rs, bK, 2)
    """
    rs = reconn_sz
    size_dC = bM * bN * bP_dc_b
    size_B = bK * bN * bP_dc_b
    size_dH = bM * bK * bP_dh
    size_A = bM * bK
    size_R = rs * bK * 2
    half_elems = size_dC + size_B + size_dH + size_A + size_R
    return half_elems * 2 + 256


def compute_smem_bytes_bwd_db(
    bM: int,
    bK: int,
    group_size: int,
    reconn_sz: int,
    bP_a: int,
    bP_ar: int,
    bP_dc: int,
) -> int:
    """Compute shared memory usage for backward dB kernel.

    Mirrors the smem calculation in oft_backward_dB_launch().
    sAR_pipe(bM, bK, bP_ar) replaces sAR_temp(bM, rs) + sARt(bK, bM, bP_ar).
    """
    gs, rs = group_size, reconn_sz
    smem = (bM * bK * bP_a + rs * bK + bM * bK * bP_ar + gs * bM * bP_dc) * 2 + 256
    return smem


def check_smem_limit(smem_bytes: int, device: int = 0) -> None:
    """Check that shared memory requirement doesn't exceed device limit."""
    props = torch.cuda.get_device_properties(device)
    max_smem = getattr(
        props, "shared_memory_per_block_optin",
        props.total_memory,  # fallback if attr not available
    )
    if smem_bytes > max_smem:
        raise RuntimeError(
            f"Kernel requires {smem_bytes} bytes of shared memory, "
            f"but device {props.name} supports at most {max_smem} bytes. "
            f"Try reducing tile sizes (bM, bN, bK) or pipeline depths."
        )
