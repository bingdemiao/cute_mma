"""cute_oft — OFT (Orthogonal Fine-Tuning) CUDA kernels with multiple backends.

Usage::

    import cute_oft

    # Default backend is 'cute' (JIT-compiled CuTe kernel)
    C = cute_oft.forward(A, B, R, group_size=256, reconn_sz=8)

    # Use cuBLAS backend (supports AR and RW modes)
    C = cute_oft.forward(A, B, R, group_size=256, reconn_sz=8, backend="cublas")
    C = cute_oft.forward(A, B, R, group_size=256, reconn_sz=8, backend="cublas", mode="rw")

    # Use pure PyTorch backend (no compilation required)
    C = cute_oft.forward(A, B, R, group_size=256, reconn_sz=8, backend="pytorch")

    # Clear all cached compilations
    cute_oft.clear_cache()
"""

from __future__ import annotations

from typing import Literal

import torch

from ._compiler import CompilationError, clear_cache
from ._config import CompParams
from ._loader import get_or_compile
from ._validate import (
    check_smem_limit,
    compute_smem_bytes,
    validate_kernel_params,
    validate_tensor_params,
)

__all__ = [
    "forward",
    "clear_cache",
    "CompParams",
    "CompilationError",
]

BACKENDS = ("cute", "cublas", "pytorch")
MODES = ("ar", "rw")


def forward(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int = 256,
    reconn_sz: int = 8,
    backend: Literal["cute", "cublas", "pytorch"] = "cute",
    mode: Literal["ar", "rw"] = "ar",
    comp_params: CompParams | None = None,
) -> torch.Tensor:
    """Compute C = A @ diag(R) @ B^T with OFT structure.

    Args:
        A: Input tensor of shape (M, K), float16, CUDA.
        B: Weight tensor of shape (N, K), float16, CUDA.
        R: Reconnection matrix of shape (n_groups * reconn_sz, K), float16, CUDA.
        group_size: Number of output channels per group. Must be a multiple of 8.
        reconn_sz: Reconnection block size. Must be a multiple of 8.
        backend: Computation backend. One of:
            - "cute": JIT-compiled CuTe cooperative kernel (default, fastest).
            - "cublas": cuBLAS-based implementation.
            - "pytorch": Pure PyTorch (no compilation, useful for debugging).
        mode: Computation mode. One of:
            - "ar": Compute (A @ R^T) first, then multiply by B^T.
            - "rw": Transform weights (R @ B) first, then multiply by A.
            Only supported by "cublas" and "pytorch" backends. The "cute" backend
            always uses "ar" mode.
        comp_params: Performance tuning parameters for the 'cute' backend. Ignored
            by other backends. Uses defaults if None.

    Returns:
        Output tensor of shape (M, N), float16, CUDA.

    Raises:
        ValueError: If parameters, tensor shapes, or backend name are invalid.
        CompilationError: If kernel compilation fails (cute/cublas backends).
        RuntimeError: If shared memory requirement exceeds device limit (cute backend).
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}, must be one of {BACKENDS}"
        )
    if mode not in MODES:
        raise ValueError(
            f"Unknown mode {mode!r}, must be one of {MODES}"
        )
    if backend == "cute" and mode != "ar":
        raise ValueError("The 'cute' backend only supports mode='ar'")

    validate_kernel_params(group_size, reconn_sz)
    validate_tensor_params(A, B, R, group_size, reconn_sz)

    if backend == "pytorch":
        from ._pytorch_backend import pytorch_forward
        return pytorch_forward(A, B, R, group_size, reconn_sz, rw_mode=(mode == "rw"))

    if backend == "cute":
        if comp_params is None:
            comp_params = CompParams()
        smem = compute_smem_bytes(
            comp_params.bM, comp_params.bN, comp_params.bK,
            group_size, reconn_sz,
            comp_params.c_width,
            comp_params.bP_a_r, comp_params.bP_ar, comp_params.bP_b,
        )
        check_smem_limit(smem, A.device.index or 0)
        module = get_or_compile(group_size, reconn_sz, backend, comp_params)
        return module.forward(A, B, R, group_size, reconn_sz)

    # cublas backend
    module = get_or_compile(group_size, reconn_sz, backend, comp_params)
    return module.forward(A, B, R, group_size, reconn_sz, mode == "rw")
