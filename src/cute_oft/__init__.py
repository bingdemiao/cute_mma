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

from ._autotune import autotune
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
    "backward",
    "autotune",
    "clear_cache",
    "CompParams",
    "CompilationError",
]

BACKENDS = ("cute", "cublas", "pytorch")
MODES = ("ar", "rw")


ACTIVATIONS = (None, "silu_gate")


def forward(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int = 256,
    reconn_sz: int = 8,
    backend: Literal["cute", "cublas", "pytorch"] = "cute",
    mode: Literal["ar", "rw"] = "ar",
    comp_params: CompParams | Literal["auto"] | None = None,
    activation: Literal["silu_gate"] | None = None,
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
            by other backends. Pass "auto" to run autotuning for the given problem
            shape. Uses defaults if None.
        activation: Optional non-linearity inserted between AR and B stages.
            - None: Standard OFT: C = (A @ R^T) @ B^T
            - "silu_gate": Gated OFT: C = (A * SiLU(A @ R^T)) @ B^T
            Only supported in "ar" mode.

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
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {activation!r}, must be one of {ACTIVATIONS}"
        )
    if activation is not None and mode != "ar":
        raise ValueError(
            f"activation={activation!r} is only supported in mode='ar'"
        )

    gated = activation == "silu_gate"

    validate_kernel_params(group_size, reconn_sz)
    validate_tensor_params(A, B, R, group_size, reconn_sz)

    if backend == "pytorch":
        from ._pytorch_backend import pytorch_forward
        return pytorch_forward(
            A, B, R, group_size, reconn_sz,
            rw_mode=(mode == "rw"), activation=activation,
        )

    if backend == "cute":
        if comp_params == "auto":
            M, K = A.shape
            N = B.shape[0]
            comp_params = autotune(
                M, N, K, group_size, reconn_sz,
                device=A.device.index or 0,
                gated=gated,
            )
        elif comp_params is None:
            comp_params = CompParams()
        smem = compute_smem_bytes(
            comp_params.bM, comp_params.bN, comp_params.bK,
            group_size, reconn_sz,
            comp_params.c_width,
            comp_params.bP_a_r, comp_params.bP_ar, comp_params.bP_b,
            gated=gated,
        )
        check_smem_limit(smem, A.device.index or 0)
        module = get_or_compile(group_size, reconn_sz, backend, comp_params, gated=gated)
        return module.forward(A, B, R, group_size, reconn_sz)

    # cublas backend
    module = get_or_compile(group_size, reconn_sz, backend, comp_params)
    return module.forward(A, B, R, group_size, reconn_sz, mode == "rw", gated)


def backward(
    dC: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int = 256,
    reconn_sz: int = 8,
    backend: Literal["cute", "cublas", "pytorch"] = "pytorch",
    activation: Literal["silu_gate"] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute gradients for OFT backward pass.

    Recomputes intermediate activations (AR or A*SiLU(AR)) from inputs
    rather than storing them, to minimize memory usage.

    When activation is None (non-gated / fine-tuning mode), dB is not
    computed and returned as None, since B is frozen.

    Args:
        dC: Gradient of loss w.r.t. output C, shape (M, N), float16, CUDA.
        A: Input tensor of shape (M, K), float16, CUDA.
        B: Weight tensor of shape (N, K), float16, CUDA.
        R: Reconnection matrix of shape (n_groups * reconn_sz, K), float16, CUDA.
        group_size: Number of output channels per group.
        reconn_sz: Reconnection block size.
        backend: Computation backend. "pytorch", "cublas", or "cute".
        activation: Must match the activation used in the forward pass.

    Returns:
        (dA, dR, dB) — gradients w.r.t. A, R, B respectively.
        dB is None when activation is None (non-gated mode).
    """
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {activation!r}, must be one of {ACTIVATIONS}"
        )

    validate_kernel_params(group_size, reconn_sz)
    validate_tensor_params(A, B, R, group_size, reconn_sz)

    gated = activation == "silu_gate"

    if backend == "pytorch":
        from ._pytorch_backend import pytorch_backward
        return pytorch_backward(dC, A, B, R, group_size, reconn_sz, activation=activation)

    if backend == "cublas":
        module = get_or_compile(group_size, reconn_sz, backend)
        grads = module.backward_dA_dR(dC, A, B, R, group_size, reconn_sz, gated)
        dA, dR = grads[0], grads[1]
        if gated:
            dB = module.backward_dB(dC, A, R, group_size, reconn_sz, gated)
        else:
            dB = None
        return dA, dR, dB

    if backend == "cute":
        module = get_or_compile(group_size, reconn_sz, backend, gated=gated)
        grads = module.backward_dA_dR(dC, A, B, R, group_size, reconn_sz, gated)
        dA, dR = grads[0], grads[1]
        if gated:
            dB = module.backward_dB(dC, A, R, group_size, reconn_sz, gated)
        else:
            dB = None
        return dA, dR, dB

    raise ValueError(
        f"Backend {backend!r} is not yet supported for backward pass."
    )
