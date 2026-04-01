"""cute_prism — Prism linear layer with multiple backends.

Usage::

    import cute_prism

    # Default backend is 'cute' (JIT-compiled CuTe kernel)
    C = cute_prism.forward(A, B, R, group_size=256, reconn_sz=8)

    # Use cuBLAS backend (supports AR and RW modes)
    C = cute_prism.forward(A, B, R, group_size=256, reconn_sz=8, backend="cublas")
    C = cute_prism.forward(A, B, R, group_size=256, reconn_sz=8, backend="cublas", mode="rw")

    # Use pure PyTorch backend (no compilation required)
    C = cute_prism.forward(A, B, R, group_size=256, reconn_sz=8, backend="pytorch")

    # Autotuning
    C = cute_prism.forward(A, B, R, 256, 8, backend="cute", autotuning=True)
    dA, dR, dB = cute_prism.backward(dC, A, B, R, 256, 8, backend="cute", autotuning=True)

    # Clear all cached compilations
    cute_prism.clear_cache()
"""

from __future__ import annotations

from typing import Callable, Iterable, Literal

import torch

from ._autotune import (
    autotune,
    autotune_bwd_dadr,
    autotune_bwd_db,
    clear_autotune_cache,
    get_autotune_cache,
)
from ._compiler import CompilationError, clear_cache
from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams
from ._module import PrismLinear, mup_fix_prism_shapes
from ._loader import get_or_compile
from ._validate import (
    check_smem_limit,
    compute_smem_bytes,
    compute_smem_bytes_bwd_dadr,
    compute_smem_bytes_bwd_db,
    validate_kernel_params,
    validate_tensor_params,
)

__all__ = [
    "forward",
    "backward",
    "autotune",
    "autotune_bwd_dadr",
    "autotune_bwd_db",
    "clear_autotune_cache",
    "get_autotune_cache",
    "clear_cache",
    "CompParams",
    "BwdDAdRCompParams",
    "BwdDBCompParams",
    "CompilationError",
    "PrismLinear",
    "mup_fix_prism_shapes",
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
    autotuning: bool = False,
    autotuning_search_space: Iterable[CompParams | tuple[CompParams, Callable]] | None = None,
    force_rebenchmark: bool = False,
) -> torch.Tensor:
    """Compute C = A @ diag(R) @ B^T with Prism structure.

    Args:
        A: Input tensor of shape (M, K), float16 or bfloat16, CUDA.
        B: Weight tensor of shape (N, K), same dtype as A, CUDA.
        R: Reconnection matrix of shape (n_groups * reconn_sz, K), same dtype as A, CUDA.
        group_size: Number of output channels per group. Must be a multiple of 8.
        reconn_sz: Reconnection block size. Must be a multiple of 8.
        backend: Computation backend ("cute", "cublas", "pytorch").
        mode: Computation mode ("ar" or "rw"). Only cublas/pytorch support "rw".
        comp_params: Explicit CompParams for the 'cute' backend. Pass "auto" for
            autotuning. Uses defaults if None.
        activation: Optional non-linearity ("silu_gate" or None).
        autotuning: When True, check cache first, autotune if miss, use best config.
        autotuning_search_space: Custom search space for autotuning (uses default if None).

    Returns:
        Output tensor of shape (M, N), same dtype as inputs, CUDA.
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
        if comp_params == "auto" or (autotuning and comp_params is None):
            M, K = A.shape
            N = B.shape[0]
            comp_params = autotune(
                M, N, K, group_size, reconn_sz,
                device=A.device.index or 0,
                gated=gated,
                search_space=autotuning_search_space,
                force_rebenchmark=force_rebenchmark,
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
        try:
            module = get_or_compile(
                group_size, reconn_sz, backend, comp_params,
                gated=gated, kernel_type="fwd",
            )
        except CompilationError:
            comp_params = CompParams.safe_defaults()
            module = get_or_compile(
                group_size, reconn_sz, backend, comp_params,
                gated=gated, kernel_type="fwd",
            )
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
    comp_params: CompParams | Literal["auto"] | None = None,
    autotuning: bool = False,
    autotuning_search_space_dadr: Iterable[BwdDAdRCompParams | tuple[BwdDAdRCompParams, Callable]] | None = None,
    autotuning_search_space_db: Iterable[BwdDBCompParams | tuple[BwdDBCompParams, Callable]] | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    bwd_db_params: BwdDBCompParams | None = None,
    force_rebenchmark: bool = False,
    dR_dtype: torch.dtype | None = None,
    dB_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute gradients for Prism backward pass.

    Recomputes intermediate activations (AR or A*SiLU(AR)) from inputs
    rather than storing them, to minimize memory usage.

    When activation is None (non-gated / fine-tuning mode), dB is not
    computed and returned as None, since B is frozen.

    Args:
        dC: Gradient of loss w.r.t. output C, shape (M, N), float16/bfloat16, CUDA.
        A: Input tensor of shape (M, K), same dtype as dC.
        B: Weight tensor of shape (N, K), same dtype as dC.
        R: Reconnection matrix of shape (n_groups * reconn_sz, K), same dtype as dC.
        group_size: Number of output channels per group.
        reconn_sz: Reconnection block size.
        backend: Computation backend. "pytorch", "cublas", or "cute".
        activation: Must match the activation used in the forward pass.
        comp_params: Forward CompParams for the 'cute' backend (legacy, for compilation).
        autotuning: When True, autotune each backward kernel independently.
        autotuning_search_space_dadr: Custom search space for dAdR kernel autotuning.
        autotuning_search_space_db: Custom search space for dB kernel autotuning.
        bwd_dadr_params: Explicit BwdDAdRCompParams override.
        bwd_db_params: Explicit BwdDBCompParams override.
        dR_dtype: Optional dtype for dR gradient (default: R.dtype).
            Useful for mixed-precision training (e.g. torch.float32).
        dB_dtype: Optional dtype for dB gradient (default: B.dtype).
            Useful for mixed-precision training (e.g. torch.float32).

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
        grads = module.backward_dA_dR(dC, A, B, R, group_size, reconn_sz, gated, dR_dtype)
        dA, dR = grads[0], grads[1]
        if gated:
            dB_out = module.backward_dB(dC, A, R, group_size, reconn_sz, gated, dB_dtype)
        else:
            dB_out = None
        return dA, dR, dB_out

    if backend == "cute":
        M, K = A.shape
        N = B.shape[0]
        dev = A.device.index or 0

        # Resolve dAdR params
        if bwd_dadr_params is None:
            if comp_params == "auto" or autotuning:
                bwd_dadr_params = autotune_bwd_dadr(
                    M, N, K, group_size, reconn_sz,
                    device=dev, gated=gated,
                    search_space=autotuning_search_space_dadr,
                    force_rebenchmark=force_rebenchmark,
                )
            else:
                bwd_dadr_params = BwdDAdRCompParams()

        # Load dAdR module
        try:
            dadr_module = get_or_compile(
                group_size, reconn_sz, backend,
                gated=gated, kernel_type="bwd_dadr",
                bwd_dadr_params=bwd_dadr_params,
            )
        except CompilationError:
            bwd_dadr_params = BwdDAdRCompParams.safe_defaults()
            dadr_module = get_or_compile(
                group_size, reconn_sz, backend,
                gated=gated, kernel_type="bwd_dadr",
                bwd_dadr_params=bwd_dadr_params,
            )
        grads = dadr_module.backward_dA_dR(dC, A, B, R, group_size, reconn_sz)
        dA, dR = grads[0], grads[1]

        if gated:
            # Resolve dB params
            if bwd_db_params is None:
                if comp_params == "auto" or autotuning:
                    bwd_db_params = autotune_bwd_db(
                        M, N, K, group_size, reconn_sz,
                        device=dev, gated=gated,
                        search_space=autotuning_search_space_db,
                        force_rebenchmark=force_rebenchmark,
                    )
                else:
                    bwd_db_params = BwdDBCompParams()

            try:
                db_module = get_or_compile(
                    group_size, reconn_sz, backend,
                    gated=gated, kernel_type="bwd_db",
                    bwd_db_params=bwd_db_params,
                )
            except CompilationError:
                bwd_db_params = BwdDBCompParams.safe_defaults()
                db_module = get_or_compile(
                    group_size, reconn_sz, backend,
                    gated=gated, kernel_type="bwd_db",
                    bwd_db_params=bwd_db_params,
                )
            dB_out = db_module.backward_dB(dC, A, R, group_size, reconn_sz)
        else:
            dB_out = None

        return dA, dR, dB_out

    raise ValueError(
        f"Backend {backend!r} is not yet supported for backward pass."
    )
