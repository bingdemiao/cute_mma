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
ACTIVATIONS = (None, "silu_gate", "sigmoid_gate")

# Runtime gate_kind values passed to the cublas kernels. Must stay in sync
# with the `GATE_*` enum in torch_ext/cublas_prism_torch.cu.
_GATE_KIND = {None: 0, "silu_gate": 1, "sigmoid_gate": 2}


def forward(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int = 256,
    reconn_sz: int = 8,
    backend: Literal["cute", "cublas", "pytorch"] = "cute",
    mode: Literal["ar", "rw"] = "ar",
    comp_params: CompParams | Literal["auto"] | None = None,
    activation: Literal["silu_gate", "sigmoid_gate"] | None = None,
    autotuning: bool = False,
    autotuning_search_space: Iterable[CompParams | tuple[CompParams, Callable]] | None = None,
    force_rebenchmark: bool = False,
    internal_bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    shuffle_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int] | torch.Tensor]:
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
        internal_bias: Optional bias added after A @ R^T, before SiLU gate.
            Shape (n_groups, K). Only used in gated mode.
        dropout_p: Dropout probability applied to H = A * SiLU(AR + bias).
        training: Whether in training mode (dropout only active when True).
        shuffle_masks: Optional (n_groups, n_chunks, n_rounds) int64 tensor of
            per-group butterfly shuffle masks. Supported by all backends
            (``cute``, ``cublas``, ``pytorch``) in AR mode — the cute branch
            below forwards ``shuffle_masks`` into the kernel.

    Returns:
        (C, dropout_seeds) — C is the output tensor of shape (M, N).
        dropout_seeds is a list[int] (pytorch) or Tensor (cublas) of per-group
        seeds for backward replay. Empty/zeros when dropout is inactive.
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

    gated = activation is not None
    if activation == "sigmoid_gate" and backend == "cute":
        raise ValueError(
            "activation='sigmoid_gate' is not supported by the 'cute' backend"
        )

    validate_kernel_params(group_size, reconn_sz)

    if backend == "pytorch":
        from ._pytorch_backend import pytorch_forward
        return pytorch_forward(
            A, B, R, group_size, reconn_sz,
            rw_mode=(mode == "rw"), activation=activation,
            internal_bias=internal_bias,
            dropout_p=dropout_p, training=training,
            shuffle_masks=shuffle_masks,
        )

    validate_tensor_params(A, B, R, group_size, reconn_sz)

    if backend == "cute":
        # internal_bias is supported in cute fwd via a separate compiled kernel
        # (PRISM_INTERNAL_BIAS=1). Requires gated mode.
        cute_internal_bias = internal_bias is not None
        if cute_internal_bias and not gated:
            raise ValueError(
                "internal_bias requires a gated activation; pass activation='silu_gate'"
            )
        # Dropout is supported in cute fwd via a separate compiled kernel
        # (PRISM_DROPOUT=1). The mask is hash-derived and identical to cublas
        # so seeds can be replayed in backward (once the bwd kernels also
        # support dropout — currently only fwd does).
        cute_dropout = dropout_p > 0.0 and training
        if cute_dropout and not gated:
            raise ValueError(
                "dropout requires a gated activation in cute backend; "
                "pass activation='silu_gate'"
            )
        # Pick compiled binary based on input dtype (each binary is built for
        # one dtype via PRISM_DTYPE; both dtypes share the same Python API).
        if A.dtype == torch.float16:
            cute_dtype = "fp16"
        elif A.dtype == torch.bfloat16:
            cute_dtype = "bf16"
        else:
            raise ValueError(f"cute backend requires fp16 or bf16, got {A.dtype}")
        if comp_params == "auto" or (autotuning and comp_params is None):
            M, K = A.shape
            N = B.shape[0]
            comp_params = autotune(
                M, N, K, group_size, reconn_sz,
                device=A.device.index or 0,
                gated=gated,
                search_space=autotuning_search_space,
                force_rebenchmark=force_rebenchmark,
                dtype=cute_dtype,
                internal_bias=cute_internal_bias,
                dropout_p=float(dropout_p) if cute_dropout else 0.0,
            )
        elif comp_params is None:
            comp_params = CompParams()
        # input_shuffle correctness requires each CTA to map to a single output
        # group: the shuffled-A buffer is selected per group via a single
        # per-CTA stride, so when group_size < bN a CTA spans multiple groups
        # and all but the first read the wrong shuffled A (silent ~100% error).
        if shuffle_masks is not None and group_size < comp_params.bN:
            raise ValueError(
                f"input_shuffle on the cute backend requires group_size >= bN "
                f"(N tile); got group_size={group_size}, bN={comp_params.bN}. "
                f"A CTA spanning multiple groups cannot select per-group shuffled "
                f"A. Use group_size >= {comp_params.bN}, pass comp_params with "
                f"bN <= group_size, or use the cublas backend for input_shuffle "
                f"at small group_size."
            )
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
                internal_bias=cute_internal_bias,
                dtype=cute_dtype,
                dropout=cute_dropout,
            )
        except CompilationError:
            comp_params = CompParams.safe_defaults()
            module = get_or_compile(
                group_size, reconn_sz, backend, comp_params,
                gated=gated, kernel_type="fwd",
                internal_bias=cute_internal_bias,
                dtype=cute_dtype,
                dropout=cute_dropout,
            )
        # Allocate per-group seeds (one uint64 per group, must be nonzero).
        # Returned alongside C so the caller can replay the same mask in bwd.
        N_dim = B.shape[0]
        n_groups = N_dim // group_size
        seeds_extra: list = []
        if cute_dropout:
            seeds = torch.randint(
                low=1, high=(1 << 62),
                size=(n_groups,), dtype=torch.int64, device=A.device,
            )
            seeds_extra = [seeds]
        else:
            seeds = None
        return module.forward(
            A, B, R, group_size, reconn_sz,
            internal_bias if cute_internal_bias else None,
            shuffle_masks,
            seeds,
            float(dropout_p) if cute_dropout else 0.0,
        ), seeds_extra

    # cublas backend
    module = get_or_compile(group_size, reconn_sz, backend, comp_params)
    results = module.forward(
        A, B, R, group_size, reconn_sz, mode == "rw", _GATE_KIND[activation],
        internal_bias, dropout_p, training, shuffle_masks,
    )
    return results[0], results[1]


def backward(
    dC: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int = 256,
    reconn_sz: int = 8,
    backend: Literal["cute", "cublas", "pytorch"] = "pytorch",
    activation: Literal["silu_gate", "sigmoid_gate"] | None = None,
    comp_params: CompParams | Literal["auto"] | None = None,
    autotuning: bool = False,
    autotuning_search_space_dadr: Iterable[BwdDAdRCompParams | tuple[BwdDAdRCompParams, Callable]] | None = None,
    autotuning_search_space_db: Iterable[BwdDBCompParams | tuple[BwdDBCompParams, Callable]] | None = None,
    bwd_dadr_params: BwdDAdRCompParams | None = None,
    bwd_db_params: BwdDBCompParams | None = None,
    force_rebenchmark: bool = False,
    dR_dtype: torch.dtype | None = None,
    dB_dtype: torch.dtype | None = None,
    internal_bias: torch.Tensor | None = None,
    dropout_seeds: list[int] | torch.Tensor | None = None,
    dropout_p: float = 0.0,
    shuffle_masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        internal_bias: Same internal_bias used in forward (for recompute).
        dropout_seeds: Per-group seeds from forward pass for mask replay.
        dropout_p: Dropout probability (must match forward).

    Returns:
        (dA, dR, dB, d_internal_bias) — gradients w.r.t. A, R, B, internal_bias.
        dB is None when activation is None (non-gated mode).
        d_internal_bias is None when internal_bias is None.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {activation!r}, must be one of {ACTIVATIONS}"
        )

    validate_kernel_params(group_size, reconn_sz)

    gated = activation is not None
    if activation == "sigmoid_gate" and backend == "cute":
        raise ValueError(
            "activation='sigmoid_gate' is not supported by the 'cute' backend"
        )

    if backend == "pytorch":
        from ._pytorch_backend import pytorch_backward
        # Convert tensor seeds to list for pytorch backend
        py_seeds = None
        if dropout_seeds is not None:
            if isinstance(dropout_seeds, torch.Tensor):
                py_seeds = dropout_seeds.tolist()
            else:
                py_seeds = dropout_seeds
        return pytorch_backward(
            dC, A, B, R, group_size, reconn_sz,
            activation=activation, internal_bias=internal_bias,
            dropout_seeds=py_seeds, dropout_p=dropout_p,
            shuffle_masks=shuffle_masks,
        )

    validate_tensor_params(A, B, R, group_size, reconn_sz)

    if backend == "cublas":
        module = get_or_compile(group_size, reconn_sz, backend)
        gk = _GATE_KIND[activation]
        # dropout_seeds is a Tensor from cublas forward
        ds_tensor = dropout_seeds if isinstance(dropout_seeds, torch.Tensor) else None
        grads = module.backward_dA_dR(
            dC, A, B, R, group_size, reconn_sz, gk, dR_dtype,
            internal_bias, ds_tensor, dropout_p, shuffle_masks,
        )
        dA, dR = grads[0], grads[1]
        d_ib = grads[2] if (len(grads) > 2 and grads[2] is not None and grads[2].numel() > 0) else None
        if gated:
            dB_out = module.backward_dB(
                dC, A, R, group_size, reconn_sz, gk, dB_dtype,
                internal_bias, ds_tensor, dropout_p, shuffle_masks,
            )
        else:
            dB_out = None
        return dA, dR, dB_out, d_ib

    if backend == "cute":
        M, K = A.shape
        N = B.shape[0]
        dev = A.device.index or 0
        cute_internal_bias = internal_bias is not None
        if cute_internal_bias and not gated:
            raise ValueError(
                "internal_bias requires a gated activation; pass activation='silu_gate'"
            )
        # Dropout is supported in cute bwd via PRISM_DROPOUT=1 binaries.
        # Seeds must be a CUDA int64 tensor of shape (n_groups,) — same as
        # the forward returned. The Tensor form (cublas convention) is what
        # the cute extension expects directly.
        cute_dropout = dropout_seeds is not None and dropout_p > 0.0
        if cute_dropout and not gated:
            raise ValueError(
                "dropout backward requires a gated activation in cute; "
                "pass activation='silu_gate'"
            )
        if cute_dropout and not isinstance(dropout_seeds, torch.Tensor):
            # Allow list[int] for cross-backend symmetry; convert to tensor.
            dropout_seeds = torch.tensor(dropout_seeds, dtype=torch.int64, device=A.device)
        if A.dtype == torch.float16:
            cute_dtype = "fp16"
        elif A.dtype == torch.bfloat16:
            cute_dtype = "bf16"
        else:
            raise ValueError(f"cute backend requires fp16 or bf16, got {A.dtype}")

        # Resolve dAdR params
        if bwd_dadr_params is None:
            if comp_params == "auto" or autotuning:
                bwd_dadr_params = autotune_bwd_dadr(
                    M, N, K, group_size, reconn_sz,
                    device=dev, gated=gated,
                    search_space=autotuning_search_space_dadr,
                    force_rebenchmark=force_rebenchmark,
                    dtype=cute_dtype,
                    internal_bias=cute_internal_bias,
                    dropout_p=float(dropout_p) if cute_dropout else 0.0,
                )
            else:
                bwd_dadr_params = BwdDAdRCompParams()

        # Load dAdR module
        try:
            dadr_module = get_or_compile(
                group_size, reconn_sz, backend,
                gated=gated, kernel_type="bwd_dadr",
                bwd_dadr_params=bwd_dadr_params,
                internal_bias=cute_internal_bias,
                dtype=cute_dtype,
                dropout=cute_dropout,
            )
        except CompilationError:
            bwd_dadr_params = BwdDAdRCompParams.safe_defaults()
            dadr_module = get_or_compile(
                group_size, reconn_sz, backend,
                gated=gated, kernel_type="bwd_dadr",
                bwd_dadr_params=bwd_dadr_params,
                internal_bias=cute_internal_bias,
                dtype=cute_dtype,
                dropout=cute_dropout,
            )
        grads = dadr_module.backward_dA_dR(
            dC, A, B, R, group_size, reconn_sz,
            internal_bias if cute_internal_bias else None,
            shuffle_masks,
            dropout_seeds if cute_dropout else None,
            float(dropout_p) if cute_dropout else 0.0,
        )
        dA, dR = grads[0], grads[1]
        d_ib = grads[2] if (len(grads) > 2 and grads[2] is not None and grads[2].numel() > 0) else None

        if gated:
            # Resolve dB params
            if bwd_db_params is None:
                if comp_params == "auto" or autotuning:
                    bwd_db_params = autotune_bwd_db(
                        M, N, K, group_size, reconn_sz,
                        device=dev, gated=gated,
                        search_space=autotuning_search_space_db,
                        force_rebenchmark=force_rebenchmark,
                        dtype=cute_dtype,
                        internal_bias=cute_internal_bias,
                        dropout_p=float(dropout_p) if cute_dropout else 0.0,
                    )
                else:
                    bwd_db_params = BwdDBCompParams()

            try:
                db_module = get_or_compile(
                    group_size, reconn_sz, backend,
                    gated=gated, kernel_type="bwd_db",
                    bwd_db_params=bwd_db_params,
                    internal_bias=cute_internal_bias,
                    dtype=cute_dtype,
                    dropout=cute_dropout,
                )
            except CompilationError:
                bwd_db_params = BwdDBCompParams.safe_defaults()
                db_module = get_or_compile(
                    group_size, reconn_sz, backend,
                    gated=gated, kernel_type="bwd_db",
                    bwd_db_params=bwd_db_params,
                    internal_bias=cute_internal_bias,
                    dtype=cute_dtype,
                    dropout=cute_dropout,
                )
            dB_out = db_module.backward_dB(
                dC, A, R, group_size, reconn_sz,
                internal_bias if cute_internal_bias else None,
                shuffle_masks,
                dropout_seeds if cute_dropout else None,
                float(dropout_p) if cute_dropout else 0.0,
            )
        else:
            dB_out = None

        return dA, dR, dB_out, d_ib

    raise ValueError(
        f"Backend {backend!r} is not yet supported for backward pass."
    )
