"""torch.nn.Module wrapper for OFT linear layers."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Literal

import torch
import torch.nn as nn

from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams


# ---------------------------------------------------------------------------
# Custom ops: opaque to torch.compile, with fake-tensor + autograd support
# ---------------------------------------------------------------------------

@torch.library.custom_op("cute_oft::forward", mutates_args=())
def _oft_forward_op(
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    backend: str,
    activation: str,
    autotuning: bool,
    force_rebenchmark: bool,
) -> torch.Tensor:
    from . import forward as oft_forward

    return oft_forward(
        A, B, R, group_size, reconn_sz,
        backend=backend,
        activation=activation or None,
        autotuning=autotuning,
        force_rebenchmark=force_rebenchmark,
    )


@_oft_forward_op.register_fake
def _(A, B, R, group_size, reconn_sz, backend, activation, autotuning,
      force_rebenchmark):
    return A.new_empty(A.shape[0], B.shape[0])


@torch.library.custom_op("cute_oft::backward", mutates_args=())
def _oft_backward_op(
    dC: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    R: torch.Tensor,
    group_size: int,
    reconn_sz: int,
    backend: str,
    activation: str,
    autotuning: bool,
    force_rebenchmark: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from . import backward as oft_backward

    dA, dR, dB = oft_backward(
        dC.contiguous(), A.contiguous(), B.contiguous(), R.contiguous(),
        group_size, reconn_sz,
        backend=backend,
        activation=activation or None,
        autotuning=autotuning,
        force_rebenchmark=force_rebenchmark,
    )

    if dB is None:
        dB = torch.zeros_like(B)
    if dR is None:
        dR = torch.zeros_like(R)

    return dA, dR, dB


@_oft_backward_op.register_fake
def _(dC, A, B, R, group_size, reconn_sz, backend, activation, autotuning,
      force_rebenchmark):
    return A.new_empty(A.shape), R.new_empty(R.shape), B.new_empty(B.shape)


def _oft_setup_context(ctx, inputs, output):
    A, B, R, group_size, reconn_sz, backend, activation, autotuning, \
        force_rebenchmark = inputs
    ctx.save_for_backward(A, B, R)
    ctx.group_size = group_size
    ctx.reconn_sz = reconn_sz
    ctx.backend = backend
    ctx.activation = activation
    ctx.autotuning = autotuning
    ctx.force_rebenchmark = force_rebenchmark


def _oft_backward(ctx, dC):
    A, B, R = ctx.saved_tensors

    dA, dR, dB = _oft_backward_op(
        dC, A, B, R,
        ctx.group_size, ctx.reconn_sz,
        ctx.backend, ctx.activation,
        ctx.autotuning, ctx.force_rebenchmark,
    )

    # Grads for: A, B, R, then None for the 6 non-tensor args
    return dA, dB, dR, None, None, None, None, None, None


_oft_forward_op.register_autograd(_oft_backward, setup_context=_oft_setup_context)


class OFTLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using OFT structure.

    Computes ``C = (A @ R^T) @ B^T`` per group (standard OFT), or
    ``C = (A * SiLU(A @ R^T)) @ B^T`` per group (gated OFT / width expansion).

    In standard mode (``activation=None``), only R is trainable and B (weight)
    is frozen — this is the classic OFT fine-tuning setup. In gated mode
    (``activation="silu_gate"``), both R and B are trainable, enabling the
    layer to act as a width expansion with ``n_groups`` independent pathways.

    Compatible with ``torch.compile``. The OFT kernel call is registered as
    a custom op, so the compiler treats it as opaque and traces through the
    surrounding standard torch operations (μP multipliers, reshape, bias)
    normally.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        group_size: Number of output channels per reconnection group.
        reconn_sz: Block size of the orthogonal reconnection matrix R.
        bias: If True, adds a learnable bias to the output.
        activation: ``None`` for standard OFT, ``"silu_gate"`` for gated OFT.
        backend: Computation backend (``"cute"``, ``"cublas"``, ``"pytorch"``).
        autotuning: Enable automatic kernel autotuning.
        force_rebenchmark: Force re-benchmarking when autotuning.
        comp_params: Explicit forward kernel configuration.
        bwd_dadr_params: Explicit backward dA+dR kernel configuration.
        bwd_db_params: Explicit backward dB kernel configuration.

    Shape:
        - Input: ``(*, in_features)`` where ``*`` means any number of leading dimensions.
        - Output: ``(*, out_features)``.

    Examples::

        # Standard OFT fine-tuning (freeze pretrained weights, train R only)
        layer = cute_oft.OFTLinear(256, 1024, group_size=256, reconn_sz=8)
        layer.load_pretrained_weight(pretrained_linear.weight)
        output = layer(input)

        # Gated OFT as width expansion (train both R and B)
        layer = cute_oft.OFTLinear(256, 1024, group_size=256, reconn_sz=8,
                                   activation="silu_gate")
        output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 256,
        reconn_sz: int = 8,
        bias: bool = True,
        activation: Literal["silu_gate"] | None = None,
        cayley_order: int | float = float("inf"),
        backend: Literal["cute", "cublas", "pytorch"] = "cute",
        autotuning: bool = False,
        force_rebenchmark: bool = False,
        comp_params: CompParams | None = None,
        bwd_dadr_params: BwdDAdRCompParams | None = None,
        bwd_db_params: BwdDBCompParams | None = None,
    ):
        super().__init__()

        if out_features % group_size != 0:
            raise ValueError(
                f"out_features ({out_features}) must be divisible by "
                f"group_size ({group_size})"
            )
        if in_features % reconn_sz != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"reconn_sz ({reconn_sz})"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.reconn_sz = reconn_sz
        self.activation = activation
        self.cayley_order = cayley_order
        self.backend = backend
        self.autotuning = autotuning
        self.force_rebenchmark = force_rebenchmark
        self.comp_params = comp_params
        self.bwd_dadr_params = bwd_dadr_params
        self.bwd_db_params = bwd_db_params

        n_groups = out_features // group_size
        gated = activation == "silu_gate"

        # μP multipliers.
        #
        # In gated mode, NO runtime multipliers are applied for either B or R.
        # All scaling is absorbed into initialization so that parameters live
        # at O(1/sqrt(fan_in)) — matching nn.Linear's convention. This makes
        # OFTLinear fully compatible with muP's optimizer-level LR scaling
        # (MuAdamW) and allows mixing with nn.Linear in the same model.
        #
        # B: init at N(0, alpha/sqrt(K)) where alpha ≈ 1.66 compensates for
        #    gated SiLU variance reduction. Same muP behavior as nn.Linear.
        # R: init at N(0, 1/sqrt(r)) where r = reconn_sz is the block fan-in.
        #    mup_fix_oft_shapes() marks R's K-dim as finite so MuAdamW
        #    doesn't over-scale its LR based on the full tensor width.
        self._weight_multiplier = 1.0 if gated else 1.0
        self._reconn_multiplier = 1.0 if gated else 1.0 / math.sqrt(reconn_sz)

        # Weight B: (out_features, in_features)
        # Frozen in standard OFT, trainable in gated mode.
        # In gated mode, stores O(1) raw parameter; effective B = multiplier * weight.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=gated,
        )

        # Reconnection parameter.
        # Gated mode (pretraining): reconn stores raw R directly.
        # Finetuning mode: reconn stores M, a (reconn_sz x reconn_sz) matrix
        #   per group per block. S = M - M^T is skew-symmetric, and R is
        #   constructed as an orthogonal matrix from S:
        #     cayley_order=inf:  R = (I + S)(I - S)^{-1}  (exact Cayley)
        #     cayley_order=k:    R = I + 2 * sum_{i=1}^{k} S^i  (k-th order approx)
        self.reconn = nn.Parameter(
            torch.empty(n_groups * reconn_sz, in_features),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with μP-correct scaling.

        Gated mode (pretraining): B is initialized with variance
        alpha^2 / K (where alpha ≈ 1.66 compensates for the gated SiLU
        variance reduction). This keeps B at O(1/sqrt(K)), matching
        nn.Linear's parameter scale so that muP's LR scaling gives
        identical per-step output change. R is O(1) with runtime
        multiplier 1/sqrt(r).

        Standard mode (finetuning): Weight B gets Kaiming init (will typically
        be overwritten by pretrained weights). Reconn is initialized to zero
        (delta from identity), so R starts as the block-diagonal identity.
        """
        gated = self.activation == "silu_gate"

        if gated:
            _GATED_SILU_CORRECTION = 1.0 / 0.6024
            # B: absorb alpha/sqrt(K) into init. No runtime multiplier.
            nn.init.normal_(self.weight, std=_GATED_SILU_CORRECTION / math.sqrt(self.in_features))
            # R: init as block-diagonal skew-symmetric.
            # Form S = M - M^T from random M, so S is skew-symmetric.
            # Var(S_ij) = 2*sigma^2 for off-diagonal, 0 on diagonal.
            # We want Var(R_ij) = 1/r (fan-in per block), so sigma = 1/sqrt(2r).
            n_groups = self.out_features // self.group_size
            n_blocks = self.in_features // self.reconn_sz
            r = self.reconn_sz
            sigma = 1.0 / math.sqrt(2 * r)
            M = torch.randn(n_groups, n_blocks, r, r, device=self.reconn.device,
                            dtype=self.reconn.dtype) * sigma
            S = M - M.transpose(-1, -2)
            with torch.no_grad():
                self.reconn.copy_(S.permute(0, 2, 1, 3).reshape(n_groups * r, self.in_features))
        else:
            # Finetuning: Kaiming B (frozen, will be overwritten),
            # zero delta (R starts at identity).
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.zeros_(self.reconn)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _build_R(self) -> torch.Tensor:
        """Construct the reconnection matrix R from the stored parameter.

        Gated mode: R = reconn (used directly, no orthogonal constraint).

        Finetuning mode: reconn stores M. We form the skew-symmetric matrix
        S = M - M^T per block, then build an orthogonal R:
          - cayley_order=inf:  R = (I + S)(I - S)^{-1}   (exact Cayley transform)
          - cayley_order=k:    R = I + 2 * sum_{i=1}^{k} S^i  (k-th order approx)

        Returns:
            R tensor of shape (n_groups * reconn_sz, in_features), same dtype/device.
        """
        if self.activation == "silu_gate":
            return self.reconn

        n_groups = self.out_features // self.group_size
        n_blocks = self.in_features // self.reconn_sz
        r = self.reconn_sz

        # Reshape M to blocks: (n_groups, n_blocks, r, r)
        M_blocks = self.reconn.reshape(n_groups, r, n_blocks, r).permute(0, 2, 1, 3)

        # Skew-symmetric: S = M - M^T
        S = M_blocks - M_blocks.transpose(-1, -2)

        if self.cayley_order == float("inf"):
            # Exact Cayley: R = (I + S)(I - S)^{-1}
            I = torch.eye(r, device=S.device, dtype=S.dtype).expand_as(S)
            R_blocks = torch.linalg.solve(I - S, I + S)
        else:
            # Approximate: R = I + 2 * sum_{i=1}^{order} S^i
            I = torch.eye(r, device=S.device, dtype=S.dtype).expand_as(S)
            R_blocks = I.clone()
            S_power = S  # S^1
            for _ in range(int(self.cayley_order)):
                R_blocks = R_blocks + 2 * S_power
                S_power = S_power @ S  # S^{i+1}

        # Reshape back: (n_groups, n_blocks, r, r) -> (n_groups * r, n_blocks * r)
        R = R_blocks.permute(0, 2, 1, 3).reshape(n_groups * r, self.in_features)
        return R

    def reconn_diag_sq_sum(self) -> torch.Tensor:
        """Sum of squared diagonal elements of R blocks.

        For a perfectly skew-symmetric R, all diagonal elements are zero,
        so this returns 0. During training, nonzero values indicate R is
        drifting from skew-symmetry. For the Cayley/approximation modes
        (finetuning), R is orthogonal so diagonals should be close to 1
        and this returns approximately n_groups * n_blocks * reconn_sz.

        Returns:
            Scalar tensor: sum of R_block[i,i]^2 across all blocks and groups.
        """
        R = self._build_R()
        n_groups = self.out_features // self.group_size
        n_blocks = self.in_features // self.reconn_sz
        r = self.reconn_sz
        R_blocks = R.reshape(n_groups, r, n_blocks, r).permute(0, 2, 1, 3)
        diag = R_blocks.diagonal(dim1=-2, dim2=-1)  # (n_groups, n_blocks, r)
        return (diag ** 2).sum()

    def load_pretrained_weight(
        self, weight: torch.Tensor, bias: torch.Tensor | None = None,
    ) -> None:
        """Load pretrained weights from a standard ``nn.Linear`` layer.

        This copies the weight (and optionally bias) from a pretrained linear
        layer. Useful for fine-tuning: load the pretrained weights, then train
        only R (standard OFT) or both R and B (gated OFT).

        Args:
            weight: Pretrained weight tensor of shape ``(out_features, in_features)``.
            bias: Optional pretrained bias tensor of shape ``(out_features,)``.
        """
        if weight.shape != self.weight.shape:
            raise ValueError(
                f"Weight shape mismatch: expected {self.weight.shape}, "
                f"got {weight.shape}"
            )
        with torch.no_grad():
            if self._weight_multiplier != 1.0:
                # In gated μP mode, weight stores the raw O(1) param;
                # convert from actual weight space: raw = weight / multiplier.
                self.weight.copy_(weight / self._weight_multiplier)
            else:
                self.weight.copy_(weight)
        if bias is not None:
            if self.bias is None:
                raise ValueError("This layer has no bias, but bias was provided")
            if bias.shape != self.bias.shape:
                raise ValueError(
                    f"Bias shape mismatch: expected {self.bias.shape}, "
                    f"got {bias.shape}"
                )
            with torch.no_grad():
                self.bias.copy_(bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 256,
        reconn_sz: int = 8,
        activation: Literal["silu_gate"] | None = None,
        backend: Literal["cute", "cublas", "pytorch"] = "cute",
        autotuning: bool = False,
        **kwargs,
    ) -> "OFTLinear":
        """Create an OFTLinear layer from an existing ``nn.Linear`` layer.

        Copies the pretrained weight and bias, initializes R as identity.

        Args:
            linear: The source nn.Linear layer.
            group_size: Number of output channels per reconnection group.
            reconn_sz: Block size of the orthogonal reconnection matrix.
            activation: ``None`` for fine-tuning, ``"silu_gate"`` for width expansion.
            backend: Computation backend.
            autotuning: Enable automatic kernel autotuning.
            **kwargs: Additional arguments passed to ``OFTLinear.__init__``.

        Returns:
            A new OFTLinear layer with pretrained weights loaded.
        """
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            group_size=group_size,
            reconn_sz=reconn_sz,
            bias=linear.bias is not None,
            activation=activation,
            backend=backend,
            autotuning=autotuning,
            **kwargs,
        )
        layer.load_pretrained_weight(linear.weight.data,
                                     linear.bias.data if linear.bias is not None else None)
        return layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Flatten leading dimensions to 2D
        orig_shape = input.shape
        A = input.reshape(-1, self.in_features)

        if self._weight_multiplier != 1.0:
            B = self._weight_multiplier * self.weight
        else:
            B = self.weight

        R = self._build_R()

        # The custom op call is opaque to torch.compile — no graph breaks.
        # activation="" is used as sentinel for None (custom_op requires str).
        C = _oft_forward_op(
            A, B, R,
            self.group_size, self.reconn_sz,
            self.backend, self.activation or "",
            self.autotuning, self.force_rebenchmark,
        )

        # Restore leading dimensions
        out_shape = orig_shape[:-1] + (self.out_features,)
        C = C.reshape(out_shape)

        if self.bias is not None:
            C = C + self.bias

        return C

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"group_size={self.group_size}",
            f"reconn_sz={self.reconn_sz}",
            f"bias={self.bias is not None}",
        ]
        if self.activation is not None:
            parts.append(f"activation={self.activation!r}")
        if self.activation is None:
            parts.append(f"cayley_order={self.cayley_order}")
        if self.backend != "cute":
            parts.append(f"backend={self.backend!r}")
        if self.autotuning:
            parts.append("autotuning=True")
        return ", ".join(parts)


def mup_fix_oft_shapes(model: nn.Module) -> None:
    """Fix OFTLinear infshapes for correct ``mup`` LR scaling.

    Must be called **after** ``mup.set_base_shapes()`` and **before**
    creating the optimizer::

        mup.set_base_shapes(model, base_model)
        mup_fix_oft_shapes(model)
        optimizer = MuAdam(model.parameters(), lr=base_lr)

    **Weight B** (``weight``): No adjustment needed. B has no runtime
    multiplier (scaling is absorbed into init), so it behaves identically
    to ``nn.Linear`` — ``MuAdamW`` scales its LR by ``1/width_mult``
    where ``width_mult = K / base_K``, giving O(1/√K) output change
    per Adam step.

    **Reconnection R** (``reconn``): R is block-diagonal with fixed
    ``(reconn_sz × reconn_sz)`` blocks. Its actual fan-in per block is
    ``reconn_sz`` (constant), not ``K`` (which scales with width).
    ``set_base_shapes`` incorrectly marks R's K-dimension as infinite
    (since ``K != base_K``), causing ``MuAdamW`` to over-scale R's LR.
    We fix this by marking R's K-dimension as finite, so ``ninf() == 1``
    and ``MuAdamW`` leaves R's LR unscaled — matching the fact that
    each block's learning dynamics don't depend on total width.

    Args:
        model: The model containing OFTLinear layers (already processed
            by ``mup.set_base_shapes``).
    """
    from mup.infshape import InfDim, InfShape

    for module in model.modules():
        if isinstance(module, OFTLinear):
            for name, param in module.named_parameters():
                if not hasattr(param, "infshape"):
                    continue
                if name == "reconn":
                    # R shape: (n_groups * reconn_sz, K)
                    # Mark K dim (last) as finite — block fan_in = reconn_sz, not K
                    dims = list(param.infshape)
                    dims[-1] = InfDim(None, dims[-1].dim)
                    param.infshape = InfShape(dims)
                # weight (B) and bias: keep infshape from set_base_shapes
