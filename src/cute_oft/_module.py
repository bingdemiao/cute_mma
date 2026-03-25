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
        self.backend = backend
        self.autotuning = autotuning
        self.force_rebenchmark = force_rebenchmark
        self.comp_params = comp_params
        self.bwd_dadr_params = bwd_dadr_params
        self.bwd_db_params = bwd_db_params

        n_groups = out_features // group_size
        gated = activation == "silu_gate"

        # μP multipliers: absorb fan-in scaling so all raw params are O(1)
        # and a single base learning rate works for both B and R.
        # B has effective fan-in = K, R has effective fan-in = reconn_sz.
        self._weight_multiplier = 1.0 / math.sqrt(in_features) if gated else 1.0
        self._reconn_multiplier = 1.0 / math.sqrt(reconn_sz)

        # Weight B: (out_features, in_features)
        # Frozen in standard OFT, trainable in gated mode.
        # In gated mode, stores O(1) raw parameter; effective B = multiplier * weight.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=gated,
        )

        # Reconnection parameter: (n_groups * reconn_sz, in_features)
        # In gated mode (pretraining): stores O(1) raw param, R = multiplier * reconn.
        # In standard mode (finetuning): stores delta from identity, R = I + multiplier * reconn.
        self.reconn = nn.Parameter(
            torch.empty(n_groups * reconn_sz, in_features),
        )

        # For finetuning (non-gated): store block-diagonal identity as buffer
        if not gated:
            n_blocks = in_features // reconn_sz
            reconn_base = torch.eye(reconn_sz).repeat(1, n_blocks).repeat(n_groups, 1)
            self.register_buffer("_reconn_base", reconn_base)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters with μP-correct scaling.

        Gated mode (pretraining): Both weight and reconn are initialized as
        O(1) random values. The ``1/sqrt(fan_in)`` multipliers are applied in
        the forward pass, so a single base learning rate works for all params.

        Standard mode (finetuning): Weight B gets Kaiming init (will typically
        be overwritten by pretrained weights). Reconn is initialized to zero
        (delta from identity), so R starts as the block-diagonal identity.
        """
        gated = self.activation == "silu_gate"

        if gated:
            # Pretraining: O(1) random init for both B_raw and R_raw.
            # Effective B = (1/√K) * weight, R = (1/√r) * reconn.
            nn.init.normal_(self.weight)
            nn.init.normal_(self.reconn)
        else:
            # Finetuning: Kaiming B (frozen, will be overwritten),
            # zero delta (R starts at identity).
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.zeros_(self.reconn)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

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

        # Apply μP multipliers. These are standard torch ops that the
        # compiler can trace and fuse with surrounding operations.
        if self._weight_multiplier != 1.0:
            B = self._weight_multiplier * self.weight
        else:
            B = self.weight

        R = self._reconn_multiplier * self.reconn
        if hasattr(self, "_reconn_base"):
            R = self._reconn_base + R

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
        if self.backend != "cute":
            parts.append(f"backend={self.backend!r}")
        if self.autotuning:
            parts.append("autotuning=True")
        return ", ".join(parts)


def mup_fix_oft_shapes(model: nn.Module) -> None:
    """Mark OFTLinear parameters as finite for ``mup`` compatibility.

    OFTLinear already applies μP-correct ``1/sqrt(fan_in)`` multipliers in
    the forward pass, so ``mup``'s optimizer-level LR scaling must be
    disabled for these parameters.  Call this **after**
    ``mup.set_base_shapes()`` and **before** creating the optimizer::

        mup.set_base_shapes(model, base_model)
        mup_fix_oft_shapes(model)
        optimizer = MuAdam(model.parameters(), lr=base_lr)

    Args:
        model: The model containing OFTLinear layers (already processed
            by ``mup.set_base_shapes``).
    """
    from mup.infshape import InfDim, InfShape

    for module in model.modules():
        if isinstance(module, OFTLinear):
            for param in module.parameters():
                if hasattr(param, "infshape"):
                    param.infshape = InfShape(
                        [InfDim(None, d) for d in param.shape]
                    )
