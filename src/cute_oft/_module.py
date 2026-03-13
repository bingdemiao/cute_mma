"""torch.nn.Module wrapper for OFT linear layers."""

from __future__ import annotations

import math
from typing import Callable, Iterable, Literal

import torch
import torch.nn as nn

from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams


class _OFTFunction(torch.autograd.Function):
    """Custom autograd function wiring cute_oft forward/backward."""

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        R: torch.Tensor,
        group_size: int,
        reconn_sz: int,
        backend: str,
        activation: str | None,
        autotuning: bool,
        force_rebenchmark: bool,
        comp_params,
        bwd_dadr_params,
        bwd_db_params,
    ) -> torch.Tensor:
        from . import forward as oft_forward

        ctx.save_for_backward(A, B, R)
        ctx.group_size = group_size
        ctx.reconn_sz = reconn_sz
        ctx.backend = backend
        ctx.activation = activation
        ctx.autotuning = autotuning
        ctx.force_rebenchmark = force_rebenchmark
        ctx.comp_params = comp_params
        ctx.bwd_dadr_params = bwd_dadr_params
        ctx.bwd_db_params = bwd_db_params

        return oft_forward(
            A, B, R, group_size, reconn_sz,
            backend=backend, comp_params=comp_params,
            activation=activation, autotuning=autotuning,
            force_rebenchmark=force_rebenchmark,
        )

    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        from . import backward as oft_backward

        A, B, R = ctx.saved_tensors

        dA, dR, dB = oft_backward(
            dC, A, B, R,
            ctx.group_size, ctx.reconn_sz,
            backend=ctx.backend,
            activation=ctx.activation,
            autotuning=ctx.autotuning,
            force_rebenchmark=ctx.force_rebenchmark,
            bwd_dadr_params=ctx.bwd_dadr_params,
            bwd_db_params=ctx.bwd_db_params,
        )

        # Return grads for: A, B, R, and None for the non-tensor args
        return dA, dB, dR, None, None, None, None, None, None, None, None, None


class OFTLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using OFT structure.

    Computes ``C = (A @ R^T) @ B^T`` per group (standard OFT), or
    ``C = (A * SiLU(A @ R^T)) @ B^T`` per group (gated OFT / width expansion).

    In standard mode (``activation=None``), only R is trainable and B (weight)
    is frozen — this is the classic OFT fine-tuning setup. In gated mode
    (``activation="silu_gate"``), both R and B are trainable, enabling the
    layer to act as a width expansion with ``n_groups`` independent pathways.

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

        # Weight B: (out_features, in_features)
        # Frozen in standard OFT, trainable in gated mode
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=gated,
        )

        # Reconnection matrix R: (n_groups * reconn_sz, in_features)
        # Always trainable
        self.reconn = nn.Parameter(
            torch.empty(n_groups * reconn_sz, in_features),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters.

        Weight B is initialized with Kaiming uniform (same as nn.Linear).
        Reconnection R is initialized as block-diagonal identity matrices,
        so the layer starts as a standard linear transform.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Initialize R as block-diagonal identity
        n_groups = self.out_features // self.group_size
        n_blocks = self.in_features // self.reconn_sz
        with torch.no_grad():
            self.reconn.zero_()
            for g in range(n_groups):
                for b in range(n_blocks):
                    for i in range(self.reconn_sz):
                        self.reconn[
                            g * self.reconn_sz + i,
                            b * self.reconn_sz + i,
                        ] = 1.0

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

        # Ensure half precision for CUDA kernels
        if A.dtype != torch.float16:
            A = A.half()

        C = _OFTFunction.apply(
            A, self.weight, self.reconn,
            self.group_size, self.reconn_sz,
            self.backend, self.activation,
            self.autotuning, self.force_rebenchmark,
            self.comp_params, self.bwd_dadr_params, self.bwd_db_params,
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
