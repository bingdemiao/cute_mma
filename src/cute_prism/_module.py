"""torch.nn.Module wrapper for Prism linear layers."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from ._config import BwdDAdRCompParams, BwdDBCompParams, CompParams

SHUFFLE_SEGMENT_SZ = 8  # 128-bit vectorized load = 8 x fp16


_GATED_SILU_BWD_GAIN = 1.437  # measured bwd/fwd gain ratio for gated SiLU (M=32768, 10 seeds)


class _BwdScaleFn(torch.autograd.Function):
    """Multiply gradients by a fixed scalar in the backward pass only."""

    @staticmethod
    def forward(ctx, x, bwd_scale):
        ctx.bwd_scale = bwd_scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.bwd_scale, None



# ---------------------------------------------------------------------------
# Custom ops: opaque to torch.compile, with fake-tensor + autograd support
# ---------------------------------------------------------------------------

@torch.library.custom_op("cute_prism::forward", mutates_args=())
def _prism_forward_op(
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
    from . import forward as prism_forward

    C, _seeds = prism_forward(
        A, B, R, group_size, reconn_sz,
        backend=backend,
        activation=activation or None,
        autotuning=autotuning,
        force_rebenchmark=force_rebenchmark,
    )
    return C


@_prism_forward_op.register_fake
def _(A, B, R, group_size, reconn_sz, backend, activation, autotuning,
      force_rebenchmark):
    return A.new_empty(A.shape[0], B.shape[0])


@torch.library.custom_op("cute_prism::backward", mutates_args=())
def _prism_backward_op(
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
    from . import backward as prism_backward

    dA, dR, dB, _d_ib = prism_backward(
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


@_prism_backward_op.register_fake
def _(dC, A, B, R, group_size, reconn_sz, backend, activation, autotuning,
      force_rebenchmark):
    return A.new_empty(A.shape), R.new_empty(R.shape), B.new_empty(B.shape)


def _prism_setup_context(ctx, inputs, output):
    A, B, R, group_size, reconn_sz, backend, activation, autotuning, \
        force_rebenchmark = inputs
    ctx.save_for_backward(A, B, R)
    ctx.group_size = group_size
    ctx.reconn_sz = reconn_sz
    ctx.backend = backend
    ctx.activation = activation
    ctx.autotuning = autotuning
    ctx.force_rebenchmark = force_rebenchmark


def _prism_backward(ctx, dC):
    A, B, R = ctx.saved_tensors

    dA, dR, dB = _prism_backward_op(
        dC, A, B, R,
        ctx.group_size, ctx.reconn_sz,
        ctx.backend, ctx.activation,
        ctx.autotuning, ctx.force_rebenchmark,
    )

    # Grads for: A, B, R, then None for the 6 non-tensor args
    return dA, dB, dR, None, None, None, None, None, None


_prism_forward_op.register_autograd(_prism_backward, setup_context=_prism_setup_context)


class _PrismFn(torch.autograd.Function):
    """Unified autograd wrapper for Prism forward/backward.

    Threads internal_bias, dropout, and seg_pairs (input shuffle) through
    the main ``forward()`` / ``backward()`` entry points.
    """

    @staticmethod
    def forward(ctx, A, B, R, group_size, reconn_sz, backend, activation,
                autotuning, force_rebenchmark, internal_bias, dropout_p, training,
                seg_pairs):
        from . import forward as prism_forward

        C, dropout_seeds = prism_forward(
            A, B, R, group_size, reconn_sz,
            backend=backend,
            activation=activation or None,
            autotuning=autotuning,
            force_rebenchmark=force_rebenchmark,
            internal_bias=internal_bias,
            dropout_p=dropout_p,
            training=training,
            seg_pairs=seg_pairs,
        )
        ctx.save_for_backward(A, B, R, internal_bias, seg_pairs)
        ctx.group_size = group_size
        ctx.reconn_sz = reconn_sz
        ctx.backend = backend
        ctx.activation = activation
        ctx.autotuning = autotuning
        ctx.force_rebenchmark = force_rebenchmark
        ctx.dropout_p = dropout_p
        ctx.dropout_seeds = dropout_seeds
        return C

    @staticmethod
    def backward(ctx, dC):
        from . import backward as prism_backward

        A, B, R, internal_bias, seg_pairs = ctx.saved_tensors

        dA, dR, dB, d_ib = prism_backward(
            dC.contiguous(), A.contiguous(), B.contiguous(), R.contiguous(),
            ctx.group_size, ctx.reconn_sz,
            backend=ctx.backend,
            activation=ctx.activation or None,
            autotuning=ctx.autotuning,
            force_rebenchmark=ctx.force_rebenchmark,
            internal_bias=internal_bias,
            dropout_seeds=ctx.dropout_seeds,
            dropout_p=ctx.dropout_p,
            seg_pairs=seg_pairs,
        )

        if dB is None:
            dB = torch.zeros_like(B)

        # Grads: A, B, R, then None x 6 scalar args, internal_bias, None x 2, seg_pairs
        return dA, dB, dR, None, None, None, None, None, None, d_ib, None, None, None



def _round_robin_matchings(n: int) -> list[list[tuple[int, int]]]:
    """Generate all unique perfect matchings via round-robin tournament.

    For n players (even), produces n-1 rounds where each round is a perfect
    matching and no pair appears in more than one round.
    """
    assert n % 2 == 0 and n >= 2
    matchings = []
    players = list(range(n))
    for _ in range(n - 1):
        matching = [(players[0], players[-1])]
        for i in range(1, n // 2):
            matching.append((players[i], players[n - 1 - i]))
        matchings.append(matching)
        players = [players[0]] + [players[-1]] + players[1:-1]
    return matchings


def _build_seg_pairs(
    in_features: int, out_features: int, group_size: int,
    reconn_sz: int, shuffle_blk_k: int,
) -> torch.Tensor:
    """Build per-group segment pair indices for input shuffling.

    Group 0 uses identity pairing. Other groups get per-chunk randomized
    matchings from a round-robin schedule, ensuring maximum diversity
    even when n_groups exceeds the number of available matchings.

    Returns:
        (n_groups, n_blocks, 2) int64 tensor of global segment indices.
    """
    seg_sz = SHUFFLE_SEGMENT_SZ
    n_groups = out_features // group_size
    n_chunks = in_features // shuffle_blk_k
    n_segments_per_chunk = shuffle_blk_k // seg_sz
    n_blocks = in_features // reconn_sz
    blocks_per_chunk = shuffle_blk_k // reconn_sz

    all_matchings = _round_robin_matchings(n_segments_per_chunk)
    n_matchings = len(all_matchings)

    # Deterministic affine code construction.
    #
    # Map each non-identity group g (1-indexed) to (a, b):
    #   a = (g-1) // n_matchings,  b = (g-1) % n_matchings
    #
    # Assignment for group g at chunk c:
    #   matching_idx = (a * c + b) % n_matchings
    #
    # Diversity guarantees:
    #   - Same a, different b: 0 chunk collisions (constant offset)
    #   - Different a: ≤ gcd(|a1-a2|, n_matchings) collisions per
    #     period of n_matchings chunks
    #   - Supports up to n_matchings^2 + 1 groups deterministically
    if n_groups - 1 > n_matchings * n_matchings:
        raise ValueError(
            f"Too many groups ({n_groups}) for affine construction. "
            f"Max supported: {n_matchings ** 2 + 1}"
        )

    seg_pairs = torch.zeros(n_groups, n_blocks, 2, dtype=torch.long)

    for g in range(n_groups):
        if g == 0:
            # Identity: natural segment pairing
            for c in range(n_chunks):
                seg_offset = c * n_segments_per_chunk
                for b_local in range(blocks_per_chunk):
                    b_global = c * blocks_per_chunk + b_local
                    seg_pairs[g, b_global, 0] = seg_offset + 2 * b_local
                    seg_pairs[g, b_global, 1] = seg_offset + 2 * b_local + 1
        else:
            a = (g - 1) // n_matchings
            b = (g - 1) % n_matchings
            for c in range(n_chunks):
                matching_idx = (a * c + b) % n_matchings
                matching = all_matchings[matching_idx]
                seg_offset = c * n_segments_per_chunk
                for b_local, (sa, sb) in enumerate(matching):
                    b_global = c * blocks_per_chunk + b_local
                    seg_pairs[g, b_global, 0] = seg_offset + sa
                    seg_pairs[g, b_global, 1] = seg_offset + sb

    return seg_pairs


class PrismLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using Prism structure.

    Computes ``C = (A @ R^T) @ B^T`` per group (standard mode), or
    ``C = (A * SiLU(A @ R^T)) @ B^T`` per group (gated mode / width expansion).

    In standard mode (``activation=None``), only R is trainable and B (weight)
    is frozen — this is the classic fine-tuning setup. In gated mode
    (``activation="silu_gate"``), both R and B are trainable, enabling the
    layer to act as a width expansion with ``n_groups`` independent pathways.

    Compatible with ``torch.compile``. The Prism kernel call is registered as
    a custom op, so the compiler treats it as opaque and traces through the
    surrounding standard torch operations (μP multipliers, reshape, bias)
    normally.

    Args:
        in_features: Input dimension (K).
        out_features: Output dimension (N).
        group_size: Number of output channels per reconnection group.
        reconn_sz: Block size of the orthogonal reconnection matrix R.
        bias: If True, adds a learnable bias to the output.
        activation: ``None`` for standard mode, ``"silu_gate"`` for gated mode.
        backend: Computation backend (``"cute"``, ``"cublas"``, ``"pytorch"``).
        autotuning: Enable automatic kernel autotuning.
        force_rebenchmark: Force re-benchmarking when autotuning.
        comp_params: Explicit forward kernel configuration.
        bwd_dadr_params: Explicit backward dA+dR kernel configuration.
        bwd_db_params: Explicit backward dB kernel configuration.
        internal_bias: If True, adds a learnable bias after A @ R^T, before
            the SiLU gate (gated mode only). Shape (n_groups, in_features).
        dropout: Dropout probability applied to the gated activation
            H = A * SiLU(A @ R^T + b) before projection through B.

    Shape:
        - Input: ``(*, in_features)`` where ``*`` means any number of leading dimensions.
        - Output: ``(*, out_features)``.

    Examples::

        # Standard fine-tuning (freeze pretrained weights, train R only)
        layer = cute_prism.PrismLinear(256, 1024, group_size=256, reconn_sz=8)
        layer.load_pretrained_weight(pretrained_linear.weight)
        output = layer(input)

        # Gated mode as width expansion (train both R and B)
        layer = cute_prism.PrismLinear(256, 1024, group_size=256, reconn_sz=8,
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
        input_shuffle: bool = False,
        shuffle_blk_k: int = 128,
        backend: Literal["cute", "cublas", "pytorch"] = "cute",
        autotuning: bool = False,
        force_rebenchmark: bool = False,
        comp_params: CompParams | None = None,
        bwd_dadr_params: BwdDAdRCompParams | None = None,
        bwd_db_params: BwdDBCompParams | None = None,
        internal_bias: bool = False,
        dropout: float = 0.0,
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
        if input_shuffle:
            if reconn_sz != 16:
                raise ValueError("input_shuffle requires reconn_sz=16")
            if in_features % shuffle_blk_k != 0:
                raise ValueError(
                    f"in_features ({in_features}) must be divisible by "
                    f"shuffle_blk_k ({shuffle_blk_k})"
                )
            if backend == "cute":
                raise ValueError(
                    "input_shuffle is not supported with the 'cute' backend. "
                    "Use 'cublas' or 'pytorch' instead."
                )

        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.reconn_sz = reconn_sz
        self.activation = activation
        self.cayley_order = cayley_order
        self.input_shuffle = input_shuffle
        self.shuffle_blk_k = shuffle_blk_k
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
        # PrismLinear fully compatible with muP's optimizer-level LR scaling
        # (MuAdamW) and allows mixing with nn.Linear in the same model.
        #
        # B: init at N(0, alpha/sqrt(K)) where alpha ≈ 1.66 compensates for
        #    gated SiLU variance reduction. Same muP behavior as nn.Linear.
        # R: init at N(0, 1/sqrt(r)) where r = reconn_sz is the block fan-in.
        #    mup_fix_prism_shapes() marks R's K-dim as finite so MuAdamW
        #    doesn't over-scale its LR based on the full tensor width.
        self._weight_multiplier = 1.0 if gated else 1.0
        self._reconn_multiplier = 1.0 if gated else 1.0 / math.sqrt(reconn_sz)

        # Weight B: (out_features, in_features)
        # Frozen in standard mode, trainable in gated mode.
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

        # Internal bias: added after A @ R^T, before SiLU gate (gated mode only)
        if internal_bias and gated:
            self._internal_bias = nn.Parameter(
                torch.empty(n_groups, in_features))
        else:
            self.register_parameter("_internal_bias", None)

        self.dropout = dropout

        # Input shuffle: build per-group segment pair indices
        if input_shuffle:
            seg_pairs = _build_seg_pairs(
                in_features, out_features, group_size, reconn_sz, shuffle_blk_k,
            )
            self.register_buffer("_seg_pairs", seg_pairs)

        # Backward scalar correction for gated SiLU's multiplicative gain
        bwd_gain = _GATED_SILU_BWD_GAIN if activation == "silu_gate" else 1.0
        self._bwd_scale = 1.0 / bwd_gain

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
            # Gain of a * SiLU(a @ r^T) with skew-symmetric R at std=1/sqrt(r).
            # Measured empirically: 0.5638 for r=16 (NOT 0.6024 from the
            # independent x*SiLU(y) approximation, which ignores within-block
            # correlation between a and ar).
            _GATED_SILU_CORRECTION = 1.0 / 0.5638
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
            # Finetuning: B at 1/sqrt(K) for variance-preserving forward
            # (R starts at identity via Cayley with M=0).
            # B will typically be overwritten by load_pretrained_weight().
            nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.in_features))
            nn.init.zeros_(self.reconn)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self._internal_bias is not None:
            # Scale so that std(AR + bias) matches std(AR) at init.
            # bias std ~ std(R @ x) to keep total variance ~ 2x original.
            nn.init.normal_(self._internal_bias,
                            std=1.0 / math.sqrt(self.reconn_sz))

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
        only R (standard mode) or both R and B (gated mode).

        When ``input_shuffle=True`` and not gated (finetuning mode), each
        group's B columns are permuted to match the shuffled AR layout.
        This is a one-time cost at load time so the forward path needs no
        runtime scatter or extra gather.

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
                self.weight.copy_(weight / self._weight_multiplier)
            elif self.input_shuffle and self.activation != "silu_gate":
                # Non-gated finetuning with shuffle: permute B columns per group
                # so that frozen B is compatible with the shuffled AR layout.
                n_groups = self.out_features // self.group_size
                w = weight.clone()
                for g in range(n_groups):
                    elem_perm = self._seg_pairs_to_elem_perm(g)
                    w[g * self.group_size : (g + 1) * self.group_size] = \
                        weight[g * self.group_size : (g + 1) * self.group_size][:, elem_perm]
                self.weight.copy_(w)
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
    ) -> "PrismLinear":
        """Create an PrismLinear layer from an existing ``nn.Linear`` layer.

        Copies the pretrained weight and bias, initializes R as identity.

        Args:
            linear: The source nn.Linear layer.
            group_size: Number of output channels per reconnection group.
            reconn_sz: Block size of the orthogonal reconnection matrix.
            activation: ``None`` for fine-tuning, ``"silu_gate"`` for width expansion.
            backend: Computation backend.
            autotuning: Enable automatic kernel autotuning.
            **kwargs: Additional arguments passed to ``PrismLinear.__init__``.

        Returns:
            A new PrismLinear layer with pretrained weights loaded.
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

        needs_full_fn = (
            self.input_shuffle
            or self._internal_bias is not None
            or self.dropout > 0.0
        )

        if needs_full_fn:
            seg_pairs = self._seg_pairs if self.input_shuffle else None
            C = _PrismFn.apply(
                A, B, R,
                self.group_size, self.reconn_sz,
                self.backend, self.activation or "",
                self.autotuning, self.force_rebenchmark,
                self._internal_bias, self.dropout, self.training,
                seg_pairs,
            )
        else:
            # The custom op call is opaque to torch.compile — no graph breaks.
            # activation="" is used as sentinel for None (custom_op requires str).
            C = _prism_forward_op(
                A, B, R,
                self.group_size, self.reconn_sz,
                self.backend, self.activation or "",
                self.autotuning, self.force_rebenchmark,
            )

        # Correct backward gain from gated SiLU's multiplicative structure
        if self._bwd_scale != 1.0:
            C = _BwdScaleFn.apply(C, self._bwd_scale)

        # Restore leading dimensions
        out_shape = orig_shape[:-1] + (self.out_features,)
        C = C.reshape(out_shape)

        if self.bias is not None:
            C = C + self.bias

        return C

    def _seg_pairs_to_elem_perm(self, g: int) -> torch.Tensor:
        """Convert seg_pairs[g] to an element-level permutation of K columns.

        seg_pairs[g, b] = (seg_a, seg_b) means R block b reads segments
        seg_a and seg_b. The permuted layout puts them at positions
        [b*reconn_sz .. (b+1)*reconn_sz). This returns the element indices
        that map natural order to permuted order.
        """
        seg_sz = SHUFFLE_SEGMENT_SZ
        n_blocks = self._seg_pairs.shape[1]
        perm = torch.zeros(self.in_features, dtype=torch.long,
                           device=self._seg_pairs.device)
        for b in range(n_blocks):
            sa = self._seg_pairs[g, b, 0]
            sb = self._seg_pairs[g, b, 1]
            dst = b * self.reconn_sz
            offsets = torch.arange(seg_sz, device=self._seg_pairs.device)
            perm[dst:dst + seg_sz] = sa * seg_sz + offsets
            perm[dst + seg_sz:dst + 2 * seg_sz] = sb * seg_sz + offsets
        return perm

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
        if self._internal_bias is not None:
            parts.append("internal_bias=True")
        if self.dropout > 0.0:
            parts.append(f"dropout={self.dropout}")
        if self.input_shuffle:
            parts.append(f"input_shuffle=True, shuffle_blk_k={self.shuffle_blk_k}")
        if self.backend != "cute":
            parts.append(f"backend={self.backend!r}")
        if self.autotuning:
            parts.append("autotuning=True")
        return ", ".join(parts)


def mup_fix_prism_shapes(model: nn.Module) -> None:
    """Fix PrismLinear infshapes for correct ``mup`` LR scaling.

    Must be called **after** ``mup.set_base_shapes()`` and **before**
    creating the optimizer::

        mup.set_base_shapes(model, base_model)
        mup_fix_prism_shapes(model)
        optimizer = MuAdam(model.parameters(), lr=base_lr)

    **Weight B** (``weight``): No adjustment needed. B has no runtime
    multiplier (scaling is absorbed into init), so it behaves identically
    to ``nn.Linear`` — ``MuAdamW`` scales its LR by ``1/width_mult``
    where ``width_mult = K / base_K``, giving O(1/√K) output change
    per Adam step.

    **Reconnection R** (``reconn``): R is block-diagonal with K/r blocks
    of size r × r (r = reconn_sz). Its fan-in and fan-out are both r,
    independent of the model width K. In muP, each parameter's LR
    scaling depends on its own width dimension, not on downstream layers.
    Since R's effective width is r (constant), its LR should have no
    K-dependent scaling — analogous to how attention Wq/Wk/Wv (which
    operate at head_dim, not d_model) don't get width-dependent LR
    scaling while Wo does.

    We achieve this by marking R's K-dimension as finite (``base_dim = K``),
    so MuAdamW treats it as a fixed-size parameter and applies no LR
    scaling. The O(1) per-step output perturbation from R is damped to
    O(1/√K) by the downstream B projection, just as attention head
    outputs are damped by Wo.

    Args:
        model: The model containing PrismLinear layers (already processed
            by ``mup.set_base_shapes``).
    """
    from mup.infshape import InfDim, InfShape

    for module in model.modules():
        if isinstance(module, PrismLinear):
            for name, param in module.named_parameters():
                if not hasattr(param, "infshape"):
                    continue
                if name == "reconn":
                    # R shape: (n_groups * reconn_sz, K)
                    # Mark K dim as finite so width_mult = 1 and MuAdamW
                    # applies no LR scaling. R's block-diagonal structure
                    # means its effective fan-in is reconn_sz (constant),
                    # and it acts as the input layer in each group's
                    # two-layer sub-network.
                    dims = list(param.infshape)
                    K = dims[-1].dim
                    dims[-1] = InfDim(K, K)  # base_dim = dim → width_mult = 1
                    param.infshape = InfShape(dims)
                # weight (B) and bias: keep infshape from set_base_shapes


