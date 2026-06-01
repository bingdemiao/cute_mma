"""GPT-2 architecture with optional PrismLinear MLP up-projection.

Layout follows Karpathy's nanoGPT (model.py). The only structural change is
``PrismMLP``, which replaces the GELU MLP with::

    LayerNorm(d) -> PrismLinear(d, 4d, silu_gate, cublas, bf16)
                 -> GroupNorm(n_groups=4d/group_size)
                 -> Linear(4d, d)

per the guidance in ``cute_mma/TRAINING_GATED_PRISM.md``: the gate is built
into PrismLinear and the required GroupNorm sits *after* it. The down-projection
stays a plain ``nn.Linear`` for an apples-to-apples MLP swap.

Set ``config.mlp_type = "vanilla"`` for the original GELU MLP to use as a
baseline. Set ``"prism"`` (default) to enable the Prism up-projection.
"""

from dataclasses import dataclass, field
import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import cute_prism


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias, matching nanoGPT's convention."""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class GroupNormLast(nn.Module):
    """nn.GroupNorm wrapper for inputs whose channel dim is the last dim.

    GPT activations are ``(B, T, C)``; ``F.group_norm`` wants ``(N, C, *)``.
    We unfold (B,T) into the batch and add a singleton spatial dim so the
    normalization runs across each (B, T, group) slice independently of T.
    """

    def __init__(self, num_groups: int, num_channels: int, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = 1e-5
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.num_channels).unsqueeze(-1)  # (B*T, C, 1)
        out = F.group_norm(flat, self.num_groups, self.weight, self.bias, self.eps)
        return out.squeeze(-1).reshape(*leading, self.num_channels)


# ---------------------------------------------------------------------------
# Attention (standard nanoGPT)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


# ---------------------------------------------------------------------------
# MLPs
# ---------------------------------------------------------------------------

class VanillaMLP(nn.Module):
    """Original GPT-2 GELU MLP. Kept as a baseline for ablation."""

    def __init__(self, config):
        super().__init__()
        d_ff = config.mlp_expansion * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(d_ff, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class PrismMLP(nn.Module):
    """MLP with a PrismLinear up-projection and a plain Linear down-projection.

    Per ``cute_mma/TRAINING_GATED_PRISM.md``:
      * §1: a normalization layer must follow PrismLinear.
        We use ``GroupNorm(num_groups=4d/group_size)`` so the n_groups
        pathways stay independent.
      * §5: dropout between PrismLinears is forbidden, but dropout *after*
        the down projection (on the residual path) is fine — equivalent
        to the existing nn.Dropout in VanillaMLP.
      * §9: cublas + bf16 is the supported training configuration.
    """

    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        d_ff = config.mlp_expansion * d
        if d_ff % config.group_size != 0:
            raise ValueError(
                f"mlp_expansion*n_embd ({d_ff}) must be divisible by group_size "
                f"({config.group_size})"
            )
        if d % config.reconn_sz != 0:
            raise ValueError(
                f"n_embd ({d}) must be divisible by reconn_sz "
                f"({config.reconn_sz})"
            )
        if config.prism_input_shuffle:
            # Constraints (cute_prism @ this revision):
            #   - reconn_sz must be 16
            #   - in_features must be divisible by shuffle_blk_k
            # NOTE: the original nanogpt copy also raised for backend=="cute"
            # here, based on the stale forward() docstring claiming shuffle is
            # "cublas/pytorch only". The cute forward DOES wire shuffle_masks
            # into the kernel (see cute_prism/__init__.py forward, cute branch),
            # so this integration copy allows backend=="cute" on purpose — the
            # cute-vs-cublas harness exists precisely to validate it end-to-end.
            if config.reconn_sz != 16:
                raise ValueError(
                    f"prism_input_shuffle=True requires reconn_sz=16, "
                    f"got {config.reconn_sz}"
                )
            if d % config.prism_shuffle_blk_k != 0:
                raise ValueError(
                    f"n_embd ({d}) must be divisible by "
                    f"prism_shuffle_blk_k ({config.prism_shuffle_blk_k})"
                )
        n_groups = d_ff // config.group_size

        prism_kwargs = dict(
            in_features=d,
            out_features=d_ff,
            group_size=config.group_size,
            reconn_sz=config.reconn_sz,
            bias=config.bias,
            activation=config.prism_activation,
            backend=config.prism_backend,
            internal_bias=config.prism_internal_bias,
            dropout=config.prism_internal_dropout,
            r_init_scale=config.r_init_scale,
            input_shuffle=config.prism_input_shuffle,
            shuffle_blk_k=config.prism_shuffle_blk_k,
        )
        # Only pass ar_scale_init when set (>0), so we stay backwards-
        # compatible with the container's baked-in cute_prism (which doesn't
        # have the kwarg).
        if config.prism_ar_scale_init > 0:
            prism_kwargs["ar_scale_init"] = config.prism_ar_scale_init
        self.up = cute_prism.PrismLinear(**prism_kwargs)
        self.gn = GroupNormLast(n_groups, d_ff, affine=True)
        self.down = nn.Linear(d_ff, d, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.up(x)
        x = self.gn(x)
        x = self.down(x)
        return self.dropout(x)


class PrismMLPAdditive(nn.Module):
    """Prism MLP variant that composes additively instead of multiplicatively.

    The standard PrismMLP (gated) computes:
        preact = A @ B.T         in R^(d_ff)
        gate   = SiLU(AR)        in R^(n_groups, r)  broadcast to R^(d_ff)
        H      = preact * gate                            <-- multiplicative

    Chapter 8's diagnostic showed that the SiLU(AR) gate drifts into its
    near-linear saturated regime during pretraining, so the operation
    degenerates into a bilinear interaction (H approx preact * AR). All
    parameter-level interventions we tested (r_lr_mult, ar_scale, sigmoid
    gate) failed to escape this regime because the multiplicative coupling
    has a "linear tail" the model can lean on.

    The additive variant replaces the multiplicative gate with an additive
    contribution:
        preact = A @ B.T                         in R^(d_ff)
        gate_branch = SiLU(AR), broadcast to R^(d_ff)
        H      = preact + gate_branch                     <-- additive

    Properties:
    - No multiplicative coupling between preact and AR, so no path for SGD
      to drive the operation into a degenerate linear regime via R growth.
    - Same parameter count as PrismMLP (B, R, GN, down, dropout).
    - The block-diagonal structure of R is preserved -- this is still
      "prism" structurally, just with additive composition.

    Limitation: cannot "turn off" channels (gate=0 case for multiplicative).
    SGD has to use the additive branch as a shift rather than a mask.

    Implementation note: we do NOT use cute_prism's fused kernel here
    because that kernel computes the multiplicative H internally. Instead
    we use a plain nn.Linear for B and a separate Parameter for R, and
    compute the additive forward in pure PyTorch. This is slower than the
    fused kernel but only matters for the additive ablation experiment.
    """

    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        d_ff = config.mlp_expansion * d
        if d_ff % config.group_size != 0:
            raise ValueError(
                f"mlp_expansion*n_embd ({d_ff}) must be divisible by "
                f"group_size ({config.group_size})"
            )
        if d % config.reconn_sz != 0:
            raise ValueError(
                f"n_embd ({d}) must be divisible by reconn_sz "
                f"({config.reconn_sz})"
            )
        if config.group_size % config.reconn_sz != 0:
            raise ValueError(
                f"group_size ({config.group_size}) must be divisible by "
                f"reconn_sz ({config.reconn_sz}) for the per-group "
                f"broadcast in the additive variant"
            )

        n_groups = d_ff // config.group_size
        r = config.reconn_sz
        n_blocks = d // r
        per_gate = config.group_size // r  # channels each gate value covers

        # Up-projection (dense, same as VanillaMLP.c_fc).
        self.up_B = nn.Linear(d, d_ff, bias=config.bias)

        # Reconnection R: same shape as PrismLinear.reconn.
        self.reconn = nn.Parameter(torch.empty(n_groups * r, d))
        # Match cute_prism's gated-mode init: N(0, r_init_scale/sqrt(r)).
        nn.init.normal_(self.reconn, std=config.r_init_scale / math.sqrt(r))

        # Group dimensions cached for the forward.
        self.n_groups = n_groups
        self.r = r
        self.n_blocks = n_blocks
        self.group_size = config.group_size
        self.per_gate = per_gate
        self.d_ff = d_ff

        # Activation choice: silu_gate or sigmoid_gate (matches PrismLinear).
        if config.prism_activation == "silu_gate":
            self.activation = F.silu
        elif config.prism_activation == "sigmoid_gate":
            self.activation = torch.sigmoid
        else:
            raise ValueError(
                f"prism_activation must be 'silu_gate' or 'sigmoid_gate', "
                f"got {config.prism_activation!r}"
            )

        self.gn = GroupNormLast(n_groups, d_ff, affine=True)
        self.down = nn.Linear(d_ff, d, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x: (B, T, d)
        orig_shape = x.shape

        # Dense up-projection.
        preact = self.up_B(x)  # (B, T, d_ff)

        # Per-group reconnection product. Block-diagonal mathematically.
        flat = x.reshape(-1, orig_shape[-1])  # (BT, d)
        A_blocks = flat.reshape(-1, self.n_blocks, self.r)  # (BT, n_blocks, r)
        R_blocks = self.reconn.reshape(
            self.n_groups, self.r, self.n_blocks, self.r
        ).permute(0, 2, 1, 3).contiguous()  # (n_groups, n_blocks, r, r)
        AR = torch.einsum("tbi,gbij->tgj", A_blocks, R_blocks)
        # AR: (BT, n_groups, r)

        gate = self.activation(AR)  # (BT, n_groups, r)

        # Broadcast to d_ff: each gate value covers `per_gate` consecutive
        # channels within its group.
        gate_expanded = gate.unsqueeze(-1).expand(
            -1, -1, -1, self.per_gate
        )  # (BT, n_groups, r, per_gate)
        gate_expanded = gate_expanded.reshape(-1, self.d_ff)  # (BT, d_ff)
        gate_expanded = gate_expanded.reshape(*orig_shape[:-1], self.d_ff)

        # Additive composition.
        h = preact + gate_expanded
        h = self.gn(h)
        h = self.down(h)
        return self.dropout(h)


def make_mlp(config):
    if config.mlp_type == "vanilla":
        return VanillaMLP(config)
    if config.mlp_type == "prism":
        return PrismMLP(config)
    if config.mlp_type == "prism_additive":
        return PrismMLPAdditive(config)
    raise ValueError(f"unknown mlp_type {config.mlp_type!r}")


# ---------------------------------------------------------------------------
# Block + GPT
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = make_mlp(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # GPT-2 vocab padded up to nearest multiple of 64
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    # MLP options
    mlp_type: str = "prism"               # "prism" | "vanilla"
    mlp_expansion: int = 4                # hidden width = mlp_expansion * n_embd
    group_size: int = 64                  # PrismLinear group_size
    reconn_sz: int = 8                    # PrismLinear reconn block size
    prism_backend: str = "cublas"         # "cublas" (default), "cute", "pytorch"
    prism_activation: str = "silu_gate"   # "silu_gate" | "sigmoid_gate"
    prism_internal_bias: bool = False
    prism_internal_dropout: float = 0.0   # PrismLinear's internal dropout (post-gate)
    r_init_scale: float = 1.0
    # Input shuffle (per-group butterfly shuffle of A before A@R).
    # Per cute_prism: requires reconn_sz=16; in_features must be divisible by shuffle_blk_k.
    prism_input_shuffle: bool = False
    prism_shuffle_blk_k: int = 128
    # LoRA-α-style learnable scalar that multiplies R before the gating
    # non-linearity. 0.0 = standard prism (no extra parameter). Positive
    # value (e.g. 0.1) registers a learnable ar_scale Parameter and keeps
    # gate inputs in SiLU's informative band even if R grows during
    # training. Float (not Optional) so configurator's type-check works.
    prism_ar_scale_init: float = 0.0

    # muP (maximal update parameterization). When use_mup=True, train.py
    # calls mup_setup.apply_mup() which assigns infshapes (via a base model
    # at width mup_base_width) and switches the optimizer to MuAdamW. The
    # readout (tied lm_head) gets a 1/width_mult output multiplier applied
    # in forward() — option B: keep weight tying, no MuReadout, so the
    # model stays comparable to the (tied) vanilla baseline. See mup_setup.py.
    use_mup: bool = False
    mup_base_width: int = 128
    mup_output_mult: float = 1.0    # set by apply_mup() to base_width / n_embd


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tied embedding (weight sharing between wte and lm_head).
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Apply the GPT-2 scaled init to residual projections (c_proj only;
        # we leave PrismLinear alone since its reset_parameters() is calibrated).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        # Subtract wpe; wte is tied with lm_head so it's already counted once.
        n_params -= self.transformer.wpe.weight.numel()
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # PrismLinear has its own calibrated reset_parameters() — do not touch.

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # muP readout multiplier (option B): scale logits by base_width/n_embd
        # so the output's per-step change is width-stable without a MuReadout.
        # Default mult is 1.0 (no-op) unless apply_mup() set it.
        om = self.config.mup_output_mult
        if targets is not None:
            logits = self.lm_head(x)
            if om != 1.0:
                logits = logits * om
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference micro-optimization: only compute the last position.
            logits = self.lm_head(x[:, [-1], :])
            if om != 1.0:
                logits = logits * om
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        """Decrease the model's block_size in place (e.g. for fine-tuning)."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type,
                             r_wd_factor=0.1, r_lr_mult=1.0):
        """Build AdamW with the parameter groups recommended in
        ``cute_mma/TRAINING_GATED_PRISM.md`` §4:

        | group         | weight decay                             |
        |---------------|------------------------------------------|
        | weight (B)    | base wd                                  |
        | reconn (R)    | base wd * r_wd_factor (~0.1× by default) |
        | _internal_bias| base wd                                  |
        | bias / norms  | 0                                        |
        """
        norm_types = (nn.LayerNorm, nn.GroupNorm, GroupNormLast, LayerNorm)
        norm_param_ids = set()
        for m in self.modules():
            if isinstance(m, norm_types):
                for p in m.parameters(recurse=False):
                    norm_param_ids.add(id(p))

        decay, r_decay, ib_decay, no_decay = [], [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in norm_param_ids:
                no_decay.append(p)
            elif "reconn" in name:
                r_decay.append(p)
            elif "_internal_bias" in name:
                ib_decay.append(p)
            elif name.endswith(".bias"):
                no_decay.append(p)
            elif p.dim() < 2:
                # Catch-all for 1-D params that aren't tagged above (rare).
                no_decay.append(p)
            else:
                decay.append(p)

        groups = [
            {"params": decay,    "weight_decay": weight_decay,                  "lr": learning_rate,                   "name": "decay"},
            {"params": r_decay,  "weight_decay": weight_decay * r_wd_factor,    "lr": learning_rate * r_lr_mult,       "name": "r_decay"},
            {"params": ib_decay, "weight_decay": weight_decay,                  "lr": learning_rate * r_lr_mult,       "name": "ib_decay"},
            {"params": no_decay, "weight_decay": 0.0,                           "lr": learning_rate,                   "name": "no_decay"},
        ]
        groups = [g for g in groups if len(g["params"]) > 0]

        for g in groups:
            n = sum(p.numel() for p in g["params"])
            print(f"  param group {g['name']!r}: {len(g['params'])} tensors, "
                  f"{n:,} params, wd={g['weight_decay']}, lr={g['lr']}")

        # Use fused AdamW when available.
        fused_ok = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused_ok and device_type == "cuda") else {}
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)
        print(f"  using fused AdamW: {bool(extra.get('fused', False))}")
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
