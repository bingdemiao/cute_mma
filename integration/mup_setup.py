"""muP (maximal update parameterization) setup for the prism nanoGPT.

Isolated from model.py / train.py so the non-muP path is untouched. The
goal: principled LR scaling across (width, gs, r) so we don't have to
re-sweep LR for every shape (see result.md §3.4 / §6).

Design choice (option B): keep the weight-tied lm_head, do NOT use
mup.MuReadout. Instead apply a scalar `mup_output_mult = base_width/n_embd`
to the logits in GPT.forward, which gives a width-stable readout while
staying comparable to the (tied) vanilla 4x baseline. set_base_shapes
still assigns infshapes so MuAdamW scales each parameter's LR correctly;
mup_fix_prism_shapes marks R's K-dim finite so R gets no width scaling.

Usage (train.py):
    from mup_setup import apply_mup, make_mup_optimizer
    if use_mup:
        apply_mup(model, config)               # sets infshapes + output_mult
        optimizer = make_mup_optimizer(model, weight_decay, learning_rate,
                                       betas, r_wd_factor, r_lr_mult)

Validate before any real run with the coordinate check:
    python mup_setup.py --coord_check
A correct muP setup produces activation magnitudes that are flat across
width at init; a broken one shows them growing/shrinking with width.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import torch

from model import GPT, GPTConfig


def make_base_config(config: GPTConfig) -> GPTConfig:
    """A narrow copy of `config` at width = mup_base_width.

    Only n_embd changes; n_head, n_layer, gs, r, etc. stay fixed (muP
    varies width = n_embd, holding the rest). The base must satisfy the
    same divisibility constraints as the target.
    """
    base_w = config.mup_base_width
    if base_w % config.n_head != 0:
        raise ValueError(
            f"mup_base_width ({base_w}) must be divisible by n_head "
            f"({config.n_head})")
    if base_w % config.reconn_sz != 0:
        raise ValueError(
            f"mup_base_width ({base_w}) must be divisible by reconn_sz "
            f"({config.reconn_sz})")
    d_ff_base = config.mlp_expansion * base_w
    if d_ff_base % config.group_size != 0:
        raise ValueError(
            f"mlp_expansion*mup_base_width ({d_ff_base}) must be divisible "
            f"by group_size ({config.group_size})")
    # Disable muP knobs on the base model itself (it's only used for shapes).
    return replace(config, n_embd=base_w, use_mup=False, mup_output_mult=1.0)


def apply_mup(model: GPT, config: GPTConfig) -> None:
    """Assign muP infshapes to `model` in place and set the readout mult.

    After this, build the optimizer with `make_mup_optimizer` (MuAdamW).
    """
    import mup
    from cute_prism import mup_fix_prism_shapes

    base_cfg = make_base_config(config)
    base_model = GPT(base_cfg)
    # set_base_shapes attaches .infshape to every parameter of `model`
    # by comparing against `base_model`. It returns the model.
    mup.set_base_shapes(model, base_model)
    # Mark R's K-dimension finite so MuAdamW gives R no width scaling
    # (R's effective fan-in is reconn_sz, constant in width).
    mup_fix_prism_shapes(model)
    # Option B readout: width-stable logits without MuReadout.
    config.mup_output_mult = config.mup_base_width / config.n_embd
    print(f"[mup] base_width={config.mup_base_width} target_width={config.n_embd} "
          f"output_mult={config.mup_output_mult:.4f}")


def make_mup_optimizer(model: GPT, weight_decay: float, learning_rate: float,
                       betas, device_type: str,
                       r_wd_factor: float = 0.1, r_lr_mult: float = 1.0):
    """Build MuAdamW with the same param-group structure as
    GPT.configure_optimizers, but LR-scaled per-parameter by muP infshapes.
    """
    import mup
    import torch.nn as nn
    from model import GroupNormLast, LayerNorm

    norm_types = (nn.LayerNorm, nn.GroupNorm, GroupNormLast, LayerNorm)
    norm_param_ids = set()
    for m in model.modules():
        if isinstance(m, norm_types):
            for p in m.parameters(recurse=False):
                norm_param_ids.add(id(p))

    decay, r_decay, ib_decay, no_decay = [], [], [], []
    for name, p in model.named_parameters():
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
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay,    "weight_decay": weight_decay,               "lr": learning_rate},
        {"params": r_decay,  "weight_decay": weight_decay * r_wd_factor, "lr": learning_rate * r_lr_mult},
        {"params": ib_decay, "weight_decay": weight_decay,               "lr": learning_rate * r_lr_mult},
        {"params": no_decay, "weight_decay": 0.0,                        "lr": learning_rate},
    ]
    groups = [g for g in groups if len(g["params"]) > 0]
    # MuAdamW reads each param's .infshape and scales its LR by 1/width_mult.
    optimizer = mup.MuAdamW(groups, lr=learning_rate, betas=betas)
    print(f"[mup] MuAdamW with {len(groups)} param groups")
    return optimizer


# ---------------------------------------------------------------------------
# Coordinate check: the standard muP validation. Activations at init should
# be ~width-invariant. We report the mean absolute logit and a mid-block
# activation across a few widths; flat across width = muP wiring is correct.
# ---------------------------------------------------------------------------
def coord_check(widths=(128, 256, 512, 1024), seed=0, device="cuda"):
    import mup
    base_cfg_kwargs = dict(
        block_size=128, vocab_size=50304, n_layer=4, n_head=8,
        dropout=0.0, bias=False, mlp_type="prism", prism_backend="cublas",
        prism_activation="silu_gate", mlp_expansion=3, group_size=32,
        reconn_sz=32, r_init_scale=0.5, use_mup=True, mup_base_width=128,
    )
    print(f"{'width':>8} {'|logits|':>12} {'|act blk2|':>12}")
    for w in widths:
        torch.manual_seed(seed)
        cfg = GPTConfig(n_embd=w, **base_cfg_kwargs)
        model = GPT(cfg).to(device).to(torch.bfloat16)
        apply_mup(model, cfg)
        model.eval()
        x = torch.randint(0, cfg.vocab_size, (4, 64), device=device)
        acts = {}
        h = model.transformer.h[len(model.transformer.h) // 2].register_forward_hook(
            lambda m, i, o: acts.__setitem__("blk", o))
        with torch.no_grad():
            logits, _ = model(x)
        h.remove()
        amax_logit = logits.float().abs().mean().item()
        amax_act = acts["blk"].float().abs().mean().item()
        print(f"{w:>8} {amax_logit:>12.4f} {amax_act:>12.4f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord_check", action="store_true")
    args = ap.parse_args()
    if args.coord_check:
        coord_check()
    else:
        print("nothing to do; pass --coord_check")
