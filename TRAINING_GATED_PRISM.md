# Training guidance: gated PrismLinear (pretraining)

This document captures the practical guidance for configuring a training
script that uses `cute_prism.PrismLinear` in gated pretraining mode
(`activation="silu_gate"`, both `R` and `B` trainable). It assumes the
caller already has a pretraining loop and wants to know how to wire
PrismLinear into it correctly.

Out of scope: finetuning mode (orthogonal R, frozen B) and the `cute`
backend. Training is expected to use the **`cublas` backend** with bf16.

## 1. Architecture requirements

**Every `PrismLinear` must be followed by a normalization layer** —
`nn.GroupNorm(num_groups=n_groups, num_channels=out_features, affine=False)`
where `n_groups = out_features // group_size` — or by a `LayerNorm` /
`RMSNorm` whose output feeds into the next sublayer's norm via a residual
path. Rationale:

- Forward and backward gains of `PrismLinear → GroupNorm` are empirically
  verified to be ≈1.0 under the current initialization (measured on the
  `library` branch). Without a norm layer downstream, the layer's raw
  forward and backward gains are *not* unit, and gradient scales will be
  mis-calibrated for anything that depends on them.
- The norm layer is also the primary stabilizer against R drift (see §4).

Prefer `GroupNorm` with `num_groups == n_groups` over `LayerNorm` over all
channels. PrismLinear keeps the `n_groups` pathways independent by design;
`LayerNorm` couples them, which defeats part of the inductive bias.

## 2. Initialization

**Already handled by `PrismLinear.reset_parameters()` on the `library`
branch at or after commit `77aa4e2`.** The training script should not
override it. For reference, the current scheme is:

- `B`: `N(0, 1/sqrt(K))` — matches `nn.Linear`.
- `R`: block-diagonal skew-symmetric with per-entry std `4/sqrt(2r)`
  (or `4/sqrt(2(r-1))` when `internal_bias=True`), where `r = reconn_sz`.
- `_internal_bias` (if enabled): `N(0, 1/sqrt(r-1))`.
- Output bias: uniform over `±1/sqrt(K)`.

Do **not** pass a custom `_weight_multiplier` or `_reconn_multiplier`.
Both are 1.0 in gated mode; all scaling is absorbed into init so that
parameters live at O(1/√fan_in) and are compatible with muP's LR scaling.

## 3. muP LR scaling

If the training script uses `mup.set_base_shapes(model, base_model)`,
**you must call `cute_prism.mup_fix_prism_shapes(model)` immediately
after `set_base_shapes` and before constructing the optimizer**. This
marks `R`'s K-dimension as finite so `MuAdamW` does not scale R's LR
with model width. Without this call, R's LR will be decayed
incorrectly as you scale width, and training will be unstable at large
widths.

```python
mup.set_base_shapes(model, base_model)
cute_prism.mup_fix_prism_shapes(model)      # REQUIRED with muP
optimizer = MuAdamW(param_groups, lr=base_lr)
```

B and bias behave like a normal `nn.Linear` under muP.

## 4. Weight decay

Split parameters into groups with different weight decay:

| Parameter | Recommended WD |
|---|---|
| `weight` (B) | Normal pretraining value (e.g., 0.1) |
| `reconn` (R) | Small — `0.1×` to `0.01×` of B's WD, but **not zero** |
| `_internal_bias` (if used) | Normal-to-aggressive (≥ B's WD) |
| `bias` (output bias) | 0 |
| Norm layers (GroupNorm/LN affine) | 0 |

**Why R needs a small amount of WD, not zero:**

- R is not in danger of blowing up. Its gradient is bounded by `||a||² ·
  SiLU'(·) · ||dH||`, Adam's second moment adapts, and the downstream
  norm layer absorbs any scale drift. So zero WD *almost always* trains
  stably.
- The real failure mode is **asymmetric drift**. SiLU has near-zero
  gradient in the deep-negative regime, so R entries that push `(Ra)_i`
  into that regime receive no corrective signal and freeze. Their
  positive-side siblings then grow to compensate, and over long runs
  you silently lose R capacity. A tiny WD continuously pulls frozen
  entries back toward zero where they can get unstuck.
- Do NOT use large WD on R. R → 0 makes `SiLU(AR) → 0.5·AR`, collapsing
  the gate into a scaled bilinear map and killing the nonlinearity that
  justifies gating in the first place. Keep WD on R at least 10× smaller
  than on B.

**Why `_internal_bias` should be regularized aggressively:**

It is the fastest way to shift `Ra + b` off-center, has no natural
restoring force, and once it pushes the gate deep into saturation it
gets stuck there because the gradient is zero. Regularizing it hard is
cheap insurance.

Example with AdamW:

```python
decay, r_decay, ib_decay, no_decay = [], [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "reconn" in name:
        r_decay.append(p)
    elif "_internal_bias" in name:
        ib_decay.append(p)
    elif "bias" in name or "norm" in name.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = AdamW([
    {"params": decay,    "weight_decay": 0.1},
    {"params": r_decay,  "weight_decay": 0.01},    # 10× smaller
    {"params": ib_decay, "weight_decay": 0.1},
    {"params": no_decay, "weight_decay": 0.0},
], lr=base_lr)
```

## 5. Dropout

**Do not insert `nn.Dropout` between stacked PrismLinear layers.** It
regularizes the downstream layer's `R`, which has fan-in `r` (8 or 16).
For `r=16, p=0.1` the relative noise on the pre-activation is ~8× higher
than on a dense layer with fan-in 1024, and for `r=8` it's ~12× higher.
This produces a late-training convergence plateau where R's gradient
signal is dominated by dropout noise and the optimizer oscillates instead
of converging.

If you want activation dropout inside a PrismLinear, use the layer's
**built-in `dropout` kwarg**. It drops `H = A · SiLU(AR + b)` — the
post-gate activation — which regularizes B's input directly without
corrupting R's gradient.

```python
layer = cute_prism.PrismLinear(..., activation="silu_gate", dropout=0.1)
```

External `nn.Dropout` between the PrismLinear and its downstream
GroupNorm is fine but rarely buys you anything beyond the internal
dropout above.

## 6. Gradient clipping

Use standard global gradient clipping (e.g., `torch.nn.utils.clip_grad_norm_`
with `max_norm=1.0`). This catches outlier-batch spikes for all
parameters and is cheap. It's more important than any specific WD tuning
as a guard against rare training incidents.

## 7. Rank bottleneck

`R` is block-diagonal with blocks of size `r`. Each output element of
`AR` depends on only `r` input features. This is a real capacity
ceiling on a single PrismLinear, so:

- **Do not judge PrismLinear from the first few thousand steps.** Training
  loss will plateau higher than a dense `nn.Linear` of the same fan-in
  for a while, then catch up as the stack of layers mixes features
  across blocks. Early-curve comparisons with baselines will be
  misleading.
- If you need faster mixing, set `input_shuffle=True` in the cublas
  backend (only; not supported in cute backend). This permutes A's
  columns per group so consecutive blocks see different feature
  combinations.
- Alternatively, increase `n_groups` by decreasing `group_size`.

## 8. Training monitors

Add these to your training loop's logging. They catch the common
failure modes before they get expensive.

1. **`|AR|` distribution per group.** Target: std(AR) in roughly [0.3, 3.0]
   per group. Drift below 0.3 means the gate has linearized; drift
   above 3.0 means SiLU is saturating. Either indicates something is
   off (bad LR, bad init override, bias runaway).
2. **Per-group output norms.** If one of the `n_groups` pathways has
   consistently 5–10× smaller output norm than the others after a few
   thousand steps, that group has dead gates and likely won't recover.
   Consider re-seeding or adding per-group LR warmup.
3. **`||reconn.grad||` RMS.** Should decay smoothly over training. A
   noise floor (flat gradient RMS that won't drop) is a sign that
   regularization is too strong or that you have an inter-layer
   dropout somewhere it shouldn't be.
4. **`||_internal_bias||`** (if used). Flag if it doubles from its init
   scale — usually a symptom of a dead gate trying to re-center.
5. **Fraction of `H = A·SiLU(AR+b)` elements with `|H| < 0.01`.** Rising
   fraction means gates are dying.

## 9. Things to explicitly NOT do

- **Do not use the `cute` backend for training.** It is fp16-only, and
  SiLU's exp can overflow at the new init's aggressive R scale. Use
  `backend="cublas"` with bf16.
- **Do not override `reset_parameters`** unless you know the current
  init is wrong for your config. The current scheme is calibrated to
  produce ~unit fwd/bwd gain through PrismLinear → GroupNorm.
- **Do not apply weight decay to R with the same strength as B.** 10×
  smaller is a reasonable default.
- **Do not insert `nn.Dropout` between stacked PrismLinears.** Use the
  layer's `dropout` kwarg instead.
- **Do not omit `mup_fix_prism_shapes`** when using muP. Without it,
  R's LR scales incorrectly with width.
- **Do not use `LayerNorm` over all out_features** as the downstream
  norm. Use `GroupNorm(num_groups=n_groups)` so the pathways stay
  independent.

## 10. Quick sanity check before a long run

On a single node, before launching a multi-hour pretraining run:

1. Build the model and print `sum(p.numel() for p in model.parameters()
   if p.requires_grad)`. Sanity-check against expectations (R adds
   `n_groups * K * r` params per PrismLinear).
2. Run 10 forward+backward passes on random data and log:
   - `var(y_gn) / var(x)` at each PrismLinear→GN output (should be ~1)
   - `||reconn.grad||` and `||weight.grad||` (both should be nonzero,
     finite, comparable in magnitude)
   - `|AR|` std per group (should be in [0.3, 3.0])
3. If any of the above look off, fix the config before scaling up.

---

*Document scope: practical training-script guidance as of library
branch commit `77aa4e2`. Does not cover kernel autotuning, finetuning
mode, or the cute backend. For those see `CLAUDE.md` and the public
API reference in `src/cute_prism/__init__.py`.*
