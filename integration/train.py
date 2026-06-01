"""nanoGPT training script with optional PrismLinear MLP.

Adapted from Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT).

Single-process::

    python train.py config/train_shakespeare_char.py

Multi-GPU on one node (DDP)::

    torchrun --standalone --nproc_per_node=4 train.py \\
        config/train_fineweb.py

Multi-node DDP — the standard torchrun env vars must be set by the launcher
(e.g. submit.sh).

Precision policy
----------------
Following ``cute_mma/TRAINING_GATED_PRISM.md`` §9, we run with bf16 weights
and bf16 activations end-to-end. PrismLinear's cublas backend requires
matching dtypes for A/B/R, so we cast the entire model once at construction
and skip ``GradScaler``. AdamW keeps fp32 ``exp_avg``/``exp_avg_sq`` state
internally regardless of param dtype.
"""

import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

try:
    import cute_prism
except ImportError:
    cute_prism = None

# ----------------------------------------------------------------------------
# Default config — override via configurator.py.
# ----------------------------------------------------------------------------

# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' | 'resume'

# wandb logging
wandb_log = False
wandb_project = "nanogpt"
wandb_entity = None  # falls back to WANDB_ENTITY env var, then the user's default entity
wandb_run_name = "run"

# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Prism MLP
mlp_type = "prism"             # "prism" | "vanilla"
mlp_expansion = 4              # hidden width = mlp_expansion * n_embd
prism_input_shuffle = False    # per-group butterfly shuffle; requires reconn_sz=16
prism_shuffle_blk_k = 128
prism_ar_scale_init = 0.0      # >0 (e.g. 0.1) enables learnable ar_scale; 0 = disabled
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

# AdamW
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
r_wd_factor = 0.1              # R's WD as a fraction of base WD (§4)
r_lr_mult = 1.0                # R LR multiplier

# muP (maximal update parameterization) — see mup_setup.py. When True,
# infshapes are assigned via a base model and the optimizer becomes MuAdamW
# so LR scales correctly with width. Keeps weight tying (option B).
use_mup = False
mup_base_width = 128

# LR schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP / system
backend = "nccl"
device = "cuda"
dtype = "bfloat16"             # bf16 throughout (per training guide §9)
compile = False                # torch.compile is opt-in (cute_prism is opaque)

# ----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())
config = {k: globals()[k] for k in config_keys}
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Distributed setup
# ----------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# When the whole model is cast to bf16 (the supported PrismLinear training
# config), autocast(bf16) is a net negative: ops in the fp32-promotion list
# (LayerNorm, GroupNorm, ...) would emit fp32 outputs which then collide
# with PrismLinear's strict same-dtype A/B/R requirement. Skip autocast in
# that case and run everything at the parameter dtype.
if device_type == "cpu" or ptdtype != torch.float32:
    ctx = nullcontext()
else:
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "data", dataset)


def get_batch(split):
    """Memory-mapped uniform-sample minibatch loader.

    Re-opens the .bin file each call so we don't hold a stale fd if the
    underlying file rotates (e.g. in the streaming fineweb prepare).
    """
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    mlp_type=mlp_type, mlp_expansion=mlp_expansion,
    group_size=group_size, reconn_sz=reconn_sz,
    prism_backend=prism_backend, prism_activation=prism_activation,
    prism_internal_bias=prism_internal_bias,
    prism_internal_dropout=prism_internal_dropout,
    r_init_scale=r_init_scale,
    prism_input_shuffle=prism_input_shuffle,
    prism_shuffle_blk_k=prism_shuffle_blk_k,
    prism_ar_scale_init=prism_ar_scale_init,
    use_mup=use_mup,
    mup_base_width=mup_base_width,
)

if init_from == "scratch":
    if master_process:
        print("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ck_args = checkpoint["model_args"]
    for k in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size",
              "mlp_type", "mlp_expansion", "group_size", "reconn_sz",
              "prism_backend", "prism_activation", "prism_internal_bias",
              "prism_internal_dropout", "r_init_scale",
              "prism_input_shuffle", "prism_shuffle_blk_k",
              "prism_ar_scale_init"):
        if k in ck_args:
            model_args[k] = ck_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Strip DDP/compile prefixes if present.
    for prefix in ("_orig_mod.", "module."):
        bad = [k for k in state_dict if k.startswith(prefix)]
        for k in bad:
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
else:
    raise ValueError(f"unknown init_from: {init_from}")

# Optionally crop block_size for finetuning (rare).
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

# muP: assign infshapes (via a base model) and set the readout multiplier
# BEFORE the dtype cast / optimizer build. Isolated in mup_setup.py; no-op
# unless use_mup=True. See result.md §6.
if use_mup:
    from mup_setup import apply_mup
    apply_mup(model, gptconf)

# Cast the entire model to the training dtype. PrismLinear's cublas backend
# requires A/B/R to share dtype; the simplest way to guarantee that without
# threading autocast through a custom op is to keep all params in bf16.
if ptdtype != torch.float32:
    model = model.to(ptdtype)
model.to(device)

# Optimizer (param groups per training guide §4). MuAdamW under muP.
if use_mup:
    from mup_setup import make_mup_optimizer
    optimizer = make_mup_optimizer(
        model, weight_decay, learning_rate, (beta1, beta2), device_type,
        r_wd_factor=r_wd_factor, r_lr_mult=r_lr_mult,
    )
else:
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type,
        r_wd_factor=r_wd_factor, r_lr_mult=r_lr_mult,
    )
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

if compile:
    if master_process:
        print("compiling the model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# ----------------------------------------------------------------------------
# Eval / LR schedule helpers
# ----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def compute_metrics(model, iter_num, elapsed_sec, last_grad_norm):
    """Compute additional training metrics for wandb logging.

    grad_norm is passed in from the training loop (captured as the return
    value of clip_grad_norm_), because by the time the eval block runs,
    optimizer.zero_grad(set_to_none=True) has cleared every p.grad to None
    and recomputing it here would always yield zero.
    """
    metrics = {"grad_norm": last_grad_norm}

    total_param_norm = 0.0
    for p in model.parameters():
        if p.data is not None:
            total_param_norm += (p.data.norm(2) ** 2).item()
    metrics["param_norm"] = math.sqrt(total_param_norm)

    if elapsed_sec > 0:
        tokens_per_iter = batch_size * block_size * gradient_accumulation_steps
        if ddp:
            tokens_per_iter *= dist.get_world_size()
        metrics["throughput"] = tokens_per_iter / elapsed_sec

    return metrics


@torch.no_grad()
def compute_prism_monitors(model, sample_X, device):
    """Per-PrismLinear training-health diagnostics (TRAINING_GATED_PRISM.md §8).

    For every PrismLinear in the model, logs:
      * |AR| std per group (target [0.3, 3.0])
      * dead-gate fraction: |H = A * SiLU(AR + b)| < 0.01
      * per-group output norm ratio (max/min, flag > 5-10x)
      * R diag/offdiag RMS (drift from skew-symmetric init; ratio ~ 0 at init)
      * ||_internal_bias|| if the layer has one

    Hooks each PrismLinear's input, runs one forward on `sample_X`, then
    reconstructs AR and H in fp32 to keep statistics independent of the
    kernel's bf16 precision.
    """
    if cute_prism is None:
        return {}
    target_cls = cute_prism.PrismLinear
    prism_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, target_cls)]
    if not prism_layers:
        return {}

    captured = {}
    hooks = []

    def _make_pre_hook(name):
        def pre_hook(_module, inputs):
            captured[name] = inputs[0].detach()
        return pre_hook

    for name, layer in prism_layers:
        hooks.append(layer.register_forward_pre_hook(_make_pre_hook(name)))

    was_training = model.training
    model.eval()
    try:
        model(sample_X.to(device))
    finally:
        model.train(was_training)
        for h in hooks:
            h.remove()

    monitors = {}
    ar_std_means, dead_fracs, out_ratios, r_drift_ratios = [], [], [], []
    for idx, (name, layer) in enumerate(prism_layers):
        if name not in captured:
            continue
        tag = f"L{idx}"

        A = captured[name].reshape(-1, layer.in_features).float()
        n_groups = layer.out_features // layer.group_size
        r = layer.reconn_sz
        n_blocks = layer.in_features // r

        R = layer.reconn.detach().float()
        R_blocks = R.reshape(n_groups, r, n_blocks, r).permute(0, 2, 1, 3).contiguous()
        A_blocks = A.reshape(-1, n_blocks, r)

        # R drifts from its skew-symmetric init: diag=0, offdiag std=4/sqrt(r).
        # ratio diag/offdiag tracks how much self-feedback R has acquired.
        n_diag = n_groups * n_blocks * r
        n_offdiag = n_groups * n_blocks * r * (r - 1)
        r_diag = R_blocks.diagonal(dim1=-2, dim2=-1)
        diag_sq_sum = (r_diag ** 2).sum().item()
        total_sq_sum = (R_blocks ** 2).sum().item()
        offdiag_sq_sum = max(0.0, total_sq_sum - diag_sq_sum)
        r_diag_rms = math.sqrt(diag_sq_sum / n_diag)
        r_offdiag_rms = math.sqrt(offdiag_sq_sum / max(1, n_offdiag))
        monitors[f"r_diag_rms/{tag}"] = r_diag_rms
        monitors[f"r_offdiag_rms/{tag}"] = r_offdiag_rms
        r_drift_ratios.append(r_diag_rms / max(1e-12, r_offdiag_rms))

        AR_all = torch.einsum("mbr,gbsr->gmbs", A_blocks, R_blocks)
        AR_all = AR_all.reshape(n_groups, A.shape[0], layer.in_features)

        if layer._internal_bias is not None:
            ib = layer._internal_bias.detach().float()
            AR_all = AR_all + ib.unsqueeze(1)
            monitors[f"ib_norm/{tag}"] = ib.norm().item()

        std_per_group = AR_all.reshape(n_groups, -1).std(dim=1)
        ar_mean = std_per_group.mean().item()
        monitors[f"ar_std_mean/{tag}"] = ar_mean
        monitors[f"ar_std_min/{tag}"] = std_per_group.min().item()
        monitors[f"ar_std_max/{tag}"] = std_per_group.max().item()
        ar_std_means.append(ar_mean)

        H = A.unsqueeze(0) * F.silu(AR_all)
        dead = (H.abs() < 0.01).float().mean().item()
        monitors[f"dead_gate_frac/{tag}"] = dead
        dead_fracs.append(dead)

        B = layer.weight.detach().float()
        gs = layer.group_size
        out_norms = torch.empty(n_groups, device=A.device)
        for g in range(n_groups):
            Bg = B[g * gs : (g + 1) * gs, :]
            Cg = H[g] @ Bg.T
            out_norms[g] = Cg.norm()
        monitors[f"out_norm_mean/{tag}"] = out_norms.mean().item()
        ratio = (out_norms.max() / out_norms.min().clamp_min(1e-12)).item()
        monitors[f"out_norm_ratio/{tag}"] = ratio
        out_ratios.append(ratio)

    # Cross-layer summary scalars (easier to alert on than the per-layer panels).
    if ar_std_means:
        monitors["prism_summary/ar_std_mean"] = sum(ar_std_means) / len(ar_std_means)
        monitors["prism_summary/ar_std_min_layer"] = min(ar_std_means)
        monitors["prism_summary/ar_std_max_layer"] = max(ar_std_means)
    if dead_fracs:
        monitors["prism_summary/dead_gate_frac_max"] = max(dead_fracs)
    if out_ratios:
        monitors["prism_summary/out_norm_ratio_max"] = max(out_ratios)
    if r_drift_ratios:
        monitors["prism_summary/r_drift_ratio_max"] = max(r_drift_ratios)

    return monitors


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
if wandb_log and master_process:
    import wandb
    # Continuous wandb runs across requeues: persist the run_id in out_dir
    # the first time we wandb.init, then on resume reopen the SAME run.
    # Otherwise every requeue creates a fresh run that re-starts at step 0.
    wandb_run_id = None
    wandb_run_id_path = os.path.join(out_dir, "wandb_run_id.txt")
    if init_from == "resume" and os.path.exists(wandb_run_id_path):
        with open(wandb_run_id_path) as f:
            wandb_run_id = f.read().strip()
        print(f"resuming wandb run {wandb_run_id}")
    init_kwargs = dict(
        project=wandb_project, entity=wandb_entity, name=wandb_run_name,
        config=config,
    )
    if wandb_run_id:
        init_kwargs["id"] = wandb_run_id
        init_kwargs["resume"] = "must"
    wandb.init(**init_kwargs)
    os.makedirs(out_dir, exist_ok=True)
    with open(wandb_run_id_path, "w") as f:
        f.write(wandb.run.id)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
X, Y = get_batch("train")
t0 = time.time()
local_iter_num = 0
running_mfu = -1.0
last_grad_norm = float("nan")       # total clipped norm
last_b_grad_norm = float("nan")     # MLP dense projections (B, down, vanilla c_fc/c_proj)
last_r_grad_norm = float("nan")     # PrismLinear.reconn (R) only
last_other_grad_norm = float("nan") # attn + embeddings + norms
last_grad_norm_diff = 0.0           # current grad_norm minus the previous step's
# Relative per-step update ratios ||Δθ|| / ||θ|| for R vs the dense up-projection
# B it gates. Tests whether R "leads" B early in training (see thesis §5 dynamics
# hypothesis). Sampled on logged steps to keep overhead negligible.
last_R_update_ratio = float("nan")
last_B_update_ratio = float("nan")

def _classify_grad_param(name: str) -> str:
    """Bucket a parameter name into one of: 'R', 'B_mlp', 'other'."""
    if "reconn" in name or "ar_scale" in name:
        return "R"
    if "mlp" in name:
        return "B_mlp"
    return "other"

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        # Preserve r_lr_mult and ib_lr_mult ratios across LR schedule.
        if pg.get("name") in ("r_decay", "ib_decay"):
            pg["lr"] = lr * r_lr_mult
        else:
            pg["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}")
        if wandb_log:
            t_eval = time.time()
            metrics = compute_metrics(raw_model, iter_num, t_eval - t0, last_grad_norm)
            log_dict = {
                "iter": iter_num,
                "train/loss": losses["train"],
                "val/loss": losses["val"],
                "lr": lr,
                "grad_norm": metrics["grad_norm"],
                "grad_norm/B_mlp": last_b_grad_norm,
                "grad_norm/R": last_r_grad_norm,
                "grad_norm/other": last_other_grad_norm,
                "grad_norm/diff": last_grad_norm_diff,
                "grad_norm/R_over_B": (last_r_grad_norm / last_b_grad_norm
                                       if last_b_grad_norm > 0 else 0.0),
                "param_norm": metrics["param_norm"],
                "throughput": metrics.get("throughput", 0),
                # Relative per-step update ratios (sampled on logged steps).
                "update_ratio/R": last_R_update_ratio,
                "update_ratio/B_up": last_B_update_ratio,
                "update_ratio/R_over_B": (last_R_update_ratio / last_B_update_ratio
                                          if last_B_update_ratio > 0 else 0.0),
            }
            if mlp_type == "prism":
                sample_X, _ = get_batch("val")
                log_dict.update(compute_prism_monitors(raw_model, sample_X, device))
            wandb.log(log_dict, step=iter_num)
            t0 = t_eval
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, float(losses["val"]))
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")  # prefetch overlaps with bwd
        loss.backward()

    # Early-stop on non-finite loss. Cheap LR / shape sweeps shouldn't waste
    # GPU hours running for full max_iters once training has clearly diverged.
    if not torch.isfinite(loss):
        if master_process:
            print(f"training stopped: non-finite loss "
                  f"(value={loss.item()}, iter={iter_num}, lr={lr:.2e})")
        break

    # Per-group grad-norm decomposition — accumulate squared norms on-GPU,
    # then a single sync at the end. Done BEFORE clip_grad_norm so the
    # numbers reflect the raw (pre-clip) gradient magnitudes per group.
    b_sq = torch.zeros((), device=device)
    r_sq = torch.zeros((), device=device)
    o_sq = torch.zeros((), device=device)
    for n, p in raw_model.named_parameters():
        if p.grad is None:
            continue
        bucket = _classify_grad_param(n)
        contrib = (p.grad.detach() ** 2).sum()
        if bucket == "R":
            r_sq = r_sq + contrib
        elif bucket == "B_mlp":
            b_sq = b_sq + contrib
        else:
            o_sq = o_sq + contrib
    last_b_grad_norm = b_sq.sqrt().item()
    last_r_grad_norm = r_sq.sqrt().item()
    last_other_grad_norm = o_sq.sqrt().item()

    prev_total = last_grad_norm
    if grad_clip != 0.0:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        last_grad_norm = grad_norm_tensor.item()
    if not math.isnan(prev_total):
        last_grad_norm_diff = last_grad_norm - prev_total

    # Sample the R-vs-B relative update ratio on logged steps. Snapshot the
    # reconn (R) and prism up-projection (B) tensors before the step, measure
    # ||Δθ|| / ||θ|| after. Only on logged steps -> one clone per ~log_interval
    # steps, negligible overhead.
    _track_update_ratio = (iter_num % log_interval == 0 and master_process
                           and mlp_type == "prism")
    if _track_update_ratio:
        _snap = {n: p.detach().clone()
                 for n, p in raw_model.named_parameters()
                 if ("reconn" in n) or n.endswith("mlp.up.weight")}

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if _track_update_ratio:
        dR2 = R2 = dB2 = B2 = 0.0
        for n, p in raw_model.named_parameters():
            if n not in _snap:
                continue
            old = _snap[n]
            d2 = ((p.detach() - old) ** 2).sum().item()
            n2 = (old ** 2).sum().item()
            if "reconn" in n:
                dR2 += d2; R2 += n2
            else:                       # mlp.up.weight = the dense B that R gates
                dB2 += d2; B2 += n2
        last_R_update_ratio = (dR2 ** 0.5) / (R2 ** 0.5 + 1e-12)
        last_B_update_ratio = (dB2 ** 0.5) / (B2 ** 0.5 + 1e-12)
        del _snap

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        ur = ""
        if mlp_type == "prism" and not math.isnan(last_R_update_ratio):
            ur = (f", upd R/B {last_R_update_ratio:.2e}/{last_B_update_ratio:.2e}"
                  f" (R/B={last_R_update_ratio / last_B_update_ratio:.2f})"
                  if last_B_update_ratio > 0 else "")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
              f"lr {lr:.2e}{ur}")
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
