"""TinyStories — prism-3× with ADDITIVE composition (Chapter 8 follow-up).

Replaces the multiplicative gate `H = preact * SiLU(AR)` with the additive
combination `H = preact + SiLU(AR)`. Block-diagonal R structure is
preserved; only the composition with preact changes.

Motivation (Chapter 8):
  * The standard gated PrismMLP degenerates to a bilinear interaction
    (H ≈ preact * AR) because SiLU saturates as R grows during training.
  * 7+ parameter-level interventions failed to escape this regime.
  * Additive composition removes the multiplicative coupling: SGD has no
    path to drive the operation into a "linear tail" through R growth.

If this also lands at val_loss ≈ 1.31, the structural conclusion is
robust: it's not the multiplicative gate per se, it's PrismLinear's
expressivity as an MLP from scratch.

Parameter count is similar to gated PrismMLP-3× (down: 4d², up_B: 4d²,
R: (r·n_groups/d)·d² = 0.5d²; total per layer ~6.5d² vs 6.4d² gated).
"""

out_dir = "out-tinystories-prism-3x-additive"
eval_interval = 1000
eval_iters = 100
log_interval = 50

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-3x-additive-tinystories"

dataset = "tinystories"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

mlp_type = "prism_additive"   # <-- the headline change
mlp_expansion = 3
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 0.5

learning_rate = 1.5e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1.5e-4
warmup_iters = 1000

dtype = "bfloat16"
compile = False
