"""Tiny char-level Shakespeare config — smoke test for the whole pipeline.

Trains in a few minutes on a single GH200. Designed to confirm:
  * PrismLinear forward + backward both fire under ``cublas`` + bf16
  * GroupNorm-after-Prism produces stable activations
  * Loss decreases (target: ~1.4 train, ~1.5 val by iter 5000)

Numbers chosen so that ``4*n_embd / group_size`` and ``n_embd / reconn_sz``
divide cleanly: d=384, 4d=1536, group_size=64 → 24 groups; reconn_sz=8.
"""

out_dir = "out-shakespeare-char-prism"
eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-shakespeare-char"

dataset = "shakespeare_char"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

mlp_type = "prism"
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

dtype = "bfloat16"
compile = False
