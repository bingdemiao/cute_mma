"""GPT-2 small (124M) on FineWeb-Edu sample-10BT.

Default config sized for 4x GH200 with bf16. Adjust
``gradient_accumulation_steps`` to keep tokens-per-iter near the
nanoGPT reference (~500K).
"""

out_dir = "out-fineweb-prism"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "nanogpt-prism-fineweb"

dataset = "fineweb"
# 5 * 8 = 40 micro-batches. Per-rank micro-batch = 12, block_size = 1024.
# Effective batch tokens = 12 * 1024 * 40 = 491,520 ≈ nanoGPT's 0.5M.
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "prism"
mlp_expansion = 4               # hidden width = mlp_expansion * n_embd
group_size = 64                 # 4d=3072 → 48 groups
reconn_sz = 8                   # d=768   → 96 blocks per group
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

learning_rate = 6e-4
max_iters = 100000              # ~50B tokens at the default batch
lr_decay_iters = 100000
min_lr = 6e-5
warmup_iters = 2000

weight_decay = 0.1
r_wd_factor = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dtype = "bfloat16"
compile = False
