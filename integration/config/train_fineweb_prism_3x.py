"""GPT-2 small (124M) with prism-3× MLPs on FineWeb-Edu sample-10BT.

Step-6 param-matched ablation at GPT-2 scale. At d=768, gs=64, r=8:
  vanilla 4× MLP : 56.62 M params
  prism   3× MLP : 45.30 M params   (-20.0% MLP, -9.1% total)

If the TinyStories LR sweep (Step 5) shows prism-3× tying vanilla-4× at
fixed recipe, this run confirms the result survives the jump to a
real-language dataset and a 2.5x larger model.

Wallclock: ~17-20 hours on 4x GH200 (bf16, 100k iters at ~700 ms/iter).
"""

out_dir = "out-fineweb-prism-3x"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "nanogpt-prism-3x-fineweb"

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
mlp_expansion = 3               # hidden width = 3 * 768 = 2304
group_size = 64                 # 2304 / 64 = 36 groups
reconn_sz = 8                   # 768 / 8 = 96 blocks per group
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

learning_rate = 6e-4
max_iters = 100000
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
