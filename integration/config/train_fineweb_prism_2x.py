"""GPT-2 small (124M) with prism-2× MLPs on FineWeb-Edu sample-10BT.

Aggressive param-reduction at GPT-2 scale. At d=768, gs=64, r=8:
  vanilla 4× MLP : 56.62 M params
  prism   2× MLP : 30.20 M params   (-46.7% MLP, -21.3% total)

Companion to train_fineweb_prism_3x.py for the Pareto curve.
"""

out_dir = "out-fineweb-prism-2x"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "nanogpt-prism-2x-fineweb"

dataset = "fineweb"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "prism"
mlp_expansion = 2               # hidden width = 2 * 768 = 1536
group_size = 64                 # 1536 / 64 = 24 groups
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
