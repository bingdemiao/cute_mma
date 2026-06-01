"""TinyStories — prism with 2x MLP expansion (aggressive param-reduction ablation).

At gs=64, r=8: prism MLP params = 2.125*x * d^2.
x=2 → 4.25*d^2 per MLP — about 47% fewer MLP params than vanilla 4x.

Total params (TinyStories, d=512, 8 layers):
  vanilla 4x ≈ 51.2M
  prism 2x   ≈ 43.3M  (≈ -15% total, -47% MLP)

Likely loses some perplexity vs vanilla 4x, but quantifies the cost of the most
aggressive shrink. The full Pareto front needs this datapoint.
"""

out_dir = "out-tinystories-prism-2x"
eval_interval = 1000
eval_iters = 100
log_interval = 50

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-2x-tinystories"

dataset = "tinystories"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

mlp_type = "prism"
mlp_expansion = 2
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

learning_rate = 6e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 6e-5
warmup_iters = 1000

dtype = "bfloat16"
compile = False
