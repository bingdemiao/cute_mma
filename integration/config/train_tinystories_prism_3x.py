"""TinyStories — prism with 3x MLP expansion (param-matched ablation, Step 4).

At gs=64, r=8, prism MLP params = (2*x + x*r/gs) * d^2 = 2.125*x * d^2.
Vanilla MLP params at 4x = 8 * d^2.
So prism at x=3 has 6.375*d^2 per MLP — about 20% fewer MLP params than vanilla 4x.

Total params (TinyStories, d=512, 8 layers):
  vanilla 4x ≈ 51.2M
  prism 3x   ≈ 47.8M  (≈ -6.6% total, -20% MLP)

Comparison target: prism 3x val_loss vs vanilla 4x val_loss at iter 50k.
If prism 3x ties or beats vanilla 4x → "fewer params, same/better performance".
"""

out_dir = "out-tinystories-prism-3x"
eval_interval = 1000
eval_iters = 100
log_interval = 50

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-3x-tinystories"

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
mlp_expansion = 3
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
