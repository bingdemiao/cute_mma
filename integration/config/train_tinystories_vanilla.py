"""Vanilla GELU MLP baseline for TinyStories."""

out_dir = "out-tinystories-vanilla"
eval_interval = 1000
eval_iters = 100
log_interval = 50

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-vanilla-tinystories"

dataset = "tinystories"
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

mlp_type = "vanilla"

learning_rate = 6e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 6e-5
warmup_iters = 1000

dtype = "bfloat16"
compile = False
