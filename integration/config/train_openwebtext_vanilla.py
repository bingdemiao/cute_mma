"""Vanilla GELU MLP baseline for OpenWebText (GPT-2 small, canonical nanoGPT)."""

out_dir = "out-openwebtext-vanilla"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-vanilla-openwebtext"

dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "vanilla"

learning_rate = 6e-4
max_iters = 600000
lr_decay_iters = 600000
min_lr = 6e-5
warmup_iters = 2000

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dtype = "bfloat16"
compile = False
