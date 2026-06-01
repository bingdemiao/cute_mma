"""GPT-2 small (124M) on OpenWebText.

Same hyperparameters as the canonical nanoGPT recipe, with the prism MLP
swapped in. Provided as the head-to-head baseline against published
nanoGPT numbers.
"""

out_dir = "out-openwebtext-prism"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-openwebtext"

dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

mlp_type = "prism"
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0

learning_rate = 6e-4
max_iters = 600000
lr_decay_iters = 600000
min_lr = 6e-5
warmup_iters = 2000

weight_decay = 0.1
r_wd_factor = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

dtype = "bfloat16"
compile = False
