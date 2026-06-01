"""TinyStories config — fast iteration with real BPE.

Smaller-than-GPT-2 model so single-GPU runs finish in a couple of hours.
"""

out_dir = "out-tinystories-prism"
eval_interval = 1000
eval_iters = 100
log_interval = 50

wandb_log = True
wandb_project = "nanogpt"
wandb_run_name = "nanogpt-prism-tinystories"

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
