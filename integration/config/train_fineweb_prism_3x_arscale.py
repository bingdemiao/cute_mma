"""GPT-2 small (124M) with prism-3× + learnable ar_scale, on FineWeb-Edu.

Same as train_fineweb_prism_3x.py BUT adds `prism_ar_scale_init = 0.1`,
the LoRA-α-style scalar that multiplies R before the SiLU gate.

Motivation (from TinyStories diagnostic work):
  * At fixed LR=1.5e-3, r_init_scale=0.5, prism-3× converges to a state
    where R.std grows 4× from init (0.177 → 0.71), driving AR.std to ~16.
  * SiLU saturates: only ~7% of gates in the informative [0.1, 0.9] band,
    gate values average ~2.5 (linear regime of SiLU).
  * r_lr_mult sweeps in {0.1, 0.25, 0.5, 1.0} ALL converge to the same
    final R.std (within 0.5%), so constraining R's learning rate doesn't
    escape the saturated attractor.
  * ar_scale decouples R magnitude from gate input magnitude: it's a
    learnable scalar applied to AR before SiLU, initialized at 0.1.

Comparison plan (all at default LR=6e-4, matching the other FineWeb
runs):
  * train_fineweb_vanilla.py            → 2.9653 (done, baseline)
  * train_fineweb_prism_3x.py           → pending (no ar_scale)
  * train_fineweb_prism_3x_arscale.py   → this config
"""

out_dir = "out-fineweb-prism-3x-arscale"
eval_interval = 2000
eval_iters = 200
log_interval = 10

wandb_log = True
wandb_project = "nanogpt_fineweb"
wandb_run_name = "nanogpt-prism-3x-arscale-fineweb"

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
mlp_expansion = 3
group_size = 64
reconn_sz = 8
prism_backend = "cublas"
prism_activation = "silu_gate"
prism_internal_bias = False
prism_internal_dropout = 0.0
r_init_scale = 1.0
prism_ar_scale_init = 0.1            # the headline change

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
