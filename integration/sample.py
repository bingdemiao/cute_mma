"""Sample from a trained nanoGPT-Prism checkpoint."""

import os
import pickle
from contextlib import nullcontext

import torch
import tiktoken

from model import GPT, GPTConfig

# defaults — override via configurator.py
init_from = "resume"
out_dir = "out"
start = "\n"
num_samples = 5
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 1337
device = "cuda"
dtype = "bfloat16"
compile = False

exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
for prefix in ("_orig_mod.", "module."):
    bad = [k for k in state_dict if k.startswith(prefix)]
    for k in bad:
        state_dict[k[len(prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
if ptdtype != torch.float32:
    model = model.to(ptdtype)
model.to(device)
if compile:
    model = torch.compile(model)

# Try to load a custom char-level tokenizer from data_dir's meta.pkl;
# fall back to GPT-2 BPE.
meta_path = os.path.join("data", checkpoint["config"].get("dataset", ""), "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad(), ctx:
    for _ in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print("---------------")
