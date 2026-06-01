"""Char-level Shakespeare. Tiny smoke test (~1M tokens).

Downloads tinyshakespeare, builds a 65-symbol vocab, writes
``train.bin``/``val.bin`` (uint16) plus a ``meta.pkl`` so train.py can
recover ``vocab_size`` and the encode/decode tables.
"""

import os
import pickle
import urllib.request

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

here = os.path.dirname(__file__)
input_path = os.path.join(here, "input.txt")
if not os.path.exists(input_path):
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, input_path)

with open(input_path, "r") as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)!r}")
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

train_ids = np.array([stoi[c] for c in train_data], dtype=np.uint16)
val_ids = np.array([stoi[c] for c in val_data], dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has   {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(here, "train.bin"))
val_ids.tofile(os.path.join(here, "val.bin"))

meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
with open(os.path.join(here, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print("wrote train.bin, val.bin, meta.pkl")
