"""TinyStories → GPT-2 BPE tokens.

Smaller than FineWeb (~500M tokens) and grammatically simpler — useful for
fast end-to-end debugging at small scale where shakespeare_char is too
limiting (no real BPE to test).

Run once::

    python data/tinystories/prepare.py
"""

import os
from tqdm import tqdm

import numpy as np
import tiktoken
from datasets import load_dataset

NUM_PROC = 8
NUM_PROC_LOAD = 8

enc = tiktoken.get_encoding("gpt2")
here = os.path.dirname(__file__)


def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
    return {"ids": ids, "len": len(ids)}


if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories", num_proc=NUM_PROC_LOAD)

    tokenized = {}
    for split in ("train", "validation"):
        out_split = "train" if split == "train" else "val"
        tokenized[out_split] = dataset[split].map(
            process,
            remove_columns=["text"],
            desc=f"tokenizing {split}",
            num_proc=NUM_PROC,
        )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(here, f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = max(1, min(1024, len(dset) // 1024))
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx,
                               contiguous=True).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
