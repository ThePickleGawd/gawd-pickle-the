import time

import numpy as np
from tqdm import tqdm

from gpt.tokenizer.bpe import Tokenizer

vocab_path = "checkpoints/tokenizer/owt/vocab.txt"
merges_path = "checkpoints/tokenizer/owt/merges.txt"
special_tokens = ["<|endoftext|>"]

input_data_path = "data/owt_train.txt"
input_val_path = "data/owt_valid.txt"
train_out_path = "data/owt_train.npy"
val_out_path = "data/owt_val.npy"

t0 = time.time()

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# Encode data
with open(input_data_path, "r") as f:
    train_ids = list(tqdm(tokenizer.encode_iterable(f), desc="train", unit="tok"))

with open(input_val_path, "r") as f:
    val_ids = list(tqdm(tokenizer.encode_iterable(f), desc="val", unit="tok"))

print(
    f"Train has {len(train_ids)} tokens",
)
print(f"Val has {len(val_ids)} tokens")

# Export
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(train_out_path)
val_ids.tofile(val_out_path)

print(f"Finished in {time.time() - t0}s")
