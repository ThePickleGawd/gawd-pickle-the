import numpy as np

from gpt.tokenizer.bpe import Tokenizer

vocab_path = "checkpoints/tokenizer/vocab.txt"
merges_path = "checkpoints/tokenizer/merges.txt"
special_tokens = ["<|endoftext|>"]

input_path = "tests/fixtures/tinystories_sample_5M.txt"
train_out_path = "data/train.npy"
val_out_path = "data/val.npy"

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# Load data
with open(input_path, "r") as f:
    data = f.read()

n = len(data)

train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Encode data
train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)

print(
    f"Train has {len(train_ids)} tokens",
)
print(f"Val has {len(val_ids)} tokens")

# Export
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(train_out_path)
val_ids.tofile(val_out_path)
