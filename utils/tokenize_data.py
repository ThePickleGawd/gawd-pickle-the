import numpy as np

from gpt.tokenizer.bpe import Tokenizer

vocab_path = "checkpoints/tokenizer/vocab.txt"
merges_path = "checkpoints/tokenizer/merges.txt"
special_tokens = ["<|endoftext|>"]

input_data_path = "data/TinyStoriesV2-GPT4-train.txt"
input_val_path = "data/TinyStoriesV2-GPT4-valid.txt"
train_out_path = "data/train.npy"
val_out_path = "data/val.npy"

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

# Load data
with open(input_data_path, "r") as f:
    train_data = f.read()

with open(input_val_path, "r") as f:
    val_data = f.read()

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
