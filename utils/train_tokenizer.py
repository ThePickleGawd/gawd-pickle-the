from pathlib import Path
import time

from gpt.tokenizer.bpe import Tokenizer, train_bpe

vocab_path = "checkpoints/tokenizer/tinystories/vocab.txt"
merges_path = "checkpoints/tokenizer/tinystories/merges.txt"

input_path = "data/TinyStoriesV2-GPT4-train.txt"

special_tokens = ["<|endoftext|>"]

t0 = time.time()

vocab, merges = train_bpe(input_path, 10000, special_tokens, num_processes=32)
tokenizer = Tokenizer(vocab, merges, special_tokens)

print(f"Exporting to:\n vocab: {vocab_path}\n merges: {merges_path}")
Path(vocab_path).parent.mkdir(parents=True, exist_ok=True)
tokenizer.export(vocab_path, merges_path)

print(f"Finished in {time.time() - t0}s")
