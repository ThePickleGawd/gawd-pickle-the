from gpt.tokenizer.bpe import Tokenizer, train_bpe

vocab_path = "checkpoints/tokenizer/vocab.txt"
merges_path = "checkpoints/tokenizer/merges.txt"

input_path = "tests/fixtures/tinystories_sample_5M.txt"

special_tokens = ["<|endoftext|>"]

vocab, merges = train_bpe(input_path, 1024, special_tokens)
tokenizer = Tokenizer(vocab, merges, special_tokens)

print(f"Exporting to:\n vocab: {vocab_path}\n merges: {merges_path}")
tokenizer.export(vocab_path, merges_path)
