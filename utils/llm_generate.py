import json

import torch

from gpt.model import TransformerLM
from gpt.tokenizer.bpe import Tokenizer

vocab_path = "checkpoints/tokenizer/tinystories/vocab.txt"
merges_path = "checkpoints/tokenizer/tinystories/merges.txt"
model_path = "tests/fixtures/ts_tests/model.pt"
model_config_path = "tests/fixtures/ts_tests/model_config.json"
special_tokens = ["<|endoftext|>"]

prompt = "Once upon a time"
max_new_tokens = 100
temperature = 1.0
top_p = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

with open(model_config_path) as f:
    config = json.load(f)

model = TransformerLM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    d_model=config["d_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    d_ff=config["d_ff"],
    theta=config["rope_theta"],
    device=device,
)

state_dict = torch.load(model_path, map_location=device)
state_dict = {
    key.removeprefix("_orig_mod."): value
    for key, value in state_dict.items()
}
model.load_state_dict(state_dict)
model.eval()

token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    output_ids = model.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

print(tokenizer.decode(output_ids[0].tolist()))
