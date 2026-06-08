import torch

from gpt.model import TransformerLM
from gpt.tokenizer.bpe import Tokenizer

vocab_path = "checkpoints/tokenizer/tinystories/vocab.txt"
merges_path = "checkpoints/tokenizer/tinystories/merges.txt"
model_path = "checkpoints/model/9000.pth"
special_tokens = ["<|endoftext|>"]

prompt = "Once upon a time"
max_new_tokens = 500
temperature = 1.0
top_p = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
eos_token_id = tokenizer.byte2int[special_tokens[0].encode("utf-8")]

model = TransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=16,
    d_ff=1344,
    theta=1e4,
    device=device,
)

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint["model"]
state_dict = {
    key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
}
model.load_state_dict(state_dict)
model.eval()

token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    output_ids = model.generate(
        token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )

output = tokenizer.decode(output_ids[0].tolist())
print(output.split(special_tokens[0], 1)[0])
