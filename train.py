import argparse

from einops import rearrange
import torch
import numpy as np

from gpt.data import get_batch
from gpt.model import TransformerLM
from gpt.optim import AdamW, cross_entropy, gradient_clip
from gpt.tokenizer.bpe import Tokenizer

# Tokenizer
vocab_path = "checkpoints/tokenizer/vocab.txt"
merges_path = "checkpoints/tokenizer/merges.txt"

tokenizer = Tokenizer.from_files(vocab_path, merges_path)

# Data
train_path = "data/train.npy"

# Settings
device = "cpu"


def train_gpt():
    # Setup
    model = TransformerLM(
        vocab_size=1024,
        context_length=256,
        d_model=384,
        num_layers=6,
        num_heads=6,
        d_ff=1536,
        theta=1e4,
        device=device,
        dtype=torch.float32,
    )

    optim = AdamW(model.parameters())

    for t in range(5):
        model.train()
        optim.zero_grad()

        # Get data
        train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        inputs, targets = get_batch(
            train_data, batch_size=8, context_length=256, device=device
        )  # (batch_size, seq_len)

        inputs = inputs.to(device=device, dtype=torch.long)
        targets = targets.to(device=device, dtype=torch.long)

        # Run forward pass
        logits = model.forward(inputs)  # (batch_size, seq_len, vocab_size)

        # Loss
        loss = cross_entropy(
            rearrange(logits, "b s v -> (b s) v"), rearrange(targets, "b s -> (b s)")
        )
        loss.backward()

        # Optimize
        gradient_clip(model.parameters(), max_l2_norm=1.0)
        optim.step()

        # Logs
        print(f"Loss: {loss.item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)

    train_gpt()
