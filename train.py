import argparse
import csv
from datetime import datetime
from pathlib import Path

from einops import rearrange
import torch
import numpy as np

from gpt.data import get_batch, save_checkpoint
from gpt.model import TransformerLM
from gpt.optim import AdamW, cross_entropy, gradient_clip
from gpt.tokenizer.bpe import Tokenizer

# Tokenizer
vocab_path = "checkpoints/tokenizer/vocab.txt"
merges_path = "checkpoints/tokenizer/merges.txt"

tokenizer = Tokenizer.from_files(vocab_path, merges_path)

# Data
train_path = "data/tinystories_train.npy"

# Settings
device = "cuda"
log_dir = "logs"
log_every = 15


def train_gpt():
    # Setup
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        theta=1e4,
        device=device,
        dtype=torch.float32,
    )

    optim = AdamW(model.parameters())

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    print(f"Logging to: {log_path}")

    with open(log_path, "w", newline="") as log_f:
        logger = csv.DictWriter(log_f, fieldnames=["step", "loss"])
        logger.writeheader()

        for t in range(10000):
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
            if t % log_every == 0:
                loss_value = loss.item()
                logger.writerow({"step": t, "loss": loss_value})
                log_f.flush()

                print(f"Loss: {loss_value}")
                sample_out = logits[0].argmax(dim=-1).detach().cpu().tolist()
                print(tokenizer.decode(sample_out))

            if t % 1000 == 0:
                print("=== Saving checkpoint ===")
                save_checkpoint(model, optim, t, f"checkpoints/model/{t}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)

    train_gpt()
