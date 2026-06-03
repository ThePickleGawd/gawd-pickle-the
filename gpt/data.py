import os
from typing import IO, BinaryIO

import numpy.typing as npt
import torch


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    starts = torch.randint(0, len(x) - context_length, (batch_size,), device=device)
    offsets = torch.arange(0, context_length, device=device)

    indices = starts[:, None] + offsets[None, :]
    inputs = torch.tensor(x[indices], device=device)
    targets = torch.tensor(x[indices + 1], device=device)

    return (inputs, targets)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]
