import numpy.typing as npt
import torch


def get_batch(x: npt.NDArray, batch_size: int, context_length: int, device: str):
    starts = torch.randint(0, len(x) - context_length, (batch_size,), device=device)
    offsets = torch.arange(0, context_length, device=device)

    indices = starts[:, None] + offsets[None, :]
    inputs = torch.tensor(x[indices], device=device)
    targets = torch.tensor(x[indices + 1], device=device)

    return (inputs, targets)
