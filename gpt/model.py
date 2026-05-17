import torch
import torch.nn as nn
import math
from einops import rearrange, reduce


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # Initialize weights, following assignment hyperparams
        self.W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        sigma_w = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma_w**2, a=-sigma_w, b=sigma_w)

    def forward(self, x: torch.Tensor):
        return x @ self.W.T


"""
def __init__(self, in_features, out_features, device=None, dtype=None) Construct a lineartransformation module. This function should accept the following parameters:in_features: int final dimension of the inputout_features: int final dimension of the outputdevice: torch.device | None = None Device to store the parameters ondtype: torch.dtype | None = None Data type of the parametersdef forward(self, x: torch.Tensor) -> torch.Tensor Apply the linear transformation to theinput.
"""
