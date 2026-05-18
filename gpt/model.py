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
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-sigma, b=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


"""
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct anembedding module. This function should accept the following parameters:num_embeddings: int Size of the vocabularyembedding_dim: int Dimension of the embedding vectors, i.e., d_modeldevice: torch.device | None = None Device to store the parameters ondtype: torch.dtype | None = None Data type of the parametersdef forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectorsfor the given token IDs.
"""


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0, std=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch_size, sequence_length)
        # output: (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]
