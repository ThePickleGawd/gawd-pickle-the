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
        """
        token_ids: (batch_size, seq_len)
        output: (batch_size, seq_len, embedding_dim)
        """

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """

        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Apply RMS(a)
        x2 = torch.square(x)
        x2_sum = reduce(x2, "b s d -> b s 1", "sum")
        rms = torch.sqrt(1 / self.d_model * x2_sum + self.eps)  # (batch_size, seq_len)

        # Return RMSNorm(a)
        result = x / rms * self.gain
        return result.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.w1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        output: (batch_size, seq_len, d_model)
        """

        # SwiGLU = W_2 * (SiLU(W_1 * x) * W_3 * x))
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()

        assert d_k % 2 == 0, "RoPE d_k must be even"
        self.d_k = d_k

        tok_pos = torch.arange(max_seq_len, dtype=torch.float32)
        pair_pos = torch.arange(d_k // 2, dtype=torch.float32)

        # Allows broadcasting to assign every (i,j) pair to rope_angles
        tok_pos = rearrange(tok_pos, "i -> i 1")
        pair_pos = rearrange(pair_pos, "j -> 1 j")

        rope_angles = tok_pos / (theta ** (2 * pair_pos / d_k))
        rope_cos = torch.cos(rope_angles)  # (max_seq_len, d_k//2)
        rope_sin = torch.sin(rope_angles)  # (max_seq_len, d_k//2)

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor, tok_pos: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_k)
        tok_pos: (batch_size, seq_len)
        output: (batch_size, seq_len, d_k)
        """

        # Split x into even and odd pairs
        # (..., seq_len, d_k / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply the precomputed rotation 2x2 matrix
        cos = self.rope_cos[tok_pos]
        sin = self.rope_sin[tok_pos]
        new_even = cos * x_even - sin * x_odd
        new_odd = sin * x_even + cos * x_odd

        out = torch.empty_like(x)
        out[..., 0::2] = new_even
        out[..., 1::2] = new_odd

        return out


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """x: (batch_size, seq_len, d_k)"""

    x_max, _ = torch.max(x, dim=dim, keepdim=True)  # (batch_size, seq_len, 1)
    x -= x_max  # (batch_size, seq_len, d_k)

    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
):
    """
    Q: (batch_size, ..., seq_len, d_k)
    K: (batch_size, ..., seq_len*, d_k)
    V: (batch_size, ..., seq_len*, d_v)
    mask: (batch_size, seq_len, seq_len*)
    """

    d_k = Q.size(-1)
    K_t = rearrange(K, "... m d_k -> ... d_k m")
    scores = (Q @ K_t) / math.sqrt(d_k)  # (..., seq_len, seq_len*)

    if mask is not None:
        assert mask.dtype == torch.bool, "Only boolean masks are supported right now"

        attn_bias = torch.zeros_like(mask, dtype=Q.dtype)
        attn_bias.masked_fill_(~mask, float("-inf"))
        scores += attn_bias

    attn = softmax(scores, dim=-1)
    return attn @ V
