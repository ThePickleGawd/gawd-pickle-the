from collections.abc import Iterable
import math

import torch
from einops import reduce


def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    """
    logits: (batch_size, vocab_size)
    target: (batch_size,)
    """

    # Get shift logits for numerical stability
    logits_max, _ = logits.max(dim=-1, keepdim=True)
    logits = logits - logits_max

    # Compute negative log likelihood (try to cancel log where possible)
    target_logits = torch.gather(logits, dim=-1, index=target.unsqueeze(-1))
    logsumexp_logits = torch.log(reduce(logits.exp(), "b vocab_size -> b 1", "sum"))

    nll = logsumexp_logits - target_logits

    # Compute loss
    loss = reduce(nll, "... -> 1", "mean")

    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr, weight_decay, betas, eps = (
                group["lr"],
                group["weight_decay"],
                group["betas"],
                group["eps"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Get iteration (t) and moments (m, v)
                if not self.state[p]:
                    self.state[p] = {
                        "t": 0,
                        "m": torch.zeros_like(p),
                        "v": torch.zeros_like(p),
                    }

                t = self.state[p]["t"]
                m = self.state[p]["m"]
                v = self.state[p]["v"]

                # 1. Compute gradients for loss
                grad = p.grad.data
                # 2. Compute adjusted alpha for this iteration
                beta1, beta2 = betas[0], betas[1]
                lr_t = lr * math.sqrt(1 - (beta2 ** (t + 1))) / (1 - (beta1 ** (t + 1)))
                # 3. Weight decay
                p.data -= lr * weight_decay * p.data
                # 4. Update momentum
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                # 5. Apply momentum
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                self.state[p]["t"] = t + 1
                self.state[p]["m"] = m
                self.state[p]["v"] = v

        return loss


"""
(Warm-up) If 𝑡 < 𝑇𝑤, then 𝛼𝑡 = 𝑡𝑇𝑤𝛼max.(Cosine annealing) If 𝑇𝑤 ≤ 𝑡 ≤ 𝑇𝑐, then 𝛼𝑡 = 𝛼min + 12 (1 + cos( 𝑡−𝑇𝑤𝑇𝑐−𝑇𝑤𝜋))(𝛼max − 𝛼min).(Post-annealing) If 𝑡 > 𝑇𝑐, then 𝛼𝑡 = 𝛼min.Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup(1 point)8It’s sometimes common to use a schedule where the learning rate rises back up (restarts) to help get past local minima.33Write a function that takes 𝑡, 𝛼max, 𝛼min, 𝑇𝑤 and 𝑇𝑐, and returns the learning rate 𝛼𝑡 according to
"""


def lr_cosine_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        return t / T_w * a_max
    elif t < T_c:
        return a_min + 0.5 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
            a_max - a_min
        )
    else:
        return a_min


def gradient_clip(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 10e-6
):
    """
    Check if the L2 norm over all parameters exceeds max_l2_norm.
    If so, apply normalization factor to all params
    """
    squared_norm = 0
    param_to_update = []
    for param in parameters:
        if param.grad is None:
            continue

        squared_norm += torch.sum(param.grad**2)
        param_to_update.append(param)

    norm = math.sqrt(squared_norm)

    if norm > max_l2_norm:
        for param in param_to_update:
            param.grad = param.grad * max_l2_norm / (norm + eps)
