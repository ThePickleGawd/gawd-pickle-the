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
