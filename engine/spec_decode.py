"""Speculative decoding: batched draft proposal and lossless verification.

Per sequence and step, the draft model proposes γ tokens (γ may differ per sequence; γ = 0 means
plain decoding), the target scores all γ+1 positions in one forward pass, and standard rejection
sampling (Leviathan et al. 2023; Chen et al. 2023) accepts a prefix. The output distribution is
exactly the target model's, so speculation itself cannot cost quality; with temperature 0 this
reduces to "accept while the target's argmax equals the draft token".
"""
import torch


def sample(logits: torch.Tensor, temperature: float, generator=None):
    """logits [N, V] -> (tokens [N], probs [N, V] or None for greedy)."""
    if temperature <= 0:
        return logits.argmax(-1), None
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1), probs


def verify(target_logits: torch.Tensor, draft_tokens: list[int], draft_probs, temperature: float,
           generator=None) -> list[int]:
    """Rejection sampling for one sequence.

    target_logits: [γ+1, V] target logits at the pending token and each draft token
    draft_tokens:  γ proposed tokens
    draft_probs:   [γ, V] draft distributions (None when greedy)
    Returns the accepted draft prefix plus one token from the target (correction or bonus).
    """
    g = len(draft_tokens)
    if temperature <= 0:
        tgt = target_logits.argmax(-1).tolist()
        k = 0
        while k < g and tgt[k] == draft_tokens[k]:
            k += 1
        return draft_tokens[:k] + [tgt[k]]

    p = torch.softmax(target_logits / temperature, dim=-1)
    d = torch.tensor(draft_tokens, device=p.device)
    idx = torch.arange(g, device=p.device)
    ratio = (p[idx, d] / draft_probs[idx, d].clamp_min(1e-20)).clamp(max=1.0)
    u = torch.rand(g, device=p.device, generator=generator)
    rejected = (u >= ratio).nonzero()
    if len(rejected) == 0:
        bonus = torch.multinomial(p[g], 1, generator=generator).item()
        return draft_tokens + [bonus]
    k = rejected[0].item()
    residual = (p[k] - draft_probs[k]).clamp_min(0)
    residual = residual / residual.sum() if residual.sum() > 0 else p[k]
    return draft_tokens[:k] + [torch.multinomial(residual, 1, generator=generator).item()]
