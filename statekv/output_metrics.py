"""Exact full-vocabulary output metrics used by active closed-loop runs."""
from __future__ import annotations

from typing import Dict

import torch


def exact_distribution_metrics(
    full_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    target_token: int,
) -> Dict[str, float]:
    """Return exact KL, JS, NLL, logit distance, and Fisher diagnostics."""

    full = full_logits.detach().float().cpu()
    perturbed = perturbed_logits.detach().float().cpu()
    if torch.equal(full, perturbed):
        full_log = torch.log_softmax(full, dim=-1)
        nll = -float(full_log[int(target_token)].item())
        return {
            "exact_kl": 0.0,
            "js": 0.0,
            "full_nll": nll,
            "perturbed_nll": nll,
            "delta_nll": 0.0,
            "logit_l2_sq": 0.0,
            "fisher_quadratic": 0.0,
        }

    full_log = torch.log_softmax(full, dim=-1)
    perturbed_log = torch.log_softmax(perturbed, dim=-1)
    full_probability = full_log.exp()
    perturbed_probability = perturbed_log.exp()
    midpoint = 0.5 * (full_probability + perturbed_probability)
    midpoint_log = torch.log(midpoint.clamp_min(1e-30))
    delta_logits = perturbed - full
    centered = delta_logits - torch.sum(full_probability * delta_logits)
    fisher = 0.5 * torch.sum(full_probability * centered.square())
    return {
        "exact_kl": float(
            torch.sum(full_probability * (full_log - perturbed_log)).item()
        ),
        "js": float(
            (
                0.5 * torch.sum(full_probability * (full_log - midpoint_log))
                + 0.5
                * torch.sum(perturbed_probability * (perturbed_log - midpoint_log))
            ).item()
        ),
        "full_nll": -float(full_log[int(target_token)].item()),
        "perturbed_nll": -float(perturbed_log[int(target_token)].item()),
        "delta_nll": float(
            full_log[int(target_token)].item()
            - perturbed_log[int(target_token)].item()
        ),
        "logit_l2_sq": float(delta_logits.square().sum().item()),
        "fisher_quadratic": float(fisher.item()),
    }
