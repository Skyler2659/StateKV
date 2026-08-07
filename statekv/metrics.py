"""Numerically checked teacher-forced distortion metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


def approximate_kl(
    full_top_ids: torch.Tensor,
    full_top_probabilities: torch.Tensor,
    compressed_logits: torch.Tensor,
    floor: float = 1e-12,
) -> Dict[str, float]:
    ids = full_top_ids.detach().long().flatten().to(compressed_logits.device)
    p = full_top_probabilities.detach().double().flatten()
    if p.numel() != ids.numel():
        raise ValueError("full top-k IDs/probabilities do not align")
    if not torch.isfinite(p).all() or (p < 0).any():
        raise FloatingPointError("full top-k probabilities are invalid")
    q_distribution = torch.softmax(compressed_logits.detach().double().flatten(), dim=0)
    q = q_distribution.index_select(0, ids).cpu()
    p_sum = float(p.sum().item())
    q_sum = float(q.sum().item())
    # Stored top-k probabilities may originate from float32 softmax output.
    # A 128-term sum can exceed one by a few ulps even though the underlying
    # distribution was normalized. Correct only that explicitly bounded
    # rounding case; larger violations remain hard failures.
    tolerance = 1e-5
    if p_sum > 1.0 + tolerance or q_sum > 1.0 + tolerance:
        raise FloatingPointError(
            "top-k probability mass exceeds one beyond rounding tolerance "
            "(p=%.12g, q=%.12g)" % (p_sum, q_sum)
        )
    p_rounding_correction = max(0.0, p_sum - 1.0)
    q_rounding_correction = max(0.0, q_sum - 1.0)
    if p_sum > 1.0:
        p = p / p_sum
        p_sum = 1.0
    if q_sum > 1.0:
        q = q / q_sum
        q_sum = 1.0
    p_other = max(0.0, 1.0 - p_sum)
    q_other = max(0.0, 1.0 - q_sum)
    p_bucket = torch.cat([p.cpu(), torch.tensor([p_other], dtype=torch.float64)])
    q_bucket = torch.cat([q, torch.tensor([q_other], dtype=torch.float64)])
    p_bucket = p_bucket / p_bucket.sum().clamp_min(float(floor))
    q_bucket = q_bucket / q_bucket.sum().clamp_min(float(floor))
    mask = p_bucket > 0
    value = (
        p_bucket[mask]
        * (
            torch.log(p_bucket[mask].clamp_min(float(floor)))
            - torch.log(q_bucket[mask].clamp_min(float(floor)))
        )
    ).sum()
    if not torch.isfinite(value):
        raise FloatingPointError("approximate KL is NaN/Inf")
    return {
        "approx_kl": float(value.item()),
        "full_top_mass": p_sum,
        "compressed_on_full_top_mass": q_sum,
        "full_other_mass": p_other,
        "compressed_other_mass": q_other,
        "full_top_mass_rounding_correction": p_rounding_correction,
        "compressed_top_mass_rounding_correction": q_rounding_correction,
    }


def attention_output_relative_errors(
    reference: Dict[str, torch.Tensor],
    compressed: Dict[str, torch.Tensor],
    epsilon: float,
) -> List[Dict[str, float]]:
    records = []
    for key, full_value in reference.items():
        if key not in compressed:
            continue
        candidate = compressed[key].detach().float().cpu()
        full = full_value.detach().float().cpu()
        numerator = torch.linalg.vector_norm(full - candidate)
        denominator = torch.linalg.vector_norm(full) + float(epsilon)
        error = numerator / denominator
        if not torch.isfinite(error):
            raise FloatingPointError("attention-output error is NaN/Inf")
        layer_text, head_text = key.split(":")
        records.append(
            {
                "layer": int(layer_text),
                "head": int(head_text),
                "relative_error": float(error.item()),
            }
        )
    return records


def loss_shape(
    delta_nll: Sequence[float],
    large_spike_threshold: float,
) -> Dict[str, Any]:
    values = np.asarray(delta_nll, dtype=np.float64)
    if values.size == 0:
        return {
            "cumulative": [],
            "running_average": [],
            "running_max": [],
            "slope": [],
            "curvature": [],
            "first_large_loss_spike": None,
            "change_point": None,
        }
    if not np.isfinite(values).all():
        raise FloatingPointError("loss curve contains NaN/Inf")
    cumulative = np.cumsum(values)
    average = cumulative / np.arange(1, values.size + 1)
    running_max = np.maximum.accumulate(values)
    slope = np.diff(values, prepend=values[0])
    curvature = np.diff(slope, prepend=slope[0])
    spikes = np.flatnonzero(values > float(large_spike_threshold))
    change_point: Optional[int] = None
    if values.size >= 5:
        candidates = []
        for split in range(2, values.size - 1):
            candidates.append(
                (abs(float(values[:split].mean() - values[split:].mean())), split + 1)
            )
        if candidates:
            change_point = max(candidates)[1]
    return {
        "cumulative": cumulative.tolist(),
        "running_average": average.tolist(),
        "running_max": running_max.tolist(),
        "slope": slope.tolist(),
        "curvature": curvature.tolist(),
        "first_large_loss_spike": int(spikes[0] + 1) if spikes.size else None,
        "change_point": change_point,
        "change_point_method": "max_absolute_prefix_suffix_mean_difference",
    }


def validity_observations(
    steps: Sequence[Dict[str, Any]],
    thresholds: Dict[str, Iterable[float]],
    max_measured_horizon: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(steps, key=lambda row: int(row["future_step"]))
    delta = np.asarray([float(row["delta_nll"]) for row in ordered], dtype=np.float64)
    kl = np.asarray([float(row["approx_kl"]) for row in ordered], dtype=np.float64)
    output = []
    fields = {
        "avg_delta_nll": np.cumsum(delta) / np.arange(1, delta.size + 1),
        "max_delta_nll": np.maximum.accumulate(delta),
        "avg_approx_kl": np.cumsum(kl) / np.arange(1, kl.size + 1),
    }
    for metric, values in fields.items():
        for threshold in thresholds.get(metric, []):
            valid = np.flatnonzero(values <= float(threshold))
            horizon = int(valid[-1] + 1) if valid.size else 0
            output.append(
                {
                    "metric": metric,
                    "threshold": float(threshold),
                    "observed_horizon": horizon,
                    "is_right_censored": bool(
                        horizon == int(max_measured_horizon)
                        and int(max_measured_horizon) == len(ordered)
                    ),
                }
            )
    return output
