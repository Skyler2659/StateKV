"""Causal temporal-importance estimators for adaptive KV-cache research.

The estimators in this module consume one attention observation per token and
decode step.  They never read future observations.  Offline diagnostics may
compare their outputs with future utility after the causal scores have been
materialized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class AdaptiveTemporalConfig:
    fast_rho: float = 0.5
    slow_rho: float = 0.99
    variance_rho: float = 0.9
    rho_short: float = 0.5
    rho_long: float = 0.99
    threshold: float = 1.0
    smooth_alpha: float = 4.0
    epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        for name in (
            "fast_rho",
            "slow_rho",
            "variance_rho",
            "rho_short",
            "rho_long",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.fast_rho >= self.slow_rho:
            raise ValueError("fast_rho must be smaller than slow_rho")
        if self.rho_short > self.rho_long:
            raise ValueError("rho_short must not exceed rho_long")
        if self.threshold < 0.0 or self.smooth_alpha <= 0.0:
            raise ValueError("threshold must be non-negative and alpha positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("observations must have shape [steps, tokens]")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0.0):
        raise ValueError("attention observations must be non-negative")
    return values


def fixed_ema(observations: np.ndarray, rho: float) -> np.ndarray:
    """Return a causal tokenwise EMA, initializing each token on first sight."""

    values = _validate_observations(observations)
    decay = float(rho)
    if not 0.0 <= decay < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    output = np.full_like(values, np.nan)
    state = np.zeros(values.shape[1], dtype=np.float64)
    initialized = np.zeros(values.shape[1], dtype=bool)
    for step in range(values.shape[0]):
        present = np.isfinite(values[step])
        new = present & ~initialized
        old = present & initialized
        state[new] = values[step, new]
        state[old] = decay * state[old] + (1.0 - decay) * values[step, old]
        initialized |= present
        output[step, initialized] = state[initialized]
    return output


def cumulative_attention(observations: np.ndarray) -> np.ndarray:
    """Return causal accumulated attention (the H2O-style saliency control)."""

    values = _validate_observations(observations)
    output = np.full_like(values, np.nan)
    state = np.zeros(values.shape[1], dtype=np.float64)
    initialized = np.zeros(values.shape[1], dtype=bool)
    for step in range(values.shape[0]):
        present = np.isfinite(values[step])
        state[present] += values[step, present]
        initialized |= present
        output[step, initialized] = state[initialized]
    return output


def percentile_rank_observations(observations: np.ndarray) -> np.ndarray:
    """Return causal within-step percentile ranks for attention observations."""

    from scipy.stats import rankdata

    values = _validate_observations(observations)
    output = np.full_like(values, np.nan)
    for step in range(values.shape[0]):
        present = np.isfinite(values[step])
        count = int(present.sum())
        if count:
            output[step, present] = (
                rankdata(values[step, present], method="average") / float(count)
            )
    return output


def rank_jump_dual_scores(
    observations: np.ndarray,
    *,
    fast_rho: float,
    slow_rho: float,
    rank_memory_rho: float,
    jump_threshold: float,
    gate_alpha: float,
    output_space: str = "attention",
) -> Dict[str, np.ndarray]:
    """Mix short/long memories with a causal percentile-rank jump gate.

    Rank jump is measured against the preceding rank memory. Large jumps
    select the fast memory; stable ranks select the slow memory. The output can
    mix attention EMAs or rank EMAs. No future observation is read.
    """

    values = _validate_observations(observations)
    if not 0.0 <= float(fast_rho) < float(slow_rho) < 1.0:
        raise ValueError("rank-jump memory requires 0 <= fast_rho < slow_rho < 1")
    if not 0.0 <= float(rank_memory_rho) < 1.0:
        raise ValueError("rank_memory_rho must lie in [0, 1)")
    if float(jump_threshold) < 0.0 or float(gate_alpha) <= 0.0:
        raise ValueError("jump threshold must be non-negative and alpha positive")
    if output_space not in {"attention", "rank"}:
        raise ValueError("output_space must be attention or rank")

    ranks = percentile_rank_observations(values)
    attention_fast = fixed_ema(values, float(fast_rho))
    attention_slow = fixed_ema(values, float(slow_rho))
    rank_fast = fixed_ema(ranks, float(fast_rho))
    rank_slow = fixed_ema(ranks, float(slow_rho))
    source_fast = attention_fast if output_space == "attention" else rank_fast
    source_slow = attention_slow if output_space == "attention" else rank_slow

    output = np.full_like(values, np.nan)
    jump = np.full_like(values, np.nan)
    stable_gate = np.full_like(values, np.nan)
    rank_memory = np.zeros(values.shape[1], dtype=np.float64)
    initialized = np.zeros(values.shape[1], dtype=bool)
    for step in range(values.shape[0]):
        present = np.isfinite(ranks[step])
        new = present & ~initialized
        old = present & initialized
        jump[step, new] = 0.0
        jump[step, old] = np.abs(ranks[step, old] - rank_memory[old])
        gate = np.full(values.shape[1], np.nan, dtype=np.float64)
        gate[present] = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    float(gate_alpha)
                    * (jump[step, present] - float(jump_threshold)),
                    -60.0,
                    60.0,
                )
            )
        )
        stable_gate[step, present] = gate[present]
        output[step, present] = (
            gate[present] * source_slow[step, present]
            + (1.0 - gate[present]) * source_fast[step, present]
        )
        rank_memory[new] = ranks[step, new]
        rank_memory[old] = (
            float(rank_memory_rho) * rank_memory[old]
            + (1.0 - float(rank_memory_rho)) * ranks[step, old]
        )
        initialized |= present
    return {"score": output, "rank_jump": jump, "stable_gate": stable_gate}


def adaptive_temporal_scores(
    observations: np.ndarray,
    config: AdaptiveTemporalConfig,
) -> Dict[str, np.ndarray]:
    """Return fast/slow drift states and three causal adaptive scores.

    ``adaptive_discrete`` changes the recursive decay at a normalized-drift
    threshold. ``adaptive_smooth`` uses the same endpoints with a logistic
    transition. ``dual_memory`` gates directly between the fast and slow
    states and is the preregistered first fallback if dynamic-rho recursion is
    unstable. ``adaptive_raw`` is the required no-normalization ablation.
    """

    values = _validate_observations(observations)
    shape = values.shape
    outputs = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in (
            "fast",
            "slow",
            "variance",
            "drift_raw",
            "drift_z",
            "rho_discrete",
            "rho_smooth",
            "adaptive_discrete",
            "adaptive_smooth",
            "adaptive_raw",
            "dual_memory",
            "surprise",
            "surprise_z",
            "adaptive_surprise",
        )
    }
    width = shape[1]
    initialized = np.zeros(width, dtype=bool)
    fast = np.zeros(width, dtype=np.float64)
    slow = np.zeros(width, dtype=np.float64)
    variance = np.zeros(width, dtype=np.float64)
    discrete = np.zeros(width, dtype=np.float64)
    smooth = np.zeros(width, dtype=np.float64)
    raw_adaptive = np.zeros(width, dtype=np.float64)
    surprise_adaptive = np.zeros(width, dtype=np.float64)

    for step in range(shape[0]):
        observation = values[step]
        present = np.isfinite(observation)
        new = present & ~initialized
        old = present & initialized

        fast[new] = observation[new]
        slow[new] = observation[new]
        discrete[new] = observation[new]
        smooth[new] = observation[new]
        raw_adaptive[new] = observation[new]
        surprise_adaptive[new] = observation[new]

        previous_slow = slow.copy()
        previous_smooth = smooth.copy()
        if np.any(old):
            residual = observation[old] - previous_slow[old]
            variance[old] = (
                config.variance_rho * variance[old]
                + (1.0 - config.variance_rho) * np.square(residual)
            )
            fast[old] = (
                config.fast_rho * fast[old]
                + (1.0 - config.fast_rho) * observation[old]
            )
            slow[old] = (
                config.slow_rho * slow[old]
                + (1.0 - config.slow_rho) * observation[old]
            )

        drift_raw = np.abs(fast - slow)
        denominator = np.sqrt(np.maximum(variance, 0.0)) + config.epsilon
        drift_z = drift_raw / denominator
        surprise = np.abs(observation - previous_smooth)
        surprise_z = surprise / denominator

        rho_discrete = np.where(
            drift_z < config.threshold, config.rho_long, config.rho_short
        )
        logistic = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    config.smooth_alpha * (drift_z - config.threshold),
                    -60.0,
                    60.0,
                )
            )
        )
        rho_smooth = config.rho_short + (
            config.rho_long - config.rho_short
        ) * logistic
        # Raw drift is scaled by the threshold used for Z. This is intentionally
        # simple: it tests whether normalization, rather than gating alone,
        # carries the effect.
        rho_raw = np.where(
            drift_raw < config.threshold, config.rho_long, config.rho_short
        )
        rho_surprise = np.where(
            surprise_z < config.threshold, config.rho_long, config.rho_short
        )

        if np.any(old):
            discrete[old] = (
                rho_discrete[old] * discrete[old]
                + (1.0 - rho_discrete[old]) * observation[old]
            )
            smooth[old] = (
                rho_smooth[old] * smooth[old]
                + (1.0 - rho_smooth[old]) * observation[old]
            )
            raw_adaptive[old] = (
                rho_raw[old] * raw_adaptive[old]
                + (1.0 - rho_raw[old]) * observation[old]
            )
            surprise_adaptive[old] = (
                rho_surprise[old] * surprise_adaptive[old]
                + (1.0 - rho_surprise[old]) * observation[old]
            )

        initialized |= present
        for name, state in (
            ("fast", fast),
            ("slow", slow),
            ("variance", variance),
            ("drift_raw", drift_raw),
            ("drift_z", drift_z),
            ("rho_discrete", rho_discrete),
            ("rho_smooth", rho_smooth),
            ("adaptive_discrete", discrete),
            ("adaptive_smooth", smooth),
            ("adaptive_raw", raw_adaptive),
            ("dual_memory", logistic * slow + (1.0 - logistic) * fast),
            ("surprise", surprise),
            ("surprise_z", surprise_z),
            ("adaptive_surprise", surprise_adaptive),
        ):
            outputs[name][step, initialized] = state[initialized]

    return outputs


def estimator_panel(
    observations: np.ndarray,
    fixed_rhos: Iterable[float],
    config: AdaptiveTemporalConfig,
) -> Mapping[str, np.ndarray]:
    """Build the causal estimator panel used by offline diagnostics."""

    values = _validate_observations(observations)
    panel: Dict[str, np.ndarray] = {
        "current": fixed_ema(values, 0.0),
        "cumulative": cumulative_attention(values),
    }
    for rho in fixed_rhos:
        value = float(rho)
        panel[f"fixed_ema_rho_{value:g}"] = fixed_ema(values, value)
    adaptive = adaptive_temporal_scores(values, config)
    panel.update(
        {
            name: adaptive[name]
            for name in (
                "adaptive_discrete",
                "adaptive_smooth",
                "adaptive_raw",
                "dual_memory",
                "adaptive_surprise",
            )
        }
    )
    return panel


def future_attention_utility(
    observations: np.ndarray, horizon: int
) -> np.ndarray:
    """Return NON_CAUSAL_ORACLE sum of attention over the next H steps."""

    values = _validate_observations(observations)
    width = int(horizon)
    if width < 1:
        raise ValueError("horizon must be positive")
    output = np.full_like(values, np.nan)
    for step in range(values.shape[0]):
        stop = min(values.shape[0], step + width + 1)
        if step + 1 >= stop:
            continue
        future = values[step + 1 : stop]
        counts = np.sum(np.isfinite(future), axis=0)
        summed = np.nansum(future, axis=0)
        valid = counts > 0
        output[step, valid] = summed[valid]
    return output


def additional_state_bytes(
    *,
    scalars_per_token: int,
    dtype_bytes: int,
    layers: int,
    tokens_per_layer: int,
) -> int:
    """Return the exact storage for scalar temporal states."""

    values = (scalars_per_token, dtype_bytes, layers, tokens_per_layer)
    if any(int(value) < 0 for value in values):
        raise ValueError("memory dimensions must be non-negative")
    return int(np.prod([int(value) for value in values], dtype=np.int64))
