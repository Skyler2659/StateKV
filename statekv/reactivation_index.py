"""Reactivation Index (RI) from full-cache attention trajectories.

Computes how much of a sequence's future importance mass is carried by
tokens that were dormant (low rank) for an extended period before becoming
important again. RI is computed ONLY from full-cache attention trajectories
(collected artifacts: per-cycle attention over all KV positions); no
compressed-policy output is read.

Definitions (parameters frozen per protocol before any test use):

- importance(p, t): attention received by KV position p at decoding cycle
  t, averaged over the probed layers and KV heads.
- future-important event: position p ENTERS the top-`top_k` importance
  ranks at cycle u (was not top-k at u-1), with u >= min_cycle.
- dormant before u: p's importance rank (0 = most important) stayed at or
  below the `dormant_rank_quantile` fraction of active positions for at
  least `dormant_window` consecutive cycles immediately before u.
- RI_fraction = dormant-reactivation events / all future-important events.

Per-sequence outputs include RI_count, RI_fraction, and per-event
reactivation distance, dormancy duration, and reactivation amplitude.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

import numpy as np


@dataclass(frozen=True)
class ReactivationParams:
    """Frozen RI parameters (selected on train/validation only)."""

    top_k: int = 10
    dormant_window: int = 16
    dormant_rank_quantile: float = 0.1
    horizon: int = 32
    min_cycle: int = 1


@dataclass
class ReactivationEvent:
    position: int
    cycle: int
    dormancy_duration: int
    reactivation_distance: int
    dormant_mean_attention: float
    reactivation_attention: float

    @property
    def amplitude(self) -> float:
        return self.reactivation_attention - self.dormant_mean_attention


@dataclass
class SequenceReactivation:
    sample_id: str
    task: str
    n_cycles: int
    n_positions: int
    n_future_important_events: int
    n_reactivation_events: int
    events: List[ReactivationEvent] = field(default_factory=list)

    @property
    def ri_fraction(self) -> float:
        if self.n_future_important_events == 0:
            return 0.0
        return self.n_reactivation_events / self.n_future_important_events

    def summary(self) -> Dict[str, Any]:
        durations = [e.dormancy_duration for e in self.events]
        distances = [e.reactivation_distance for e in self.events]
        amplitudes = [e.amplitude for e in self.events]
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "n_cycles": self.n_cycles,
            "n_positions": self.n_positions,
            "ri_count": self.n_reactivation_events,
            "ri_fraction": self.ri_fraction,
            "future_important_events": self.n_future_important_events,
            "mean_dormancy_duration": float(np.mean(durations)) if durations else 0.0,
            "mean_reactivation_distance": float(np.mean(distances)) if distances else 0.0,
            "mean_reactivation_amplitude": float(np.mean(amplitudes)) if amplitudes else 0.0,
        }


def _aggregated_importance(artifact: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-cycle per-position importance, averaged over layers and heads.

    Returns an array of shape (n_cycles, n_positions) with zeros beyond each
    cycle's active prefix length.
    """
    attention = np.asarray(artifact["attention"], dtype=np.float64)
    lengths = np.asarray(artifact["position_lengths"], dtype=np.int64)
    importance = attention.mean(axis=(1, 2))
    for cycle in range(importance.shape[0]):
        importance[cycle, int(lengths[cycle]):] = 0.0
    return importance


def compute_sequence_reactivation(
    artifact: Mapping[str, np.ndarray],
    params: ReactivationParams,
) -> SequenceReactivation:
    """Compute reactivation events for one full-cache trajectory artifact."""
    importance = _aggregated_importance(artifact)
    n_cycles, n_positions = importance.shape
    lengths = np.asarray(artifact["position_lengths"], dtype=np.int64)

    ranks = np.full((n_cycles, n_positions), np.inf)
    in_top_k = np.zeros((n_cycles, n_positions), dtype=bool)
    for cycle in range(n_cycles):
        active = int(lengths[cycle])
        if active == 0:
            continue
        order = np.argsort(-importance[cycle, :active], kind="stable")
        rank_of = np.empty(active, dtype=np.float64)
        rank_of[order] = np.arange(active, dtype=np.float64) / active
        ranks[cycle, :active] = rank_of
        top = order[: min(params.top_k, active)]
        in_top_k[cycle, top] = True

    dormant_cut = params.dormant_rank_quantile
    n_entries = 0
    events: List[ReactivationEvent] = []
    for position in range(n_positions):
        first_cycle = int(np.searchsorted(lengths, position + 1))
        for cycle in range(max(first_cycle, params.min_cycle), n_cycles):
            if not in_top_k[cycle, position]:
                continue
            if cycle > 0 and in_top_k[cycle - 1, position]:
                continue  # continuation, not an entry
            n_entries += 1
            # dormancy streak immediately before this entry
            streak = 0
            lookback = cycle - 1
            while lookback >= first_cycle and streak < cycle - first_cycle:
                rank = ranks[lookback, position]
                if np.isinf(rank) or rank < dormant_cut:
                    break
                streak += 1
                lookback -= 1
            if streak < params.dormant_window:
                continue
            # reactivation distance: cycles since last top-k membership
            # (or since the position entered the cache)
            previous = np.flatnonzero(in_top_k[:cycle, position])
            last_important = (
                int(previous[-1]) if len(previous) else first_cycle
            )
            dormant_span = importance[cycle - streak : cycle, position]
            events.append(
                ReactivationEvent(
                    position=position,
                    cycle=cycle,
                    dormancy_duration=streak,
                    reactivation_distance=cycle - last_important,
                    dormant_mean_attention=float(dormant_span.mean()),
                    reactivation_attention=float(importance[cycle, position]),
                )
            )

    return SequenceReactivation(
        sample_id=str(artifact["sample_id"]),
        task=str(artifact["task"]),
        n_cycles=n_cycles,
        n_positions=n_positions,
        n_future_important_events=n_entries,
        n_reactivation_events=len(events),
        events=events,
    )


def load_artifact(path: str) -> Dict[str, Any]:
    """Load a collected .npz artifact into a plain mapping."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
