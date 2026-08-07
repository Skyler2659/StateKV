"""Core primitives for P1 state-conditioned fixed-boundary risk closure."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
P0_SCRIPT_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"

import sys

if str(P0_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(P0_SCRIPT_DIR))

from p0_v2_core import (  # noqa: E402
    _jvp_via_autodiff,
    atomic_frame,
    atomic_json,
    exact_kl,
    fisher_variance,
    json_safe,
    pairwise_sign_accuracy,
    prefixed_metrics,
    ranking_metrics,
    sha256_array,
    sha256_file,
    spearman,
    stable_softmax,
    vector_metrics,
)


def fisher_inner(
    probability: Any,
    left: Any,
    right: Any,
) -> float:
    """Return the categorical-Fisher bilinear form at ``probability``."""
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a_centered = a - float(np.dot(p, a))
    b_centered = b - float(np.dot(p, b))
    return float(np.dot(p, a_centered * b_centered))


def state_action_scores(
    probability: Any,
    state_logits: Any,
    action_logits: Any,
) -> Dict[str, float]:
    """Compute the preregistered state/action quadratic decomposition."""
    state = np.asarray(state_logits, dtype=np.float64).reshape(-1)
    action = np.asarray(action_logits, dtype=np.float64).reshape(-1)
    action_score = 0.5 * fisher_variance(probability, action)
    cross_score = fisher_inner(probability, state, action)
    state_score = action_score + cross_score
    state_energy = 0.5 * fisher_variance(probability, state)
    total_score = 0.5 * fisher_variance(probability, state + action)
    return {
        "action_fisher_score": float(action_score),
        "cross_fisher_score": float(cross_score),
        "state_fisher_score": float(state_score),
        "state_fisher_energy": float(state_energy),
        "total_fisher_score": float(total_score),
        "decomposition_error": float(
            total_score - state_energy - state_score
        ),
    }


def polarization_cross(
    probability: Any,
    state_logits: Any,
    action_logits: Any,
) -> float:
    """Recover the cross term by Fisher polarization."""
    state = np.asarray(state_logits, dtype=np.float64).reshape(-1)
    action = np.asarray(action_logits, dtype=np.float64).reshape(-1)
    return 0.5 * (
        fisher_variance(probability, state + action)
        - fisher_variance(probability, state)
        - fisher_variance(probability, action)
    )


def euclidean_cosine(
    left: Any,
    right: Any,
    norm_floor: float = 1.0e-12,
) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = max(
        float(np.linalg.norm(a)) * float(np.linalg.norm(b)),
        float(norm_floor) ** 2,
    )
    return float(np.dot(a, b) / denominator)


def fisher_cosine(
    probability: Any,
    left: Any,
    right: Any,
    norm_floor: float = 1.0e-12,
) -> float:
    numerator = fisher_inner(probability, left, right)
    left_norm = max(fisher_variance(probability, left), 0.0) ** 0.5
    right_norm = max(fisher_variance(probability, right), 0.0) ** 0.5
    return float(
        numerator
        / max(left_norm * right_norm, float(norm_floor) ** 2)
    )


def required_reference_anchors(
    targets: Sequence[int],
    history_conditions: Mapping[str, Mapping[str, Any]],
) -> Tuple[int, ...]:
    """Return every anchor needed for stale starts and matched refreshes."""
    required = set()
    for target in targets:
        target = int(target)
        required.add(target)
        for condition in history_conditions.values():
            if (
                not isinstance(condition, Mapping)
                or "history_length" not in condition
            ):
                continue
            length = int(condition["history_length"])
            if length <= 0:
                continue
            start = target - length
            if start < 0:
                raise ValueError("history start precedes generated sequence")
            required.add(start)
            interval = condition.get("refresh_interval")
            count = int(condition.get("expected_refresh_count", 0))
            if interval is not None:
                for refresh_index in range(1, count + 1):
                    required.add(start + int(interval) * refresh_index)
    return tuple(sorted(required))


def validate_split_isolation(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    data = protocol["data"]

    def ids(section: Mapping[str, Any]) -> set[str]:
        return {
            *{
                f"gov_report:{int(value)}"
                for value in section["gov_report_indices"]
            },
            *{
                f"synthetic_niah_{int(value)}"
                for value in section["niah_offsets"]
            },
        }

    smoke = ids(data["smoke"])
    calibration = ids(data["calibration"])
    evaluation = ids(data["evaluation"])
    forbidden = {str(value) for value in data["forbidden_ids"]}
    checks = {
        "calibration_evaluation_disjoint": calibration.isdisjoint(evaluation),
        "evaluation_forbidden_disjoint": evaluation.isdisjoint(forbidden),
        "evaluation_smoke_disjoint": evaluation.isdisjoint(smoke),
        "two_tasks_in_evaluation": (
            any(value.startswith("gov_report:") for value in evaluation)
            and any(
                value.startswith("synthetic_niah_") for value in evaluation
            )
        ),
        "two_sequences_per_task": (
            sum(value.startswith("gov_report:") for value in evaluation) >= 2
            and sum(
                value.startswith("synthetic_niah_") for value in evaluation
            )
            >= 2
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"split isolation failed: {checks}")
    return {
        "checks": checks,
        "smoke_ids": sorted(smoke),
        "calibration_ids": sorted(calibration),
        "evaluation_ids": sorted(evaluation),
        "forbidden_ids": sorted(forbidden),
    }


def history_state_key(
    sample_id: str,
    target_anchor: int,
    boundary_layer: int,
    history_id: str,
    delta: Any,
) -> str:
    payload = {
        "sample_id": str(sample_id),
        "target_anchor": int(target_anchor),
        "boundary_layer": int(boundary_layer),
        "history_id": str(history_id),
        "delta_sha256": sha256_array(
            np.asarray(delta, dtype=np.float32)
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HistoryObservation:
    history_id: str
    target_anchor: int
    start_anchor: int
    history_length: int
    refresh_count: int
    initial_source: str
    refresh_source: str | None
    logits: np.ndarray
    record: Any
    position_maps: Mapping[int, Any]
    replay_trace: Tuple[Mapping[str, Any], ...]


def clear_runtime_controls(backend: Any) -> None:
    """Clear all intervention controls before an ordinary replay step."""
    state = backend.runner.attention_state
    for key in (
        "temporal_projected_injections",
        "temporal_query_overrides",
        "temporal_new_key_overrides",
        "temporal_new_value_overrides",
        "temporal_attention_input_overrides",
        "temporal_layer_input_overrides",
    ):
        state[key] = {}
    state["temporal_record_diagnostics"] = True


class HistoryTrajectoryGenerator:
    """Generate observed full-network history states at a fixed target step.

    Each segment starts from a full-reference anchor, applies one jointly
    selected all-layer cache mask, then replays the exact reference tokens.
    The final target query is always included. H3 uses matched full-reference
    anchor resets at its preregistered refresh times; it is a control condition,
    not an online policy.
    """

    def __init__(
        self,
        backend: Any,
        reference: Any,
        protocol: Mapping[str, Any],
    ):
        self.backend = backend
        self.reference = reference
        self.protocol = protocol
        self._candidate_cache: Dict[int, Tuple[Any, ...]] = {}

    def candidates(self, anchor_step: int) -> Tuple[Any, ...]:
        from mlx_predictive_core import PhysicalCandidate

        anchor_step = int(anchor_step)
        if anchor_step not in self._candidate_cache:
            anchor = self.reference.anchors[anchor_step]
            positions = [
                int(value)
                for value in anchor.position_maps[0].tolist()
            ]
            if any(
                [
                    int(value)
                    for value in anchor.position_maps[layer].tolist()
                ]
                != positions
                for layer in anchor.position_maps
            ):
                raise RuntimeError(
                    "history mask requires a shared physical token universe"
                )
            cache = self.backend.cfg.cache
            current = int(anchor.logical_length - 1)
            sink = positions[: int(cache.sink_size)]
            recent = positions[-int(cache.recent_size) :]
            mandatory = set(sink + recent)
            eligible = [
                value for value in positions if value not in mandatory
            ]
            core_size = int(cache.selected_core_budget)
            if len(eligible) < core_size:
                raise RuntimeError(
                    "history anchor has insufficient eligible core positions"
                )
            candidates = []
            for source, core in (
                ("old_stale_core", eligible[:core_size]),
                ("fresh_core", eligible[-core_size:]),
            ):
                retained = tuple(sorted(mandatory | set(core)))
                if len(retained) != int(cache.total_budget):
                    raise RuntimeError("history candidate budget mismatch")
                candidates.append(
                    PhysicalCandidate(
                        candidate_id=f"history_{source}_{anchor_step}",
                        source=source,
                        core_positions=tuple(sorted(core)),
                        keep_prefix_positions=tuple(
                            value
                            for value in retained
                            if value != current
                        ),
                        retained_positions=retained,
                        seed=int(self.protocol["numeric"]["seed"]),
                    )
                )
            self._candidate_cache[anchor_step] = tuple(candidates)
        return self._candidate_cache[anchor_step]

    def _candidate(self, anchor_step: int, source: str) -> Any:
        by_source = {
            candidate.source: candidate
            for candidate in self.candidates(anchor_step)
        }
        if source not in by_source:
            raise RuntimeError(
                f"history source {source} missing at anchor {anchor_step}"
            )
        return by_source[source]

    def _segment(
        self,
        start_anchor: int,
        source: str,
        steps: int,
    ) -> Tuple[np.ndarray, Any, Mapping[int, Any], Dict[str, Any]]:
        from mlx_predictive_core import joint_candidate_selection

        start_anchor = int(start_anchor)
        candidate = self._candidate(start_anchor, source)
        selection = joint_candidate_selection(
            self.reference, start_anchor, candidate
        )
        state, fixed = self.backend.state_from_anchor(
            self.reference.anchors[start_anchor],
            selection,
            cache_config=self.backend.cfg.cache,
        )
        logits = None
        record = None
        try:
            for offset in range(1, int(steps) + 1):
                if offset > 1:
                    self.backend.prune_recent_before_query(
                        state,
                        fixed,
                        cache_config=self.backend.cfg.cache,
                    )
                target_index = start_anchor + offset - 1
                if offset == 1:
                    token = int(
                        self.reference.anchors[start_anchor].query_token_id
                    )
                else:
                    token = int(
                        self.reference.generated_token_ids[target_index - 1]
                    )
                expected_token = (
                    int(self.reference.anchors[target_index].query_token_id)
                    if target_index in self.reference.anchors
                    else token
                )
                if token != expected_token:
                    raise RuntimeError(
                        "teacher-forced history token alignment failed"
                    )
                clear_runtime_controls(self.backend)
                logits_t, record, _elapsed = self.backend.forward_one(
                    state, token, capture_attention=True
                )
                self.backend.validate_active_budget(
                    state, cache_config=self.backend.cfg.cache
                )
                logits = logits_t.double().numpy()
            if logits is None or record is None:
                raise RuntimeError("empty history segment")
            maps = {
                int(layer): value.detach().clone()
                for layer, value in state.position_maps.items()
            }
            trace = {
                "start_anchor": start_anchor,
                "source": source,
                "steps": int(steps),
                "end_target": start_anchor + int(steps) - 1,
                "candidate_id": candidate.candidate_id,
                "mask_hash": candidate.mask_hash,
            }
            return logits, record, maps, trace
        finally:
            self.backend.release(state)

    def generate(
        self,
        target_anchor: int,
        history_id: str,
        base_logits: np.ndarray,
        base_record: Any,
        base_position_maps: Mapping[int, Any],
    ) -> HistoryObservation:
        target_anchor = int(target_anchor)
        condition = self.protocol["history_conditions"][history_id]
        length = int(condition["history_length"])
        source = str(condition["initial_source"])
        refresh_source = condition.get("refresh_source")
        refresh_count = int(condition["expected_refresh_count"])
        if length == 0:
            return HistoryObservation(
                history_id=history_id,
                target_anchor=target_anchor,
                start_anchor=target_anchor,
                history_length=0,
                refresh_count=0,
                initial_source=source,
                refresh_source=None,
                logits=np.asarray(base_logits, dtype=np.float64).copy(),
                record=base_record,
                position_maps=base_position_maps,
                replay_trace=tuple(),
            )
        start = target_anchor - length
        interval = condition.get("refresh_interval")
        if interval is None:
            logits, record, maps, trace = self._segment(
                start, source, length + 1
            )
            traces = (trace,)
        else:
            interval = int(interval)
            starts = [
                start + interval * index
                for index in range(refresh_count + 1)
            ]
            traces_list = []
            logits = None
            record = None
            maps = None
            for index, segment_start in enumerate(starts):
                segment_source = (
                    source if index == 0 else str(refresh_source)
                )
                segment_steps = (
                    interval
                    if index < refresh_count
                    else target_anchor - segment_start + 1
                )
                logits, record, maps, trace = self._segment(
                    segment_start, segment_source, segment_steps
                )
                traces_list.append(trace)
            if logits is None or record is None or maps is None:
                raise RuntimeError("empty periodic history replay")
            traces = tuple(traces_list)
        if int(record.query_position) != int(
            self.reference.query_records[target_anchor].query_position
        ):
            raise RuntimeError("history target query position mismatch")
        return HistoryObservation(
            history_id=history_id,
            target_anchor=target_anchor,
            start_anchor=start,
            history_length=length,
            refresh_count=refresh_count,
            initial_source=source,
            refresh_source=(
                None if refresh_source is None else str(refresh_source)
            ),
            logits=np.asarray(logits, dtype=np.float64),
            record=record,
            position_maps=maps,
            replay_trace=traces,
        )


def select_fd_radius(
    radius_rows: Any,
    rule: Mapping[str, Any],
) -> Tuple[float | None, Any]:
    """Apply the frozen finite-difference selection rule."""
    import pandas as pd

    frame = pd.DataFrame(radius_rows)
    grouped = []
    for radius, group in frame.groupby("epsilon_relative", sort=True):
        record = {
            "epsilon_relative": float(radius),
            "row_count": int(len(group)),
            "finite_rate": float(group["finite"].mean()),
            "nonzero_fd_norm_rate": float(
                (group["fd_norm"] > 0.0).mean()
            ),
            "median_cosine": float(group["cosine"].median()),
            "median_relative_l2": float(group["relative_l2"].median()),
            "median_symmetric_norm_ratio": float(
                group["symmetric_norm_ratio"].median()
            ),
        }
        record["passes"] = bool(
            record["finite_rate"] >= float(rule["finite_rate_min"])
            and record["nonzero_fd_norm_rate"]
            >= float(rule["nonzero_fd_norm_rate_min"])
            and record["median_cosine"] >= float(rule["median_cosine_min"])
            and record["median_relative_l2"]
            <= float(rule["median_relative_l2_max"])
            and record["median_symmetric_norm_ratio"]
            >= float(rule["median_symmetric_norm_ratio_min"])
        )
        grouped.append(record)
    summary = pd.DataFrame(grouped)
    passing = summary[summary["passes"]].copy()
    if passing.empty:
        return None, summary
    passing = passing.sort_values(
        ["median_relative_l2", "epsilon_relative"],
        ascending=[True, False],
        kind="mergesort",
    )
    return float(passing.iloc[0]["epsilon_relative"]), summary


def downstream_jvp_at(
    downstream_map: Any,
    operating_point: Any,
    tangent: Any,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Evaluate a downstream JVP at an observed nonzero boundary state."""
    import mlx.core as mx

    primal = mx.array(
        np.asarray(operating_point, dtype=np.float32).reshape(-1)
    )
    direction = mx.array(
        np.asarray(tangent, dtype=np.float32).reshape(-1)
    )
    fingerprint_before = downstream_map.cache_fingerprint()
    result = _jvp_via_autodiff(
        downstream_map, primal, direction
    )
    if fingerprint_before != downstream_map.cache_fingerprint():
        raise RuntimeError(
            "state-operating-point JVP mutated frozen prefix cache"
        )
    return result


__all__ = [
    "HistoryObservation",
    "HistoryTrajectoryGenerator",
    "atomic_frame",
    "atomic_json",
    "euclidean_cosine",
    "downstream_jvp_at",
    "exact_kl",
    "fisher_cosine",
    "fisher_inner",
    "fisher_variance",
    "history_state_key",
    "json_safe",
    "pairwise_sign_accuracy",
    "polarization_cross",
    "prefixed_metrics",
    "ranking_metrics",
    "required_reference_anchors",
    "select_fd_radius",
    "sha256_array",
    "sha256_file",
    "spearman",
    "stable_softmax",
    "state_action_scores",
    "validate_split_isolation",
    "vector_metrics",
]
