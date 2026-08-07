"""Controlled trajectory interventions for stochastic-model identification.

This module is deliberately opt-in.  It replays the full-cache teacher-forced
trajectory, injects an exact projected attention-branch perturbation at one
anchor, and records how that perturbation propagates.  It does not implement a
deployment policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.artifacts import json_text
from statekv.backend import QueryRecord, ReferenceTrajectory
from statekv.config import CacheDiscoveryConfig, DiscoveryConfig
from statekv.functional_probe import FunctionalProbeRunner
from statekv.functional_probe import _condition_cache
from statekv.runner import _sample_slug
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
    mandatory_and_eligible,
    ridge_leverage,
)
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame


TRAJECTORY_TABLES = (
    "trajectory_intervention_inventory",
    "trajectory_state_rows",
)
REQUIRED_SCALING_LAYERS = (0, 7, 14, 21, 27)
SUPERPOSITION_CATEGORIES = (
    "small_mass_plus_small_mass",
    "large_mass_plus_small_mass",
    "same_direction_residual",
    "opposite_direction_residual",
    "shared_kv_head_query_heads",
    "layer27_plus_other_layer",
)
CONTROL_KEYS = (
    "temporal_projected_injections",
    "temporal_query_overrides",
    "temporal_new_key_overrides",
    "temporal_new_value_overrides",
    "temporal_attention_input_overrides",
    "temporal_layer_input_overrides",
)


def stable_trajectory_id(*parts: Any) -> str:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return "%s_%s" % (
        "_".join(
            "".join(ch if ch.isalnum() else "_" for ch in str(part))[:24]
            for part in parts[:4]
        ),
        digest,
    )


def exact_distribution_metrics(
    full_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    target_token: int,
) -> Dict[str, float]:
    """Full-vocabulary KL/JS/NLL and the local Fisher quadratic."""

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
                0.5
                * torch.sum(full_probability * (full_log - midpoint_log))
                + 0.5
                * torch.sum(
                    perturbed_probability
                    * (perturbed_log - midpoint_log)
                )
            ).item()
        ),
        "full_nll": -float(full_log[int(target_token)].item()),
        "perturbed_nll": -float(
            perturbed_log[int(target_token)].item()
        ),
        "delta_nll": float(
            full_log[int(target_token)].item()
            - perturbed_log[int(target_token)].item()
        ),
        "logit_l2_sq": float(delta_logits.square().sum().item()),
        "fisher_quadratic": float(fisher.item()),
    }


def scaling_fit(beta: np.ndarray, response: np.ndarray) -> Dict[str, float]:
    """Origin-constrained scaling diagnostics against the beta=1 response."""

    beta = np.asarray(beta, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(beta) & np.isfinite(response)
    beta = beta[finite]
    response = response[finite]
    if len(beta) < 2:
        return {"slope": float("nan"), "r2": float("nan")}
    denominator = float(np.dot(beta, beta))
    slope = float(np.dot(beta, response) / max(denominator, 1e-30))
    prediction = slope * beta
    residual = float(np.sum((response - prediction) ** 2))
    total = float(np.sum((response - response.mean()) ** 2))
    return {
        "slope": slope,
        "r2": 1.0 - residual / max(total, 1e-30),
    }


def closed_form_recursion(
    a: np.ndarray,
    b: np.ndarray,
    sigma: np.ndarray,
    q: np.ndarray,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    inputs: np.ndarray,
    input_covariances: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Mean/covariance recursion used by the closed-form validation."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mean = np.asarray(initial_mean, dtype=np.float64)
    covariance = np.asarray(initial_covariance, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    if input_covariances is None:
        input_covariances = np.zeros(
            (len(inputs), b.shape[1], b.shape[1]), dtype=np.float64
        )
    means: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    risks: List[float] = []
    for step, current_input in enumerate(inputs):
        mean = a @ mean + b @ current_input
        covariance = (
            a @ covariance @ a.T
            + b @ input_covariances[step] @ b.T
            + sigma
        )
        means.append(mean.copy())
        covariances.append(covariance.copy())
        risks.append(
            float(mean.T @ q @ mean + np.trace(q @ covariance))
        )
    return {
        "mean": np.asarray(means),
        "covariance": np.asarray(covariances),
        "step_risk": np.asarray(risks),
        "cumulative_risk": np.cumsum(np.asarray(risks)),
    }


def fit_linear_dynamics(
    states: np.ndarray,
    inputs: np.ndarray,
    next_states: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Least-squares identification for a noiseless/synthetic linear system."""

    states = np.asarray(states, dtype=np.float64)
    inputs = np.asarray(inputs, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    if (
        states.ndim != 2
        or inputs.ndim != 2
        or next_states.ndim != 2
        or len(states) != len(inputs)
        or len(states) != len(next_states)
        or states.shape[1] != next_states.shape[1]
    ):
        raise ValueError("linear dynamics arrays are not aligned")
    design = np.concatenate([states, inputs], axis=1)
    coefficient, _, _, _ = np.linalg.lstsq(
        design, next_states, rcond=None
    )
    state_dimension = int(states.shape[1])
    return {
        "A": coefficient[:state_dimension].T,
        "B": coefficient[state_dimension:].T,
    }


def fit_quadratic_form(
    states: np.ndarray, losses: np.ndarray
) -> np.ndarray:
    """Recover a symmetric Q from losses x^T Q x."""

    states = np.asarray(states, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    if states.ndim != 2 or losses.shape != (len(states),):
        raise ValueError("quadratic-form arrays are not aligned")
    dimension = int(states.shape[1])
    features = []
    pairs = []
    for left in range(dimension):
        for right in range(left, dimension):
            multiplier = 1.0 if left == right else 2.0
            features.append(
                multiplier * states[:, left] * states[:, right]
            )
            pairs.append((left, right))
    coefficient, _, _, _ = np.linalg.lstsq(
        np.stack(features, axis=1), losses, rcond=None
    )
    q = np.zeros((dimension, dimension), dtype=np.float64)
    for value, (left, right) in zip(coefficient, pairs):
        q[left, right] = value
        q[right, left] = value
    return q


def assert_sequence_split(
    train_ids: Iterable[str], test_ids: Iterable[str]
) -> None:
    overlap = set(str(value) for value in train_ids) & set(
        str(value) for value in test_ids
    )
    if overlap:
        raise RuntimeError("sequence split leakage: %s" % sorted(overlap))


def assert_teacher_alignment(
    full_query_position: int,
    perturbed_query_position: int,
    full_target_token: int,
    perturbed_target_token: int,
) -> None:
    if (
        int(full_query_position) != int(perturbed_query_position)
        or int(full_target_token) != int(perturbed_target_token)
    ):
        raise RuntimeError("teacher-forced token/position alignment failed")


def validate_hybrid_source_labels(
    arm: str, labels: Mapping[str, str]
) -> None:
    expected = {
        "query_restore": ("query", "full_reference_restored"),
        "new_kv_restore": (
            "new_key_value",
            "full_reference_restored",
        ),
        "attention_input_restore": (
            "attention_input",
            "full_reference_restored",
        ),
        "next_layer_hidden_restore": (
            "next_layer_hidden",
            "full_reference_restored",
        ),
    }
    if arm == "stateful":
        if any(
            value == "full_reference_restored"
            for value in labels.values()
        ):
            raise ValueError("stateful arm must not restore a source")
        return
    if arm not in expected:
        raise ValueError("unknown hybrid arm")
    key, value = expected[arm]
    if labels.get(key) != value:
        raise ValueError("hybrid source label does not match arm")


def validate_recent_budget(
    retained_positions: Sequence[int],
    total_budget: int,
    sink_size: int,
    protected_recent: int,
) -> None:
    expected = int(total_budget)
    if len(set(int(value) for value in retained_positions)) > expected:
        raise RuntimeError("trajectory retained mask exceeds total budget")
    if expected - int(sink_size) - int(protected_recent) <= 0:
        raise RuntimeError("trajectory recent allocation leaves no core")


def layer_regime(layer: int) -> str:
    if int(layer) == 27:
        return "layer27"
    if int(layer) <= 7:
        return "low"
    if int(layer) <= 21:
        return "middle"
    return "high"


class TrajectoryModelRunner(FunctionalProbeRunner):
    """Execute the pre-registered balanced trajectory intervention matrix."""

    def run(self) -> Path:
        if not self.cfg.trajectory_model.enabled:
            raise ValueError("trajectory_model.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "trajectory_stochastic_model_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        for table in TRAJECTORY_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "trajectory_model"
                / table
            ).mkdir(parents=True, exist_ok=True)
        (self.store.run_dir / "trajectory_states").mkdir(
            parents=True, exist_ok=True
        )
        try:
            for sample_index, sample in enumerate(samples):
                self._run_trajectory_sample(sample, sample_index)
            outputs = self._consolidate_trajectory_tables()
            self.store.status["state"] = (
                "trajectory_complete_analysis_pending"
            )
            self.store.status["trajectory_outputs"] = {
                key: str(value) for key, value in outputs.items()
            }
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _fragment_path(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "trajectory_model"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _write_sample_tables(
        self,
        sample_id: str,
        inventory: Sequence[Mapping[str, Any]],
        state_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        _atomic_frame(
            pd.DataFrame(inventory),
            self._fragment_path(
                "trajectory_intervention_inventory", sample_id
            ),
        )
        _atomic_frame(
            pd.DataFrame(state_rows),
            self._fragment_path("trajectory_state_rows", sample_id),
        )

    def _consolidate_trajectory_tables(self) -> Dict[str, Path]:
        output: Dict[str, Path] = {}
        for table in TRAJECTORY_TABLES:
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "trajectory_model"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in paths]
            frame = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
            path = self.store.run_dir / ("%s.parquet" % table)
            _atomic_frame(frame, path)
            output[table] = path
        return output

    def _stable_rng(
        self, sample_id: str, layer: int, anchor: int
    ) -> np.random.Generator:
        token = "%s:%d:%d:%d" % (
            sample_id,
            int(layer),
            int(anchor),
            int(self.cfg.trajectory_model.random_seed),
        )
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return np.random.default_rng(
            int.from_bytes(digest[:8], "little", signed=False)
        )

    @staticmethod
    def _record_queries(
        record: QueryRecord, layer: int, heads: Sequence[int]
    ) -> torch.Tensor:
        return torch.stack(
            [record.queries["%d:%d" % (layer, head)] for head in heads]
        ).float()

    @staticmethod
    def _record_keys(
        record: QueryRecord, layer: int, kv_heads: int
    ) -> torch.Tensor:
        return torch.stack(
            [record.new_keys["%d:%d" % (layer, head)] for head in range(kv_heads)]
        ).float()

    @staticmethod
    def _record_values(
        record: QueryRecord, layer: int, kv_heads: int
    ) -> torch.Tensor:
        return torch.stack(
            [record.new_values["%d:%d" % (layer, head)] for head in range(kv_heads)]
        ).float()

    def _score_bundle(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        layer: int,
    ) -> Dict[str, torch.Tensor]:
        state = reference.anchors[int(anchor)]
        record = reference.query_records[int(anchor)]
        attention = record.all_head_attention_distributions[int(layer)].float()
        values = state.values[int(layer)][0].float()
        query_heads = int(attention.shape[0])
        kv_heads = int(values.shape[0])
        group = query_heads // kv_heads
        repeated_values = values.repeat_interleave(group, dim=0)
        full_heads = record.all_head_attention_outputs[int(layer)].float()
        aov_features = attention[:, :, None] * repeated_values
        safe_denominator = (1.0 - attention).clamp_min(1e-6)
        aor_features = (
            attention[:, :, None]
            / safe_denominator[:, :, None]
            * (full_heads[:, None, :] - repeated_values)
        )
        aov_projected = self.model.project_features(
            int(layer),
            aov_features.permute(1, 0, 2).reshape(
                int(attention.shape[1]), -1
            ),
        )
        aor_projected = self.model.project_features(
            int(layer),
            aor_features.permute(1, 0, 2).reshape(
                int(attention.shape[1]), -1
            ),
        )
        v_scores = []
        for head in range(kv_heads):
            score, _ = ridge_leverage(
                values[head],
                self.cfg.selectors.ridge_lambda,
                self.cfg.selectors.ridge_lambda_mode,
            )
            v_scores.append(score)
        return {
            "attention": attention.mean(dim=0),
            "aov": aov_projected.square().sum(dim=1),
            "aor": aor_projected.square().sum(dim=1),
            "v_ridge": torch.stack(v_scores).mean(dim=0),
            "aov_projected": aov_projected,
            "aor_projected": aor_projected,
            "attention_by_head": attention,
        }

    def _cached_score_bundle(
        self,
        score_cache: Dict[
            Tuple[int, int], Dict[str, torch.Tensor]
        ],
        reference: ReferenceTrajectory,
        anchor: int,
        layer: int,
    ) -> Dict[str, torch.Tensor]:
        key = (int(anchor), int(layer))
        if key not in score_cache:
            score_cache[key] = self._score_bundle(
                reference, int(anchor), int(layer)
            )
        return score_cache[key]

    def _retained_positions(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        layer: int,
        mask_type: str,
        protected_recent: int,
        score_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    ) -> List[int]:
        state = reference.anchors[int(anchor)]
        positions = [
            int(value)
            for value in state.position_maps[int(layer)].tolist()
        ]
        sink, recent, eligible = mandatory_and_eligible(
            positions,
            int(self.cfg.cache.sink_size),
            int(protected_recent),
        )
        core_budget = (
            int(self.cfg.trajectory_model.total_budget)
            - len(set(sink + recent))
        )
        if core_budget <= 0:
            raise RuntimeError("trajectory mask leaves no selectable core")
        bundle = self._cached_score_bundle(
            score_cache, reference, int(anchor), int(layer)
        )
        row_by_position = {
            position: row for row, position in enumerate(positions)
        }
        if mask_type == "old_core":
            prior = max(
                [
                    value
                    for value in [0]
                    + list(self.cfg.trajectory_model.anchors)
                    if int(value) < int(anchor)
                ]
            )
            prior_positions = [
                int(value)
                for value in reference.anchors[prior].position_maps[
                    int(layer)
                ].tolist()
            ]
            _, _, prior_eligible = mandatory_and_eligible(
                prior_positions,
                int(self.cfg.cache.sink_size),
                int(protected_recent),
            )
            prior_bundle = self._cached_score_bundle(
                score_cache, reference, int(prior), int(layer)
            )
            prior_rows = {
                position: row
                for row, position in enumerate(prior_positions)
            }
            ordered_prior = sorted(
                prior_eligible,
                key=lambda position: (
                    -float(prior_bundle["aor"][prior_rows[position]].item()),
                    int(position),
                ),
            )
            selected = [
                position
                for position in ordered_prior[:core_budget]
                if position in set(eligible)
            ]
            fill_order = sorted(
                eligible,
                key=lambda position: (
                    -float(bundle["aor"][row_by_position[position]].item()),
                    int(position),
                ),
            )
            selected.extend(
                position
                for position in fill_order
                if position not in set(selected)
            )
            selected = selected[:core_budget]
        elif mask_type == "random":
            generator = self._stable_rng(
                reference.sample_id, int(layer), int(anchor)
            )
            selected = sorted(
                int(value)
                for value in generator.choice(
                    np.asarray(eligible, dtype=np.int64),
                    size=min(core_budget, len(eligible)),
                    replace=False,
                ).tolist()
            )
        else:
            score = bundle[str(mask_type)]
            selected = sorted(
                eligible,
                key=lambda position: (
                    -float(score[row_by_position[position]].item()),
                    int(position),
                ),
            )[:core_budget]
        retained = sorted(set(sink + recent + selected))
        validate_recent_budget(
            retained,
            int(self.cfg.trajectory_model.total_budget),
            int(self.cfg.cache.sink_size),
            int(protected_recent),
        )
        return retained

    def _direct_perturbation(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        layer: int,
        retained_positions: Sequence[int],
    ) -> Dict[str, Any]:
        state = reference.anchors[int(anchor)]
        record = reference.query_records[int(anchor)]
        positions = [
            int(value)
            for value in state.position_maps[int(layer)].tolist()
        ]
        row_by_position = {
            position: row for row, position in enumerate(positions)
        }
        retained_rows = torch.as_tensor(
            [row_by_position[int(position)] for position in retained_positions],
            dtype=torch.long,
        )
        attention = record.all_head_attention_distributions[int(layer)].float()
        values = state.values[int(layer)][0].float()
        query_heads = int(attention.shape[0])
        kv_heads = int(values.shape[0])
        group = query_heads // kv_heads
        repeated_values = values.repeat_interleave(group, dim=0)
        kept_attention = attention.index_select(1, retained_rows)
        denominator = kept_attention.sum(dim=1).clamp_min(1e-12)
        masked_heads = (
            kept_attention[:, :, None]
            * repeated_values.index_select(1, retained_rows)
        ).sum(dim=1) / denominator[:, None]
        full_heads = record.all_head_attention_outputs[int(layer)].float()
        head_delta = masked_heads - full_heads
        projected = self.model.project_features(
            int(layer), head_delta.reshape(1, -1)
        )[0]
        deleted_mask = torch.ones(
            int(attention.shape[1]), dtype=torch.bool
        )
        deleted_mask[retained_rows] = False
        deleted_mass_by_head = attention[:, deleted_mask].sum(dim=1)
        return {
            "projected": projected.detach().float().cpu(),
            "head_delta": head_delta.detach().float().cpu(),
            "deleted_attention_mass_mean": float(
                deleted_mass_by_head.mean().item()
            ),
            "deleted_attention_mass_max": float(
                deleted_mass_by_head.max().item()
            ),
            "deleted_positions": [
                positions[index]
                for index in torch.nonzero(
                    deleted_mask, as_tuple=False
                ).flatten().tolist()
            ],
            "retained_positions": [
                int(value) for value in retained_positions
            ],
        }

    def _deletion_perturbation(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        layer: int,
        deleted_positions: Sequence[int],
    ) -> Dict[str, Any]:
        positions = [
            int(value)
            for value in reference.anchors[int(anchor)].position_maps[
                int(layer)
            ].tolist()
        ]
        deleted = set(int(value) for value in deleted_positions)
        retained = [value for value in positions if value not in deleted]
        return self._direct_perturbation(
            reference, int(anchor), int(layer), retained
        )

    def _superposition_sets(
        self,
        reference: ReferenceTrajectory,
        anchor: int,
        category: str,
        score_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        primary_layer = 27 if category == "layer27_plus_other_layer" else 14
        state = reference.anchors[int(anchor)]
        positions = [
            int(value)
            for value in state.position_maps[int(primary_layer)].tolist()
        ]
        _, _, eligible = mandatory_and_eligible(
            positions, int(self.cfg.cache.sink_size), 0
        )
        row = {position: index for index, position in enumerate(positions)}
        bundle = self._cached_score_bundle(
            score_cache, reference, int(anchor), int(primary_layer)
        )
        mass_order = sorted(
            eligible,
            key=lambda position: (
                -float(bundle["attention"][row[position]].item()),
                int(position),
            ),
        )
        width = min(4, max(1, len(mass_order) // 8))
        if category == "small_mass_plus_small_mass":
            d1 = mass_order[-width:]
            d2 = mass_order[-2 * width : -width]
        elif category == "large_mass_plus_small_mass":
            d1 = mass_order[:width]
            d2 = mass_order[-width:]
        elif category in {
            "same_direction_residual",
            "opposite_direction_residual",
        }:
            candidates = mass_order[: min(64, len(mass_order))]
            seed = candidates[0]
            features = bundle["aor_projected"]
            seed_vector = features[row[seed]]
            seed_norm = max(float(seed_vector.norm().item()), 1e-12)
            similarities = []
            for position in candidates[1:]:
                vector = features[row[position]]
                cosine = float(
                    torch.dot(seed_vector, vector).item()
                    / max(
                        seed_norm * float(vector.norm().item()), 1e-12
                    )
                )
                similarities.append((cosine, position))
            similarities.sort()
            paired = (
                similarities[-1][1]
                if category == "same_direction_residual"
                else similarities[0][1]
            )
            d1, d2 = [seed], [paired]
        elif category == "shared_kv_head_query_heads":
            attention = bundle["attention_by_head"]
            head0 = sorted(
                eligible,
                key=lambda position: (
                    -float(attention[0, row[position]].item()),
                    int(position),
                ),
            )
            head1 = sorted(
                eligible,
                key=lambda position: (
                    -float(attention[1, row[position]].item()),
                    int(position),
                ),
            )
            d1 = head0[:width]
            d2 = [
                position for position in head1 if position not in set(d1)
            ][:width]
        elif category == "layer27_plus_other_layer":
            other_layer = 14
            other_positions = [
                int(value)
                for value in state.position_maps[other_layer].tolist()
            ]
            _, _, other_eligible = mandatory_and_eligible(
                other_positions, int(self.cfg.cache.sink_size), 0
            )
            other_rows = {
                position: index
                for index, position in enumerate(other_positions)
            }
            other_bundle = self._cached_score_bundle(
                score_cache, reference, int(anchor), other_layer
            )
            d1 = mass_order[:width]
            d2_other = sorted(
                other_eligible,
                key=lambda position: (
                    -float(
                        other_bundle["attention"][
                            other_rows[position]
                        ].item()
                    ),
                    int(position),
                ),
            )[:width]
            return {27: d1}, {14: d2_other}
        else:
            raise ValueError("unknown superposition category: %s" % category)
        if set(d1) & set(d2):
            raise RuntimeError("superposition deletion sets are not disjoint")
        return {primary_layer: list(d1)}, {primary_layer: list(d2)}

    def _inventory_base(
        self,
        sample: Any,
        trajectory_id: str,
        intervention_type: str,
        anchor: int,
        protected_recent: int,
        mask_type: str,
        beta: float,
        layers: Sequence[int],
        source_labels: Mapping[str, str],
    ) -> Dict[str, Any]:
        return {
            **self._base(sample),
            "trajectory_id": trajectory_id,
            "intervention_type": intervention_type,
            "anchor": int(anchor),
            "horizon": int(self.cfg.trajectory_model.horizon),
            "protected_recent_size": int(protected_recent),
            "total_budget": int(self.cfg.trajectory_model.total_budget),
            "mask_type": str(mask_type),
            "beta": float(beta),
            "injection_layers": json_text([int(value) for value in layers]),
            "source_labels": json_text(dict(source_labels)),
            "teacher_forced": True,
            "full_reference_immutable": True,
            "token_alignment_contract": (
                "query_records[t] predicts generated_token_ids[t]"
            ),
        }

    def _build_plans(
        self,
        sample: Any,
        sample_index: int,
        reference: ReferenceTrajectory,
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg.trajectory_model
        anchors = [int(value) for value in cfg.anchors]
        masks = [str(value) for value in cfg.mask_types]
        recent_values = [int(value) for value in cfg.protected_recent_sizes]
        score_cache: Dict[
            Tuple[int, int], Dict[str, torch.Tensor]
        ] = {}
        plans: List[Dict[str, Any]] = []

        def append_plan(
            intervention_type: str,
            anchor: int,
            recent: int,
            mask: str,
            beta: float,
            injection_map: Mapping[int, torch.Tensor],
            direct_metadata: Mapping[str, Any],
            source_labels: Mapping[str, str],
            extra: Optional[Mapping[str, Any]] = None,
        ) -> None:
            trajectory_id = stable_trajectory_id(
                sample.sample_id,
                intervention_type,
                anchor,
                mask,
                beta,
                len(plans),
            )
            layers = sorted(int(value) for value in injection_map)
            row = self._inventory_base(
                sample,
                trajectory_id,
                intervention_type,
                anchor,
                recent,
                mask,
                beta,
                layers,
                source_labels,
            )
            row.update(
                {
                    "direct_input_l2": float(
                        math.sqrt(
                            sum(
                                float(value.float().square().sum().item())
                                for value in injection_map.values()
                            )
                        )
                    ),
                    "deleted_attention_mass_mean": float(
                        direct_metadata.get(
                            "deleted_attention_mass_mean", float("nan")
                        )
                    ),
                    "deleted_attention_mass_max": float(
                        direct_metadata.get(
                            "deleted_attention_mass_max", float("nan")
                        )
                    ),
                    "retained_positions": json_text(
                        direct_metadata.get("retained_positions", {})
                    ),
                    "deleted_positions": json_text(
                        direct_metadata.get("deleted_positions", {})
                    ),
                }
            )
            if extra:
                row.update(dict(extra))
            plans.append(
                {
                    "trajectory_id": trajectory_id,
                    "inventory": row,
                    "anchor": int(anchor),
                    "beta": float(beta),
                    "injection_map": {
                        int(layer): value.detach().float().cpu()
                        for layer, value in injection_map.items()
                    },
                    "source_labels": dict(source_labels),
                    "intervention_type": intervention_type,
                }
            )

        for unit_index, layer in enumerate(REQUIRED_SCALING_LAYERS):
            anchor = anchors[unit_index % len(anchors)]
            recent = recent_values[(sample_index + unit_index) % len(recent_values)]
            mask = masks[(sample_index + unit_index) % len(masks)]
            retained = self._retained_positions(
                reference,
                anchor,
                layer,
                mask,
                recent,
                score_cache,
            )
            direct = self._direct_perturbation(
                reference, anchor, layer, retained
            )
            for beta in cfg.betas:
                append_plan(
                    "scaling_single",
                    anchor,
                    recent,
                    mask,
                    float(beta),
                    {layer: float(beta) * direct["projected"]},
                    direct,
                    {
                        "anchor_history": "full_reference",
                        "attention_branch_input": "full_reference",
                        "projected_injection": "exact_masked_minus_full",
                        "future_tokens": "full_reference_teacher_forced",
                    },
                    {"unit_key": "single_l%d_a%d" % (layer, anchor)},
                )

        multi_anchor = anchors[sample_index % len(anchors)]
        multi_recent = recent_values[sample_index % len(recent_values)]
        multi_mask = masks[(sample_index + 5) % len(masks)]
        multi_direct: Dict[int, Dict[str, Any]] = {}
        for layer in (0, 14, 27):
            retained = self._retained_positions(
                reference,
                multi_anchor,
                layer,
                multi_mask,
                multi_recent,
                score_cache,
            )
            multi_direct[layer] = self._direct_perturbation(
                reference, multi_anchor, layer, retained
            )
        multi_metadata = {
            "deleted_attention_mass_mean": float(
                np.mean(
                    [
                        value["deleted_attention_mass_mean"]
                        for value in multi_direct.values()
                    ]
                )
            ),
            "deleted_attention_mass_max": float(
                max(
                    value["deleted_attention_mass_max"]
                    for value in multi_direct.values()
                )
            ),
            "retained_positions": {
                str(layer): value["retained_positions"]
                for layer, value in multi_direct.items()
            },
            "deleted_positions": {
                str(layer): value["deleted_positions"]
                for layer, value in multi_direct.items()
            },
        }
        for beta in cfg.betas:
            append_plan(
                "scaling_multi",
                multi_anchor,
                multi_recent,
                multi_mask,
                float(beta),
                {
                    layer: float(beta) * value["projected"]
                    for layer, value in multi_direct.items()
                },
                multi_metadata,
                {
                    "anchor_history": "full_reference",
                    "attention_branch_input": "full_reference",
                    "projected_injection": (
                        "simultaneous_exact_masked_minus_full"
                    ),
                    "future_tokens": "full_reference_teacher_forced",
                },
                {"unit_key": "multi_a%d" % multi_anchor},
            )

        category = SUPERPOSITION_CATEGORIES[
            sample_index % len(SUPERPOSITION_CATEGORIES)
        ]
        super_anchor = anchors[(sample_index + 2) % len(anchors)]
        first_sets, second_sets = self._superposition_sets(
            reference, super_anchor, category, score_cache
        )
        first_map = {
            layer: self._deletion_perturbation(
                reference, super_anchor, layer, deleted
            )
            for layer, deleted in first_sets.items()
        }
        second_map = {
            layer: self._deletion_perturbation(
                reference, super_anchor, layer, deleted
            )
            for layer, deleted in second_sets.items()
        }
        union_sets: Dict[int, List[int]] = {}
        for layer in set(first_sets) | set(second_sets):
            union_sets[layer] = sorted(
                set(first_sets.get(layer, []))
                | set(second_sets.get(layer, []))
            )
        union_map = {
            layer: self._deletion_perturbation(
                reference, super_anchor, layer, deleted
            )
            for layer, deleted in union_sets.items()
        }
        super_maps = {
            "D1": first_map,
            "D2": second_map,
            "union": union_map,
        }
        additive_map = {
            layer: (
                first_map.get(layer, {}).get(
                    "projected",
                    torch.zeros_like(next(iter(union_map.values()))["projected"]),
                )
                + second_map.get(layer, {}).get(
                    "projected",
                    torch.zeros_like(next(iter(union_map.values()))["projected"]),
                )
            )
            for layer in union_map
        }
        direct_cross_error = math.sqrt(
            sum(
                float(
                    (
                        union_map[layer]["projected"]
                        - additive_map[layer]
                    )
                    .square()
                    .sum()
                    .item()
                )
                for layer in union_map
            )
        ) / max(
            math.sqrt(
                sum(
                    float(
                        union_map[layer]["projected"]
                        .square()
                        .sum()
                        .item()
                    )
                    for layer in union_map
                )
            ),
            1e-12,
        )
        for arm, direct_map in super_maps.items():
            metadata = {
                "deleted_attention_mass_mean": float(
                    np.mean(
                        [
                            value["deleted_attention_mass_mean"]
                            for value in direct_map.values()
                        ]
                    )
                ),
                "deleted_attention_mass_max": float(
                    max(
                        value["deleted_attention_mass_max"]
                        for value in direct_map.values()
                    )
                ),
                "retained_positions": {
                    str(layer): value["retained_positions"]
                    for layer, value in direct_map.items()
                },
                "deleted_positions": {
                    str(layer): value["deleted_positions"]
                    for layer, value in direct_map.items()
                },
            }
            append_plan(
                "superposition",
                super_anchor,
                0,
                "disjoint_deletion",
                1.0,
                {
                    layer: value["projected"]
                    for layer, value in direct_map.items()
                },
                metadata,
                {
                    "anchor_history": "full_reference",
                    "projected_injection": "exact_deletion_mask",
                    "future_tokens": "full_reference_teacher_forced",
                },
                {
                    "unit_key": "super_%s_a%d"
                    % (category, super_anchor),
                    "superposition_category": category,
                    "superposition_arm": arm,
                    "deletion_set_1": json_text(first_sets),
                    "deletion_set_2": json_text(second_sets),
                    "direct_cross_interaction_relative": float(
                        direct_cross_error
                    ),
                },
            )

        hybrid_anchor = 32
        hybrid_layer = 14
        hybrid_recent = recent_values[sample_index % len(recent_values)]
        retained = self._retained_positions(
            reference,
            hybrid_anchor,
            hybrid_layer,
            "old_core",
            hybrid_recent,
            score_cache,
        )
        hybrid_direct = self._direct_perturbation(
            reference, hybrid_anchor, hybrid_layer, retained
        )
        hybrid_arms = (
            "stateful",
            "query_restore",
            "new_kv_restore",
            "attention_input_restore",
            "next_layer_hidden_restore",
        )
        for arm in hybrid_arms:
            labels = {
                "anchor_history": "full_reference",
                "future_tokens": "full_reference_teacher_forced",
                "query": (
                    "full_reference_restored"
                    if arm == "query_restore"
                    else "trajectory"
                ),
                "new_key_value": (
                    "full_reference_restored"
                    if arm == "new_kv_restore"
                    else "trajectory"
                ),
                "attention_input": (
                    "full_reference_restored"
                    if arm == "attention_input_restore"
                    else "trajectory"
                ),
                "next_layer_hidden": (
                    "full_reference_restored"
                    if arm == "next_layer_hidden_restore"
                    else "trajectory"
                ),
            }
            append_plan(
                "hybrid_%s" % arm,
                hybrid_anchor,
                hybrid_recent,
                "old_core",
                1.0,
                {hybrid_layer: hybrid_direct["projected"]},
                hybrid_direct,
                labels,
                {
                    "unit_key": "hybrid_a%d_l%d"
                    % (hybrid_anchor, hybrid_layer),
                    "hybrid_arm": arm,
                    "hybrid_layer": hybrid_layer,
                    "hidden_restore_layer": hybrid_layer + 1,
                },
            )
        return plans

    @staticmethod
    def _all_history_selection(
        reference: ReferenceTrajectory, anchor: int
    ) -> CoreSelection:
        by_layer: Dict[int, LayerSelection] = {}
        state = reference.anchors[int(anchor)]
        for layer, position_map in state.position_maps.items():
            positions = [int(value) for value in position_map.tolist()]
            by_layer[int(layer)] = LayerSelection(
                layer=int(layer),
                selected_positions=positions,
                eligible_positions=positions,
                aggregate_scores=[1.0] * len(positions),
                metadata={"source": "full_reference_history"},
            )
        return CoreSelection(
            strategy="full_reference_history",
            horizon_condition=None,
            by_layer=by_layer,
            metadata={"compressed": False},
        )

    def _clear_controls(self) -> None:
        for key in CONTROL_KEYS:
            self.model.runner.attention_state[key] = {}

    def _apply_hybrid_controls(
        self,
        plan: Mapping[str, Any],
        reference_record: QueryRecord,
    ) -> None:
        state = self.model.runner.attention_state
        arm = str(plan["inventory"].get("hybrid_arm", ""))
        layer = int(plan["inventory"].get("hybrid_layer", 14))
        heads = self.model.selected_heads[layer]
        kv_heads = int(self.model.model_info["num_key_value_heads"])
        if arm == "query_restore":
            state["temporal_query_overrides"] = {
                layer: self._record_queries(
                    reference_record, layer, heads
                ).numpy()
            }
        elif arm == "new_kv_restore":
            state["temporal_new_key_overrides"] = {
                layer: self._record_keys(
                    reference_record, layer, kv_heads
                ).numpy()
            }
            state["temporal_new_value_overrides"] = {
                layer: self._record_values(
                    reference_record, layer, kv_heads
                ).numpy()
            }
        elif arm == "attention_input_restore":
            state["temporal_attention_input_overrides"] = {
                layer: reference_record.attention_inputs[
                    layer
                ].float().numpy()
            }
        elif arm == "next_layer_hidden_restore":
            restore_layer = int(
                plan["inventory"]["hidden_restore_layer"]
            )
            state["temporal_layer_input_overrides"] = {
                restore_layer: reference_record.residual_inputs[
                    restore_layer
                ].float().numpy()
            }

    def _save_state_npz(
        self, trajectory_id: str, arrays: Mapping[str, np.ndarray]
    ) -> Path:
        path = (
            self.store.run_dir
            / "trajectory_states"
            / ("%s.npz" % trajectory_id)
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
        )
        os.close(descriptor)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def _replay_plan(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
        plan: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        anchor = int(plan["anchor"])
        horizon = int(self.cfg.trajectory_model.horizon)
        selected_layers = [
            int(value) for value in self.model.selected_layers
        ]
        heads_by_layer = self.model.selected_heads
        kv_heads = int(self.model.model_info["num_key_value_heads"])
        hidden = int(self.model.model_info["hidden_size"])
        head_dim = int(
            self.model.model_info.get("head_dim")
            or hidden // int(self.model.model_info["num_attention_heads"])
        )
        qdim = int(self.model.model_info["num_attention_heads"]) * head_dim
        kvdim = kv_heads * head_dim
        dtype = np.dtype(self.cfg.trajectory_model.state_storage_dtype)
        shape_hidden = (horizon, len(selected_layers), hidden)
        shape_query = (horizon, len(selected_layers), qdim)
        shape_kv = (horizon, len(selected_layers), kvdim)
        arrays: Dict[str, np.ndarray] = {
            "residual_drift": np.empty(shape_hidden, dtype=dtype),
            "attention_input_drift": np.empty(shape_hidden, dtype=dtype),
            "query_drift": np.empty(shape_query, dtype=dtype),
            "new_key_drift": np.empty(shape_kv, dtype=dtype),
            "new_value_drift": np.empty(shape_kv, dtype=dtype),
            "attention_output_drift": np.empty(shape_query, dtype=dtype),
            "projected_attention_output_drift": np.empty(
                shape_hidden, dtype=dtype
            ),
            "layer_output_drift": np.empty(shape_hidden, dtype=dtype),
            "logit_top128_drift": np.empty((horizon, 129), dtype=dtype),
            "logit_top128_ids": np.empty((horizon, 129), dtype=np.int32),
            "direct_input": np.zeros(
                (len(selected_layers), hidden), dtype=dtype
            ),
            "selected_layers": np.asarray(
                selected_layers, dtype=np.int16
            ),
        }
        for layer_index, layer in enumerate(selected_layers):
            if layer in plan["injection_map"]:
                arrays["direct_input"][layer_index] = (
                    plan["injection_map"][layer].numpy().astype(dtype)
                )

        # The frozen anchor cache is serialized through CPU float16.  A
        # reconstructed no-op replay therefore contains round-trip noise and
        # cannot serve as an exact zero control.  beta=0 is the immutable
        # reference trajectory itself; represent it as an exact zero drift
        # alias and record that source explicitly.
        if float(plan["beta"]) == 0.0:
            for key in (
                "residual_drift",
                "attention_input_drift",
                "query_drift",
                "new_key_drift",
                "new_value_drift",
                "attention_output_drift",
                "projected_attention_output_drift",
                "layer_output_drift",
                "logit_top128_drift",
            ):
                arrays[key].fill(0)
            rows: List[Dict[str, Any]] = []
            for offset in range(1, horizon + 1):
                target_index = anchor + offset - 1
                full_logits = reference.probe_logits[target_index].float()
                target_token = int(
                    reference.generated_token_ids[target_index]
                )
                top_ids = torch.topk(
                    full_logits, k=min(128, int(full_logits.numel()))
                ).indices
                if int(target_token) not in set(top_ids.tolist()):
                    ids = torch.cat(
                        [top_ids, torch.tensor([target_token])]
                    )
                else:
                    ids = torch.cat([top_ids, top_ids[-1:]])
                arrays["logit_top128_ids"][offset - 1] = ids[:129].numpy()
                full_nll = -float(
                    torch.log_softmax(full_logits, dim=-1)[
                        target_token
                    ].item()
                )
                for layer in selected_layers:
                    recent = int(
                        plan["inventory"]["protected_recent_size"]
                    )
                    rows.append(
                        {
                            **{
                                key: value
                                for key, value in plan[
                                    "inventory"
                                ].items()
                                if key
                                not in {
                                    "retained_positions",
                                    "deleted_positions",
                                }
                            },
                            "state_npz_path": "",
                            "horizon_offset": int(offset),
                            "target_index": int(target_index),
                            "target_token_id": int(target_token),
                            "query_position_full": int(
                                reference.query_records[
                                    target_index
                                ].query_position
                            ),
                            "query_position_perturbed": int(
                                reference.query_records[
                                    target_index
                                ].query_position
                            ),
                            "token_position_aligned": True,
                            "layer": int(layer),
                            "layer_regime": layer_regime(layer),
                            "recent_exit_event": bool(
                                recent > 0 and offset == recent
                            ),
                            "post_recent_exit": bool(
                                recent > 0 and offset >= recent
                            ),
                            "residual_l2": 0.0,
                            "attention_input_l2": 0.0,
                            "query_l2": 0.0,
                            "new_key_l2": 0.0,
                            "new_value_l2": 0.0,
                            "attention_output_l2": 0.0,
                            "projected_attention_output_l2": 0.0,
                            "layer_output_l2": 0.0,
                            "exact_kl": 0.0,
                            "js": 0.0,
                            "full_nll": full_nll,
                            "perturbed_nll": full_nll,
                            "delta_nll": 0.0,
                            "logit_l2_sq": 0.0,
                            "fisher_quadratic": 0.0,
                            "forward_time_s": 0.0,
                            "beta0_source": (
                                "immutable_full_reference_exact_alias"
                            ),
                        }
                    )
            state_path = self._save_state_npz(
                str(plan["trajectory_id"]), arrays
            )
            for row in rows:
                row["state_npz_path"] = str(state_path)
            inventory = dict(plan["inventory"])
            inventory.update(
                {
                    "state_npz_path": str(state_path),
                    "trajectory_forward_time_s": 0.0,
                    "beta0_max_abs_drift": 0.0,
                    "beta1_injection_l2_error": float("nan"),
                    "rows_saved": int(len(rows)),
                    "beta0_source": (
                        "immutable_full_reference_exact_alias"
                    ),
                }
            )
            return inventory, rows

        cache_mode = str(plan.get("cache_mode", "full_pulse_replay"))
        if cache_mode == "compressed_recent_fifo":
            full_selection = plan["compressed_selection"]
            full_cache = plan["compressed_cache_config"]
        else:
            full_selection = self._all_history_selection(reference, anchor)
            full_cache = CacheDiscoveryConfig(
                total_budget=int(
                    reference.anchors[anchor].logical_length + horizon + 4
                ),
                sink_size=0,
                recent_size=1,
                selected_core_budget=int(
                    reference.anchors[anchor].logical_length + horizon + 3
                ),
            )
        state, fixed = self.model.state_from_anchor(
            reference.anchors[anchor],
            full_selection,
            cache_config=full_cache,
        )
        current_token = int(reference.anchors[anchor].query_token_id)
        rows: List[Dict[str, Any]] = []
        maximum_beta0_drift = 0.0
        beta1_injection_error = float("nan")
        started = time.perf_counter()
        try:
            for offset in range(1, horizon + 1):
                target_index = anchor + offset - 1
                if target_index >= len(reference.generated_token_ids):
                    raise RuntimeError(
                        "reference ended before trajectory horizon"
                    )
                reference_record = reference.query_records[target_index]
                full_logits = reference.probe_logits[target_index].float()
                target_token = int(
                    reference.generated_token_ids[target_index]
                )
                self._clear_controls()
                if cache_mode == "compressed_recent_fifo" and offset > 1:
                    self.model.prune_recent_before_query(
                        state, fixed, cache_config=full_cache
                    )
                if offset == 1:
                    self.model.runner.attention_state[
                        "temporal_projected_injections"
                    ] = {
                        int(layer): value.numpy()
                        for layer, value in plan.get(
                            "control_injection_map",
                            plan["injection_map"],
                        ).items()
                    }
                if str(plan["intervention_type"]).startswith("hybrid_"):
                    self._apply_hybrid_controls(plan, reference_record)
                logits, record, forward_s = self.model.forward_one(
                    state, current_token, capture_attention=True
                )
                if cache_mode == "compressed_recent_fifo":
                    self.model.validate_active_budget(
                        state, cache_config=full_cache
                    )
                metrics = exact_distribution_metrics(
                    full_logits, logits, target_token
                )
                top_ids = torch.topk(
                    full_logits, k=min(128, int(full_logits.numel()))
                ).indices
                if int(target_token) not in set(top_ids.tolist()):
                    ids = torch.cat(
                        [top_ids, torch.tensor([target_token])]
                    )
                else:
                    ids = torch.cat([top_ids, top_ids[-1:]])
                arrays["logit_top128_ids"][offset - 1] = ids[:129].numpy()
                arrays["logit_top128_drift"][offset - 1] = (
                    (logits[ids[:129]] - full_logits[ids[:129]])
                    .numpy()
                    .astype(dtype)
                )

                layer_norms: Dict[str, List[float]] = {
                    name: []
                    for name in (
                        "residual",
                        "attention_input",
                        "query",
                        "new_key",
                        "new_value",
                        "attention_output",
                        "projected_attention_output",
                        "layer_output",
                    )
                }
                for layer_index, layer in enumerate(selected_layers):
                    heads = heads_by_layer[layer]
                    differences = {
                        "residual": (
                            record.residual_inputs[layer].float()
                            - reference_record.residual_inputs[layer].float()
                        ),
                        "attention_input": (
                            record.attention_inputs[layer].float()
                            - reference_record.attention_inputs[layer].float()
                        ),
                        "query": (
                            self._record_queries(record, layer, heads)
                            - self._record_queries(
                                reference_record, layer, heads
                            )
                        ).reshape(-1),
                        "new_key": (
                            self._record_keys(record, layer, kv_heads)
                            - self._record_keys(
                                reference_record, layer, kv_heads
                            )
                        ).reshape(-1),
                        "new_value": (
                            self._record_values(record, layer, kv_heads)
                            - self._record_values(
                                reference_record, layer, kv_heads
                            )
                        ).reshape(-1),
                        "attention_output": (
                            record.all_head_attention_outputs[
                                layer
                            ].float()
                            - reference_record.all_head_attention_outputs[
                                layer
                            ].float()
                        ).reshape(-1),
                        "projected_attention_output": (
                            record.projected_attention_outputs[
                                layer
                            ].float()
                            - reference_record.projected_attention_outputs[
                                layer
                            ].float()
                        ),
                        "layer_output": (
                            record.layer_outputs[layer].float()
                            - reference_record.layer_outputs[layer].float()
                        ),
                    }
                    array_names = {
                        "residual": "residual_drift",
                        "attention_input": "attention_input_drift",
                        "query": "query_drift",
                        "new_key": "new_key_drift",
                        "new_value": "new_value_drift",
                        "attention_output": "attention_output_drift",
                        "projected_attention_output": (
                            "projected_attention_output_drift"
                        ),
                        "layer_output": "layer_output_drift",
                    }
                    for name, difference in differences.items():
                        values = difference.detach().float().cpu().reshape(-1)
                        arrays[array_names[name]][
                            offset - 1, layer_index
                        ] = values.numpy().astype(dtype)
                        norm = float(values.norm().item())
                        layer_norms[name].append(norm)
                    exit_offset = (
                        int(
                            plan["inventory"][
                                "protected_recent_size"
                            ]
                        )
                        + 1
                        if cache_mode == "compressed_recent_fifo"
                        else int(
                            plan["inventory"][
                                "protected_recent_size"
                            ]
                        )
                    )
                    recent_exit = bool(
                        int(plan["inventory"][
                            "protected_recent_size"
                        ])
                        > 0
                        and offset == exit_offset
                    )
                    rows.append(
                        {
                            **{
                                key: value
                                for key, value in plan[
                                    "inventory"
                                ].items()
                                if key
                                not in {
                                    "retained_positions",
                                    "deleted_positions",
                                }
                            },
                            "state_npz_path": "",
                            "horizon_offset": int(offset),
                            "target_index": int(target_index),
                            "target_token_id": int(target_token),
                            "query_position_full": int(
                                reference_record.query_position
                            ),
                            "query_position_perturbed": int(
                                record.query_position
                            ),
                            "token_position_aligned": bool(
                                reference_record.query_position
                                == record.query_position
                            ),
                            "layer": int(layer),
                            "layer_regime": layer_regime(layer),
                            "recent_exit_event": recent_exit,
                            "post_recent_exit": bool(
                                int(
                                    plan["inventory"][
                                        "protected_recent_size"
                                    ]
                                )
                                > 0
                                and offset >= exit_offset
                            ),
                            "residual_l2": layer_norms["residual"][-1],
                            "attention_input_l2": layer_norms[
                                "attention_input"
                            ][-1],
                            "query_l2": layer_norms["query"][-1],
                            "new_key_l2": layer_norms["new_key"][-1],
                            "new_value_l2": layer_norms["new_value"][-1],
                            "attention_output_l2": layer_norms[
                                "attention_output"
                            ][-1],
                            "projected_attention_output_l2": layer_norms[
                                "projected_attention_output"
                            ][-1],
                            "layer_output_l2": layer_norms[
                                "layer_output"
                            ][-1],
                            "exact_kl": metrics["exact_kl"],
                            "js": metrics["js"],
                            "full_nll": metrics["full_nll"],
                            "perturbed_nll": metrics[
                                "perturbed_nll"
                            ],
                            "delta_nll": metrics["delta_nll"],
                            "logit_l2_sq": metrics["logit_l2_sq"],
                            "fisher_quadratic": metrics[
                                "fisher_quadratic"
                            ],
                            "forward_time_s": float(forward_s),
                        }
                    )
                aggregate_norm = math.sqrt(
                    sum(value * value for value in layer_norms["layer_output"])
                )
                if float(plan["beta"]) == 0.0:
                    maximum_beta0_drift = max(
                        maximum_beta0_drift,
                        aggregate_norm,
                        math.sqrt(max(metrics["logit_l2_sq"], 0.0)),
                    )
                if (
                    offset == 1
                    and float(plan["beta"]) == 1.0
                    and cache_mode == "full_pulse_replay"
                ):
                    errors = []
                    for layer, injection in plan["injection_map"].items():
                        actual = (
                            record.projected_attention_outputs[layer].float()
                            - reference_record.projected_attention_outputs[
                                layer
                            ].float()
                        )
                        errors.append(float((actual - injection).norm().item()))
                    beta1_injection_error = max(errors, default=0.0)
                current_token = target_token
        finally:
            self._clear_controls()
            self.model.release(state)
        state_path = self._save_state_npz(
            str(plan["trajectory_id"]), arrays
        )
        for row in rows:
            row["state_npz_path"] = str(state_path)
        inventory = dict(plan["inventory"])
        inventory.update(
            {
                "state_npz_path": str(state_path),
                "trajectory_forward_time_s": float(
                    time.perf_counter() - started
                ),
                "beta0_max_abs_drift": float(maximum_beta0_drift),
                "beta1_injection_l2_error": float(beta1_injection_error),
                "rows_saved": int(len(rows)),
            }
        )
        return inventory, rows

    def run_recent_exit_extension(self) -> Path:
        """Append one real recent-FIFO exit trajectory per sequence."""

        samples, _ = load_discovery_tasks(self.cfg)
        self.model.load()
        inventory_path = (
            self.store.run_dir
            / "trajectory_intervention_inventory.parquet"
        )
        states_path = (
            self.store.run_dir / "trajectory_state_rows.parquet"
        )
        existing_inventory = pd.read_parquet(inventory_path)
        existing_states = pd.read_parquet(states_path)
        new_inventory: List[Dict[str, Any]] = []
        new_states: List[Dict[str, Any]] = []
        try:
            for sample in samples:
                if (
                    (existing_inventory["sample_id"] == sample.sample_id)
                    & (
                        existing_inventory["intervention_type"]
                        == "recent_exit_stateful"
                    )
                ).any():
                    continue
                reference = self.model.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                anchor = 32
                recent = 32
                cache_cfg = _condition_cache(
                    self.cfg,
                    int(self.cfg.trajectory_model.total_budget),
                    recent,
                )
                selector = self.selector.__class__(
                    replace(self.cfg, cache=cache_cfg)
                )
                selection = selector.select(
                    reference.anchors[anchor].snapshot(
                        reference.sample_id
                    ),
                    "v_ridge_leverage",
                )
                layer = 14
                positions = [
                    int(value)
                    for value in reference.anchors[
                        anchor
                    ].position_maps[layer].tolist()
                ]
                sink = positions[: int(cache_cfg.sink_size)]
                recent_positions = positions[
                    -int(cache_cfg.recent_size) :
                ]
                retained = sorted(
                    set(
                        sink
                        + recent_positions
                        + selection.by_layer[layer].selected_positions
                    )
                )
                direct = self._direct_perturbation(
                    reference, anchor, layer, retained
                )
                trajectory_id = stable_trajectory_id(
                    sample.sample_id,
                    "recent_exit_stateful",
                    anchor,
                    "v_ridge",
                    1.0,
                )
                labels = {
                    "anchor_history": "compressed_v_ridge_core",
                    "recent_policy": "real_fifo_prune",
                    "future_tokens": "full_reference_teacher_forced",
                    "query": "trajectory",
                    "new_key_value": "trajectory",
                }
                row = self._inventory_base(
                    sample,
                    trajectory_id,
                    "recent_exit_stateful",
                    anchor,
                    recent,
                    "v_ridge",
                    1.0,
                    [layer],
                    labels,
                )
                row.update(
                    {
                        "unit_key": "recent_exit_a32_r32",
                        "direct_input_l2": float(
                            direct["projected"].norm().item()
                        ),
                        "deleted_attention_mass_mean": direct[
                            "deleted_attention_mass_mean"
                        ],
                        "deleted_attention_mass_max": direct[
                            "deleted_attention_mass_max"
                        ],
                        "retained_positions": json_text(
                            {str(layer): retained}
                        ),
                        "deleted_positions": json_text(
                            {str(layer): direct["deleted_positions"]}
                        ),
                        "cache_mode": "compressed_recent_fifo",
                        "actual_exit_offset": recent + 1,
                    }
                )
                plan = {
                    "trajectory_id": trajectory_id,
                    "inventory": row,
                    "anchor": anchor,
                    "beta": 1.0,
                    "injection_map": {
                        layer: direct["projected"]
                    },
                    "control_injection_map": {},
                    "source_labels": labels,
                    "intervention_type": "recent_exit_stateful",
                    "cache_mode": "compressed_recent_fifo",
                    "compressed_selection": selection,
                    "compressed_cache_config": cache_cfg,
                }
                inventory_row, state_rows = self._replay_plan(
                    sample, reference, plan
                )
                new_inventory.append(inventory_row)
                new_states.extend(state_rows)
                fragment_inventory = self._fragment_path(
                    "trajectory_intervention_inventory",
                    sample.sample_id,
                )
                fragment_states = self._fragment_path(
                    "trajectory_state_rows", sample.sample_id
                )
                _atomic_frame(
                    pd.concat(
                        [
                            pd.read_parquet(fragment_inventory),
                            pd.DataFrame([inventory_row]),
                        ],
                        ignore_index=True,
                        sort=False,
                    ),
                    fragment_inventory,
                )
                _atomic_frame(
                    pd.concat(
                        [
                            pd.read_parquet(fragment_states),
                            pd.DataFrame(state_rows),
                        ],
                        ignore_index=True,
                        sort=False,
                    ),
                    fragment_states,
                )
                self.model.release(reference)
        finally:
            self.model.close()
        if new_inventory:
            _atomic_frame(
                pd.concat(
                    [
                        existing_inventory,
                        pd.DataFrame(new_inventory),
                    ],
                    ignore_index=True,
                    sort=False,
                ),
                inventory_path,
            )
            _atomic_frame(
                pd.concat(
                    [existing_states, pd.DataFrame(new_states)],
                    ignore_index=True,
                    sort=False,
                ),
                states_path,
            )
        self.store.status["recent_exit_extension"] = {
            "complete": True,
            "trajectories": len(new_inventory),
            "state_rows": len(new_states),
            "actual_exit_offset": 33,
        }
        self.store.save_status()
        return self.store.run_dir

    def _run_trajectory_sample(self, sample: Any, sample_index: int) -> None:
        key = "trajectory_model:%s" % _sample_slug(sample.sample_id)
        if self.cfg.runtime.resume and self.store.is_complete(key):
            return
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            plans = self._build_plans(
                sample, sample_index, reference
            )
            inventory: List[Dict[str, Any]] = []
            state_rows: List[Dict[str, Any]] = []
            for plan_index, plan in enumerate(plans):
                row, trajectory_rows = self._replay_plan(
                    sample, reference, plan
                )
                inventory.append(row)
                state_rows.extend(trajectory_rows)
                self.store.status["current"] = {
                    "sample_id": sample.sample_id,
                    "trajectory": plan_index + 1,
                    "trajectory_total": len(plans),
                    "trajectory_id": plan["trajectory_id"],
                }
                self.store.save_status()
            self._write_sample_tables(
                sample.sample_id, inventory, state_rows
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(time.perf_counter() - started),
                    "trajectory_count": len(inventory),
                    "state_rows": len(state_rows),
                },
            )
            self.model.release(reference)
        except Exception as exc:
            self.store.append_error(
                {
                    "key": key,
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.store.mark_failed(key, "%s: %s" % (type(exc).__name__, exc))
            if self.cfg.runtime.fail_on_error:
                raise
