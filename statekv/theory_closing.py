"""Strict small theory-closing experiments for selection--refresh.

The module deliberately separates fixed-QKV direct interventions from
stateful teacher-forced replay.  It is opt-in and does not alter the default
benchmark path.
"""
from __future__ import annotations

import itertools
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
from statekv.backend import ReferenceTrajectory
from statekv.config import CacheDiscoveryConfig, DiscoveryConfig
from statekv.functional_probe import (
    FunctionalProbeRunner,
    ProbeStep,
    _condition_cache,
    _distribution_metrics,
)
from statekv.runner import _sample_slug
from statekv.selectors import (
    CoreSelection,
    CoreSelector,
    mandatory_and_eligible,
    ridge_leverage,
)
from statekv.tasks import load_discovery_tasks


THEORY_TABLES = (
    "subset_objective_rows",
    "subset_unit_inventory",
    "future_oracle_horizon_rows",
    "direct_stateful_decomposition",
    "theory_runtime",
)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary_path, index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary_path, index=False)
        else:
            raise ValueError("unsupported output table: %s" % path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def enumerate_fixed_subsets(pool_size: int, subset_size: int) -> np.ndarray:
    """Return the lexicographically ordered exhaustive subset index matrix."""

    if not 0 < int(subset_size) <= int(pool_size):
        raise ValueError("subset size must lie in [1, pool_size]")
    return np.asarray(
        list(itertools.combinations(range(int(pool_size)), int(subset_size))),
        dtype=np.int16,
    )


def subset_masks(combinations: np.ndarray) -> np.ndarray:
    """Encode combinations as stable unsigned bit masks."""

    if combinations.ndim != 2:
        raise ValueError("combinations must be [subset, selected-index]")
    masks = np.zeros((len(combinations),), dtype=np.uint64)
    for column in range(int(combinations.shape[1])):
        masks |= np.left_shift(
            np.uint64(1), combinations[:, column].astype(np.uint64)
        )
    return masks


def _fixed_relative_ridge(
    base: torch.Tensor,
    pool: torch.Tensor,
    coefficient: float,
) -> float:
    reference = torch.cat([base, pool], dim=0).double()
    trace = float((reference * reference).sum().item())
    dimension = max(1, int(reference.shape[1]))
    return max(
        float(coefficient) * max(trace / dimension, 1e-12),
        1e-12,
    )


def exhaustive_ridge_subset_risk(
    base: torch.Tensor,
    pool: torch.Tensor,
    combinations: np.ndarray,
    coefficient: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Exact squared ridge-residual risk for every ``base + subset``.

    The relative ridge is fixed once using ``base + pool``.  A base Cholesky
    factor and a small Schur complement per subset give exactly the same
    projection as refitting the combined history with that fixed ridge.
    No feature-dimension inverse is formed.
    """

    p = base.detach().to(dtype=torch.float64, device="cpu")
    u = pool.detach().to(dtype=torch.float64, device="cpu")
    combos = torch.as_tensor(combinations, dtype=torch.long)
    if p.ndim != 2 or u.ndim != 2 or int(p.shape[1]) != int(u.shape[1]):
        raise ValueError("base and pool must be aligned rank-2 matrices")
    if combos.ndim != 2 or int(combos.max().item()) >= int(u.shape[0]):
        raise ValueError("subset combinations do not index the pool")
    if not torch.isfinite(p).all() or not torch.isfinite(u).all():
        raise FloatingPointError("ridge subset feature contains NaN/Inf")
    ridge = _fixed_relative_ridge(p, u, coefficient)
    if len(p):
        gram = p @ p.T + ridge * torch.eye(len(p), dtype=torch.float64)
        base_factor, info = torch.linalg.cholesky_ex(gram)
        if int(info.max().item()) != 0:
            raise FloatingPointError("base ridge Cholesky failed")

        def residualize(rows: torch.Tensor) -> torch.Tensor:
            coefficients = torch.cholesky_solve(
                p @ rows.T, base_factor
            )
            return rows - coefficients.T @ p

        pool_residual = residualize(u)
        calculation = "base_cholesky_plus_subset_schur"
    else:
        pool_residual = u.clone()
        calculation = "empty_base_subset_cholesky"
    candidate = u.index_select(0, combos.reshape(-1)).reshape(
        len(combos), int(combos.shape[1]), int(u.shape[1])
    )
    candidate_residual = pool_residual.index_select(
        0, combos.reshape(-1)
    ).reshape_as(candidate)
    schur = torch.matmul(candidate, candidate_residual.transpose(1, 2))
    schur = 0.5 * (schur + schur.transpose(1, 2))
    schur = schur + ridge * torch.eye(
        int(combos.shape[1]), dtype=torch.float64
    ).unsqueeze(0)
    factor, info = torch.linalg.cholesky_ex(schur)
    if int(info.max().item()) != 0:
        raise FloatingPointError("subset Schur Cholesky failed")
    # Exact final residual is R - (U C_tilde^T) S^-1 C_tilde.  Compute its
    # norm without materializing [num_subsets, |U|, feature_dimension].
    cross = torch.einsum("nd,bkd->bnk", u, candidate_residual)
    coefficients = torch.cholesky_solve(
        cross.transpose(1, 2), factor
    ).transpose(1, 2)
    residual_dot_candidate = torch.einsum(
        "nd,bkd->bnk", pool_residual, candidate_residual
    )
    candidate_gram = torch.matmul(
        candidate_residual, candidate_residual.transpose(1, 2)
    )
    base_norm = (pool_residual * pool_residual).sum(dim=1)
    risk_by_evaluation_row = (
        base_norm.unsqueeze(0)
        - 2.0 * (coefficients * residual_dot_candidate).sum(dim=2)
        + torch.einsum(
            "bnj,bjk,bnk->bn",
            coefficients,
            candidate_gram,
            coefficients,
        )
    ).clamp_min(0.0)
    risk = risk_by_evaluation_row.sum(dim=1)
    if not torch.isfinite(risk).all():
        raise FloatingPointError("subset ridge risk contains NaN/Inf")
    return risk.numpy(), {
        "ridge": float(ridge),
        "calculation": calculation,
        "base_rows": int(p.shape[0]),
        "pool_rows": int(u.shape[0]),
        "feature_dimension": int(u.shape[1]),
        "subset_count": int(len(combos)),
        "subset_size": int(combos.shape[1]),
    }


def _rows_from_positions(
    positions: Sequence[int], selected: Iterable[int]
) -> List[int]:
    lookup = {int(position): row for row, position in enumerate(positions)}
    return [
        lookup[int(position)]
        for position in sorted(set(int(value) for value in selected))
        if int(position) in lookup
    ]


def fixed_qkv_subset_metrics(
    attention: torch.Tensor,
    values: torch.Tensor,
    projected_basis: torch.Tensor,
    base_rows: Sequence[int],
    pool_rows: Sequence[int],
    combinations: np.ndarray,
    epsilon: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """Vectorized exact fixed-mask risks and deletion decomposition."""

    alpha = attention.detach().to(dtype=torch.float64, device="cpu").flatten()
    vectors = values.detach().to(dtype=torch.float64, device="cpu")
    basis = projected_basis.detach().to(dtype=torch.float64, device="cpu")
    combos = torch.as_tensor(combinations, dtype=torch.long)
    if vectors.ndim != 2 or int(vectors.shape[0]) != int(alpha.shape[0]):
        raise ValueError("fixed-QKV attention/value rows are not aligned")
    if basis.ndim != 2 or int(basis.shape[0]) != int(vectors.shape[1]):
        raise ValueError("projected basis must map head_dim to hidden_size")
    alpha = alpha / alpha.sum().clamp_min(float(epsilon))
    full = alpha @ vectors
    base_index = torch.as_tensor(list(base_rows), dtype=torch.long)
    pool_index = torch.as_tensor(list(pool_rows), dtype=torch.long)
    if len(base_index):
        base_mass = alpha.index_select(0, base_index).sum()
        base_numerator = (
            alpha.index_select(0, base_index).unsqueeze(1)
            * vectors.index_select(0, base_index)
        ).sum(dim=0)
    else:
        base_mass = torch.tensor(0.0, dtype=torch.float64)
        base_numerator = torch.zeros_like(full)
    pool_alpha = alpha.index_select(0, pool_index)
    pool_values = vectors.index_select(0, pool_index)
    chosen_alpha = pool_alpha.index_select(0, combos.reshape(-1)).reshape(
        len(combos), int(combos.shape[1])
    )
    chosen_values = pool_values.index_select(
        0, combos.reshape(-1)
    ).reshape(len(combos), int(combos.shape[1]), int(vectors.shape[1]))
    retained_mass = base_mass + chosen_alpha.sum(dim=1)
    if bool((retained_mass <= float(epsilon)).any()):
        raise FloatingPointError("a subset has zero retained attention mass")
    retained_numerator = base_numerator.unsqueeze(0) + (
        chosen_alpha.unsqueeze(2) * chosen_values
    ).sum(dim=1)
    masked_output = retained_numerator / retained_mass.unsqueeze(1)
    delta = masked_output - full.unsqueeze(0)
    projected_delta = delta @ basis
    true_head = (delta * delta).sum(dim=1)
    true_projected = (projected_delta * projected_delta).sum(dim=1)
    deleted_mass = 1.0 - retained_mass
    deleted_value = full.unsqueeze(0) - retained_numerator
    identity_delta = (
        deleted_mass.unsqueeze(1) * full.unsqueeze(0) - deleted_value
    ) / retained_mass.unsqueeze(1)
    identity_relative_error = torch.linalg.vector_norm(
        identity_delta - delta, dim=1
    ) / (torch.linalg.vector_norm(delta, dim=1) + float(epsilon))

    individual = alpha.unsqueeze(1) * (full.unsqueeze(0) - vectors)
    individual_projected = individual @ basis
    individual_head_energy = (individual * individual).sum(dim=1)
    individual_projected_energy = (
        individual_projected * individual_projected
    ).sum(dim=1)
    total_individual_head = individual_head_energy.sum()
    total_individual_projected = individual_projected_energy.sum()
    if len(base_index):
        base_individual_head = individual_head_energy.index_select(
            0, base_index
        ).sum()
        base_individual_projected = individual_projected_energy.index_select(
            0, base_index
        ).sum()
    else:
        base_individual_head = torch.tensor(0.0, dtype=torch.float64)
        base_individual_projected = torch.tensor(
            0.0, dtype=torch.float64
        )
    pool_individual_head = individual_head_energy.index_select(0, pool_index)
    pool_individual_projected = individual_projected_energy.index_select(
        0, pool_index
    )
    retained_individual_head = base_individual_head + (
        pool_individual_head.index_select(0, combos.reshape(-1)).reshape(
            len(combos), int(combos.shape[1])
        )
    ).sum(dim=1)
    retained_individual_projected = base_individual_projected + (
        pool_individual_projected.index_select(
            0, combos.reshape(-1)
        ).reshape(len(combos), int(combos.shape[1]))
    ).sum(dim=1)
    additive_head = (
        total_individual_head - retained_individual_head
    ) / retained_mass.square()
    additive_projected = (
        total_individual_projected - retained_individual_projected
    ) / retained_mass.square()
    return {
        "retained_attention_mass": retained_mass.numpy(),
        "deleted_attention_mass": deleted_mass.numpy(),
        "true_head_risk": true_head.numpy(),
        "true_proj_head_risk": true_projected.numpy(),
        "identity_head_risk": (identity_delta * identity_delta)
        .sum(dim=1)
        .numpy(),
        "identity_relative_error": identity_relative_error.numpy(),
        "individual_head_energy_sum": additive_head.numpy(),
        "individual_proj_energy_sum": additive_projected.numpy(),
        "cross_head_interaction": (true_head - additive_head).numpy(),
        "cross_proj_interaction": (
            true_projected - additive_projected
        ).numpy(),
        "actual_fixed_output_l2": torch.linalg.vector_norm(
            masked_output, dim=1
        ).numpy(),
        "identity_delta_l2": torch.linalg.vector_norm(
            identity_delta, dim=1
        ).numpy(),
        "_head_delta": delta.numpy(),
        "_projected_delta": projected_delta.numpy(),
    }


def direct_mask_risk(
    attention: torch.Tensor,
    values: torch.Tensor,
    projected_basis: torch.Tensor,
    positions: Sequence[int],
    retained_positions: Iterable[int],
    epsilon: float = 1e-12,
) -> Tuple[float, float]:
    """Single-mask fixed-QKV head and projected-head risk."""

    retained_rows = _rows_from_positions(positions, retained_positions)
    if not retained_rows:
        return float("nan"), float("nan")
    alpha = attention.detach().double().cpu().flatten()
    alpha = alpha / alpha.sum().clamp_min(float(epsilon))
    vectors = values.detach().double().cpu()
    rows = torch.as_tensor(retained_rows, dtype=torch.long)
    mass = alpha.index_select(0, rows).sum()
    if float(mass) <= float(epsilon):
        return float("nan"), float("nan")
    full = alpha @ vectors
    masked = (
        alpha.index_select(0, rows).unsqueeze(1)
        * vectors.index_select(0, rows)
    ).sum(dim=0) / mass
    delta = masked - full
    projected = delta @ projected_basis.detach().double().cpu()
    return (
        float((delta * delta).sum().item()),
        float((projected * projected).sum().item()),
    )


def fixed_query_feature_matrices(
    attention_by_head: torch.Tensor,
    values_by_kv_head: torch.Tensor,
    projected_bases: Mapping[int, torch.Tensor],
    gqa_query_heads_per_kv_head: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Construct aligned Raw-V/OV/AOV/AOR matrices for every query head."""

    attention = attention_by_head.detach().float().cpu()
    values = values_by_kv_head.detach().float().cpu()
    if attention.ndim != 2 or values.ndim != 3:
        raise ValueError("attention/values must be [qh,n] and [kvh,n,d]")
    if int(attention.shape[1]) != int(values.shape[1]):
        raise ValueError("attention and value token axes differ")
    group = int(gqa_query_heads_per_kv_head)
    if int(attention.shape[0]) != int(values.shape[0]) * group:
        raise ValueError("invalid GQA query/KV-head mapping")
    output: Dict[int, Dict[str, torch.Tensor]] = {}
    for head in range(int(attention.shape[0])):
        kv_head = int(head // group)
        alpha = attention[head]
        alpha = alpha / alpha.sum().clamp_min(1e-30)
        raw = values[kv_head]
        basis = projected_bases[head].detach().float().cpu()
        projected = raw @ basis
        mean = alpha @ projected
        output[head] = {
            "raw_v": raw,
            "projected_v": projected,
            "aov": torch.sqrt(alpha).unsqueeze(1) * projected,
            "aor": torch.sqrt(alpha).unsqueeze(1)
            * (projected - mean.unsqueeze(0)),
        }
    return output


def paired_gap(old: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """Signed old-minus-fresh gap with exact identity at equal arms."""

    left = np.asarray(old, dtype=np.float64)
    right = np.asarray(fresh, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired gap arms are not aligned")
    return left - right


def validate_recent_budget(
    position_maps: Mapping[int, Sequence[int]], total_budget: int
) -> None:
    """Assert that every physical layer cache satisfies its rolling budget."""

    lengths = {
        int(layer): len(positions)
        for layer, positions in position_maps.items()
    }
    if any(length > int(total_budget) for length in lengths.values()):
        raise RuntimeError(
            "recent rolling window exceeded cache budget: %s" % lengths
        )


def cumulative_rows(
    step_rows: pd.DataFrame,
    horizons: Sequence[int],
    value_columns: Sequence[str],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Create exact prefix cumulative rows; H=1 equals the first step."""

    output: List[Dict[str, Any]] = []
    for keys, group in step_rows.groupby(
        list(group_columns), dropna=False, sort=False
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ordered = group.sort_values("horizon_offset")
        for horizon in horizons:
            prefix = ordered[
                ordered["horizon_offset"] <= int(horizon)
            ]
            if len(prefix) != int(horizon):
                continue
            row = dict(zip(group_columns, keys))
            row["horizon"] = int(horizon)
            for column in value_columns:
                row[column] = float(prefix[column].sum())
            output.append(row)
    return pd.DataFrame(output)


def assert_replay_alignment(
    old: ProbeStep, fresh: ProbeStep
) -> None:
    """Reject token-misaligned direct/stateful arm comparisons."""

    if (
        int(old.target_index) != int(fresh.target_index)
        or int(old.target_token_id) != int(fresh.target_token_id)
        or int(old.target_token_position)
        != int(fresh.target_token_position)
    ):
        raise RuntimeError("direct/stateful replay token alignment failed")


def leave_one_sequence_out(
    frame: pd.DataFrame, sequence_column: str = "sample_id"
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return leakage-free outer folds indexed by whole sequence."""

    folds = []
    values = frame[sequence_column].astype(str).to_numpy()
    for sequence in sorted(set(values.tolist())):
        test = np.flatnonzero(values == sequence)
        train = np.flatnonzero(values != sequence)
        if set(values[train]) & set(values[test]):
            raise AssertionError("LOSO split leaked a sequence")
        folds.append((train, test))
    return folds


ANCHOR_PREDICTOR_FORBIDDEN_COLUMNS = {
    "future_aor_gap",
    "future_raw_v_gap",
    "future_projected_v_gap",
    "future_aov_gap",
    "cumulative_direct_proj_benefit",
    "cumulative_stateful_proj_benefit",
    "cumulative_nll_benefit",
    "fresh_selected_positions",
    "fresh_output",
    "future_oracle_feature",
}


def validate_anchor_predictor_columns(columns: Iterable[str]) -> None:
    overlap = set(str(value) for value in columns) & (
        ANCHOR_PREDICTOR_FORBIDDEN_COLUMNS
    )
    if overlap:
        raise ValueError(
            "future oracle/label leaked into anchor predictor: %s"
            % sorted(overlap)
        )


class NoBackingMonitoringState:
    """Minimal accumulator that never owns evicted K/V tensors."""

    def __init__(self, gamma: float):
        self.gamma = float(gamma)
        self.arrival_scores: Dict[int, float] = {}
        self.retained_scores: Dict[int, float] = {}

    def observe_arrival(self, position: int, residual_score: float) -> None:
        self.arrival_scores = {
            int(key): self.gamma * float(value)
            for key, value in self.arrival_scores.items()
        }
        self.arrival_scores[int(position)] = float(residual_score)

    def update_retained(
        self, retained_positions: Iterable[int], scores: Mapping[int, float]
    ) -> None:
        retained = set(int(value) for value in retained_positions)
        self.retained_scores = {
            position: float(scores[position])
            for position in retained
            if position in scores
        }

    def evict(self, positions: Iterable[int]) -> None:
        for position in positions:
            self.retained_scores.pop(int(position), None)

    def schema(self) -> Dict[str, str]:
        return {
            "arrival_scores": "position_to_scalar_only",
            "retained_scores": "retained_position_to_scalar_only",
            "stores_evicted_kv": "false",
        }


class TheoryClosingRunner(FunctionalProbeRunner):
    """Run the pre-registered small theory-closing matrix."""

    def run(self) -> Path:
        if not self.cfg.theory_closing.enabled:
            raise ValueError("theory_closing.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "theory_closing_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        self._projection_bases = self._build_projection_bases()
        for table in THEORY_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "theory_closing"
                / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample in samples:
                self._run_theory_sample(sample)
            outputs = self._consolidate_theory_tables()
            schema = self._write_artifact_schema(outputs)
            self.store.status["state"] = "mechanism_complete_analysis_pending"
            self.store.status["theory_outputs"] = {
                key: str(value) for key, value in outputs.items()
            }
            self.store.status["artifact_schema"] = str(schema)
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir

    def _build_projection_bases(
        self,
    ) -> Dict[int, Dict[int, torch.Tensor]]:
        head_dim = int(
            self.model.model_info.get("head_dim")
            or int(self.model.model_info["hidden_size"])
            // int(self.model.model_info["num_attention_heads"])
        )
        identity = torch.eye(head_dim, dtype=torch.float32)
        result: Dict[int, Dict[int, torch.Tensor]] = {}
        for layer in self.model.selected_layers:
            result[int(layer)] = {}
            for head in self.model.selected_heads[int(layer)]:
                result[int(layer)][int(head)] = (
                    self.model.project_features(
                        int(layer), identity, head=int(head)
                    )
                    .detach()
                    .float()
                    .cpu()
                )
        return result

    def _theory_fragment_path(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "theory_closing"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _write_theory_tables(
        self, sample_id: str, tables: Mapping[str, pd.DataFrame]
    ) -> None:
        for table in THEORY_TABLES:
            _atomic_frame(
                tables.get(table, pd.DataFrame()),
                self._theory_fragment_path(table, sample_id),
            )

    def _consolidate_theory_tables(self) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        for table in THEORY_TABLES:
            fragments = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "theory_closing"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in fragments]
            frame = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
            parquet = self.store.run_dir / ("%s.parquet" % table)
            csv = self.store.run_dir / ("%s.csv" % table)
            _atomic_frame(frame, parquet)
            _atomic_frame(frame, csv)
            outputs[table] = parquet
        return outputs

    def _write_artifact_schema(
        self, outputs: Mapping[str, Path]
    ) -> Path:
        payload: Dict[str, Any] = {
            "schema_version": "theory_closing_v1",
            "primary_independent_unit": "sequence",
            "tables": {},
            "fixed_qkv": {
                "query_state": "full_reference",
                "intervention": "retained_mask_only",
                "ridge": "relative_to_P_union_U_then_fixed_across_subsets",
            },
            "stateful": {
                "trajectory": "teacher_forced_reference_tokens",
                "fresh_references": [
                    "per_step_fresh",
                    "horizon_start_once_fresh",
                ],
            },
        }
        for name, path in outputs.items():
            frame = pd.read_parquet(path)
            payload["tables"][name] = {
                "path": str(path),
                "rows": int(len(frame)),
                "columns": {
                    column: str(dtype)
                    for column, dtype in frame.dtypes.items()
                },
            }
        path = self.store.run_dir / "artifact_schema.json"
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name + ".", dir=str(path.parent)
        )
        os.close(descriptor)
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return path

    def _stable_rng(self, sample_id: str, layer: int) -> np.random.Generator:
        import hashlib

        token = "%s:%d:%d" % (
            sample_id,
            int(layer),
            int(self.cfg.theory_closing.candidate_random_seed),
        )
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        return np.random.default_rng(seed)

    def _candidate_pool(
        self,
        sample_id: str,
        layer: int,
        positions: Sequence[int],
        values: torch.Tensor,
        attention_by_head: torch.Tensor,
        full_outputs: torch.Tensor,
        old_selection: CoreSelection,
        fresh_selection: CoreSelection,
        protected_recent: int,
    ) -> Tuple[List[int], Dict[int, List[str]], List[int]]:
        theory = self.cfg.theory_closing
        sink, recent, eligible = mandatory_and_eligible(
            positions,
            int(self.cfg.cache.sink_size),
            int(protected_recent),
        )
        mandatory = sorted(set(sink + recent))
        eligible_set = set(eligible)
        old_layer = old_selection.by_layer[int(layer)]
        fresh_layer = fresh_selection.by_layer[int(layer)]
        old_set = set(old_layer.selected_positions)
        fresh_set = set(fresh_layer.selected_positions)
        entrants = sorted((fresh_set - old_set) & eligible_set)
        exits = sorted((old_set - fresh_set) & eligible_set)

        def boundary(selection: Any) -> List[int]:
            score_by_position = dict(
                zip(positions, selection.aggregate_scores)
            )
            valid = [
                position
                for position in eligible
                if math.isfinite(
                    float(score_by_position.get(position, float("nan")))
                )
            ]
            ordered = sorted(
                valid,
                key=lambda position: (
                    -float(score_by_position[position]),
                    int(position),
                ),
            )
            cutoff = min(
                int(self.cfg.theory_closing.total_budget)
                - int(self.cfg.cache.sink_size)
                - max(1, int(protected_recent)),
                len(ordered),
            )
            return ordered[max(0, cutoff - 3) : min(len(ordered), cutoff + 3)]

        row_by_position = {
            int(position): row for row, position in enumerate(positions)
        }
        eligible_rows = torch.as_tensor(
            [row_by_position[position] for position in eligible],
            dtype=torch.long,
        )
        mean_attention = attention_by_head.detach().float().cpu().mean(dim=0)
        attention_order = [
            eligible[int(index)]
            for index in torch.argsort(
                mean_attention.index_select(0, eligible_rows),
                descending=True,
                stable=True,
            ).tolist()
        ]
        query_heads = int(attention_by_head.shape[0])
        group = int(self.model.model_info["gqa_query_heads_per_kv_head"])
        aor_score = torch.zeros(len(positions), dtype=torch.float64)
        for head in range(query_heads):
            kv_head = int(head // group)
            vector = values[kv_head].detach().double().cpu()
            delta = vector - full_outputs[head].detach().double().cpu()
            basis = self._projection_bases[int(layer)][head].double()
            projected = delta @ basis
            aor_score += (
                attention_by_head[head].detach().double().cpu()
                * (projected * projected).sum(dim=1)
            )
        aor_score /= max(1, query_heads)
        aor_order = [
            eligible[int(index)]
            for index in torch.argsort(
                aor_score.index_select(0, eligible_rows),
                descending=True,
                stable=True,
            ).tolist()
        ]
        flattened = values.detach().float().cpu().permute(1, 0, 2).reshape(
            len(positions), -1
        )
        raw_scores, _ = ridge_leverage(
            flattened.index_select(0, eligible_rows),
            self.cfg.theory_closing.ridge_coefficient,
            "relative",
        )
        raw_order = [
            eligible[int(index)]
            for index in torch.argsort(
                raw_scores, descending=True, stable=True
            ).tolist()
        ]
        random_order = list(eligible)
        self._stable_rng(sample_id, layer).shuffle(random_order)
        sources: Dict[str, List[int]] = {
            "fresh_entrant": entrants,
            "old_exit": exits,
            "old_fresh_boundary": list(
                dict.fromkeys(boundary(old_layer) + boundary(fresh_layer))
            ),
            "high_attention": attention_order,
            "high_aor": aor_order,
            "high_raw_v_leverage": raw_order,
            "random_control": random_order,
        }
        pool: List[int] = []
        provenance: Dict[int, List[str]] = {}
        for source, candidates in sources.items():
            for position in candidates:
                provenance.setdefault(int(position), []).append(source)
        rank = 0
        source_names = list(sources)
        while len(pool) < int(theory.candidate_pool_size):
            progressed = False
            for source in source_names:
                candidates = sources[source]
                while rank < len(candidates) and candidates[rank] in pool:
                    break
                if rank < len(candidates):
                    position = int(candidates[rank])
                    if position not in pool and position in eligible_set:
                        pool.append(position)
                        progressed = True
                        if len(pool) == int(theory.candidate_pool_size):
                            break
            if len(pool) == int(theory.candidate_pool_size):
                break
            rank += 1
            if not progressed and rank > max(
                (len(value) for value in sources.values()), default=0
            ):
                break
        if len(pool) < int(theory.candidate_pool_size):
            for position in eligible:
                if position not in pool:
                    pool.append(int(position))
                    provenance.setdefault(int(position), []).append(
                        "deterministic_fill"
                    )
                if len(pool) == int(theory.candidate_pool_size):
                    break
        if len(pool) != int(theory.candidate_pool_size):
            raise RuntimeError("candidate pool could not reach configured size")
        return pool, provenance, mandatory

    @staticmethod
    def _frame_base(
        base: Mapping[str, Any],
        combinations: np.ndarray,
        masks: np.ndarray,
        unit_id: str,
        layer: int,
        granularity: str,
        head: int,
        kv_group: int,
    ) -> Dict[str, Any]:
        count = len(combinations)
        return {
            **{key: [value] * count for key, value in base.items()},
            "unit_id": [unit_id] * count,
            "layer": np.full(count, int(layer), dtype=np.int16),
            "granularity": [granularity] * count,
            "head": np.full(count, int(head), dtype=np.int16),
            "kv_head_group": np.full(
                count, int(kv_group), dtype=np.int16
            ),
            "subset_id": np.arange(count, dtype=np.int32),
            "subset_mask": masks,
            **{
                "candidate_%d" % column: combinations[:, column]
                for column in range(int(combinations.shape[1]))
            },
        }

    def _subset_experiment(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        theory = self.cfg.theory_closing
        anchor_step = int(theory.subset_anchor_step)
        probe_step = int(theory.subset_probe_step)
        cache_cfg = _condition_cache(
            self.cfg,
            int(theory.total_budget),
            int(theory.subset_recent_size),
        )
        condition_cfg = replace(self.cfg, cache=cache_cfg)
        selector = CoreSelector(condition_cfg)
        old_selection = selector.select(
            reference.anchors[anchor_step].snapshot(reference.sample_id),
            theory.selector,
        )
        fresh_selection = selector.select(
            reference.anchors[probe_step].snapshot(reference.sample_id),
            theory.selector,
        )
        current_anchor = reference.anchors[probe_step]
        record = reference.query_records[probe_step]
        combinations = enumerate_fixed_subsets(
            int(theory.candidate_pool_size),
            int(theory.candidate_subset_size),
        )
        masks = subset_masks(combinations)
        frames: List[pd.DataFrame] = []
        inventory: List[Dict[str, Any]] = []
        common = {
            **self._base(sample),
            "anchor_step": anchor_step,
            "probe_step": probe_step,
            "total_budget": int(theory.total_budget),
            "protected_recent_size": int(theory.subset_recent_size),
            "candidate_pool_size": int(theory.candidate_pool_size),
            "candidate_subset_size": int(theory.candidate_subset_size),
            "fixed_qkv": True,
            "subset_space_exhaustive": True,
        }
        group_size = int(
            self.model.model_info["gqa_query_heads_per_kv_head"]
        )
        for layer in self.model.selected_layers:
            layer = int(layer)
            positions = [
                int(value)
                for value in current_anchor.position_maps[layer].tolist()
            ]
            values = current_anchor.values[layer].detach()[0].float().cpu()
            attention = (
                record.all_head_attention_distributions[layer]
                .detach()
                .float()
                .cpu()
            )
            full_outputs = (
                record.all_head_attention_outputs[layer]
                .detach()
                .float()
                .cpu()
            )
            pool_positions, provenance, mandatory_positions = (
                self._candidate_pool(
                    sample.sample_id,
                    layer,
                    positions,
                    values,
                    attention,
                    full_outputs,
                    old_selection,
                    fresh_selection,
                    int(theory.subset_recent_size),
                )
            )
            pool_rows = _rows_from_positions(positions, pool_positions)
            base_rows = _rows_from_positions(
                positions, mandatory_positions
            )
            inventory.append(
                {
                    **common,
                    "layer": layer,
                    "pool_positions": json_text(pool_positions),
                    "pool_sources": json_text(
                        {
                            str(index): provenance.get(position, [])
                            for index, position in enumerate(pool_positions)
                        }
                    ),
                    "mandatory_positions": json_text(mandatory_positions),
                    "old_selected_positions": json_text(
                        old_selection.by_layer[layer].selected_positions
                    ),
                    "fresh_selected_positions": json_text(
                        fresh_selection.by_layer[layer].selected_positions
                    ),
                    "old_fresh_jaccard": float(
                        len(
                            set(
                                old_selection.by_layer[
                                    layer
                                ].selected_positions
                            )
                            & set(
                                fresh_selection.by_layer[
                                    layer
                                ].selected_positions
                            )
                        )
                        / max(
                            1,
                            len(
                                set(
                                    old_selection.by_layer[
                                        layer
                                    ].selected_positions
                                )
                                | set(
                                    fresh_selection.by_layer[
                                        layer
                                    ].selected_positions
                                )
                            ),
                        )
                    ),
                }
            )
            aggregate: Dict[str, Any] = {
                "layer_projected_delta": np.zeros(
                    (
                        len(combinations),
                        int(self.model.model_info["hidden_size"]),
                    ),
                    dtype=np.float64,
                ),
                "head_projected_risk_sum": np.zeros(
                    len(combinations), dtype=np.float64
                ),
                "head_risk_sum": np.zeros(
                    len(combinations), dtype=np.float64
                ),
                "surrogates": {
                    variant: np.zeros(len(combinations), dtype=np.float64)
                    for variant in theory.feature_variants
                },
                "scalars": {},
                "groups": {},
            }
            for group_id in range(
                int(self.model.model_info["num_key_value_heads"])
            ):
                aggregate["groups"][group_id] = {
                    "projected_delta": np.zeros(
                        (
                            len(combinations),
                            int(self.model.model_info["hidden_size"]),
                        ),
                        dtype=np.float64,
                    ),
                    "head_projected_risk_sum": np.zeros(
                        len(combinations), dtype=np.float64
                    ),
                    "head_risk_sum": np.zeros(
                        len(combinations), dtype=np.float64
                    ),
                    "surrogates": {
                        variant: np.zeros(
                            len(combinations), dtype=np.float64
                        )
                        for variant in theory.feature_variants
                    },
                    "scalars": {},
                }
            for head in self.model.selected_heads[layer]:
                head = int(head)
                kv_group = int(head // group_size)
                raw = values[kv_group].double()
                alpha = attention[head].double()
                alpha = alpha / alpha.sum().clamp_min(
                    float(theory.identity_epsilon)
                )
                basis = self._projection_bases[layer][head].double()
                projected = raw @ basis
                full = alpha @ raw
                projected_full = full @ basis
                sqrt_alpha = torch.sqrt(alpha.clamp_min(0.0))
                matrices = {
                    "raw_v": raw,
                    "projected_v": projected,
                    "aov": sqrt_alpha.unsqueeze(1) * projected,
                    "aor": sqrt_alpha.unsqueeze(1)
                    * (projected - projected_full.unsqueeze(0)),
                }
                surrogate: Dict[str, np.ndarray] = {}
                ridge_diagnostics: Dict[str, Dict[str, Any]] = {}
                for variant, matrix in matrices.items():
                    risk, diagnostics = exhaustive_ridge_subset_risk(
                        matrix.index_select(
                            0, torch.as_tensor(base_rows, dtype=torch.long)
                        ),
                        matrix.index_select(
                            0, torch.as_tensor(pool_rows, dtype=torch.long)
                        ),
                        combinations,
                        float(theory.ridge_coefficient),
                    )
                    surrogate[variant] = risk
                    ridge_diagnostics[variant] = diagnostics
                exact = fixed_qkv_subset_metrics(
                    alpha,
                    raw,
                    basis,
                    base_rows,
                    pool_rows,
                    combinations,
                    float(theory.identity_epsilon),
                )
                unit_id = "%s:l%d:h%d" % (
                    sample.sample_id,
                    layer,
                    head,
                )
                data = self._frame_base(
                    common,
                    combinations,
                    masks,
                    unit_id,
                    layer,
                    "query_head",
                    head,
                    kv_group,
                )
                for key, values_array in exact.items():
                    if not key.startswith("_"):
                        data[key] = values_array
                for variant, values_array in surrogate.items():
                    data["surrogate_%s" % variant] = values_array
                    data["ridge_%s" % variant] = np.full(
                        len(combinations),
                        ridge_diagnostics[variant]["ridge"],
                    )
                data["attention_only_surrogate"] = exact[
                    "deleted_attention_mass"
                ]
                data["cross_head_cancellation"] = np.zeros(
                    len(combinations), dtype=np.float64
                )
                frames.append(pd.DataFrame(data))

                aggregate["layer_projected_delta"] += exact[
                    "_projected_delta"
                ]
                aggregate["head_projected_risk_sum"] += exact[
                    "true_proj_head_risk"
                ]
                aggregate["head_risk_sum"] += exact["true_head_risk"]
                group_aggregate = aggregate["groups"][kv_group]
                group_aggregate["projected_delta"] += exact[
                    "_projected_delta"
                ]
                group_aggregate["head_projected_risk_sum"] += exact[
                    "true_proj_head_risk"
                ]
                group_aggregate["head_risk_sum"] += exact[
                    "true_head_risk"
                ]
                for variant in theory.feature_variants:
                    aggregate["surrogates"][variant] += surrogate[variant]
                    group_aggregate["surrogates"][variant] += surrogate[
                        variant
                    ]
                scalar_keys = [
                    "retained_attention_mass",
                    "deleted_attention_mass",
                    "identity_relative_error",
                    "individual_head_energy_sum",
                    "individual_proj_energy_sum",
                    "cross_head_interaction",
                    "cross_proj_interaction",
                ]
                for key in scalar_keys:
                    aggregate["scalars"].setdefault(
                        key, np.zeros(len(combinations), dtype=np.float64)
                    )
                    aggregate["scalars"][key] += exact[key]
                    group_aggregate["scalars"].setdefault(
                        key, np.zeros(len(combinations), dtype=np.float64)
                    )
                    group_aggregate["scalars"][key] += exact[key]
            for group_id, group_aggregate in aggregate["groups"].items():
                projected_risk = (
                    group_aggregate["projected_delta"] ** 2
                ).sum(axis=1)
                data = self._frame_base(
                    common,
                    combinations,
                    masks,
                    "%s:l%d:g%d" % (
                        sample.sample_id,
                        layer,
                        group_id,
                    ),
                    layer,
                    "kv_head_group",
                    -1,
                    group_id,
                )
                data["true_head_risk"] = group_aggregate["head_risk_sum"]
                data["true_proj_head_risk"] = projected_risk
                data["identity_head_risk"] = group_aggregate[
                    "head_risk_sum"
                ]
                for key, value in group_aggregate["scalars"].items():
                    data[key] = value / group_size
                for variant, value in group_aggregate[
                    "surrogates"
                ].items():
                    data["surrogate_%s" % variant] = value
                data["attention_only_surrogate"] = group_aggregate[
                    "scalars"
                ]["deleted_attention_mass"] / group_size
                data["cross_head_cancellation"] = (
                    projected_risk
                    - group_aggregate["head_projected_risk_sum"]
                )
                data["actual_fixed_output_l2"] = np.full(
                    len(combinations), np.nan
                )
                data["identity_delta_l2"] = np.full(
                    len(combinations), np.nan
                )
                frames.append(pd.DataFrame(data))
            projected_risk = (
                aggregate["layer_projected_delta"] ** 2
            ).sum(axis=1)
            data = self._frame_base(
                common,
                combinations,
                masks,
                "%s:l%d:layer" % (sample.sample_id, layer),
                layer,
                "layer",
                -1,
                -1,
            )
            data["true_head_risk"] = aggregate["head_risk_sum"]
            data["true_proj_head_risk"] = projected_risk
            data["identity_head_risk"] = aggregate["head_risk_sum"]
            for key, value in aggregate["scalars"].items():
                data[key] = value / len(self.model.selected_heads[layer])
            for variant, value in aggregate["surrogates"].items():
                data["surrogate_%s" % variant] = value
            data["attention_only_surrogate"] = aggregate["scalars"][
                "deleted_attention_mass"
            ] / len(self.model.selected_heads[layer])
            data["cross_head_cancellation"] = (
                projected_risk - aggregate["head_projected_risk_sum"]
            )
            data["actual_fixed_output_l2"] = np.full(
                len(combinations), np.nan
            )
            data["identity_delta_l2"] = np.full(
                len(combinations), np.nan
            )
            frames.append(pd.DataFrame(data))
        return (
            pd.concat(frames, ignore_index=True, sort=False),
            pd.DataFrame(inventory),
        )

    @staticmethod
    def _squared_distance(left: torch.Tensor, right: torch.Tensor) -> float:
        delta = left.detach().double().cpu() - right.detach().double().cpu()
        return float((delta * delta).sum().item())

    def _uncovered_feature_energy(
        self,
        attention: torch.Tensor,
        values: torch.Tensor,
        basis: torch.Tensor,
        positions: Sequence[int],
        retained_positions: Iterable[int],
    ) -> Dict[str, float]:
        alpha = attention.detach().double().cpu().flatten()
        alpha = alpha / alpha.sum().clamp_min(
            float(self.cfg.theory_closing.identity_epsilon)
        )
        vectors = values.detach().double().cpu()
        projection = basis.detach().double().cpu()
        metric = projection @ projection.T
        full = alpha @ vectors
        raw_energy = (vectors * vectors).sum(dim=1)
        projected_energy = torch.einsum(
            "nd,de,ne->n", vectors, metric, vectors
        ).clamp_min(0.0)
        centered = vectors - full.unsqueeze(0)
        residual_energy = torch.einsum(
            "nd,de,ne->n", centered, metric, centered
        ).clamp_min(0.0)
        energies = {
            "raw_v": raw_energy,
            "projected_v": projected_energy,
            "aov": alpha * projected_energy,
            "aor": alpha * residual_energy,
        }
        retained_rows = set(
            _rows_from_positions(positions, retained_positions)
        )
        deleted = torch.as_tensor(
            [
                row
                for row in range(len(positions))
                if row not in retained_rows
            ],
            dtype=torch.long,
        )
        return {
            variant: (
                float(energy.index_select(0, deleted).sum().item())
                if len(deleted)
                else 0.0
            )
            for variant, energy in energies.items()
        }

    def _anchor_history_features(
        self,
        reference: ReferenceTrajectory,
        layer: int,
        head: int,
        old_core: Iterable[int],
        fresh_core: Iterable[int],
        protected_recent: int,
    ) -> Dict[str, float]:
        theory = self.cfg.theory_closing
        start = int(theory.horizon_start_step)
        anchor = reference.anchors[start]
        full_positions = [
            int(value) for value in anchor.position_maps[int(layer)].tolist()
        ]
        group = int(self.model.model_info["gqa_query_heads_per_kv_head"])
        kv_head = int(head // group)
        full_values = anchor.values[int(layer)][0, kv_head].float().cpu()
        basis = self._projection_bases[int(layer)][int(head)]
        gaps: List[float] = []
        for target in range(
            max(0, start - max(theory.prediction_windows)), start
        ):
            record = reference.query_records[target]
            alpha = record.all_head_attention_distributions[int(layer)][
                int(head)
            ]
            length = int(alpha.shape[-1])
            positions = full_positions[:length]
            values = full_values[:length]
            _, recent, _ = mandatory_and_eligible(
                positions,
                int(self.cfg.cache.sink_size),
                int(protected_recent),
            )
            sink = positions[: int(self.cfg.cache.sink_size)]
            mandatory = set(sink + recent)
            old_retained = mandatory | (
                set(int(value) for value in old_core) & set(positions)
            )
            fresh_retained = mandatory | (
                set(int(value) for value in fresh_core) & set(positions)
            )
            old_energy = self._uncovered_feature_energy(
                alpha, values, basis, positions, old_retained
            )["aor"]
            fresh_energy = self._uncovered_feature_energy(
                alpha, values, basis, positions, fresh_retained
            )["aor"]
            gaps.append(float(old_energy - fresh_energy))
        output: Dict[str, float] = {}
        maximum = max(int(value) for value in theory.prediction_windows)
        if len(gaps) != maximum:
            raise RuntimeError("anchor observation history is incomplete")
        for window in theory.prediction_windows:
            window = int(window)
            values = np.asarray(gaps[-window:], dtype=np.float64)
            output["anchor_obs_w%d_mean" % window] = float(values.mean())
            output["anchor_obs_w%d_std" % window] = float(
                values.std(ddof=0)
            )
            x = np.arange(window, dtype=np.float64)
            output["anchor_obs_w%d_trend" % window] = float(
                np.polyfit(x, values, 1)[0] if window > 1 else 0.0
            )
            # Stable scalar moment baseline corresponding to a diagonal
            # shrinkage Gaussian functional state.  The raw q-moment model is
            # evaluated by the analysis script from the stored query moments.
            shrink = window / (window + 10.0)
            output["anchor_gaussian_w%d" % window] = float(
                values.mean()
                + 0.5 * shrink * values.var(ddof=0)
                / (abs(values.mean()) + 1e-12)
            )
            query_key = "%d:%d" % (int(layer), int(head))
            query_rows = torch.stack(
                [
                    reference.query_records[target].queries[
                        query_key
                    ].detach().double().cpu()
                    for target in range(start - window, start)
                ],
                dim=0,
            )
            current_query = (
                reference.query_records[start]
                .queries[query_key]
                .detach()
                .double()
                .cpu()
            )
            query_mean = query_rows.mean(dim=0)
            centered_queries = query_rows - query_mean.unsqueeze(0)
            covariance = (
                centered_queries.T @ centered_queries
            ) / max(1, window - 1)
            dimension = int(query_rows.shape[1])
            isotropic_variance = float(
                torch.trace(covariance).item() / max(1, dimension)
            )
            # Pre-registered dimension-aware shrinkage.  With W << d this
            # strongly shrinks the empirical covariance toward isotropic,
            # and Cholesky solve is used instead of an inverse.
            shrinkage = float(dimension / (dimension + window))
            regularized = (
                (1.0 - shrinkage) * covariance
                + shrinkage
                * isotropic_variance
                * torch.eye(dimension, dtype=torch.float64)
                + 1e-6 * torch.eye(dimension, dtype=torch.float64)
            )
            factor, info = torch.linalg.cholesky_ex(regularized)
            if int(info.max().item()) != 0:
                raise FloatingPointError(
                    "Gaussian-query shrinkage Cholesky failed"
                )
            difference = current_query - query_mean
            solved = torch.cholesky_solve(
                difference.reshape(-1, 1), factor
            ).flatten()
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            output[
                "anchor_gaussian_q_w%d_mean_norm" % window
            ] = float(torch.linalg.vector_norm(query_mean).item())
            output[
                "anchor_gaussian_q_w%d_cov_trace" % window
            ] = float(torch.trace(covariance).item())
            output[
                "anchor_gaussian_q_w%d_current_mahal" % window
            ] = float(torch.dot(difference, solved).item())
            output[
                "anchor_gaussian_q_w%d_top_eigen_fraction" % window
            ] = float(
                eigenvalues[-min(8, dimension) :].sum().item()
                / max(eigenvalues.sum().item(), 1e-30)
            )
            output[
                "anchor_gaussian_q_w%d_shrinkage" % window
            ] = shrinkage
        for gamma in theory.ema_gammas:
            state = 0.0
            for value in gaps:
                state = float(gamma) * state + (
                    1.0 - float(gamma)
                ) * float(value)
            output[
                "anchor_ema_g%s"
                % str(float(gamma)).replace(".", "_")
            ] = float(state)
        return output

    def _horizon_experiment(
        self,
        sample: Any,
        reference: ReferenceTrajectory,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        theory = self.cfg.theory_closing
        start = int(theory.horizon_start_step)
        maximum = max(int(value) for value in theory.horizons)
        selector_name = str(theory.selector)
        head_step_rows: List[Dict[str, Any]] = []
        global_step_rows: List[Dict[str, Any]] = []
        decomposition_rows: List[Dict[str, Any]] = []
        runtime_rows: List[Dict[str, Any]] = []
        base_common = {
            **self._base(sample),
            "anchor_step": int(theory.horizon_anchor_step),
            "horizon_start_step": start,
            "total_budget": int(theory.total_budget),
            "selector": selector_name,
        }
        for protected_recent in theory.protected_recent_sizes:
            protected_recent = int(protected_recent)
            cache_cfg = _condition_cache(
                self.cfg, int(theory.total_budget), protected_recent
            )
            condition_cfg = replace(self.cfg, cache=cache_cfg)
            selector = CoreSelector(condition_cfg)
            selections: Dict[int, CoreSelection] = {}

            def selection(step: int) -> CoreSelection:
                step = int(step)
                if step not in selections:
                    selections[step] = selector.select(
                        reference.anchors[step].snapshot(
                            reference.sample_id
                        ),
                        selector_name,
                    )
                return selections[step]

            replay_started = time.perf_counter()
            old_capture_steps = set(range(start + 1, start + maximum + 1))
            old_probes = self._replay_probes(
                reference,
                int(theory.horizon_anchor_step),
                selection(int(theory.horizon_anchor_step)),
                cache_cfg,
                old_capture_steps,
            )
            start_probes = self._replay_probes(
                reference,
                start,
                selection(start),
                cache_cfg,
                set(range(1, maximum + 1)),
            )
            per_step_probes: Dict[int, ProbeStep] = {}
            for target in range(start, start + maximum):
                per_step_probes[target] = self._replay_probes(
                    reference,
                    target,
                    selection(target),
                    cache_cfg,
                    {1},
                )[1]
            runtime_rows.append(
                {
                    **base_common,
                    "protected_recent_size": protected_recent,
                    "stage": "dense_stateful_replay",
                    "wall_time_s": float(
                        time.perf_counter() - replay_started
                    ),
                    "old_trajectory_forwards": start + maximum,
                    "start_once_forwards": maximum,
                    "per_step_fresh_forwards": maximum,
                    "fresh_global_selections": maximum,
                }
            )
            old_selection = selection(int(theory.horizon_anchor_step))
            start_selection = selection(start)
            anchor_features: Dict[
                Tuple[int, int], Dict[str, float]
            ] = {}
            for layer in self.model.selected_layers:
                for head in self.model.selected_heads[int(layer)]:
                    anchor_features[(int(layer), int(head))] = (
                        self._anchor_history_features(
                            reference,
                            int(layer),
                            int(head),
                            old_selection.by_layer[
                                int(layer)
                            ].selected_positions,
                            start_selection.by_layer[
                                int(layer)
                            ].selected_positions,
                            protected_recent,
                        )
                    )
            for offset in range(1, maximum + 1):
                target = start + offset - 1
                old_probe = old_probes[target + 1]
                fresh_arms = {
                    "horizon_start_once_fresh": start_probes[offset],
                    "per_step_fresh": per_step_probes[target],
                }
                full_record = reference.query_records[target]
                full_logits = reference.probe_logits[target]
                current_anchor = reference.anchors[target]
                for fresh_reference, fresh_probe in fresh_arms.items():
                    assert_replay_alignment(old_probe, fresh_probe)
                    validate_recent_budget(
                        old_probe.position_maps, int(theory.total_budget)
                    )
                    validate_recent_budget(
                        fresh_probe.position_maps, int(theory.total_budget)
                    )
                    distribution = _distribution_metrics(
                        full_logits,
                        old_probe.logits,
                        fresh_probe.logits,
                        int(old_probe.target_token_id),
                        float(self.cfg.metrics.probability_floor),
                    )
                    layer_projected_benefit = 0.0
                    for layer in self.model.selected_layers:
                        layer = int(layer)
                        full_layer = full_record.projected_attention_outputs[
                            layer
                        ]
                        old_error = self._squared_distance(
                            old_probe.diagnostic.projected_attention_outputs[
                                layer
                            ],
                            full_layer,
                        )
                        fresh_error = self._squared_distance(
                            fresh_probe.diagnostic.projected_attention_outputs[
                                layer
                            ],
                            full_layer,
                        )
                        layer_projected_benefit += old_error - fresh_error
                    global_step_rows.append(
                        {
                            **base_common,
                            "protected_recent_size": protected_recent,
                            "fresh_reference": fresh_reference,
                            "horizon_offset": offset,
                            "target_index": target,
                            "target_token_id": int(
                                old_probe.target_token_id
                            ),
                            "same_reference_token_verified": True,
                            "stateful_layer_projected_benefit": float(
                                layer_projected_benefit
                            ),
                            "stateful_nll_benefit": float(
                                distribution["refresh_benefit_nll"]
                            ),
                            "stateful_kl_benefit": float(
                                distribution[
                                    "refresh_benefit_exact_kl"
                                ]
                            ),
                            "old_delta_nll": float(
                                distribution["old_delta_nll"]
                            ),
                            "fresh_delta_nll": float(
                                distribution["fresh_delta_nll"]
                            ),
                            "post_first_recent_exit": bool(offset > 32),
                            "recent_exit_distance": int(offset - 33),
                        }
                    )
                    positions = [
                        int(value)
                        for value in current_anchor.position_maps[
                            int(self.model.selected_layers[0])
                        ].tolist()
                    ]
                    for layer in self.model.selected_layers:
                        layer = int(layer)
                        positions = [
                            int(value)
                            for value in current_anchor.position_maps[
                                layer
                            ].tolist()
                        ]
                        old_retained = set(
                            int(value)
                            for value in old_probe.position_maps[
                                layer
                            ].tolist()
                        )
                        fresh_retained = set(
                            int(value)
                            for value in fresh_probe.position_maps[
                                layer
                            ].tolist()
                        )
                        values_by_kv = (
                            current_anchor.values[layer]
                            .detach()[0]
                            .float()
                            .cpu()
                        )
                        attention_by_head = (
                            full_record.all_head_attention_distributions[
                                layer
                            ]
                            .detach()
                            .float()
                            .cpu()
                        )
                        full_outputs = (
                            full_record.all_head_attention_outputs[layer]
                            .detach()
                            .float()
                            .cpu()
                        )
                        group_size = int(
                            self.model.model_info[
                                "gqa_query_heads_per_kv_head"
                            ]
                        )
                        for head in self.model.selected_heads[layer]:
                            head = int(head)
                            kv_group = int(head // group_size)
                            values = values_by_kv[kv_group]
                            attention = attention_by_head[head]
                            basis = self._projection_bases[layer][head]
                            old_energy = self._uncovered_feature_energy(
                                attention,
                                values,
                                basis,
                                positions,
                                old_retained,
                            )
                            fresh_energy = self._uncovered_feature_energy(
                                attention,
                                values,
                                basis,
                                positions,
                                fresh_retained,
                            )
                            old_head_risk, old_proj_risk = direct_mask_risk(
                                attention,
                                values,
                                basis,
                                positions,
                                old_retained,
                                float(theory.identity_epsilon),
                            )
                            fresh_head_risk, fresh_proj_risk = (
                                direct_mask_risk(
                                    attention,
                                    values,
                                    basis,
                                    positions,
                                    fresh_retained,
                                    float(theory.identity_epsilon),
                                )
                            )
                            full_head = full_outputs[head]
                            old_stateful_error = self._squared_distance(
                                old_probe.diagnostic.all_head_attention_outputs[
                                    layer
                                ][head],
                                full_head,
                            )
                            fresh_stateful_error = self._squared_distance(
                                fresh_probe.diagnostic.all_head_attention_outputs[
                                    layer
                                ][head],
                                full_head,
                            )
                            old_stateful_delta = (
                                old_probe.diagnostic.all_head_attention_outputs[
                                    layer
                                ][head]
                                .detach()
                                .double()
                                .cpu()
                                - full_head.detach().double().cpu()
                            )
                            fresh_stateful_delta = (
                                fresh_probe.diagnostic.all_head_attention_outputs[
                                    layer
                                ][head]
                                .detach()
                                .double()
                                .cpu()
                                - full_head.detach().double().cpu()
                            )
                            old_stateful_proj_error = float(
                                (
                                    old_stateful_delta @ basis.double()
                                ).square().sum().item()
                            )
                            fresh_stateful_proj_error = float(
                                (
                                    fresh_stateful_delta @ basis.double()
                                ).square().sum().item()
                            )
                            feature_gaps = {
                                "%s_gap" % variant: float(
                                    old_energy[variant]
                                    - fresh_energy[variant]
                                )
                                for variant in theory.feature_variants
                            }
                            direct_head_benefit = float(
                                old_head_risk - fresh_head_risk
                            )
                            direct_proj_benefit = float(
                                old_proj_risk - fresh_proj_risk
                            )
                            stateful_head_benefit = float(
                                old_stateful_error
                                - fresh_stateful_error
                            )
                            stateful_proj_benefit = float(
                                old_stateful_proj_error
                                - fresh_stateful_proj_error
                            )
                            head_step_rows.append(
                                {
                                    **base_common,
                                    "protected_recent_size": (
                                        protected_recent
                                    ),
                                    "fresh_reference": fresh_reference,
                                    "horizon_offset": offset,
                                    "target_index": target,
                                    "target_token_id": int(
                                        old_probe.target_token_id
                                    ),
                                    "layer": layer,
                                    "head": head,
                                    "kv_head_group": kv_group,
                                    **feature_gaps,
                                    "direct_head_benefit": (
                                        direct_head_benefit
                                    ),
                                    "direct_proj_benefit": (
                                        direct_proj_benefit
                                    ),
                                    "stateful_head_benefit": (
                                        stateful_head_benefit
                                    ),
                                    "stateful_proj_benefit": (
                                        stateful_proj_benefit
                                    ),
                                    "old_direct_proj_risk": (
                                        old_proj_risk
                                    ),
                                    "fresh_direct_proj_risk": (
                                        fresh_proj_risk
                                    ),
                                    "same_reference_token_verified": True,
                                    "post_first_recent_exit": bool(
                                        offset > 32
                                    ),
                                    "recent_exit_distance": int(
                                        offset - 33
                                    ),
                                    **anchor_features[(layer, head)],
                                }
                            )
                            decomposition_rows.append(
                                {
                                    **base_common,
                                    "source_matrix": (
                                        "theory_closing_dense_budget128"
                                    ),
                                    "protected_recent_size": (
                                        protected_recent
                                    ),
                                    "fresh_reference": fresh_reference,
                                    "lag": offset,
                                    "target_index": target,
                                    "target_token_id": int(
                                        old_probe.target_token_id
                                    ),
                                    "layer": layer,
                                    "head": head,
                                    "kv_head_group": kv_group,
                                    "direct_head_benefit": (
                                        direct_head_benefit
                                    ),
                                    "stateful_head_benefit": (
                                        stateful_head_benefit
                                    ),
                                    "feedback_head": (
                                        stateful_head_benefit
                                        - direct_head_benefit
                                    ),
                                    "direct_proj_benefit": (
                                        direct_proj_benefit
                                    ),
                                    "stateful_proj_benefit": (
                                        stateful_proj_benefit
                                    ),
                                    "feedback_proj": (
                                        stateful_proj_benefit
                                        - direct_proj_benefit
                                    ),
                                    **feature_gaps,
                                    "same_reference_token_verified": True,
                                }
                            )
        head_steps = pd.DataFrame(head_step_rows)
        global_steps = pd.DataFrame(global_step_rows)
        head_values = [
            "%s_gap" % variant
            for variant in theory.feature_variants
        ] + [
            "direct_head_benefit",
            "direct_proj_benefit",
            "stateful_head_benefit",
            "stateful_proj_benefit",
        ]
        head_horizons = cumulative_rows(
            head_steps,
            theory.horizons,
            head_values,
            [
                "run_id",
                "model",
                "task",
                "sample_id",
                "seed",
                "config_hash",
                "git_commit",
                "anchor_step",
                "horizon_start_step",
                "total_budget",
                "selector",
                "protected_recent_size",
                "fresh_reference",
                "layer",
                "head",
                "kv_head_group",
            ],
        )
        head_horizons = head_horizons.rename(
            columns={
                "raw_v_gap": "future_raw_v_gap",
                "projected_v_gap": "future_projected_v_gap",
                "aov_gap": "future_aov_gap",
                "aor_gap": "future_aor_gap",
                "direct_head_benefit": (
                    "cumulative_direct_head_benefit"
                ),
                "direct_proj_benefit": (
                    "cumulative_direct_proj_benefit"
                ),
                "stateful_head_benefit": (
                    "cumulative_stateful_head_benefit"
                ),
                "stateful_proj_benefit": (
                    "cumulative_stateful_proj_benefit"
                ),
            }
        )
        head_horizons["row_type"] = "head_horizon"
        feature_columns = [
            column
            for column in head_steps.columns
            if column.startswith("anchor_")
        ]
        feature_lookup = (
            head_steps.sort_values("horizon_offset")
            .drop_duplicates(
                [
                    "sample_id",
                    "protected_recent_size",
                    "fresh_reference",
                    "layer",
                    "head",
                ]
            )
            [
                [
                    "sample_id",
                    "protected_recent_size",
                    "fresh_reference",
                    "layer",
                    "head",
                ]
                + feature_columns
            ]
        )
        head_horizons = head_horizons.merge(
            feature_lookup,
            on=[
                "sample_id",
                "protected_recent_size",
                "fresh_reference",
                "layer",
                "head",
            ],
            how="left",
            validate="many_to_one",
        )
        global_horizons = cumulative_rows(
            global_steps,
            theory.horizons,
            [
                "stateful_layer_projected_benefit",
                "stateful_nll_benefit",
                "stateful_kl_benefit",
                "old_delta_nll",
                "fresh_delta_nll",
            ],
            [
                "run_id",
                "model",
                "task",
                "sample_id",
                "seed",
                "config_hash",
                "git_commit",
                "anchor_step",
                "horizon_start_step",
                "total_budget",
                "selector",
                "protected_recent_size",
                "fresh_reference",
            ],
        ).rename(
            columns={
                "stateful_layer_projected_benefit": (
                    "cumulative_stateful_layer_projected_benefit"
                ),
                "stateful_nll_benefit": "cumulative_nll_benefit",
                "stateful_kl_benefit": "cumulative_kl_benefit",
                "old_delta_nll": "cumulative_old_delta_nll",
                "fresh_delta_nll": "cumulative_fresh_delta_nll",
            }
        )
        global_horizons["row_type"] = "global_stateful_horizon"
        horizon_rows = pd.concat(
            [head_horizons, global_horizons],
            ignore_index=True,
            sort=False,
        )
        horizon_rows["recent_exit_event_count"] = (
            horizon_rows["horizon"].astype(int) - 32
        ).clip(lower=0)
        return (
            horizon_rows,
            pd.DataFrame(decomposition_rows),
            pd.DataFrame(runtime_rows),
        )

    def _run_theory_sample(self, sample: Any) -> None:
        slug = _sample_slug(sample.sample_id)
        completion_key = "theory_closing:%s" % slug
        if self.cfg.runtime.resume and self.store.is_complete(completion_key):
            if all(
                self._theory_fragment_path(
                    table, sample.sample_id
                ).exists()
                for table in THEORY_TABLES
            ):
                return
        reference: Optional[ReferenceTrajectory] = None
        started = time.perf_counter()
        try:
            reference_started = time.perf_counter()
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            required = set(self.cfg.captured_anchor_steps())
            missing = sorted(
                value
                for value in required
                if value <= len(reference.generated_token_ids)
                and value not in reference.anchors
            )
            if missing:
                raise RuntimeError(
                    "theory-closing reference anchors are missing: %s"
                    % missing
                )
            reference_runtime = {
                **self._base(sample),
                "stage": "reference_generation",
                "wall_time_s": float(
                    time.perf_counter() - reference_started
                ),
                "prompt_length": int(reference.prompt_length),
                "generated_length": int(
                    len(reference.generated_token_ids)
                ),
                "captured_anchor_count": int(len(reference.anchors)),
                "selected_layers": json_text(reference.selected_layers),
                "selected_heads": json_text(reference.selected_heads),
                "peak_rss_bytes": int(reference.peak_rss_bytes),
                "peak_accelerator_bytes": (
                    reference.peak_accelerator_bytes
                ),
            }
            subset_started = time.perf_counter()
            subset_rows, inventory = self._subset_experiment(
                sample, reference
            )
            subset_runtime = {
                **self._base(sample),
                "stage": "fixed_qkv_subset_enumeration",
                "wall_time_s": float(
                    time.perf_counter() - subset_started
                ),
                "subset_rows": int(len(subset_rows)),
                "subset_units": int(subset_rows["unit_id"].nunique()),
            }
            horizon_started = time.perf_counter()
            horizon_rows, decomposition, replay_runtime = (
                self._horizon_experiment(sample, reference)
            )
            horizon_runtime = {
                **self._base(sample),
                "stage": "future_oracle_horizon",
                "wall_time_s": float(
                    time.perf_counter() - horizon_started
                ),
                "horizon_rows": int(len(horizon_rows)),
                "decomposition_rows": int(len(decomposition)),
            }
            runtime = pd.concat(
                [
                    pd.DataFrame(
                        [
                            reference_runtime,
                            subset_runtime,
                            horizon_runtime,
                            {
                                **self._base(sample),
                                "stage": "sample_total",
                                "wall_time_s": float(
                                    time.perf_counter() - started
                                ),
                            },
                        ]
                    ),
                    replay_runtime,
                ],
                ignore_index=True,
                sort=False,
            )
            self._write_theory_tables(
                sample.sample_id,
                {
                    "subset_objective_rows": subset_rows,
                    "subset_unit_inventory": inventory,
                    "future_oracle_horizon_rows": horizon_rows,
                    "direct_stateful_decomposition": decomposition,
                    "theory_runtime": runtime,
                },
            )
            self.store.mark_complete(
                completion_key,
                {
                    "valid": True,
                    "subset_rows": int(len(subset_rows)),
                    "horizon_rows": int(len(horizon_rows)),
                    "elapsed_s": float(time.perf_counter() - started),
                },
            )
        except Exception as exc:
            self.store.append_error(
                {
                    **self._base(sample),
                    "key": completion_key,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "time": time.time(),
                }
            )
            self.store.mark_failed(
                completion_key,
                "%s: %s" % (type(exc).__name__, exc),
            )
            if self.cfg.runtime.fail_on_error:
                raise
        finally:
            if reference is not None:
                self.model.release(reference)
