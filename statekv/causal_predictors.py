"""Learned causal future-utility predictors for the StateKV existence study."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from statekv.causal_existence_analysis import (
    aggregate_sequence_metrics,
    boundary_metrics,
    topk_indices,
)
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json


HISTORY_WIDTH = 8
HISTORY_SCALAR_WIDTH = 16


def feature_groups(width: int = 120) -> Dict[str, np.ndarray]:
    if int(width) != 120:
        raise ValueError("unexpected causal feature width")
    return {
        "A_history_J_statekv": np.arange(0, 16),
        "B_current_query": np.r_[0:16, 40:56],
        "C_token_key": np.r_[0:16, 20:32, 40:56],
        "D_qk_geometry": np.r_[0:32, 40:56],
        "E_value_state": np.arange(0, 56),
        "F_current_state": np.arange(0, 80),
        "H_query_trajectory": np.arange(0, 112),
        "I_global_full": np.arange(0, 120),
    }


@dataclass
class Boundary:
    sample_id: str
    task: str
    split: str
    cycle: int
    layer: int
    head: int
    features: np.ndarray
    history: np.ndarray
    truth: np.ndarray
    binary: np.ndarray
    baseline: np.ndarray


class FixedProjector:
    """Public-seed projections; fitted normalization remains train-only."""

    def __init__(self, seed: int = 20260820, width: int = 8):
        rng = np.random.default_rng(int(seed))
        self.k = rng.normal(size=(128, width)).astype(np.float32) / math.sqrt(128)
        self.v = rng.normal(size=(128, width)).astype(np.float32) / math.sqrt(128)
        self.q = rng.normal(size=(128, width)).astype(np.float32) / math.sqrt(128)
        self.state = rng.normal(size=(4096, width)).astype(np.float32) / math.sqrt(4096)


def _history_features(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Token features using only rows at or before the current cycle."""

    rows = np.asarray(rows, dtype=np.float32)
    observed = np.isfinite(rows)
    values = np.nan_to_num(rows, nan=0.0)
    current = values[-1]
    lag1 = values[-2] if len(values) >= 2 else current
    lag4 = values[-5] if len(values) >= 5 else values[0]
    counts = np.maximum(observed.sum(axis=0), 1)
    mean = values.sum(axis=0) / counts
    variance = (
        np.where(observed, (values - mean[None, :]) ** 2, 0.0).sum(axis=0)
        / counts
    )
    maximum = np.where(observed, values, -np.inf).max(axis=0)
    maximum[~np.isfinite(maximum)] = 0.0
    slope = current - lag4
    ema = ema_score(rows, 0.9)
    fast = ema_score(rows, 0.5)
    slow = ema_score(rows, 0.99)
    order = np.lexsort((np.arange(current.size), -current))
    rank = np.empty(current.size, dtype=np.float32)
    rank[order] = (np.arange(current.size) + 1) / max(1, current.size)
    lag_order = np.lexsort((np.arange(lag1.size), -lag1))
    lag_rank = np.empty(lag1.size, dtype=np.float32)
    lag_rank[lag_order] = (np.arange(lag1.size) + 1) / max(1, lag1.size)
    previously_present = observed[-2] if len(observed) >= 2 else observed[-1]
    rank_drift = np.where(previously_present, rank - lag_rank, 0.0)
    recent4 = values[-4:]
    recent4_observed = observed[-4:]
    recent4_counts = np.maximum(recent4_observed.sum(axis=0), 1)
    recent4_mean = recent4.sum(axis=0) / recent4_counts
    recent4_variance = (
        np.where(
            recent4_observed,
            (recent4 - recent4_mean[None, :]) ** 2,
            0.0,
        ).sum(axis=0)
        / recent4_counts
    )
    surprise = np.abs(current - lag1) / np.sqrt(variance + 1.0e-12)
    scalar = np.stack(
        [
            current,
            values.sum(axis=0),
            mean,
            maximum,
            lag1,
            lag4,
            slope,
            variance,
            ema,
            fast,
            slow,
            surprise,
            rank,
            rank_drift,
            recent4_mean,
            np.sqrt(recent4_variance),
        ],
        axis=1,
    )
    temporal = np.zeros((current.size, HISTORY_WIDTH, 2), dtype=np.float32)
    recent = values[-HISTORY_WIDTH:]
    recent_observed = observed[-HISTORY_WIDTH:]
    temporal[:, -len(recent) :, 0] = recent.T
    temporal[:, -len(recent) :, 1] = recent_observed.T.astype(np.float32)
    return scalar.astype(np.float32), temporal


def ema_score(rows: np.ndarray, rho: float) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    observed = np.isfinite(rows)
    values = np.nan_to_num(rows, nan=0.0)
    memory = np.zeros(values.shape[1], dtype=np.float64)
    seen = np.zeros(values.shape[1], dtype=bool)
    for row, present in zip(values, observed):
        first = present & ~seen
        continuing = present & seen
        memory[first] = row[first]
        memory[continuing] = (
            float(rho) * memory[continuing]
            + (1.0 - float(rho)) * row[continuing]
        )
        seen |= present
    return memory.astype(np.float32)


def _rho_key(horizon: int, layer_index: int, head: int) -> str:
    return f"h{int(horizon)}:l{int(layer_index)}:head{int(head)}"


def _future_truth(
    attention: np.ndarray,
    position_ids: np.ndarray,
    position_lengths: np.ndarray,
    cycle: int,
    layer_index: int,
    head: int,
    candidate_positions: Sequence[int],
    horizons: Sequence[int],
) -> np.ndarray:
    per_step = []
    for future_cycle in range(cycle + 1, cycle + max(horizons) + 1):
        count = int(position_lengths[future_cycle])
        row_by_position = {
            int(position): row
            for row, position in enumerate(position_ids[future_cycle, :count])
        }
        columns = [row_by_position[int(position)] for position in candidate_positions]
        per_step.append(
            np.take(
                attention[future_cycle, layer_index, head], columns, axis=-1
            )
        )
    return np.stack(
        [np.asarray(per_step[: int(horizon)], dtype=np.float64).sum(axis=0) for horizon in horizons],
        axis=1,
    ).astype(np.float32)


def artifact_boundary(
    artifact: Mapping[str, np.ndarray],
    cycle: int,
    layer_index: int,
    head: int,
    horizons: Sequence[int],
    sink_size: int,
    recent_size: int,
    core_budget: int,
    projector: FixedProjector,
    baseline_rhos: Optional[Mapping[str, float]] = None,
    feature_only: bool = False,
) -> Boundary:
    count = int(artifact["position_lengths"][cycle])
    positions = [int(value) for value in artifact["position_ids"][cycle, :count]]
    _, _, eligible = mandatory_and_eligible(positions, sink_size, recent_size)
    row_by_position = {position: row for row, position in enumerate(positions)}
    candidate_rows = np.asarray(
        [row_by_position[int(position)] for position in eligible], dtype=np.int64
    )
    layer = int(artifact["layers"][layer_index])
    attention = artifact["attention"]
    history_rows = attention[: cycle + 1, layer_index, head, candidate_rows]
    scalar, temporal = _history_features(history_rows)
    if feature_only:
        truth = np.empty((len(candidate_rows), 0), dtype=np.float32)
        binary = np.empty_like(truth)
        baseline = np.empty_like(truth)
    else:
        truth = _future_truth(
            attention,
            artifact["position_ids"],
            artifact["position_lengths"],
            cycle,
            layer_index,
            head,
            eligible,
            horizons,
        )
        binary = np.zeros_like(truth, dtype=np.float32)
        for column in range(len(horizons)):
            binary[topk_indices(truth[:, column], core_budget), column] = 1.0
        baseline = np.stack(
            [
                ema_score(
                    history_rows,
                    (
                        0.0
                        if baseline_rhos is None
                        else float(
                            baseline_rhos[
                                _rho_key(horizon, layer_index, head)
                            ]
                        )
                    ),
                )
                for horizon in horizons
            ],
            axis=1,
        )

    kv_row_by_position = {
        int(position): row
        for row, position in enumerate(artifact["kv_position_ids"])
    }
    kv_rows = np.asarray(
        [kv_row_by_position[int(position)] for position in eligible], dtype=np.int64
    )
    keys = np.asarray(
        artifact["keys"][layer_index, head, kv_rows], dtype=np.float32
    )
    values = np.asarray(
        artifact["values"][layer_index, head, kv_rows], dtype=np.float32
    )
    query_heads = int(artifact["query_post"].shape[2])
    kv_heads = int(artifact["attention"].shape[2])
    group = query_heads // kv_heads
    query = np.asarray(
        artifact["query_post"][
            cycle, layer_index, head * group : (head + 1) * group
        ],
        dtype=np.float32,
    )
    dots = keys @ query.T / math.sqrt(keys.shape[1])
    geometry = np.stack(
        [dots.mean(axis=1), dots.max(axis=1), dots.min(axis=1), dots.std(axis=1)],
        axis=1,
    ).astype(np.float32)
    k_projected = keys @ projector.k
    v_projected = values @ projector.v
    query_mean = query.mean(axis=0) @ projector.q
    query_std = query.std(axis=0) @ projector.q
    residual_raw = np.asarray(
        artifact["residual"][cycle, layer_index], dtype=np.float32
    )
    attention_input_raw = np.asarray(
        artifact["attention_input"][cycle, layer_index], dtype=np.float32
    )
    residual = residual_raw @ projector.state
    attention_input = attention_input_raw @ projector.state
    state_statistics = np.asarray(
        [
            residual_raw.mean(),
            residual_raw.std(),
            np.linalg.norm(residual_raw) / math.sqrt(residual_raw.size),
            np.max(np.abs(residual_raw)),
            attention_input_raw.mean(),
            attention_input_raw.std(),
            np.linalg.norm(attention_input_raw) / math.sqrt(attention_input_raw.size),
            np.max(np.abs(attention_input_raw)),
        ],
        dtype=np.float32,
    )
    current_query = query.mean(axis=0)
    query_trajectory = []
    for lag in (1, 2, 4, 8):
        previous_cycle = max(0, cycle - lag)
        previous_query = np.asarray(
            artifact["query_post"][
                previous_cycle,
                layer_index,
                head * group : (head + 1) * group,
            ],
            dtype=np.float32,
        ).mean(axis=0)
        query_trajectory.append((current_query - previous_query) @ projector.q)
    global_features = np.asarray(
        artifact["global_features"][cycle], dtype=np.float32
    )
    position_values = np.asarray(eligible, dtype=np.float32)
    relative_position = position_values / max(1.0, float(max(positions)))
    age = (float(max(positions)) - position_values) / max(
        1.0, float(max(positions))
    )
    token_norms = np.stack(
        [
            np.linalg.norm(keys, axis=1),
            np.linalg.norm(values, axis=1),
            relative_position,
            age,
        ],
        axis=1,
    ).astype(np.float32)
    repeated_state = np.broadcast_to(
        np.concatenate(
            [
                query_mean,
                query_std,
                residual,
                attention_input,
                state_statistics,
                *query_trajectory,
                global_features,
                np.asarray(
                    [
                        cycle / max(1, int(attention.shape[0]) - 1),
                        math.log1p(count) / math.log1p(4096),
                        layer_index / max(1, int(attention.shape[1]) - 1),
                        head / max(1, kv_heads - 1),
                    ],
                    dtype=np.float32,
                ),
            ]
        ),
        (len(candidate_rows), 80),
    )
    features = np.concatenate(
        [scalar, geometry, token_norms, k_projected, v_projected, repeated_state],
        axis=1,
    ).astype(np.float32)
    return Boundary(
        sample_id=str(artifact["sample_id"].item()),
        task=str(artifact["task"].item()),
        split=str(artifact["split"].item()),
        cycle=int(cycle),
        layer=layer,
        head=int(head),
        features=features,
        history=temporal,
        truth=truth,
        binary=binary,
        baseline=baseline,
    )


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def boundary_specs(
    paths: Sequence[Path], cycles: Sequence[int]
) -> List[Tuple[Path, int, int, int]]:
    specs: List[Tuple[Path, int, int, int]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as artifact:
            layer_count = int(artifact["layers"].size)
            head_count = int(artifact["attention"].shape[2])
            total_cycles = int(artifact["attention"].shape[0])
        specs.extend(
            (path, int(cycle), layer_index, head)
            for cycle in cycles
            if int(cycle) < total_cycles
            for layer_index in range(layer_count)
            for head in range(head_count)
        )
    return specs


def sampled_training_arrays(
    paths: Sequence[Path],
    config: Mapping[str, Any],
    maximum_boundaries: int = 1500,
    tokens_per_boundary: int = 384,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(config["data_seed"]))
    cycles = list(range(0, int(config["control_cycles"]) - 32, 8))
    specs = boundary_specs(paths, cycles)
    if len(specs) > int(maximum_boundaries):
        chosen = np.sort(
            rng.choice(len(specs), size=int(maximum_boundaries), replace=False)
        )
        specs = [specs[int(index)] for index in chosen]
    projector = FixedProjector(int(config["data_seed"]))
    features: List[np.ndarray] = []
    histories: List[np.ndarray] = []
    truth: List[np.ndarray] = []
    binary: List[np.ndarray] = []
    boundary_ids: List[np.ndarray] = []
    current_path: Optional[Path] = None
    artifact: Optional[Dict[str, np.ndarray]] = None
    for boundary_id, (path, cycle, layer_index, head) in enumerate(specs):
        if current_path != path:
            artifact = _load_npz(path)
            current_path = path
        assert artifact is not None
        boundary = artifact_boundary(
            artifact,
            cycle,
            layer_index,
            head,
            config["future_utility_horizons"],
            int(config["sink_size"]),
            int(config["recent_size"]),
            int(config["core_budget"]),
            projector,
        )
        take = min(int(tokens_per_boundary), len(boundary.features))
        rows = np.sort(rng.choice(len(boundary.features), size=take, replace=False))
        features.append(boundary.features[rows])
        histories.append(boundary.history[rows])
        truth.append(boundary.truth[rows])
        binary.append(boundary.binary[rows])
        boundary_ids.append(np.full(take, boundary_id, dtype=np.int32))
    return (
        np.concatenate(features),
        np.concatenate(histories),
        np.concatenate(truth),
        np.concatenate(binary),
        np.concatenate(boundary_ids),
    )


def tune_fixed_baselines(
    paths: Sequence[Path], config: Mapping[str, Any], output_root: Path
) -> Path:
    """Tune global, task, and per-head fixed EMA only on train sequences."""

    horizons = sorted(
        {
            int(value)
            for value in (
                list(config["future_utility_horizons"])
                + list(config.get("causal_rollout", {}).get("horizons", []))
            )
        }
    )
    rhos = [float(value) for value in config["fixed_baseline_rhos"]]
    cycles = list(range(0, int(config["control_cycles"]) - max(horizons), 8))
    captures: Dict[str, Dict[float, float]] = {}
    oracle: Dict[str, float] = {}

    def add(key: str, rho: float, value: float, oracle_value: float) -> None:
        captures.setdefault(key, {candidate: 0.0 for candidate in rhos})
        captures[key][float(rho)] += float(value)
        oracle.setdefault(key, 0.0)
        if float(rho) == rhos[0]:
            oracle[key] += float(oracle_value)

    for path_ordinal, path in enumerate(paths, start=1):
        artifact = _load_npz(path)
        task = str(artifact["task"].item())
        for cycle in cycles:
            count = int(artifact["position_lengths"][cycle])
            positions = [
                int(value) for value in artifact["position_ids"][cycle, :count]
            ]
            _, _, eligible = mandatory_and_eligible(
                positions, int(config["sink_size"]), int(config["recent_size"])
            )
            row_by_position = {
                position: row for row, position in enumerate(positions)
            }
            candidate_rows = np.asarray(
                [row_by_position[int(position)] for position in eligible],
                dtype=np.int64,
            )
            for layer_index in range(int(artifact["layers"].size)):
                for head in range(int(artifact["attention"].shape[2])):
                    history = artifact["attention"][
                        : cycle + 1, layer_index, head
                    ][:, candidate_rows]
                    for horizon in horizons:
                        truth = np.take(
                            artifact["attention"][
                                cycle + 1 : cycle + horizon + 1,
                                layer_index,
                                head,
                            ],
                            candidate_rows,
                            axis=-1,
                        ).sum(axis=0)
                        oracle_value = float(
                            truth[topk_indices(truth, int(config["core_budget"]))].sum()
                        )
                        keys = (
                            f"global:h{horizon}",
                            f"task:{task}:h{horizon}",
                            f"head:{_rho_key(horizon, layer_index, head)}",
                        )
                        for rho in rhos:
                            score = ema_score(history, rho)
                            captured = float(
                                truth[
                                    topk_indices(score, int(config["core_budget"]))
                                ].sum()
                            )
                            for key in keys:
                                add(key, rho, captured, oracle_value)
        if path_ordinal % 10 == 0 or path_ordinal == len(paths):
            print(
                f"[fixed-baseline] tuned contributions {path_ordinal}/{len(paths)}",
                flush=True,
            )
    chosen = {
        key: min(
            values,
            key=lambda rho: (-float(values[rho]), rhos.index(float(rho))),
        )
        for key, values in captures.items()
    }
    per_head = {
        key.removeprefix("head:"): float(value)
        for key, value in chosen.items()
        if key.startswith("head:")
    }
    path = output_root / "models" / "fixed_baseline_tuning.json"
    atomic_json(
        path,
        {
            "split": "train",
            "candidate_rhos": rhos,
            "cycles": cycles,
            "global": {
                key.removeprefix("global:"): float(value)
                for key, value in chosen.items()
                if key.startswith("global:")
            },
            "task": {
                key.removeprefix("task:"): float(value)
                for key, value in chosen.items()
                if key.startswith("task:")
            },
            "per_head": per_head,
            "oracle_capture": oracle,
            "future_information_at_runtime": False,
            "future_labels_used_for_train_tuning": True,
        },
    )
    return path


class MultiHorizonMLP(nn.Module):
    def __init__(self, input_width: int, horizons: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, 192),
            nn.GELU(),
            nn.LayerNorm(192),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Linear(96, horizons * 2),
        )
        self.horizons = int(horizons)

    def forward(self, features: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        del history
        return self.network(features).reshape(-1, self.horizons, 2)


class DeepSetPredictor(nn.Module):
    def __init__(self, input_width: int, horizons: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_width, 96), nn.GELU())
        self.scorer = nn.Sequential(
            nn.Linear(input_width + 192, 128),
            nn.GELU(),
            nn.Linear(128, horizons * 2),
        )
        self.horizons = int(horizons)

    def forward(self, features: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        del history
        encoded = self.encoder(features)
        context = encoded.mean(dim=0, keepdim=True).expand_as(encoded)
        maximum = encoded.max(dim=0, keepdim=True).values.expand_as(encoded)
        output = self.scorer(torch.cat([features, context, maximum], dim=1))
        return output.reshape(-1, self.horizons, 2)


class TemporalGRUPredictor(nn.Module):
    def __init__(self, input_width: int, horizons: int):
        super().__init__()
        self.gru = nn.GRU(2, 24, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(input_width + 24, 160),
            nn.GELU(),
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, horizons * 2),
        )
        self.horizons = int(horizons)

    def forward(self, features: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(history)
        output = self.scorer(torch.cat([features, state[-1]], dim=1))
        return output.reshape(-1, self.horizons, 2)


def _train_neural(
    name: str,
    model: nn.Module,
    features: np.ndarray,
    histories: np.ndarray,
    truth: np.ndarray,
    binary: np.ndarray,
    boundary_ids: np.ndarray,
    output_root: Path,
    seed: int,
    epochs: int = 3,
) -> Path:
    torch.manual_seed(int(seed))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    bce = nn.BCEWithLogitsLoss()
    smooth = nn.SmoothL1Loss()
    regression = np.log1p(truth / np.maximum(truth.mean(axis=0), 1.0e-9))
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        groups = [
            np.flatnonzero(boundary_ids == boundary_id)
            for boundary_id in rng.permutation(np.unique(boundary_ids))
        ]
        for rows in groups:
            x = torch.from_numpy(features[rows]).to(device)
            h = torch.from_numpy(histories[rows]).to(device)
            y_class = torch.from_numpy(binary[rows]).to(device)
            y_reg = torch.from_numpy(regression[rows].astype(np.float32)).to(device)
            output = model(x, h)
            loss = bce(output[:, :, 0], y_class) + 0.25 * smooth(
                output[:, :, 1], y_reg
            )
            if len(rows) > 1:
                paired = torch.roll(torch.arange(len(rows), device=device), 1)
                true_difference = y_reg - y_reg[paired]
                predicted_difference = output[:, :, 1] - output[paired, :, 1]
                informative = true_difference.abs() > 1.0e-6
                if bool(informative.any()):
                    pairwise = torch.nn.functional.softplus(
                        -true_difference.sign() * predicted_difference
                    )[informative].mean()
                    loss = loss + 0.10 * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"[causal-predictor] {name} epoch {epoch + 1}/{epochs}", flush=True)
    path = output_root / "models" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), path)
    return path


def train_causal_predictors(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    train_paths = sorted((output_root / "artifacts" / "train").glob("*.npz"))
    expected = int(config["expected_split_sizes"]["train"])
    if len(train_paths) != expected:
        raise RuntimeError(f"expected {expected} train artifacts, found {len(train_paths)}")
    started = time.perf_counter()
    features, histories, truth, binary, boundary_ids = sampled_training_arrays(
        train_paths, config
    )
    print(
        f"[causal-predictor] sampled {len(features)} token rows from {len(train_paths)} sequences",
        flush=True,
    )
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features).astype(np.float32)
    model_root = output_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    tune_fixed_baselines(train_paths, config, output_root)
    joblib.dump(scaler, model_root / "train_only_scaler.joblib")
    horizons = [int(value) for value in config["future_utility_horizons"]]
    linear_models: Dict[str, Any] = {}
    history_width = HISTORY_SCALAR_WIDTH
    groups = feature_groups(int(normalized.shape[1]))
    for column, horizon in enumerate(horizons):
        print(f"[causal-predictor] classical horizon H={horizon}", flush=True)
        linear_models[f"history_logistic_h{horizon}"] = LogisticRegression(
            max_iter=250, class_weight="balanced", random_state=int(config["data_seed"])
        ).fit(normalized[:, :history_width], binary[:, column])
        linear_models[f"ridge_h{horizon}"] = Ridge(alpha=1.0).fit(
            normalized, np.log1p(truth[:, column])
        )
        for group_name, columns in groups.items():
            linear_models[f"feature_ridge_{group_name}_h{horizon}"] = Ridge(
                alpha=1.0
            ).fit(normalized[:, columns], np.log1p(truth[:, column]))
        subset = np.linspace(
            0, len(normalized) - 1, num=min(200000, len(normalized)), dtype=np.int64
        )
        linear_models[f"hist_gbdt_h{horizon}"] = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=int(config["data_seed"]),
        ).fit(normalized[subset], binary[subset, column])
    joblib.dump(linear_models, model_root / "classical_predictors.joblib")

    input_width = int(normalized.shape[1])
    horizon_count = len(horizons)
    _train_neural(
        "query_conditioned_mlp",
        MultiHorizonMLP(input_width, horizon_count),
        normalized,
        histories,
        truth,
        binary,
        boundary_ids,
        output_root,
        int(config["data_seed"]),
    )
    # DeepSets receives one sampled decision set per optimization step and one
    # complete decision set at evaluation.
    _train_neural(
        "deepsets",
        DeepSetPredictor(input_width, horizon_count),
        normalized,
        histories,
        truth,
        binary,
        boundary_ids,
        output_root,
        int(config["data_seed"]) + 1,
    )
    _train_neural(
        "temporal_gru",
        TemporalGRUPredictor(input_width, horizon_count),
        normalized,
        histories,
        truth,
        binary,
        boundary_ids,
        output_root,
        int(config["data_seed"]) + 2,
    )
    atomic_json(
        output_root / "training_summary.json",
        {
            "train_sequences": len(train_paths),
            "sampled_token_rows": int(len(features)),
            "feature_width": input_width,
            "normalization": "train-only StandardScaler",
            "neural_objectives": [
                "future_topk_binary_cross_entropy",
                "log_future_utility_regression",
                "within_boundary_pairwise_ranking",
                "multi_horizon_auxiliary",
            ],
            "future_labels_in_runtime_features": False,
            "models": sorted(linear_models)
            + ["query_conditioned_mlp", "deepsets", "temporal_gru"],
            "feature_groups": {
                name: [int(value) for value in columns]
                for name, columns in groups.items()
            },
            "elapsed_s": float(time.perf_counter() - started),
        },
    )
    return output_root


def _load_neural(name: str, input_width: int, horizon_count: int, path: Path) -> nn.Module:
    classes = {
        "query_conditioned_mlp": MultiHorizonMLP,
        "deepsets": DeepSetPredictor,
        "temporal_gru": TemporalGRUPredictor,
    }
    model = classes[name](input_width, horizon_count)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def evaluate_causal_predictors(
    config_path: Path,
    repository_root: Path,
    split: str = "validation",
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    paths = sorted((output_root / "artifacts" / str(split)).glob("*.npz"))
    expected = int(config["expected_split_sizes"][str(split)])
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} {split} artifacts, found {len(paths)}")
    if str(split) == "fresh_test":
        from statekv.existence_reporting import register_fresh_test_component

        register_fresh_test_component(output_root, "learned_predictors")
    scaler = joblib.load(output_root / "models" / "train_only_scaler.joblib")
    classical = joblib.load(output_root / "models" / "classical_predictors.joblib")
    horizons = [int(value) for value in config["future_utility_horizons"]]
    input_width = int(scaler.n_features_in_)
    neural = {
        name: _load_neural(
            name,
            input_width,
            len(horizons),
            output_root / "models" / f"{name}.pt",
        )
        for name in ("query_conditioned_mlp", "deepsets", "temporal_gru")
    }
    projector = FixedProjector(int(config["data_seed"]))
    groups = feature_groups(input_width)
    fixed_baseline = json.loads(
        (output_root / "models" / "fixed_baseline_tuning.json").read_text(
            encoding="utf-8"
        )
    )["per_head"]
    rows: List[Dict[str, Any]] = []
    prediction_time: Dict[str, float] = {"best_per_head_fixed_ema": 0.0}
    prediction_calls: Dict[str, int] = {"best_per_head_fixed_ema": 0}
    cycles = list(range(0, int(config["control_cycles"]) - 32, 8))
    for path_ordinal, path in enumerate(paths, start=1):
        artifact = _load_npz(path)
        for cycle in cycles:
            for layer_index in range(int(artifact["layers"].size)):
                for head in range(int(artifact["attention"].shape[2])):
                    boundary = artifact_boundary(
                        artifact,
                        cycle,
                        layer_index,
                        head,
                        horizons,
                        int(config["sink_size"]),
                        int(config["recent_size"]),
                        int(config["core_budget"]),
                        projector,
                        fixed_baseline,
                    )
                    normalized = scaler.transform(boundary.features).astype(np.float32)
                    predictions: Dict[str, np.ndarray] = {
                        "best_per_head_fixed_ema": boundary.baseline.copy()
                    }
                    prediction_calls["best_per_head_fixed_ema"] += 1
                    for column, horizon in enumerate(horizons):
                        prediction_started = time.perf_counter()
                        predictions.setdefault(
                            "history_logistic", np.zeros_like(boundary.truth)
                        )[:, column] = classical[
                            f"history_logistic_h{horizon}"
                        ].predict_proba(normalized[:, :HISTORY_SCALAR_WIDTH])[:, 1]
                        prediction_time["history_logistic"] = prediction_time.get(
                            "history_logistic", 0.0
                        ) + (time.perf_counter() - prediction_started)
                        prediction_calls["history_logistic"] = prediction_calls.get(
                            "history_logistic", 0
                        ) + 1
                        prediction_started = time.perf_counter()
                        predictions.setdefault(
                            "ridge", np.zeros_like(boundary.truth)
                        )[:, column] = classical[f"ridge_h{horizon}"].predict(normalized)
                        prediction_time["ridge"] = prediction_time.get(
                            "ridge", 0.0
                        ) + (time.perf_counter() - prediction_started)
                        prediction_calls["ridge"] = prediction_calls.get("ridge", 0) + 1
                        prediction_started = time.perf_counter()
                        predictions.setdefault(
                            "hist_gbdt", np.zeros_like(boundary.truth)
                        )[:, column] = classical[
                            f"hist_gbdt_h{horizon}"
                        ].predict_proba(normalized)[:, 1]
                        prediction_time["hist_gbdt"] = prediction_time.get(
                            "hist_gbdt", 0.0
                        ) + (time.perf_counter() - prediction_started)
                        prediction_calls["hist_gbdt"] = prediction_calls.get(
                            "hist_gbdt", 0
                        ) + 1
                        for group_name, columns in groups.items():
                            method_name = f"feature_ridge_{group_name}"
                            prediction_started = time.perf_counter()
                            predictions.setdefault(
                                method_name,
                                np.zeros_like(boundary.truth),
                            )[:, column] = classical[
                                f"feature_ridge_{group_name}_h{horizon}"
                            ].predict(normalized[:, columns])
                            prediction_time[method_name] = prediction_time.get(
                                method_name, 0.0
                            ) + (time.perf_counter() - prediction_started)
                            prediction_calls[method_name] = prediction_calls.get(
                                method_name, 0
                            ) + 1
                    x = torch.from_numpy(normalized)
                    history = torch.from_numpy(boundary.history)
                    with torch.no_grad():
                        for name, model in neural.items():
                            prediction_started = time.perf_counter()
                            output = model(x, history).numpy()
                            predictions[name] = output[:, :, 0]
                            predictions[f"{name}_utility"] = output[:, :, 1]
                            prediction_time[name] = prediction_time.get(name, 0.0) + (
                                time.perf_counter() - prediction_started
                            )
                            prediction_time[f"{name}_utility"] = prediction_time.get(
                                f"{name}_utility", 0.0
                            ) + (time.perf_counter() - prediction_started)
                            prediction_calls[name] = prediction_calls.get(name, 0) + 1
                            prediction_calls[f"{name}_utility"] = prediction_calls.get(
                                f"{name}_utility", 0
                            ) + 1
                    for name, scores in predictions.items():
                        for column, horizon in enumerate(horizons):
                            metrics = boundary_metrics(
                                boundary.truth[:, column],
                                scores[:, column],
                                boundary.baseline[:, column],
                                int(config["core_budget"]),
                            )
                            rows.append(
                                {
                                    "sample_id": boundary.sample_id,
                                    "task": boundary.task,
                                    "split": split,
                                    "cycle": cycle,
                                    "layer": boundary.layer,
                                    "head": head,
                                    "method": name,
                                    "future_horizon": horizon,
                                    **metrics,
                                }
                            )
        print(
            f"[causal-evaluation] {split} sample {path_ordinal}/{len(paths)} {path.stem}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    evaluation_root = output_root / "evaluation" / str(split)
    atomic_frame(frame, evaluation_root / "boundary_metrics.parquet")
    sequence = aggregate_sequence_metrics(frame)
    atomic_frame(sequence, evaluation_root / "sequence_metrics.csv")
    summary = (
        sequence.groupby(["method", "future_horizon"], as_index=False)
        .agg(
            future_recall=("future_topk_recall", "mean"),
            spearman=("spearman", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            ndcg=("ndcg", "mean"),
            oracle_gap_recovery=("oracle_gap_recovery", "mean"),
            sequence_win_rate=("beats_baseline", "mean"),
            sequences=("sample_id", "nunique"),
        )
    )
    atomic_frame(summary, evaluation_root / "summary.csv")
    step_paths = [path.with_suffix(".steps.parquet") for path in paths]
    mean_scoring_s = float(
        np.mean(
            [
                value
                for step_path in step_paths
                for value in pd.read_parquet(step_path)[
                    "scoring_forward_time_s"
                ].tolist()
            ]
        )
    )
    cost_frame = pd.DataFrame(
        [
            {
                "method": method,
                "total_prediction_time_s": float(total),
                "prediction_calls": int(prediction_calls[method]),
                "mean_prediction_time_s": float(
                    total
                    / max(1, prediction_calls["best_per_head_fixed_ema"])
                ),
                "mean_model_scoring_forward_time_s": mean_scoring_s,
                "runtime_multiplier": float(
                    1.0
                    + total
                    / max(1, prediction_calls["best_per_head_fixed_ema"])
                    / max(mean_scoring_s, 1.0e-9)
                ),
            }
            for method, total in sorted(prediction_time.items())
        ]
    )
    atomic_frame(cost_frame, evaluation_root / "inference_costs.csv")
    return evaluation_root


__all__ = [
    "Boundary",
    "DeepSetPredictor",
    "FixedProjector",
    "MultiHorizonMLP",
    "TemporalGRUPredictor",
    "artifact_boundary",
    "evaluate_causal_predictors",
    "feature_groups",
    "sampled_training_arrays",
    "train_causal_predictors",
]
