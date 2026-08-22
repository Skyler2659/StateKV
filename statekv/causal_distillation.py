"""Distill the expensive causal rollout teacher into a state-only scorer."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from statekv.causal_existence import _safe_sample_id, sample_id_for
from statekv.causal_existence_analysis import aggregate_sequence_metrics, boundary_metrics, topk_indices
from statekv.causal_predictors import (
    FixedProjector,
    MultiHorizonMLP,
    _load_neural,
    _load_npz,
    _train_neural,
    artifact_boundary,
)
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json


def _teacher_path(teacher_root: Path, artifact_path: Path) -> Path:
    return teacher_root / artifact_path.name


def _teacher_arrays(
    artifact_paths: Sequence[Path],
    teacher_root: Path,
    config: Mapping[str, Any],
    maximum_boundaries: int = 1500,
    tokens_per_boundary: int = 384,
    return_sizes: bool = False,
):
    rng = np.random.default_rng(int(config["data_seed"]) + 41)
    horizons = [int(value) for value in config["future_utility_horizons"]]
    specs: List[Tuple[Path, Path, int, int, int]] = []
    for artifact_path in artifact_paths:
        teacher_path = _teacher_path(teacher_root, artifact_path)
        if not teacher_path.exists():
            raise RuntimeError(f"missing causal teacher scores: {teacher_path.name}")
        with np.load(artifact_path, allow_pickle=False) as artifact, np.load(
            teacher_path, allow_pickle=False
        ) as teacher:
            layer_count = int(artifact["layers"].size)
            head_count = int(artifact["attention"].shape[2])
            cycles = [int(value) for value in teacher["cycles"]]
        specs.extend(
            (artifact_path, teacher_path, cycle_index, layer_index, head)
            for cycle_index in range(len(cycles))
            for layer_index in range(layer_count)
            for head in range(head_count)
        )
    if len(specs) > int(maximum_boundaries):
        selected = np.sort(
            rng.choice(len(specs), size=int(maximum_boundaries), replace=False)
        )
        specs = [specs[int(index)] for index in selected]

    projector = FixedProjector(int(config["data_seed"]))
    features: List[np.ndarray] = []
    histories: List[np.ndarray] = []
    utilities: List[np.ndarray] = []
    binary: List[np.ndarray] = []
    boundary_ids: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    artifact_cache: Dict[Path, Dict[str, np.ndarray]] = {}
    teacher_cache: Dict[Path, Dict[str, np.ndarray]] = {}
    for boundary_id, (
        artifact_path,
        teacher_path,
        cycle_index,
        layer_index,
        head,
    ) in enumerate(specs):
        if artifact_path not in artifact_cache:
            artifact_cache[artifact_path] = _load_npz(artifact_path)
        if teacher_path not in teacher_cache:
            teacher_cache[teacher_path] = _load_npz(teacher_path)
        artifact = artifact_cache[artifact_path]
        teacher = teacher_cache[teacher_path]
        cycle = int(teacher["cycles"][cycle_index])
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
            feature_only=True,
        )
        count = int(teacher["position_lengths"][cycle_index])
        teacher_positions = [
            int(value) for value in teacher["position_ids"][cycle_index, :count]
        ]
        current_count = int(artifact["position_lengths"][cycle])
        positions = [
            int(value) for value in artifact["position_ids"][cycle, :current_count]
        ]
        _, _, eligible = mandatory_and_eligible(
            positions, int(config["sink_size"]), int(config["recent_size"])
        )
        if teacher_positions != [int(value) for value in eligible]:
            raise RuntimeError("causal teacher positions do not match feature positions")
        teacher_horizons = [int(value) for value in teacher["horizons"]]
        horizon_rows = [teacher_horizons.index(value) for value in horizons]
        truth = np.take(
            teacher["scores"][cycle_index, :, layer_index, head, :count],
            horizon_rows,
            axis=0,
        ).T.astype(np.float32)
        labels = np.zeros_like(truth, dtype=np.float32)
        for column in range(len(horizons)):
            labels[topk_indices(truth[:, column], int(config["core_budget"])), column] = 1.0
        take = min(int(tokens_per_boundary), len(boundary.features))
        selected_rows = np.sort(
            rng.choice(len(boundary.features), size=take, replace=False)
        )
        features.append(boundary.features[selected_rows])
        histories.append(boundary.history[selected_rows])
        utilities.append(truth[selected_rows])
        binary.append(labels[selected_rows])
        boundary_ids.append(np.full(take, boundary_id, dtype=np.int32))
        if return_sizes:
            sizes.append(np.full(take, len(boundary.features), dtype=np.int32))
        if len(artifact_cache) > 2:
            keep_artifact = artifact_path
            keep_teacher = teacher_path
            artifact_cache = {keep_artifact: artifact_cache[keep_artifact]}
            teacher_cache = {keep_teacher: teacher_cache[keep_teacher]}
    if return_sizes:
        return (
            np.concatenate(features),
            np.concatenate(histories),
            np.concatenate(utilities),
            np.concatenate(binary),
            np.concatenate(boundary_ids),
            np.concatenate(sizes),
        )
    return (
        np.concatenate(features),
        np.concatenate(histories),
        np.concatenate(utilities),
        np.concatenate(binary),
        np.concatenate(boundary_ids),
    )


def train_rollout_distilled_predictor(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    train_ids = [
        sample_id_for(str(family), int(index))
        for family in config["task_families"]
        for index in config["distillation"]["train_indices"]
    ]
    artifact_paths = [
        output_root / "artifacts" / "train" / f"{_safe_sample_id(sample_id)}.npz"
        for sample_id in train_ids
    ]
    expected = int(config["distillation"]["expected_train_sequences"])
    if len(artifact_paths) != expected:
        raise RuntimeError(f"expected {expected} train artifacts, found {len(artifact_paths)}")
    missing = [path.name for path in artifact_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"distillation train artifacts are missing: {missing}")
    arrays = _teacher_arrays(
        artifact_paths, output_root / "rollout" / "train" / "teacher_scores", config
    )
    features, histories, truth, binary, boundary_ids = arrays
    scaler = joblib.load(output_root / "models" / "train_only_scaler.joblib")
    normalized = scaler.transform(features).astype(np.float32)
    horizons = [int(value) for value in config["future_utility_horizons"]]
    path = _train_neural(
        "rollout_distilled_mlp",
        MultiHorizonMLP(int(normalized.shape[1]), len(horizons)),
        normalized,
        histories,
        truth,
        binary,
        boundary_ids,
        output_root,
        int(config["data_seed"]) + 41,
        epochs=4,
    )
    atomic_json(
        output_root / "distillation_summary.json",
        {
            "teacher": "CAUSAL_EXPENSIVE_ROLLOUT_R2_PREFIX_RECOMPUTE",
            "train_sequences": len(artifact_paths),
            "sampled_token_rows": int(len(features)),
            "horizons": horizons,
            "offline_real_future_used_as_training_target": False,
            "runtime_future_access": False,
            "model_path": str(path.relative_to(repository_root)),
        },
    )
    return path


def evaluate_rollout_distilled_predictor(
    config_path: Path, repository_root: Path, split: str
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    if str(split) == "fresh_test":
        from statekv.existence_reporting import register_fresh_test_component

        register_fresh_test_component(output_root, "rollout_distilled_student")
    artifact_paths = sorted((output_root / "artifacts" / str(split)).glob("*.npz"))
    expected = int(config["expected_split_sizes"][str(split)])
    if len(artifact_paths) != expected:
        raise RuntimeError(f"expected {expected} {split} artifacts, found {len(artifact_paths)}")
    scaler = joblib.load(output_root / "models" / "train_only_scaler.joblib")
    horizons = [int(value) for value in config["future_utility_horizons"]]
    model = _load_neural(
        "query_conditioned_mlp",
        int(scaler.n_features_in_),
        len(horizons),
        output_root / "models" / "rollout_distilled_mlp.pt",
    )
    fixed_rhos = json.loads(
        (output_root / "models" / "fixed_baseline_tuning.json").read_text(
            encoding="utf-8"
        )
    )["per_head"]
    projector = FixedProjector(int(config["data_seed"]))
    rows: List[Dict[str, Any]] = []
    total_prediction_s = 0.0
    prediction_calls = 0
    teacher_root = output_root / "rollout" / str(split) / "teacher_scores"
    for artifact_path in artifact_paths:
        artifact = _load_npz(artifact_path)
        teacher = _load_npz(_teacher_path(teacher_root, artifact_path))
        for cycle_index, cycle_value in enumerate(teacher["cycles"]):
            cycle = int(cycle_value)
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
                        fixed_rhos,
                    )
                    normalized = scaler.transform(boundary.features).astype(np.float32)
                    with torch.no_grad():
                        prediction_started = time.perf_counter()
                        output = model(
                            torch.from_numpy(normalized),
                            torch.from_numpy(boundary.history),
                        ).numpy()
                        total_prediction_s += time.perf_counter() - prediction_started
                        prediction_calls += 1
                    for method, prediction in (
                        ("rollout_distilled_mlp", output[:, :, 0]),
                        ("rollout_distilled_mlp_utility", output[:, :, 1]),
                    ):
                        for column, horizon in enumerate(horizons):
                            rows.append(
                                {
                                    "sample_id": boundary.sample_id,
                                    "task": boundary.task,
                                    "split": str(split),
                                    "cycle": cycle,
                                    "layer": boundary.layer,
                                    "head": head,
                                    "method": method,
                                    "future_horizon": horizon,
                                    **boundary_metrics(
                                        boundary.truth[:, column],
                                        prediction[:, column],
                                        boundary.baseline[:, column],
                                        int(config["core_budget"]),
                                    ),
                                }
                            )
    evaluation_root = output_root / "evaluation" / str(split)
    existing_path = evaluation_root / "boundary_metrics.parquet"
    existing = pd.read_parquet(existing_path)
    existing = existing[
        ~existing["method"].astype(str).str.startswith("rollout_distilled_mlp")
    ]
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    atomic_frame(combined, existing_path)
    sequence = aggregate_sequence_metrics(combined)
    atomic_frame(sequence, evaluation_root / "sequence_metrics.csv")
    cost_path = evaluation_root / "inference_costs.csv"
    costs = pd.read_csv(cost_path)
    costs = costs[
        ~costs["method"].astype(str).str.startswith("rollout_distilled_mlp")
    ]
    mean_prediction_s = float(total_prediction_s / max(1, prediction_calls))
    mean_scoring_s = float(costs["mean_model_scoring_forward_time_s"].iloc[0])
    distillation_costs = pd.DataFrame(
        [
            {
                "method": method,
                "total_prediction_time_s": float(total_prediction_s),
                "prediction_calls": int(prediction_calls),
                "mean_prediction_time_s": mean_prediction_s,
                "mean_model_scoring_forward_time_s": mean_scoring_s,
                "runtime_multiplier": float(
                    1.0 + mean_prediction_s / max(mean_scoring_s, 1.0e-9)
                ),
            }
            for method in (
                "rollout_distilled_mlp",
                "rollout_distilled_mlp_utility",
            )
        ]
    )
    atomic_frame(pd.concat([costs, distillation_costs], ignore_index=True), cost_path)
    return evaluation_root


__all__ = [
    "evaluate_rollout_distilled_predictor",
    "train_rollout_distilled_predictor",
]
