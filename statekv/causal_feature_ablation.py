"""Nonlinear H=32 feature-group ablations for causal predictability."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from statekv.causal_existence_analysis import aggregate_sequence_metrics, boundary_metrics
from statekv.causal_predictors import (
    FixedProjector,
    _load_npz,
    artifact_boundary,
    feature_groups,
    sampled_training_arrays,
)
from statekv.storage import atomic_frame, atomic_json


def train_nonlinear_feature_ablations(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    paths = sorted((output_root / "artifacts" / "train").glob("*.npz"))
    expected = int(config["expected_split_sizes"]["train"])
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} train artifacts, found {len(paths)}")
    features, _, _, binary, _ = sampled_training_arrays(paths, config)
    scaler = joblib.load(output_root / "models" / "train_only_scaler.joblib")
    normalized = scaler.transform(features).astype(np.float32)
    groups = feature_groups(int(normalized.shape[1]))
    horizons = [int(value) for value in config["future_utility_horizons"]]
    column = horizons.index(32)
    subset = np.linspace(
        0, len(normalized) - 1, num=min(200000, len(normalized)), dtype=np.int64
    )
    models: Dict[str, Any] = {}
    started = time.perf_counter()
    for name, columns in groups.items():
        models[name] = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=int(config["data_seed"]),
        ).fit(normalized[subset][:, columns], binary[subset, column])
        print(f"[nonlinear-ablation] trained {name}", flush=True)
    path = output_root / "models" / "nonlinear_feature_ablations_h32.joblib"
    joblib.dump(models, path)
    atomic_json(
        output_root / "nonlinear_feature_ablation_training.json",
        {
            "split": "train",
            "horizon": 32,
            "models": sorted(models),
            "elapsed_s": float(time.perf_counter() - started),
        },
    )
    return path


def evaluate_nonlinear_feature_ablations(
    config_path: Path, repository_root: Path, split: str
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    if str(split) == "fresh_test":
        from statekv.existence_reporting import register_fresh_test_component

        register_fresh_test_component(output_root, "nonlinear_feature_ablations_h32")
    paths = sorted((output_root / "artifacts" / str(split)).glob("*.npz"))
    expected = int(config["expected_split_sizes"][str(split)])
    if len(paths) != expected:
        raise RuntimeError(f"expected {expected} {split} artifacts, found {len(paths)}")
    scaler = joblib.load(output_root / "models" / "train_only_scaler.joblib")
    models = joblib.load(output_root / "models" / "nonlinear_feature_ablations_h32.joblib")
    groups = feature_groups(int(scaler.n_features_in_))
    fixed_rhos = json.loads(
        (output_root / "models" / "fixed_baseline_tuning.json").read_text(
            encoding="utf-8"
        )
    )["per_head"]
    projector = FixedProjector(int(config["data_seed"]))
    rows: List[Dict[str, Any]] = []
    timing = {name: 0.0 for name in groups}
    calls = 0
    cycles = list(range(0, int(config["control_cycles"]) - 32, 8))
    for ordinal, path in enumerate(paths, start=1):
        artifact = _load_npz(path)
        for cycle in cycles:
            for layer_index in range(int(artifact["layers"].size)):
                for head in range(int(artifact["attention"].shape[2])):
                    boundary = artifact_boundary(
                        artifact,
                        cycle,
                        layer_index,
                        head,
                        [32],
                        int(config["sink_size"]),
                        int(config["recent_size"]),
                        int(config["core_budget"]),
                        projector,
                        fixed_rhos,
                    )
                    normalized = scaler.transform(boundary.features).astype(np.float32)
                    calls += 1
                    for name, columns in groups.items():
                        started = time.perf_counter()
                        prediction = models[name].predict_proba(
                            normalized[:, columns]
                        )[:, 1]
                        timing[name] += time.perf_counter() - started
                        rows.append(
                            {
                                "sample_id": boundary.sample_id,
                                "task": boundary.task,
                                "split": str(split),
                                "cycle": cycle,
                                "layer": boundary.layer,
                                "head": head,
                                "method": f"feature_gbdt_{name}",
                                "future_horizon": 32,
                                **boundary_metrics(
                                    boundary.truth[:, 0],
                                    prediction,
                                    boundary.baseline[:, 0],
                                    int(config["core_budget"]),
                                ),
                            }
                        )
        print(f"[nonlinear-ablation] {split} {ordinal}/{len(paths)}", flush=True)
    evaluation_root = output_root / "evaluation" / str(split)
    boundary_path = evaluation_root / "boundary_metrics.parquet"
    existing = pd.read_parquet(boundary_path)
    existing = existing[
        ~existing["method"].astype(str).str.startswith("feature_gbdt_")
    ]
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    atomic_frame(combined, boundary_path)
    atomic_frame(
        aggregate_sequence_metrics(combined), evaluation_root / "sequence_metrics.csv"
    )
    costs_path = evaluation_root / "inference_costs.csv"
    costs = pd.read_csv(costs_path)
    costs = costs[~costs["method"].astype(str).str.startswith("feature_gbdt_")]
    mean_scoring = float(costs["mean_model_scoring_forward_time_s"].iloc[0])
    added = pd.DataFrame(
        [
            {
                "method": f"feature_gbdt_{name}",
                "total_prediction_time_s": timing[name],
                "prediction_calls": calls,
                "mean_prediction_time_s": timing[name] / max(1, calls),
                "mean_model_scoring_forward_time_s": mean_scoring,
                "runtime_multiplier": 1.0
                + timing[name] / max(1, calls) / max(mean_scoring, 1.0e-9),
            }
            for name in groups
        ]
    )
    atomic_frame(pd.concat([costs, added], ignore_index=True), costs_path)
    return evaluation_root


__all__ = ["evaluate_nonlinear_feature_ablations", "train_nonlinear_feature_ablations"]
