"""Retrospective feasibility analysis for training-free StateKV sketches."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from statekv.storage import atomic_frame, atomic_json, atomic_text


def _sample_number(sample_id: str) -> int:
    value = str(sample_id)
    separator = ":" if ":" in value else "_"
    return int(value.rsplit(separator, 1)[-1])


def _split_name(sample_id: str, splits: Mapping[str, Sequence[int]]) -> str:
    number = _sample_number(sample_id)
    matches = [name for name, members in splits.items() if number in set(members)]
    if len(matches) != 1:
        raise ValueError("sample must belong to exactly one configured split")
    return matches[0]


def _spearman(prediction: np.ndarray, truth: np.ndarray) -> float:
    if len(prediction) < 2 or np.allclose(prediction, prediction[0]):
        return float("nan")
    return float(stats.spearmanr(prediction, truth).statistic)


def _pairwise_accuracy(prediction: np.ndarray, truth: np.ndarray) -> float:
    truth_difference = truth[:, None] - truth[None, :]
    prediction_difference = prediction[:, None] - prediction[None, :]
    mask = np.triu(np.ones(truth_difference.shape, dtype=bool), k=1)
    mask &= np.abs(truth_difference) > 1.0e-15
    if not np.any(mask):
        return float("nan")
    correct = np.sign(truth_difference[mask]) == np.sign(
        prediction_difference[mask]
    )
    return float(np.mean(correct))


def _decision_metrics(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    selected = int(np.argmin(prediction))
    oracle = int(np.argmin(truth))
    span = float(np.max(truth) - np.min(truth))
    regret = float(truth[selected] - truth[oracle])
    return {
        "spearman": _spearman(prediction, truth),
        "pairwise_accuracy": _pairwise_accuracy(prediction, truth),
        "top1_accuracy": float(selected == oracle),
        "normalized_regret": regret / max(span, 1.0e-15),
        "selected_risk": float(truth[selected]),
        "oracle_risk": float(truth[oracle]),
    }


def _recursive_risk(
    metadata: pd.DataFrame,
    signatures: np.ndarray,
    decay: float,
) -> np.ndarray:
    risk = np.full(len(metadata), np.nan, dtype=np.float64)
    grouped = metadata.groupby(["anchor", "candidate_id"], sort=False).groups
    for indices in grouped.values():
        ordered = sorted(
            (int(index) for index in indices),
            key=lambda index: int(metadata.iloc[index]["horizon_offset"]),
        )
        state = np.zeros(signatures.shape[-1], dtype=np.float64)
        for index in ordered:
            action = signatures[index].astype(np.float64, copy=False)
            risk[index] = float(np.dot(state, action) + 0.5 * np.dot(action, action))
            state = float(decay) * state + action
    return risk


def _nested_sign_projection(
    input_width: int,
    maximum_width: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    signs = generator.integers(
        0, 2, size=(int(input_width), int(maximum_width)), dtype=np.int8
    )
    return (2.0 * signs.astype(np.float32) - 1.0) / math.sqrt(
        float(maximum_width)
    )


def _load_sample(
    source_root: Path,
    stem: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    geometry_path = source_root / "oracle_geometry_rows" / (stem + ".parquet")
    index_path = source_root / "gauge_vector_index" / (stem + ".parquet")
    vector_path = source_root / "vectors" / (stem + ".npz")
    geometry = pd.read_parquet(
        geometry_path,
        columns=[
            "sample_id",
            "task",
            "candidate_id",
            "candidate_source",
            "anchor",
            "horizon_offset",
            "target_index",
            "source_exact_kl",
        ],
    )
    index = pd.read_parquet(index_path)
    keys = [
        "sample_id",
        "candidate_id",
        "anchor",
        "horizon_offset",
        "target_index",
    ]
    metadata = index.merge(geometry, on=keys, how="inner", validate="one_to_one")
    metadata = metadata.sort_values("vector_row", kind="stable").reset_index(drop=True)
    expected = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["vector_row"].to_numpy(dtype=np.int64), expected):
        raise RuntimeError("vector rows are not contiguous and aligned")
    with np.load(vector_path, allow_pickle=False) as vectors:
        direct = vectors["direct_projected_l27"].astype(np.float32)
        actual_state = vectors["actual_residual_delta"].astype(np.float32)
    if len(direct) != len(metadata):
        raise RuntimeError("vector and metadata row counts differ")
    if actual_state.shape != direct.shape:
        raise RuntimeError("actual state and direct action vectors must align")
    return metadata, direct, actual_state


def _summarize_gains(gains: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "comparison",
        "split",
        "task",
        "projection_dimension",
        "projection_seed",
        "decay",
    ]
    records: List[Dict[str, Any]] = []
    expanded = [gains]
    overall = gains.copy()
    overall["task"] = "all"
    expanded.append(overall)
    combined = pd.concat(expanded, ignore_index=True)
    for values, current in combined.groupby(keys, sort=True):
        records.append(
            {
                **dict(zip(keys, values)),
                "decision_units": int(len(current)),
                "sequences": int(current["sample_id"].nunique()),
                "median_spearman_gain": float(
                    current["spearman_gain"].median()
                ),
                "mean_pairwise_accuracy_gain": float(
                    current["pairwise_accuracy_gain"].mean()
                ),
                "mean_top1_accuracy_gain": float(
                    current["top1_accuracy_gain"].mean()
                ),
                "mean_normalized_regret_gain": float(
                    current["normalized_regret_gain"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def analyze_training_free_sketch(
    config_path: Path,
    repository_root: Path,
) -> Path:
    """Run the fixed-decay sketch feasibility test on frozen real-model vectors."""

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source_run = repository_root / str(config["source_run"])
    source_root = source_run / "fragments" / "gauge_geometry"
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    dimensions = sorted(set(int(value) for value in config["projection_dimensions"]))
    seeds = sorted(set(int(value) for value in config["projection_seeds"]))
    decays = sorted(set(float(value) for value in config["decays"]))
    if not dimensions or not seeds or not decays:
        raise ValueError("dimensions, seeds, and decays must be non-empty")
    if any(value < 1 for value in dimensions):
        raise ValueError("projection dimensions must be positive")
    if any(not 0.0 <= value <= 1.0 for value in decays):
        raise ValueError("decays must lie in [0, 1]")
    splits = {
        str(name): [int(value) for value in members]
        for name, members in config["splits"].items()
    }
    minimum_horizon = int(config.get("minimum_horizon", 2))
    vector_paths = sorted((source_root / "vectors").glob("*.npz"))
    if not vector_paths:
        raise FileNotFoundError("source run contains no gauge vector fragments")

    metric_records: List[Dict[str, Any]] = []
    gain_records: List[Dict[str, Any]] = []
    maximum_width = max(dimensions)
    for vector_path in vector_paths:
        metadata, direct, actual_state = _load_sample(
            source_root, vector_path.stem
        )
        sample_id = str(metadata["sample_id"].iloc[0])
        task = str(metadata["task"].iloc[0])
        split = _split_name(sample_id, splits)
        for seed in seeds:
            projection = _nested_sign_projection(
                int(direct.shape[-1]), maximum_width, seed
            )
            projected_max = direct @ projection
            projected_actual_max = actual_state @ projection
            for dimension in dimensions:
                signatures = projected_max[:, :dimension].astype(
                    np.float64, copy=False
                ) * math.sqrt(float(maximum_width) / float(dimension))
                projected_actual = projected_actual_max[:, :dimension].astype(
                    np.float64, copy=False
                ) * math.sqrt(float(maximum_width) / float(dimension))
                action_only = 0.5 * np.square(signatures).sum(axis=-1)
                reference_state_risk = action_only + np.sum(
                    projected_actual * signatures, axis=-1
                )
                for decay in decays:
                    recursive = _recursive_risk(metadata, signatures, decay)
                    decision_groups = metadata.groupby(
                        ["anchor", "horizon_offset"], sort=True
                    ).groups
                    for (anchor, horizon), indices in decision_groups.items():
                        if int(horizon) < minimum_horizon:
                            continue
                        rows = np.asarray(sorted(int(value) for value in indices))
                        truth = metadata.iloc[rows]["source_exact_kl"].to_numpy(
                            dtype=np.float64
                        )
                        action_metrics = _decision_metrics(action_only[rows], truth)
                        recursive_metrics = _decision_metrics(recursive[rows], truth)
                        reference_metrics = _decision_metrics(
                            reference_state_risk[rows], truth
                        )
                        common = {
                            "sample_id": sample_id,
                            "task": task,
                            "split": split,
                            "anchor": int(anchor),
                            "horizon_offset": int(horizon),
                            "candidate_count": int(len(rows)),
                            "projection_dimension": int(dimension),
                            "projection_seed": int(seed),
                            "decay": float(decay),
                        }
                        for method, metrics in (
                            ("action_energy_only", action_metrics),
                            ("recursive_state_sketch", recursive_metrics),
                            ("reference_state_oracle", reference_metrics),
                        ):
                            metric_records.append(
                                {**common, "method": method, **metrics}
                            )
                        gain_records.append(
                            {
                                **common,
                                "comparison": "recursive_vs_action",
                                "spearman_gain": recursive_metrics["spearman"]
                                - action_metrics["spearman"],
                                "pairwise_accuracy_gain": recursive_metrics[
                                    "pairwise_accuracy"
                                ]
                                - action_metrics["pairwise_accuracy"],
                                "top1_accuracy_gain": recursive_metrics[
                                    "top1_accuracy"
                                ]
                                - action_metrics["top1_accuracy"],
                                "normalized_regret_gain": action_metrics[
                                    "normalized_regret"
                                ]
                                - recursive_metrics["normalized_regret"],
                            }
                        )
                        gain_records.append(
                            {
                                **common,
                                "comparison": "reference_state_vs_action",
                                "spearman_gain": reference_metrics["spearman"]
                                - action_metrics["spearman"],
                                "pairwise_accuracy_gain": reference_metrics[
                                    "pairwise_accuracy"
                                ]
                                - action_metrics["pairwise_accuracy"],
                                "top1_accuracy_gain": reference_metrics[
                                    "top1_accuracy"
                                ]
                                - action_metrics["top1_accuracy"],
                                "normalized_regret_gain": action_metrics[
                                    "normalized_regret"
                                ]
                                - reference_metrics["normalized_regret"],
                            }
                        )

    metrics = pd.DataFrame(metric_records)
    gains = pd.DataFrame(gain_records)
    summary = _summarize_gains(gains)
    primary = config["primary"]
    primary_rows = summary[
        (summary["comparison"] == "recursive_vs_action")
        & (summary["projection_dimension"] == int(primary["projection_dimension"]))
        & (summary["projection_seed"] == int(primary["projection_seed"]))
        & np.isclose(summary["decay"], float(primary["decay"]))
    ]

    gate_checks: Dict[str, bool] = {}
    gate_values: Dict[str, Dict[str, float]] = {}
    for split in ("evaluation", "replication"):
        overall = primary_rows[
            (primary_rows["split"] == split) & (primary_rows["task"] == "all")
        ]
        if len(overall) != 1:
            raise RuntimeError("primary summary is missing split=%s" % split)
        row = overall.iloc[0]
        values = {
            "median_spearman_gain": float(row["median_spearman_gain"]),
            "mean_pairwise_accuracy_gain": float(
                row["mean_pairwise_accuracy_gain"]
            ),
            "mean_top1_accuracy_gain": float(row["mean_top1_accuracy_gain"]),
            "mean_normalized_regret_gain": float(
                row["mean_normalized_regret_gain"]
            ),
        }
        gate_values[split] = values
        gate_checks[split + "_spearman_positive"] = (
            values["median_spearman_gain"] > 0.0
        )
        gate_checks[split + "_pairwise_positive"] = (
            values["mean_pairwise_accuracy_gain"] > 0.0
        )
        gate_checks[split + "_regret_positive"] = (
            values["mean_normalized_regret_gain"] > 0.0
        )
        for task in sorted(value for value in primary_rows["task"].unique() if value != "all"):
            task_row = primary_rows[
                (primary_rows["split"] == split) & (primary_rows["task"] == task)
            ]
            if len(task_row) == 1:
                gate_checks[split + "_" + task + "_spearman_positive"] = bool(
                    float(task_row.iloc[0]["median_spearman_gain"]) > 0.0
                )

    calibration = summary[
        (summary["comparison"] == "recursive_vs_action")
        & (summary["split"] == "calibration")
        & (summary["task"] == "all")
    ].sort_values(
        ["median_spearman_gain", "mean_normalized_regret_gain"],
        ascending=[False, False],
        kind="stable",
    )
    best_calibration = calibration.iloc[0].to_dict()
    gate_passed = bool(gate_checks and all(gate_checks.values()))
    gate = {
        "experiment": str(config["experiment_name"]),
        "source_run": str(config["source_run"]),
        "status": "retrospective_real_model_analysis",
        "method_scope": (
            "fixed-decay recursive sketch over stored layer-27 direct action vectors; "
            "query-cosine decay and online direct-set execution are not tested"
        ),
        "primary": {
            "projection_dimension": int(primary["projection_dimension"]),
            "projection_seed": int(primary["projection_seed"]),
            "decay": float(primary["decay"]),
        },
        "checks": gate_checks,
        "values": gate_values,
        "passed": gate_passed,
        "outcome": (
            "history_sketch_feasible_for_next_stage"
            if gate_passed
            else "history_sketch_not_stable_at_primary_configuration"
        ),
        "calibration_best_sensitivity": {
            key: (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in best_calibration.items()
        },
    }

    atomic_frame(metrics, output_root / "decision_metrics.parquet")
    atomic_frame(gains, output_root / "history_gain_rows.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = ["analyze_training_free_sketch"]
