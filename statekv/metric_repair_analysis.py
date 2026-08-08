"""Low-cost metric-repair screening on frozen real-model action vectors.

The analysis is deliberately retrospective.  It uses no teacher labels to
construct the deployable transforms: diagonal and block scales are estimated
only from action-vector second moments on the calibration sequences.  Stored
midpoint-Fisher values are retained as a non-deployable diagnostic ceiling.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.training_free_analysis import (
    _decision_metrics,
    _nested_sign_projection,
    _split_name,
)


INDEX_KEYS = [
    "sample_id",
    "candidate_id",
    "anchor",
    "horizon_offset",
    "target_index",
]


def unsupervised_sensitivity_scales(
    sum_of_squares: np.ndarray,
    count: int,
    *,
    blocks: int,
    epsilon: float = 1.0e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit-RMS diagonal and contiguous-block inverse-RMS scales."""

    values = np.asarray(sum_of_squares, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1:
        raise ValueError("sum_of_squares must be a non-empty vector")
    if int(count) < 1:
        raise ValueError("count must be positive")
    if int(blocks) < 1 or len(values) % int(blocks):
        raise ValueError("blocks must evenly divide the hidden width")
    coordinate = 1.0 / np.sqrt(values / float(count) + float(epsilon))
    coordinate /= math.sqrt(float(np.mean(np.square(coordinate))))
    width = len(values) // int(blocks)
    block_second_moment = (values / float(count)).reshape(
        int(blocks), width
    ).mean(axis=1)
    block = 1.0 / np.sqrt(block_second_moment + float(epsilon))
    block /= math.sqrt(float(np.mean(np.square(block))))
    return coordinate, block


def trajectory_risks(
    metadata: pd.DataFrame,
    signatures: np.ndarray,
    *,
    decay: float,
) -> Dict[str, np.ndarray]:
    """Score action-only, repeated, innovation, and EMA state updates."""

    vectors = np.asarray(signatures, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(metadata):
        raise ValueError("signatures must align one-to-one with metadata")
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError("decay must lie in [0, 1]")
    result = {
        name: np.full(len(metadata), np.nan, dtype=np.float64)
        for name in ("action", "repeated", "innovation", "ema")
    }
    groups = metadata.groupby(["anchor", "candidate_id"], sort=False).groups
    for indices in groups.values():
        order = sorted(
            (int(index) for index in indices),
            key=lambda index: int(metadata.iloc[index]["horizon_offset"]),
        )
        repeated = np.zeros(vectors.shape[1], dtype=np.float64)
        innovation = np.zeros_like(repeated)
        ema = np.zeros_like(repeated)
        previous = np.zeros_like(repeated)
        for index in order:
            action = vectors[index]
            energy = 0.5 * float(np.dot(action, action))
            result["action"][index] = energy
            result["repeated"][index] = float(
                np.dot(repeated, action) + energy
            )
            result["innovation"][index] = float(
                np.dot(innovation, action) + energy
            )
            result["ema"][index] = float(np.dot(ema, action) + energy)
            delta = action - previous
            repeated = float(decay) * repeated + action
            innovation = float(decay) * innovation + delta
            ema = float(decay) * ema + (1.0 - float(decay)) * action
            previous = action
    return result


def _load_geometry_and_index(source_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geometry = pd.read_parquet(
        source_root / "independent_fisher_geometry_rows.parquet",
        columns=INDEX_KEYS
        + ["task", "candidate_source", "exact_kl", "g3_midpoint_fisher"],
    )
    index = pd.read_parquet(source_root / "independent_vector_index.parquet")
    return geometry, index


def _fragment_path(source_root: Path, recorded: str) -> Path:
    path = Path(str(recorded))
    if path.exists():
        return path
    fallback = (
        source_root
        / "fragments"
        / "independent_fisher"
        / "vectors"
        / path.name
    )
    if not fallback.exists():
        raise FileNotFoundError("missing vector fragment: %s" % path.name)
    return fallback


def _sample_frame(
    geometry: pd.DataFrame,
    index: pd.DataFrame,
    fragment_rows: pd.DataFrame,
) -> pd.DataFrame:
    metadata = fragment_rows.merge(
        geometry,
        on=INDEX_KEYS,
        how="inner",
        validate="one_to_one",
    ).sort_values("vector_row", kind="stable").reset_index(drop=True)
    if len(metadata) != len(fragment_rows):
        raise RuntimeError("geometry and vector index are not one-to-one")
    expected = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["vector_row"].to_numpy(), expected):
        raise RuntimeError("fragment vector rows are not contiguous")
    return metadata


def _calibration_scales(
    source_root: Path,
    geometry: pd.DataFrame,
    index: pd.DataFrame,
    splits: Mapping[str, Sequence[int]],
    blocks: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    sum_of_squares: Optional[np.ndarray] = None
    count = 0
    for recorded, fragment_rows in index.groupby("vector_fragment", sort=False):
        sample_id = str(fragment_rows["sample_id"].iloc[0])
        if _split_name(sample_id, splits) != "calibration":
            continue
        metadata = _sample_frame(geometry, index, fragment_rows)
        path = _fragment_path(source_root, str(recorded))
        with np.load(path, allow_pickle=False) as arrays:
            direct = arrays["direct_projected_l27"].astype(np.float64)
        if len(direct) != len(metadata):
            raise RuntimeError("calibration vectors do not align with metadata")
        current = np.square(direct).sum(axis=0)
        sum_of_squares = (
            current if sum_of_squares is None else sum_of_squares + current
        )
        count += len(direct)
    if sum_of_squares is None:
        raise RuntimeError("calibration split contains no vector fragments")
    diagonal, block = unsupervised_sensitivity_scales(
        sum_of_squares, count, blocks=int(blocks)
    )
    return diagonal, block, count


def _summaries(metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    expanded = [metrics]
    overall = metrics.copy()
    overall["task"] = "all"
    expanded.append(overall)
    joined = pd.concat(expanded, ignore_index=True)
    keys = [
        "split",
        "task",
        "method",
        "projection_dimension",
        "projection_seed",
        "decay",
    ]
    for values, current in joined.groupby(keys, sort=True):
        selectors = {
            key: value for key, value in zip(keys, values) if key != "method"
        }
        base = joined[joined["method"] == baseline]
        for key, value in selectors.items():
            base = base[base[key] == value]
        if str(values[2]) == baseline:
            gains = current.assign(
                spearman_gain=0.0,
                pairwise_accuracy_gain=0.0,
                top1_accuracy_gain=0.0,
                normalized_regret_gain=0.0,
            )
        else:
            match_keys = ["sample_id", "anchor", "horizon_offset"]
            gains = current.merge(
                base[
                    match_keys
                    + [
                        "spearman",
                        "pairwise_accuracy",
                        "top1_accuracy",
                        "normalized_regret",
                    ]
                ],
                on=match_keys,
                suffixes=("", "_baseline"),
                validate="one_to_one",
            )
            gains["spearman_gain"] = (
                gains["spearman"] - gains["spearman_baseline"]
            )
            gains["pairwise_accuracy_gain"] = (
                gains["pairwise_accuracy"]
                - gains["pairwise_accuracy_baseline"]
            )
            gains["top1_accuracy_gain"] = (
                gains["top1_accuracy"] - gains["top1_accuracy_baseline"]
            )
            gains["normalized_regret_gain"] = (
                gains["normalized_regret_baseline"]
                - gains["normalized_regret"]
            )
        records.append(
            {
                **dict(zip(keys, values)),
                "decision_units": int(len(current)),
                "sequences": int(current["sample_id"].nunique()),
                "median_spearman": float(current["spearman"].median()),
                "mean_pairwise_accuracy": float(
                    current["pairwise_accuracy"].mean()
                ),
                "mean_top1_accuracy": float(current["top1_accuracy"].mean()),
                "mean_normalized_regret": float(
                    current["normalized_regret"].mean()
                ),
                "median_spearman_gain": float(gains["spearman_gain"].median()),
                "mean_pairwise_accuracy_gain": float(
                    gains["pairwise_accuracy_gain"].mean()
                ),
                "mean_top1_accuracy_gain": float(
                    gains["top1_accuracy_gain"].mean()
                ),
                "mean_normalized_regret_gain": float(
                    gains["normalized_regret_gain"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def analyze_metric_repair(config_path: Path, repository_root: Path) -> Path:
    """Run the development-only metric repair matrix."""

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    source_root = repository_root / str(config["source_run"])
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    geometry, index = _load_geometry_and_index(source_root)
    splits = {
        str(name): [int(value) for value in members]
        for name, members in config["splits"].items()
    }
    dimensions = sorted(set(int(value) for value in config["projection_dimensions"]))
    seeds = sorted(set(int(value) for value in config["projection_seeds"]))
    decay = float(config["decay"])
    blocks = int(config["residual_blocks"])
    minimum_horizon = int(config.get("minimum_horizon", 2))
    diagonal, block, calibration_rows = _calibration_scales(
        source_root, geometry, index, splits, blocks
    )
    maximum_dimension = max(dimensions)
    metric_records: List[Dict[str, Any]] = []

    for recorded, fragment_rows in index.groupby("vector_fragment", sort=False):
        metadata = _sample_frame(geometry, index, fragment_rows)
        sample_id = str(metadata["sample_id"].iloc[0])
        split = _split_name(sample_id, splits)
        path = _fragment_path(source_root, str(recorded))
        with np.load(path, allow_pickle=False) as arrays:
            direct = arrays["direct_projected_l27"].astype(np.float64)
        width = direct.shape[1] // blocks
        transformed = {
            "raw": direct,
            "diagonal_rms": direct * diagonal,
            "block_rms": (
                direct.reshape(len(direct), blocks, width)
                * block[None, :, None]
            ).reshape(direct.shape),
        }
        for seed in seeds:
            projection = _nested_sign_projection(
                direct.shape[1], maximum_dimension, seed
            )
            for dimension in dimensions:
                scale = math.sqrt(maximum_dimension / float(dimension))
                method_scores: Dict[str, np.ndarray] = {}
                for transform_name, values in transformed.items():
                    signatures = values @ projection[:, :dimension] * scale
                    for state_name, scores in trajectory_risks(
                        metadata, signatures, decay=decay
                    ).items():
                        method_scores[transform_name + "_" + state_name] = scores
                method_scores["output_fisher_oracle"] = metadata[
                    "g3_midpoint_fisher"
                ].to_numpy(dtype=np.float64)
                for (anchor, horizon), indices in metadata.groupby(
                    ["anchor", "horizon_offset"], sort=True
                ).groups.items():
                    if int(horizon) < minimum_horizon:
                        continue
                    rows = np.asarray(sorted(int(value) for value in indices))
                    truth = metadata.iloc[rows]["exact_kl"].to_numpy(
                        dtype=np.float64
                    )
                    common = {
                        "sample_id": sample_id,
                        "task": str(metadata["task"].iloc[0]),
                        "split": split,
                        "anchor": int(anchor),
                        "horizon_offset": int(horizon),
                        "candidate_count": int(len(rows)),
                        "projection_dimension": int(dimension),
                        "projection_seed": int(seed),
                        "decay": decay,
                    }
                    for method, scores in method_scores.items():
                        metric_records.append(
                            {
                                **common,
                                "method": method,
                                "deployable": method != "output_fisher_oracle",
                                **_decision_metrics(scores[rows], truth),
                            }
                        )

    metrics = pd.DataFrame(metric_records)
    baseline = str(config["baseline"])
    summary = _summaries(metrics, baseline)
    primary = config["primary"]
    selected = summary[
        (summary["method"] == str(primary["method"]))
        & (summary["projection_dimension"] == int(primary["projection_dimension"]))
        & (summary["projection_seed"] == int(primary["projection_seed"]))
        & (summary["task"] == "all")
    ]
    checks: Dict[str, bool] = {}
    values: Dict[str, Dict[str, float]] = {}
    for split in ("evaluation", "replication"):
        current = selected[selected["split"] == split]
        if len(current) != 1:
            raise RuntimeError("primary result missing split=%s" % split)
        row = current.iloc[0]
        split_values = {
            "median_spearman_gain": float(row["median_spearman_gain"]),
            "mean_pairwise_accuracy_gain": float(
                row["mean_pairwise_accuracy_gain"]
            ),
            "mean_normalized_regret_gain": float(
                row["mean_normalized_regret_gain"]
            ),
        }
        values[split] = split_values
        for metric, value in split_values.items():
            checks[split + "_" + metric + "_positive"] = bool(value > 0.0)
    passed = bool(checks and all(checks.values()))
    gate = {
        "experiment": str(config["experiment_name"]),
        "status": "retrospective_development_screen",
        "confirmatory_evidence": False,
        "selection_disclosure": (
            "Variants were inspected during development; evaluation and replication "
            "labels are descriptive partitions, not untouched confirmatory sets."
        ),
        "source_run": str(config["source_run"]),
        "baseline": baseline,
        "primary": dict(primary),
        "calibration_action_rows": int(calibration_rows),
        "diagonal_scale_min": float(np.min(diagonal)),
        "diagonal_scale_max": float(np.max(diagonal)),
        "block_scales": [float(value) for value in block],
        "checks": checks,
        "values": values,
        "passed": passed,
        "outcome": (
            "advance_to_minimal_shared_jvp_pilot"
            if passed
            else "freeze_metric_repair_without_model_probes"
        ),
        "scope": {
            "diagonal_rms": "unlabeled calibration action second moments",
            "block_rms": (
                "contiguous layer-27 residual-coordinate blocks; not attention heads"
            ),
            "output_fisher_oracle": (
                "stored candidate-specific midpoint Fisher; diagnostic only"
            ),
        },
    }
    atomic_frame(metrics, output_root / "decision_metrics.parquet")
    atomic_frame(summary, output_root / "metrics.csv")
    atomic_json(output_root / "summary.json", gate)
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    return output_root


__all__ = [
    "analyze_metric_repair",
    "trajectory_risks",
    "unsupervised_sensitivity_scales",
]
