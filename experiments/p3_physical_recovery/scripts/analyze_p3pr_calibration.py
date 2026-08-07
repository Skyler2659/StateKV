#!/usr/bin/env python3
"""Run every calibration-only P3PR model class and freeze formal predictor."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3pr_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    sequence_first_metrics,
    sha256_file,
    validate_feature_names,
)


KEYS = [
    "sample_id",
    "task",
    "target_anchor",
    "candidate_id",
    "candidate_source",
]


def transformed(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    validate_feature_names(columns)
    value = frame[list(columns)].to_numpy(dtype=np.float64)
    return np.sign(value) * np.log1p(np.abs(value))


def gate(
    sequence_rows: pd.DataFrame,
    summary: Dict[str, Any],
    action_rows: pd.DataFrame,
) -> Dict[str, Any]:
    candidate_regret = (
        sequence_rows.groupby("task")["normalized_regret"].mean().to_dict()
    )
    action_regret = (
        action_rows.groupby("task")["normalized_regret"].mean().to_dict()
    )
    checks = {
        "overall_spearman": summary["overall_spearman"] >= 0.90,
        "each_task_spearman": min(summary["task_spearman"].values()) >= 0.85,
        "pairwise_accuracy": summary["pairwise_accuracy"] >= 0.90,
        "positive_sequence_fraction": (
            summary["positive_sequence_fraction"] >= 0.75
        ),
        "top1_accuracy": summary["top1_accuracy"] >= 0.75,
        "each_task_action_only_regret_gain": all(
            candidate_regret[task] < action_regret[task]
            for task in candidate_regret
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_normalized_regret": candidate_regret,
        "task_action_only_regret": action_regret,
    }


def evaluate_score(
    rows: pd.DataFrame,
    score: np.ndarray | str,
    name: str,
    action_rows: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    current = rows.copy()
    current["_score"] = (
        current[str(score)].to_numpy()
        if isinstance(score, str)
        else np.asarray(score, dtype=np.float64)
    )
    sequence, summary = sequence_first_metrics(current, "_score")
    decision = gate(sequence, summary, action_rows)
    return {
        "name": name,
        **summary,
        **decision,
    }, sequence


def ridge_loso(
    rows: pd.DataFrame,
    features: Sequence[str],
    alpha: float,
) -> np.ndarray:
    x = transformed(rows, features)
    y = np.log10(
        np.clip(rows["exact_physical_kl"].to_numpy(dtype=np.float64), 1.0e-14, None)
    )
    groups = rows["sample_id"].astype(str).to_numpy()
    prediction = np.zeros(len(rows), dtype=np.float64)
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = ~train
        scaler = StandardScaler().fit(x[train])
        model = Ridge(alpha=float(alpha)).fit(
            scaler.transform(x[train]), y[train]
        )
        prediction[test] = model.predict(scaler.transform(x[test]))
    return prediction


def pairwise_logistic_loso(
    rows: pd.DataFrame,
    features: Sequence[str],
    regularization: float,
) -> np.ndarray:
    x = transformed(rows, features)
    y = rows["exact_physical_kl"].to_numpy(dtype=np.float64)
    groups = rows["sample_id"].astype(str).to_numpy()
    prediction = np.zeros(len(rows), dtype=np.float64)
    unit_keys = ["sample_id", "target_anchor"]
    for held_out in sorted(set(groups)):
        train_mask = groups != held_out
        test_mask = ~train_mask
        scaler = StandardScaler().fit(x[train_mask])
        standardized = scaler.transform(x)
        pair_x: List[np.ndarray] = []
        pair_y: List[int] = []
        train_rows = rows.loc[train_mask]
        for _key, unit in train_rows.groupby(unit_keys, sort=True):
            indices = unit.index.to_list()
            for left, right in itertools.combinations(indices, 2):
                if y[left] == y[right]:
                    continue
                difference = standardized[left] - standardized[right]
                label = int(y[left] > y[right])
                pair_x.extend([difference, -difference])
                pair_y.extend([label, 1 - label])
        model = LogisticRegression(
            C=float(regularization),
            fit_intercept=False,
            solver="liblinear",
            max_iter=2000,
            random_state=2026072811,
        ).fit(np.stack(pair_x), np.asarray(pair_y))
        prediction[test_mask] = (
            standardized[test_mask] @ model.coef_.reshape(-1)
        )
    return prediction


def tree_loso(
    rows: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    x = transformed(rows, features)
    y = np.log10(
        np.clip(rows["exact_physical_kl"].to_numpy(dtype=np.float64), 1.0e-14, None)
    )
    groups = rows["sample_id"].astype(str).to_numpy()
    prediction = np.zeros(len(rows), dtype=np.float64)
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = ~train
        model = DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=4,
            random_state=2026072811,
        ).fit(x[train], y[train])
        prediction[test] = model.predict(x[test])
    return prediction


def pca_interaction_loso(
    rows: pd.DataFrame,
    features: Sequence[str],
    rank: int,
) -> np.ndarray:
    x = transformed(rows, features)
    y = np.log10(
        np.clip(rows["exact_physical_kl"].to_numpy(dtype=np.float64), 1.0e-14, None)
    )
    groups = rows["sample_id"].astype(str).to_numpy()
    prediction = np.zeros(len(rows), dtype=np.float64)
    for held_out in sorted(set(groups)):
        train = groups != held_out
        test = ~train
        scaler = StandardScaler().fit(x[train])
        scaled_train = scaler.transform(x[train])
        scaled_test = scaler.transform(x[test])
        dimension = min(int(rank), scaled_train.shape[1], scaled_train.shape[0] - 1)
        pca = PCA(n_components=dimension, random_state=2026072811).fit(
            scaled_train
        )
        left = pca.transform(scaled_train)
        right = pca.transform(scaled_test)

        def expand(value: np.ndarray) -> np.ndarray:
            columns = [value]
            columns.extend(
                [
                    value[:, i : i + 1] * value[:, j : j + 1]
                    for i in range(dimension)
                    for j in range(i, dimension)
                ]
            )
            return np.concatenate(columns, axis=1)

        model = Ridge(alpha=1.0).fit(expand(left), y[train])
        prediction[test] = model.predict(expand(right))
    return prediction


def layer_wide(
    candidates: pd.DataFrame, layers: pd.DataFrame
) -> pd.DataFrame:
    feature_columns = [
        "single_boundary_theory_risk",
        "theory_u_norm",
        "local_r_norm",
        "deleted_attention_mass_mean",
        "deleted_attention_mass_std",
        "head_disagreement",
        "deleted_key_norm",
        "key_norm_mean",
        "key_norm_variance",
        "deleted_value_norm",
        "value_norm_mean",
        "value_norm_variance",
        "action_state_ratio",
        "actual_boundary_norm",
    ]
    pieces = []
    for feature in feature_columns:
        pivot = layers.pivot(index=KEYS, columns="layer", values=feature)
        pivot.columns = [
            f"layer{int(layer)}_{feature}" for layer in pivot.columns
        ]
        pieces.append(pivot)
    wide = pd.concat(pieces, axis=1).reset_index()
    return candidates.merge(wide, on=KEYS, how="left", validate="one_to_one")


def select_best_record(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(
        records,
        key=lambda row: (
            not bool(row["passed"]),
            -float(row["overall_spearman"]),
            -float(row["pairwise_accuracy"]),
            -float(row["top1_accuracy"]),
            float(row["normalized_regret"]),
            int(row.get("parameter_count", 0)),
            str(row["name"]),
        ),
    )[0]


def main() -> None:
    config_path = EXPERIMENT / "p3pr_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidates = pd.read_parquet(
        EXPERIMENT / "results/calibration/candidate_rows.parquet"
    ).reset_index(drop=True)
    layers = pd.read_parquet(
        EXPERIMENT / "results/calibration/layer_rows.parquet"
    )
    rows = layer_wide(candidates, layers)
    action_sequence, _action_summary = sequence_first_metrics(
        rows, "action_only_risk"
    )
    results: List[Dict[str, Any]] = []
    sequence_frames: List[pd.DataFrame] = []

    raw_models = [
        ("M1_O0_inherited_P3_scalar_transfer", None, 0, 1, 0),
        ("M2_action_only_current_injection", "action_only_risk", 28, 0, 0),
        ("M2_adjacent_only_current_response", "adjacent_only_risk", 28, 0, 0),
        ("M4_mechanistic_pair_7_22", "multi_pair_endpoint_risk", 2, 0, 0),
        ("M4_mechanistic_three_4_14_22", "multi_three_endpoint_risk", 3, 0, 0),
        ("M4_mechanistic_inherited_0_14_26", "multi_inherited3_endpoint_risk", 3, 0, 0),
        ("M4_mechanistic_uniform8", "multi_uniform8_endpoint_risk", 8, 0, 0),
        ("M9_dense_mechanistic_all28", "multi_all_endpoint_risk", 28, 0, 0),
    ]
    for name, column, boundaries, probes, parameters in raw_models:
        if column is None:
            # P3's inherited result is immutable external evidence.
            results.append(
                {
                    "name": name,
                    "model_class": "inherited_control",
                    "overall_spearman": 0.1876,
                    "replication_spearman": 0.1151,
                    "passed": False,
                    "boundary_count": boundaries,
                    "candidate_probe_count": probes,
                    "parameter_count": parameters,
                    "source": "P3 immutable formal/replication",
                }
            )
            continue
        record, sequence = evaluate_score(
            rows, column, name, action_sequence
        )
        record.update(
            {
                "model_class": name.split("_")[0],
                "score_column": column,
                "boundary_count": boundaries,
                "candidate_probe_count": probes,
                "parameter_count": parameters,
            }
        )
        results.append(record)
        sequence["model"] = name
        sequence_frames.append(sequence)

    # M3: every single current physical boundary, no endpoint candidate probe.
    for boundary in range(1, 28):
        column = f"theory_b{boundary}_risk"
        record, sequence = evaluate_score(
            rows,
            column,
            f"M3_single_boundary_b{boundary}",
            action_sequence,
        )
        record.update(
            {
                "model_class": "M3",
                "score_column": column,
                "boundary_count": 1,
                "candidate_probe_count": 0,
                "parameter_count": 0,
            }
        )
        results.append(record)
        sequence["model"] = record["name"]
        sequence_frames.append(sequence)

    # M4 calibration-only sparse boundary fusion, one risk scalar per boundary.
    selected_boundaries: List[int] = []
    previous = -1.0
    forward_rows = []
    for step in range(1, 5):
        candidates_step = []
        for boundary in range(1, 28):
            if boundary in selected_boundaries:
                continue
            proposal = selected_boundaries + [boundary]
            features = [f"layer{value - 1}_single_boundary_theory_risk" for value in proposal]
            prediction = ridge_loso(rows, features, alpha=1.0)
            record, sequence = evaluate_score(
                rows,
                prediction,
                "M4_sparse_boundary_ridge_" + "_".join(map(str, proposal)),
                action_sequence,
            )
            record.update(
                {
                    "model_class": "M4",
                    "features": features,
                    "boundary_count": len(proposal),
                    "candidate_probe_count": 0,
                    "parameter_count": len(features) + 1,
                }
            )
            candidates_step.append((boundary, record, sequence))
        boundary, best, sequence = sorted(
            candidates_step,
            key=lambda item: (
                -float(item[1]["overall_spearman"]),
                int(item[0]),
            ),
        )[0]
        gain = float(best["overall_spearman"]) - previous
        selected_boundaries.append(int(boundary))
        best["forward_gain"] = gain
        forward_rows.append(best)
        results.append(best)
        sequence["model"] = best["name"]
        sequence_frames.append(sequence)
        if step > 1 and gain < float(
            config["boundaries"]["forward_selection_gain_min"]
        ):
            break
        previous = float(best["overall_spearman"])

    sparse_layers = [value - 1 for value in selected_boundaries]
    boundary_features = [
        f"layer{layer}_{feature}"
        for layer in sparse_layers
        for feature in (
            "theory_u_norm",
            "local_r_norm",
            "deleted_attention_mass_mean",
            "deleted_key_norm",
            "deleted_value_norm",
            "action_state_ratio",
        )
    ]
    kv_features = boundary_features + [
        f"layer{layer}_{feature}"
        for layer in sparse_layers
        for feature in (
            "attention_entropy",
            "attention_concentration",
            "head_disagreement",
            "key_norm_mean",
            "key_norm_variance",
            "value_norm_mean",
            "value_norm_variance",
        )
    ]
    # attention_entropy/concentration are present in layer rows.
    missing = [name for name in kv_features if name not in rows.columns]
    if missing:
        # Add the two state attention fields to the wide table mechanically.
        extra = []
        for feature in ("attention_entropy", "attention_concentration"):
            pivot = layers.pivot(index=KEYS, columns="layer", values=feature)
            pivot.columns = [
                f"layer{int(layer)}_{feature}" for layer in pivot.columns
            ]
            extra.append(pivot)
        rows = rows.merge(
            pd.concat(extra, axis=1).reset_index(),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

    model_specs = [
        ("M5_KV_augmented_ridge", kv_features, "ridge"),
        ("M5_KV_augmented_pairwise_logistic", kv_features, "logistic"),
        ("M5_KV_augmented_depth3_tree", kv_features, "tree"),
    ]
    for name, features, family in model_specs:
        family_records = []
        grid = [0.1, 1.0, 10.0] if family != "tree" else [1.0]
        for regularization in grid:
            if family == "ridge":
                prediction = ridge_loso(rows, features, regularization)
            elif family == "logistic":
                prediction = pairwise_logistic_loso(
                    rows, features, regularization
                )
            else:
                prediction = tree_loso(rows, features)
            record, sequence = evaluate_score(
                rows,
                prediction,
                f"{name}_{regularization:g}",
                action_sequence,
            )
            record.update(
                {
                    "model_class": "M5",
                    "family": family,
                    "features": features,
                    "regularization": regularization,
                    "boundary_count": len(sparse_layers),
                    "kv_descriptor_dimension": len(features),
                    "candidate_probe_count": 0,
                    "parameter_count": (
                        len(features) + 1
                        if family != "tree"
                        else 15
                    ),
                }
            )
            family_records.append((record, sequence))
        best = select_best_record([item[0] for item in family_records])
        sequence = next(
            item[1] for item in family_records if item[0]["name"] == best["name"]
        )
        results.append(best)
        sequence["model"] = best["name"]
        sequence_frames.append(sequence)

    dense_theory_features = [
        f"layer{layer}_{feature}"
        for layer in range(28)
        for feature in (
            "theory_u_norm",
            "local_r_norm",
            "deleted_attention_mass_mean",
            "deleted_key_norm",
            "deleted_value_norm",
            "action_state_ratio",
        )
    ]
    for rank in (2, 4, 8):
        prediction = pca_interaction_loso(
            rows, dense_theory_features, rank
        )
        record, sequence = evaluate_score(
            rows,
            prediction,
            f"M6_low_rank_interaction_rank{rank}",
            action_sequence,
        )
        record.update(
            {
                "model_class": "M6",
                "interaction_rank": rank,
                "features": dense_theory_features,
                "boundary_count": 28,
                "candidate_probe_count": 0,
                "parameter_count": rank + rank * (rank + 1) // 2 + 1,
            }
        )
        results.append(record)
        sequence["model"] = record["name"]
        sequence_frames.append(sequence)

    dense_prediction = ridge_loso(
        rows, dense_theory_features, alpha=1.0
    )
    dense_record, dense_sequence = evaluate_score(
        rows,
        dense_prediction,
        "M8_dense_current_state_ridge",
        action_sequence,
    )
    dense_record.update(
        {
            "model_class": "M8",
            "features": dense_theory_features,
            "boundary_count": 28,
            "candidate_probe_count": 0,
            "parameter_count": len(dense_theory_features) + 1,
        }
    )
    results.append(dense_record)
    dense_sequence["model"] = dense_record["name"]
    sequence_frames.append(dense_sequence)

    # M9 candidate-specific probe scan and calibrated path.
    for boundary in range(1, 28):
        column = f"probe_b{boundary}_risk"
        record, sequence = evaluate_score(
            rows,
            column,
            f"M9_candidate_probe_b{boundary}",
            action_sequence,
        )
        record.update(
            {
                "model_class": "M9",
                "score_column": column,
                "boundary_count": 1,
                "candidate_probe_count": 1,
                "parameter_count": 0,
            }
        )
        results.append(record)
        sequence["model"] = record["name"]
        sequence_frames.append(sequence)
    for count in (1, 2, 4, 8):
        column = f"probe_b18_path_k{count}_risk"
        record, sequence = evaluate_score(
            rows,
            column,
            f"physical_path_b18_k{count}",
            action_sequence,
        )
        record.update(
            {
                "model_class": "physical_path",
                "score_column": column,
                "boundary_count": 1,
                "candidate_probe_count": 1,
                "path_midpoint_count": count,
                "parameter_count": 0,
            }
        )
        results.append(record)
        sequence["model"] = record["name"]
        sequence_frames.append(sequence)

    frame = pd.DataFrame(results)
    atomic_frame(
        EXPERIMENT / "results/calibration/model_class_results.parquet",
        frame,
    )
    frame.to_csv(
        EXPERIMENT / "results/calibration/model_class_results.csv",
        index=False,
    )
    all_sequences = pd.concat(sequence_frames, ignore_index=True)
    atomic_frame(
        EXPERIMENT / "results/calibration/model_sequence_metrics.parquet",
        all_sequences,
    )

    frozen = {
        "schema_version": 1,
        "status": "frozen_before_formal",
        "model_name": "candidate_specific_physical_probe_b18_path_k1",
        "model_class": "M9_candidate_specific_probe",
        "representation": (
            "actual candidate-conditioned residual response at boundary 18 "
            "plus current physical downstream KV readout"
        ),
        "score_column": "probe_b18_path_k1_risk",
        "boundary": 18,
        "probe_layer": 17,
        "boundary_count": 1,
        "kv_descriptor_dimension": 0,
        "candidate_probe_count": 1,
        "path_midpoint_count": 1,
        "parameter_count": 0,
        "calibration_ids": [
            "gov_report:112",
            "gov_report:113",
            "synthetic_niah_112",
            "synthetic_niah_113",
        ],
        "formal_ids": [
            "gov_report:114",
            "gov_report:115",
            "synthetic_niah_114",
            "synthetic_niah_115",
        ],
        "replication_ids": [
            "gov_report:116",
            "gov_report:117",
            "synthetic_niah_116",
            "synthetic_niah_117",
        ],
        "selection_rule": (
            "first boundary passing all calibration gates, then lowest "
            "midpoint count with identical gate result"
        ),
        "calibration_metrics": next(
            row
            for row in results
            if row["name"] == "physical_path_b18_k1"
        ),
        "dense_oracle": next(
            row
            for row in results
            if row["name"] == "M9_dense_mechanistic_all28"
        ),
        "config_sha256": sha256_file(config_path),
        "exact_physical_kl_is_input": False,
        "endpoint_physical_logits_are_input": False,
        "future_token_is_input": False,
        "future_attention_is_input": False,
        "formal_fit_allowed": False,
    }
    atomic_json(EXPERIMENT / "results/frozen_model.json", frozen)
    summary = {
        "completed": True,
        "model_class_count": int(frame["model_class"].nunique()),
        "evaluated_model_count": len(frame),
        "sparse_forward_boundaries": selected_boundaries,
        "frozen_model": frozen,
        "passing_models": frame.loc[
            frame.get("passed", False).eq(True), "name"
        ].astype(str).tolist(),
    }
    atomic_json(
        EXPERIMENT / "results/calibration/calibration_analysis.json",
        summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

