#!/usr/bin/env python3
"""Analyze P3 staleness, detectors, refresh primitives, and physical transfer."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3_core import (  # noqa: E402
    COMPONENT_COLUMNS,
    FORWARD_COST_PER_CANDIDATE,
    atomic_frame,
    atomic_json,
    choose_harmful_epsilon,
    decision_event,
    detector_metrics,
    forward_cost,
    pairwise_accuracy,
    prefilter_coverage,
    ranking_spearman,
    validate_feature_schema,
)


EXPERIMENT = ROOT / "experiments/p3_decision_validity"
RESULTS = EXPERIMENT / "results"
MODEL_PATH = RESULTS / "frozen_detector.json"
REFRESH_PATH = RESULTS / "frozen_refresh.json"
PREFILTER_PATH = RESULTS / "frozen_prefilter.json"
UNIT = [
    "sample_id",
    "task",
    "stage",
    "history_id",
    "tau_anchor",
    "target_anchor",
    "horizon",
    "layer",
]


def config() -> Dict[str, Any]:
    return yaml.safe_load(
        (EXPERIMENT / "p3_config.yaml").read_text(encoding="utf-8")
    )


def load_stage(stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = RESULTS / stage
    metadata = json.loads(
        (directory / "stage_metadata.json").read_text()
    )
    if not metadata["completed"]:
        raise RuntimeError(f"P3 stage {stage} is incomplete")
    return (
        pd.read_parquet(directory / "event_rows.parquet"),
        pd.read_parquet(directory / "candidate_rows.parquet"),
    )


def harmful_label(events: pd.DataFrame, epsilon: float) -> np.ndarray:
    return (
        events["reuse_normalized_regret"].to_numpy(dtype=float)
        > float(epsilon)
    )


def analyze_r0() -> Dict[str, Any]:
    cfg = config()
    events, candidates = load_stage("diagnostic")
    selection = choose_harmful_epsilon(
        events["reuse_normalized_regret"],
        cfg["staleness"]["harmful_regret_epsilon_grid"],
    )
    epsilon = float(selection["selected"])
    events = events.copy()
    events["harmful_stale"] = harmful_label(events, epsilon)
    summaries = []
    for horizon, group in events.groupby("horizon", sort=True):
        summaries.append(
            {
                "horizon": int(horizon),
                "unit_count": len(group),
                "top1_staleness_rate": float(
                    group["top1_stale"].mean()
                ),
                "harmful_staleness_rate": float(
                    group["harmful_stale"].mean()
                ),
                "pairwise_degradation": float(
                    group["pairwise_degradation"].mean()
                ),
                "fresh_reused_spearman": float(
                    group["fresh_reused_spearman"].mean()
                ),
                "reuse_normalized_regret": float(
                    group["reuse_normalized_regret"].mean()
                ),
                "refresh_benefit": float(
                    group["refresh_benefit"].mean()
                ),
            }
        )
    horizon = pd.DataFrame(summaries)
    stratified = (
        events.groupby(
            ["task", "layer", "history_id", "horizon"], sort=True
        )
        .agg(
            unit_count=("sample_id", "size"),
            top1_staleness_rate=("top1_stale", "mean"),
            harmful_staleness_rate=("harmful_stale", "mean"),
            reuse_normalized_regret=(
                "reuse_normalized_regret",
                "mean",
            ),
            fresh_reused_spearman=(
                "fresh_reused_spearman",
                "mean",
            ),
        )
        .reset_index()
    )
    reversal = events[
        events["top1_stale"] | events["harmful_stale"]
    ].sort_values("reuse_normalized_regret", ascending=False)
    atomic_frame(RESULTS / "r0_staleness_events.parquet", events)
    atomic_frame(RESULTS / "r0_horizon_summary.parquet", horizon)
    horizon.to_csv(RESULTS / "r0_horizon_summary.csv", index=False)
    atomic_frame(RESULTS / "r0_stratification.parquet", stratified)
    reversal.to_csv(RESULTS / "r0_reversal_cases.csv", index=False)
    result = {
        "stage": "P3-R0",
        "epsilon_selection": selection,
        "selected_epsilon": epsilon,
        "unit_count": len(events),
        "candidate_count": len(candidates),
        "harmful_event_count": int(events["harmful_stale"].sum()),
        "harmful_event_rate": float(events["harmful_stale"].mean()),
        "top1_staleness_rate": float(events["top1_stale"].mean()),
        "nondegenerate": bool(
            0 < int(events["harmful_stale"].sum()) < len(events)
        ),
        "task_event_rate": {
            str(key): float(value)
            for key, value in events.groupby("task")[
                "harmful_stale"
            ].mean().items()
        },
    }
    atomic_json(RESULTS / "r0_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def candidate_thresholds(values: np.ndarray) -> List[float]:
    quantiles = np.linspace(0.05, 0.95, 19)
    candidates = np.unique(np.quantile(values, quantiles))
    return [float(value) for value in candidates]


def _metric_record(
    *,
    name: str,
    family: str,
    specification: Mapping[str, Any],
    truth: np.ndarray,
    decision: np.ndarray,
    regret: np.ndarray,
    task: Sequence[str],
    score: np.ndarray,
    order: int,
    feature_count: int,
) -> Dict[str, Any]:
    metrics = detector_metrics(
        truth, decision, regret, task=task
    )
    try:
        auroc = float(roc_auc_score(truth, score))
    except ValueError:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(truth, score))
    except ValueError:
        auprc = float(np.mean(truth))
    return {
        "name": name,
        "family": family,
        "specification": dict(specification),
        "auroc": auroc,
        "auprc": auprc,
        "order": int(order),
        "feature_count": int(feature_count),
        **metrics,
    }


def _tree_to_dict(
    estimator: DecisionTreeClassifier,
    features: Sequence[str],
    node: int = 0,
) -> Dict[str, Any]:
    tree = estimator.tree_
    if tree.children_left[node] == tree.children_right[node]:
        counts = tree.value[node][0]
        probability = (
            float(counts[1] / counts.sum())
            if len(counts) > 1 and counts.sum() > 0
            else 0.0
        )
        return {
            "leaf": True,
            "probability": probability,
            "decision": probability >= 0.5,
        }
    return {
        "leaf": False,
        "feature": str(features[int(tree.feature[node])]),
        "threshold": float(tree.threshold[node]),
        "left": _tree_to_dict(estimator, features, tree.children_left[node]),
        "right": _tree_to_dict(
            estimator, features, tree.children_right[node]
        ),
    }


def predict_model(
    model: Mapping[str, Any], events: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    family = model["family"]
    spec = model["specification"]
    if family == "single_threshold":
        values = events[spec["feature"]].to_numpy(dtype=float)
        if spec["direction"] == "gt":
            score = values - float(spec["threshold"])
        else:
            score = float(spec["threshold"]) - values
        return score > 0.0, score
    if family == "two_threshold_or":
        decisions = []
        scores = []
        for rule in spec["rules"]:
            values = events[rule["feature"]].to_numpy(dtype=float)
            signed = (
                values - float(rule["threshold"])
                if rule["direction"] == "gt"
                else float(rule["threshold"]) - values
            )
            decisions.append(signed > 0.0)
            scores.append(signed)
        return (
            np.logical_or.reduce(decisions),
            np.maximum.reduce(scores),
        )
    if family == "two_threshold_and":
        decisions = []
        scores = []
        for rule in spec["rules"]:
            values = events[rule["feature"]].to_numpy(dtype=float)
            signed = (
                values - float(rule["threshold"])
                if rule["direction"] == "gt"
                else float(rule["threshold"]) - values
            )
            decisions.append(signed > 0.0)
            scores.append(signed)
        return (
            np.logical_and.reduce(decisions),
            np.minimum.reduce(scores),
        )
    if family == "logistic":
        values = events[spec["features"]].to_numpy(dtype=float)
        standardized = (
            values - np.asarray(spec["mean"], dtype=float)
        ) / np.asarray(spec["scale"], dtype=float)
        linear = (
            standardized @ np.asarray(spec["coefficient"], dtype=float)
            + float(spec["intercept"])
        )
        probability = 1.0 / (1.0 + np.exp(-linear))
        threshold = float(spec["probability_threshold"])
        return probability >= threshold, probability
    if family == "depth3_tree":
        decisions = []
        scores = []
        for _, row in events.iterrows():
            node = spec["tree"]
            while not node["leaf"]:
                node = (
                    node["left"]
                    if float(row[node["feature"]])
                    <= float(node["threshold"])
                    else node["right"]
                )
            decisions.append(bool(node["decision"]))
            scores.append(float(node["probability"]))
        return np.asarray(decisions), np.asarray(scores)
    raise ValueError(f"unknown detector family {family}")


def calibrate_detector() -> Dict[str, Any]:
    cfg = config()
    epsilon = cfg["staleness"]["harmful_regret_epsilon"]
    if epsilon is None:
        raise RuntimeError("freeze harmful epsilon before detector calibration")
    events, _candidates = load_stage("calibration")
    zero = validate_feature_schema(
        cfg["observables"]["zero_cost"], allow_low_cost=False
    )
    low = validate_feature_schema(
        cfg["observables"]["low_cost"], allow_low_cost=True
    )
    truth = harmful_label(events, float(epsilon))
    regret = events["reuse_normalized_regret"].to_numpy(dtype=float)
    task = events["task"].astype(str).to_numpy()
    records: List[Dict[str, Any]] = []
    threshold_records: List[Dict[str, Any]] = []
    for feature in zero:
        values = events[feature].to_numpy(dtype=float)
        for direction in ("gt", "lt"):
            for threshold in candidate_thresholds(values):
                spec = {
                    "feature": feature,
                    "threshold": threshold,
                    "direction": direction,
                }
                decision, score = predict_model(
                    {
                        "family": "single_threshold",
                        "specification": spec,
                    },
                    events,
                )
                record = _metric_record(
                    name=f"{feature}_{direction}_{threshold:.8g}",
                    family="single_threshold",
                    specification=spec,
                    truth=truth,
                    decision=decision,
                    regret=regret,
                    task=task,
                    score=score,
                    order=0,
                    feature_count=1,
                )
                records.append(record)
                threshold_records.append(record)
    top_rules = sorted(
        threshold_records,
        key=lambda row: (
            -row["harmful_recall"],
            row["missed_normalized_regret"],
            row["refresh_coverage"],
        ),
    )[:24]
    for left_index, left in enumerate(top_rules):
        for right in top_rules[left_index + 1 :]:
            if (
                left["specification"]["feature"]
                == right["specification"]["feature"]
            ):
                continue
            spec = {
                "rules": [
                    left["specification"],
                    right["specification"],
                ]
            }
            decision, score = predict_model(
                {
                    "family": "two_threshold_or",
                    "specification": spec,
                },
                events,
            )
            records.append(
                _metric_record(
                    name=f"or_{left['name']}__{right['name']}",
                    family="two_threshold_or",
                    specification=spec,
                    truth=truth,
                    decision=decision,
                    regret=regret,
                    task=task,
                    score=score,
                    order=1,
                    feature_count=2,
                )
            )
            decision, score = predict_model(
                {
                    "family": "two_threshold_and",
                    "specification": spec,
                },
                events,
            )
            records.append(
                _metric_record(
                    name=f"and_{left['name']}__{right['name']}",
                    family="two_threshold_and",
                    specification=spec,
                    truth=truth,
                    decision=decision,
                    regret=regret,
                    task=task,
                    score=score,
                    order=2,
                    feature_count=2,
                )
            )
    feature_sets = [("zero", list(zero), 2)]
    if low:
        feature_sets.append(
            ("zero_plus_one_midpoint", list(zero) + list(low), 4)
        )
    for feature_set_name, features, family_order in feature_sets:
        x = events[features].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x)
        standardized = scaler.transform(x)
        for regularization in cfg["detector"]["lambdas"]:
            estimator = LogisticRegression(
                C=1.0 / float(regularization),
                class_weight="balanced",
                random_state=int(cfg["runtime"]["seed"]),
                max_iter=1000,
            ).fit(standardized, truth)
            probability = estimator.predict_proba(standardized)[:, 1]
            for probability_threshold in np.linspace(0.1, 0.9, 17):
                spec = {
                    "features": features,
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                    "coefficient": estimator.coef_[0].tolist(),
                    "intercept": float(estimator.intercept_[0]),
                    "probability_threshold": float(
                        probability_threshold
                    ),
                    "regularization_lambda": float(regularization),
                    "feature_set": feature_set_name,
                }
                decision = probability >= probability_threshold
                records.append(
                    _metric_record(
                        name=(
                            f"logistic_{feature_set_name}_"
                            f"{regularization}_{probability_threshold:.2f}"
                        ),
                        family="logistic",
                        specification=spec,
                        truth=truth,
                        decision=decision,
                        regret=regret,
                        task=task,
                        score=probability,
                        order=family_order,
                        feature_count=len(features),
                    )
                )
        for depth in (1, 2, 3):
            estimator = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=int(cfg["runtime"]["seed"]),
            ).fit(x, truth)
            spec = {
                "features": features,
                "depth": depth,
                "feature_set": feature_set_name,
                "tree": _tree_to_dict(estimator, features),
            }
            decision = estimator.predict(x).astype(bool)
            probability = estimator.predict_proba(x)[:, 1]
            records.append(
                _metric_record(
                    name=f"tree_d{depth}_{feature_set_name}",
                    family="depth3_tree",
                    specification=spec,
                    truth=truth,
                    decision=decision,
                    regret=regret,
                    task=task,
                    score=probability,
                    order=3 if feature_set_name == "zero" else 4,
                    feature_count=len(
                        set(
                            feature
                            for feature in features
                            if feature
                            in json.dumps(spec["tree"])
                        )
                    ),
                )
            )
    table_rows = []
    for record in records:
        table_rows.append(
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in record.items()
            }
        )
    frame = pd.DataFrame(table_rows)
    atomic_frame(RESULTS / "detector_calibration_grid.parquet", frame)
    frame.to_csv(RESULTS / "detector_calibration_grid.csv", index=False)
    gates = cfg["detector"]
    feasible = [
        record
        for record in records
        if record["harmful_recall"]
        >= float(gates["harmful_recall_min"])
        and all(
            value >= float(gates["each_task_recall_min"])
            for value in record["task_recall"].values()
        )
        and record["missed_normalized_regret"]
        <= float(gates["missed_normalized_regret_max"])
        and record["refresh_coverage"]
        <= float(gates["refresh_coverage_max"])
    ]
    pool = feasible if feasible else records
    selected = sorted(
        pool,
        key=lambda row: (
            -row["harmful_recall"],
            row["missed_normalized_regret"],
            row["refresh_coverage"],
            row["order"],
            row["feature_count"],
            -row["auprc"],
            row["name"],
        ),
    )[0]
    frozen = {
        "schema_version": 1,
        "epsilon": float(epsilon),
        "family": selected["family"],
        "name": selected["name"],
        "specification": selected["specification"],
        "calibration_metrics": {
            key: value
            for key, value in selected.items()
            if key
            not in {
                "specification",
                "family",
                "name",
                "order",
                "feature_count",
            }
        },
        "zero_cost": (
            selected["specification"].get("feature_set") !=
            "zero_plus_one_midpoint"
            and "top_reused_one_midpoint_shift"
            not in json.dumps(selected["specification"])
        ),
        "forward_cost_per_unit": (
            0
            if (
                selected["specification"].get("feature_set") !=
                "zero_plus_one_midpoint"
                and "top_reused_one_midpoint_shift"
                not in json.dumps(selected["specification"])
            )
            else 2
        ),
    }
    atomic_json(MODEL_PATH, frozen)
    result = {
        "stage": "P3-R1/R2 calibration",
        "event_count": len(events),
        "harmful_event_count": int(truth.sum()),
        "candidate_model_count": len(records),
        "feasible_model_count": len(feasible),
        "selected": frozen,
        "calibration_gate_passed": bool(feasible),
        "recovery_branch": (
            "R1_zero_cost"
            if frozen["zero_cost"] and feasible
            else "R2_B_one_midpoint"
            if feasible
            else "R2_E_no_detector_signal"
        ),
    }
    atomic_json(RESULTS / "detector_calibration_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def evaluate_detector(stage: str) -> Dict[str, Any]:
    cfg = config()
    model = json.loads(MODEL_PATH.read_text())
    events, _candidates = load_stage(stage)
    truth = harmful_label(events, float(model["epsilon"]))
    decision, score = predict_model(model, events)
    metrics = detector_metrics(
        truth,
        decision,
        events["reuse_normalized_regret"],
        task=events["task"].astype(str),
    )
    try:
        metrics["auroc"] = float(roc_auc_score(truth, score))
    except ValueError:
        metrics["auroc"] = 0.5
    try:
        metrics["auprc"] = float(
            average_precision_score(truth, score)
        )
    except ValueError:
        metrics["auprc"] = float(np.mean(truth))
    baselines = []
    for interval in (1, 2, 4, 8, 16, 32):
        baseline = (
            events["horizon"].to_numpy(dtype=int) >= interval
        )
        baselines.append(
            {
                "interval": interval,
                **detector_metrics(
                    truth,
                    baseline,
                    events["reuse_normalized_regret"],
                    task=events["task"].astype(str),
                ),
            }
        )
    gates = cfg["detector"]
    checks = {
        "harmful_recall": metrics["harmful_recall"]
        >= float(gates["harmful_recall_min"]),
        "each_task_recall": all(
            value >= float(gates["each_task_recall_min"])
            for value in metrics["task_recall"].values()
        ),
        "missed_regret": metrics["missed_normalized_regret"]
        <= float(gates["missed_normalized_regret_max"]),
        "coverage": metrics["refresh_coverage"]
        <= float(gates["refresh_coverage_max"]),
        "below_always_refresh": metrics["refresh_coverage"] < 1.0,
        "pareto_vs_fixed_interval": not any(
            baseline["harmful_recall"] >= metrics["harmful_recall"]
            and baseline["missed_normalized_regret"]
            <= metrics["missed_normalized_regret"]
            and baseline["refresh_coverage"] < metrics["refresh_coverage"]
            for baseline in baselines
        ),
    }
    enriched = events.copy()
    enriched["harmful_stale"] = truth
    enriched["detector_decision"] = decision
    enriched["detector_score"] = score
    atomic_frame(
        RESULTS / stage / "detector_event_rows.parquet", enriched
    )
    result = {
        "stage": stage,
        "model_name": model["name"],
        "metrics": metrics,
        "fixed_interval_baselines": baselines,
        "checks": checks,
        "passed": all(checks.values()),
    }
    atomic_json(
        RESULTS / stage / "detector_summary.json", result
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _unit_scores(
    candidates: pd.DataFrame, score_column: str
) -> Iterable[tuple[tuple[Any, ...], pd.DataFrame, np.ndarray]]:
    for key, group in candidates.groupby(UNIT, sort=True):
        ordered = group.sort_values("candidate_id")
        yield key, ordered, ordered[score_column].to_numpy(dtype=float)


def refresh_table(stage: str) -> pd.DataFrame:
    detector = json.loads(MODEL_PATH.read_text())
    events, candidates = load_stage(stage)
    decision, _score = predict_model(detector, events)
    event_key = events[UNIT].copy()
    event_key["detector_decision"] = decision
    event_key["harmful_stale"] = harmful_label(
        events, detector["epsilon"]
    )
    candidate = candidates.merge(
        event_key,
        on=UNIT,
        validate="many_to_one",
    )
    rows = []
    for key, group in candidate.groupby(UNIT, sort=True):
        ordered = group.sort_values("candidate_id")
        exact = ordered["controlled_exact_kl"].to_numpy(dtype=float)
        old = ordered["risk_all_old"].to_numpy(dtype=float)
        fresh = ordered["risk_full_fresh"].to_numpy(dtype=float)
        base_event = decision_event(exact, fresh, old, detector["epsilon"])
        for primitive, column in COMPONENT_COLUMNS.items():
            primitive_scores = ordered[column].to_numpy(dtype=float)
            primitive_event = decision_event(
                exact, fresh, primitive_scores, detector["epsilon"]
            )
            detector_scores = (
                primitive_scores
                if bool(ordered["detector_decision"].iloc[0])
                else old
            )
            joint_event = decision_event(
                exact, fresh, detector_scores, detector["epsilon"]
            )
            row = dict(zip(UNIT, key))
            row.update(
                {
                    "primitive": primitive,
                    "harmful_stale": bool(
                        ordered["harmful_stale"].iloc[0]
                    ),
                    "detector_decision": bool(
                        ordered["detector_decision"].iloc[0]
                    ),
                    "no_refresh_regret": base_event[
                        "reuse_normalized_regret"
                    ],
                    "full_fresh_regret": base_event[
                        "fresh_normalized_regret"
                    ],
                    "primitive_regret": primitive_event[
                        "reuse_normalized_regret"
                    ],
                    "joint_regret": joint_event[
                        "reuse_normalized_regret"
                    ],
                    "primitive_spearman": ranking_spearman(
                        exact, primitive_scores
                    ),
                    "primitive_pairwise": pairwise_accuracy(
                        exact, primitive_scores
                    ),
                    "primitive_top1_exact": float(
                        int(np.argmin(exact))
                        == int(np.argmin(primitive_scores))
                    ),
                    "full_forward_cost": forward_cost(
                        primitive, candidate_count=len(ordered)
                    ),
                    "realized_forward_cost": (
                        forward_cost(
                            primitive,
                            candidate_count=len(ordered),
                        )
                        if bool(
                            ordered["detector_decision"].iloc[0]
                        )
                        else 0
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_refresh(
    stage: str, *, select: bool = False
) -> Dict[str, Any]:
    cfg = config()
    rows = refresh_table(stage)
    atomic_frame(RESULTS / stage / "refresh_rows.parquet", rows)
    summary_rows = []
    stale = rows[rows["harmful_stale"]]
    for primitive, group in rows.groupby("primitive", sort=True):
        stale_group = stale[stale["primitive"] == primitive]
        no = float(stale_group["no_refresh_regret"].mean())
        fresh = float(stale_group["full_fresh_regret"].mean())
        current = float(stale_group["primitive_regret"].mean())
        denominator = max(no - fresh, 1.0e-12)
        task_gain = {}
        for task, task_group in stale_group.groupby("task"):
            task_gain[str(task)] = float(
                task_group["no_refresh_regret"].mean()
                - task_group["primitive_regret"].mean()
            )
        summary_rows.append(
            {
                "primitive": primitive,
                "stale_unit_count": len(stale_group),
                "gain_retention": float((no - current) / denominator),
                "normalized_regret_gap": float(current - fresh),
                "task_gain": task_gain,
                "spearman": float(group["primitive_spearman"].mean()),
                "pairwise_accuracy": float(
                    group["primitive_pairwise"].mean()
                ),
                "top1_accuracy": float(
                    group["primitive_top1_exact"].mean()
                ),
                "joint_normalized_regret": float(
                    group["joint_regret"].mean()
                ),
                "mean_forward_cost": float(
                    group["realized_forward_cost"].mean()
                ),
                "full_refresh_forward_cost": int(
                    group["full_forward_cost"].iloc[0]
                ),
            }
        )
    gates = cfg["refresh"]
    for row in summary_rows:
        row["passes"] = bool(
            row["gain_retention"]
            >= float(gates["full_fresh_gain_retention_min"])
            and row["normalized_regret_gap"]
            <= float(gates["normalized_regret_gap_max"])
            and all(
                gain > float(gates["each_task_strict_gain_min"])
                for gain in row["task_gain"].values()
            )
            and row["mean_forward_cost"]
            < float(gates["maximum_average_forward_cost"])
        )
    frame = pd.DataFrame(
        [
            {
                **row,
                "task_gain": json.dumps(row["task_gain"], sort_keys=True),
            }
            for row in summary_rows
        ]
    )
    atomic_frame(RESULTS / stage / "refresh_summary.parquet", frame)
    frame.to_csv(RESULTS / stage / "refresh_summary.csv", index=False)
    feasible = [row for row in summary_rows if row["passes"]]
    selected = None
    if select:
        pool = feasible if feasible else summary_rows
        selected = sorted(
            pool,
            key=lambda row: (
                row["full_refresh_forward_cost"],
                row["mean_forward_cost"],
                -row["gain_retention"],
                row["normalized_regret_gap"],
                row["primitive"],
            ),
        )[0]
        atomic_json(
            REFRESH_PATH,
            {
                "schema_version": 1,
                "primitive": selected["primitive"],
                "column": COMPONENT_COLUMNS[selected["primitive"]],
                "forward_cost_per_candidate": (
                    FORWARD_COST_PER_CANDIDATE[
                        selected["primitive"]
                    ]
                ),
                "calibration_metrics": selected,
            },
        )
    elif REFRESH_PATH.exists():
        name = json.loads(REFRESH_PATH.read_text())["primitive"]
        selected = next(
            row for row in summary_rows if row["primitive"] == name
        )
    result = {
        "stage": stage,
        "primitive_count": len(summary_rows),
        "feasible_count": len(feasible),
        "selected": selected,
        "passed": bool(selected and selected["passes"]),
    }
    atomic_json(RESULTS / stage / "refresh_gate.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def analyze_prefilter(
    stage: str, *, select: bool = False
) -> Dict[str, Any]:
    cfg = config()
    detector = json.loads(MODEL_PATH.read_text())
    refresh = json.loads(REFRESH_PATH.read_text())
    events, candidates = load_stage(stage)
    decision, _score = predict_model(detector, events)
    event_key = events[UNIT].copy()
    event_key["detector_decision"] = decision
    candidates = candidates.merge(
        event_key, on=UNIT, validate="many_to_one"
    )
    rows = []
    for key, group in candidates.groupby(UNIT, sort=True):
        ordered = group.sort_values("candidate_id")
        exact = ordered["controlled_exact_kl"].to_numpy(dtype=float)
        fresh = ordered["risk_full_fresh"].to_numpy(dtype=float)
        cheap_sets = {
            "action_only": ordered[
                "action_only_risk"
            ].to_numpy(dtype=float),
            "reused": ordered["risk_all_old"].to_numpy(dtype=float),
            "retained_attention": -ordered[
                "candidate_retained_attention_mass"
            ].to_numpy(dtype=float),
        }
        cheap_sets["consensus"] = np.mean(
            np.stack(
                [rankdata(value) for value in cheap_sets.values()]
            ),
            axis=0,
        )
        for method, cheap in cheap_sets.items():
            for k in cfg["prefilter"]["top_k_grid"]:
                exact_coverage = prefilter_coverage(cheap, exact, k)
                fresh_coverage = prefilter_coverage(cheap, fresh, k)
                row = dict(zip(UNIT, key))
                row.update(
                    {
                        "method": method,
                        "k": int(k),
                        "exact_coverage": exact_coverage["coverage"],
                        "fresh_coverage": fresh_coverage["coverage"],
                        "false_elimination": exact_coverage[
                            "false_elimination"
                        ],
                        "probe_savings": exact_coverage[
                            "probe_savings"
                        ],
                        "detector_decision": bool(
                            ordered["detector_decision"].iloc[0]
                        ),
                        "joint_forward_cost": (
                            int(
                                refresh[
                                    "forward_cost_per_candidate"
                                ]
                            )
                            * int(k)
                            if bool(
                                ordered[
                                    "detector_decision"
                                ].iloc[0]
                            )
                            else 0
                        ),
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_frame(RESULTS / stage / "prefilter_rows.parquet", frame)
    summary = (
        frame.groupby(["method", "k"], sort=True)
        .agg(
            exact_coverage=("exact_coverage", "mean"),
            fresh_coverage=("fresh_coverage", "mean"),
            false_elimination=("false_elimination", "mean"),
            probe_savings=("probe_savings", "mean"),
            mean_joint_forward_cost=("joint_forward_cost", "mean"),
        )
        .reset_index()
    )
    exact_min = float(cfg["prefilter"]["exact_optimum_coverage_min"])
    fresh_min = float(cfg["prefilter"]["fresh_optimum_coverage_min"])
    summary["passes"] = (
        summary["exact_coverage"].ge(exact_min)
        & summary["fresh_coverage"].ge(fresh_min)
    )
    atomic_frame(RESULTS / stage / "prefilter_summary.parquet", summary)
    summary.to_csv(
        RESULTS / stage / "prefilter_summary.csv", index=False
    )
    feasible = summary[summary["passes"]]
    if select:
        pool = (
            summary
            if feasible.empty
            else feasible
        )
        selected = (
            None
            if pool.empty
            else pool.sort_values(
                [
                    "exact_coverage",
                    "fresh_coverage",
                    "k",
                    "mean_joint_forward_cost",
                    "method",
                ],
                ascending=[False, False, True, True, True],
            ).iloc[0].to_dict()
        )
        if not feasible.empty:
            selected = feasible.sort_values(
                ["k", "mean_joint_forward_cost", "method"]
            ).iloc[0].to_dict()
        if selected is not None:
            atomic_json(
                PREFILTER_PATH,
                {
                    "schema_version": 1,
                    "method": selected["method"],
                    "k": int(selected["k"]),
                    "calibration_passed": bool(selected["passes"]),
                    "calibration_metrics": selected,
                },
            )
    else:
        frozen = json.loads(PREFILTER_PATH.read_text())
        row = summary[
            summary["method"].eq(frozen["method"])
            & summary["k"].eq(int(frozen["k"]))
        ]
        selected = (
            None if row.empty else row.iloc[0].to_dict()
        )
    result = {
        "stage": stage,
        "selected": selected,
        "passed": bool(
            selected is not None and selected["passes"]
        ),
    }
    atomic_json(RESULTS / stage / "prefilter_gate.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def analyze_physical(stage: str) -> Dict[str, Any]:
    cfg = config()
    events, candidates = load_stage(stage)
    recovery_rows = []
    for key, group in candidates.groupby(UNIT, sort=True):
        ordered = group.sort_values("candidate_id")
        physical = ordered["physical_exact_kl"].to_numpy(dtype=float)
        boundary = ordered[
            "physical_boundary_effect_norm"
        ].to_numpy(dtype=float)
        action = ordered["action_only_risk"].to_numpy(dtype=float)
        boundary_event = decision_event(
            physical, boundary, action, 0.0
        )
        recovery_rows.append(
            {
                **dict(zip(UNIT, key)),
                "boundary_physical_spearman": ranking_spearman(
                    boundary, physical
                ),
                "boundary_physical_normalized_regret": boundary_event[
                    "fresh_normalized_regret"
                ],
            }
        )
    recovery = pd.DataFrame(recovery_rows)
    events = events.merge(
        recovery,
        on=UNIT,
        validate="one_to_one",
    )
    sequence = (
        events.groupby(["sample_id", "task"], sort=True)
        .agg(
            controlled_physical_spearman=(
                "controlled_physical_spearman",
                "mean",
            ),
            scalar_physical_spearman=(
                "scalar_physical_spearman",
                "mean",
            ),
            action_only_physical_spearman=(
                "action_only_physical_spearman",
                "mean",
            ),
            fresh_physical_normalized_regret=(
                "fresh_physical_normalized_regret",
                "mean",
            ),
            action_only_physical_normalized_regret=(
                "action_only_physical_normalized_regret",
                "mean",
            ),
            boundary_physical_spearman=(
                "boundary_physical_spearman",
                "mean",
            ),
            boundary_physical_normalized_regret=(
                "boundary_physical_normalized_regret",
                "mean",
            ),
        )
        .reset_index()
    )
    atomic_frame(
        RESULTS / stage / "physical_sequence_metrics.parquet",
        sequence,
    )
    overall = float(sequence["scalar_physical_spearman"].median())
    task = {
        str(key): float(value)
        for key, value in sequence.groupby("task")[
            "scalar_physical_spearman"
        ].median().items()
    }
    positive = float(
        (sequence["scalar_physical_spearman"] > 0.0).mean()
    )
    task_gain = {
        str(name): float(
            (
                group["action_only_physical_normalized_regret"]
                - group["fresh_physical_normalized_regret"]
            ).mean()
        )
        for name, group in sequence.groupby("task")
    }
    boundary_overall = float(
        sequence["boundary_physical_spearman"].median()
    )
    boundary_task = {
        str(key): float(value)
        for key, value in sequence.groupby("task")[
            "boundary_physical_spearman"
        ].median().items()
    }
    boundary_positive = float(
        (sequence["boundary_physical_spearman"] > 0.0).mean()
    )
    boundary_gain = {
        str(name): float(
            (
                group["action_only_physical_normalized_regret"]
                - group["boundary_physical_normalized_regret"]
            ).mean()
        )
        for name, group in sequence.groupby("task")
    }
    gates = cfg["physical_transfer"]
    checks = {
        "overall_spearman": overall
        >= float(gates["overall_spearman_min"]),
        "each_task_spearman": all(
            value >= float(gates["each_task_spearman_min"])
            for value in task.values()
        ),
        "positive_sequence_fraction": positive
        >= float(gates["positive_sequence_fraction_min"]),
        "action_only_gain": all(
            value > float(gates["action_only_gain_strict_min"])
            for value in task_gain.values()
        ),
    }
    recovery_checks = {
        "overall_spearman": boundary_overall
        >= float(gates["overall_spearman_min"]),
        "each_task_spearman": all(
            value >= float(gates["each_task_spearman_min"])
            for value in boundary_task.values()
        ),
        "positive_sequence_fraction": boundary_positive
        >= float(gates["positive_sequence_fraction_min"]),
        "action_only_gain": all(
            value > float(gates["action_only_gain_strict_min"])
            for value in boundary_gain.values()
        ),
    }
    primary_passed = all(checks.values())
    recovery_passed = all(recovery_checks.values())
    result = {
        "stage": stage,
        "sequence_count": len(sequence),
        "overall_scalar_physical_spearman": overall,
        "task_scalar_physical_spearman": task,
        "positive_sequence_fraction": positive,
        "task_regret_gain_vs_action_only": task_gain,
        "checks": checks,
        "passed": primary_passed,
        "recovery_descriptor": {
            "name": "physical_boundary_effect_norm_no_fitted_weight",
            "overall_spearman": boundary_overall,
            "task_spearman": boundary_task,
            "positive_sequence_fraction": boundary_positive,
            "task_regret_gain_vs_action_only": boundary_gain,
            "checks": recovery_checks,
            "passed": recovery_passed,
        },
        "transfer_outcome": (
            "Transfer_A"
            if primary_passed
            else "Transfer_B"
            if recovery_passed
            else "Transfer_C"
        ),
    }
    atomic_json(RESULTS / stage / "physical_gate.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "r0",
            "calibrate-detector",
            "formal-detector",
            "replication-detector",
            "calibrate-refresh",
            "formal-refresh",
            "replication-refresh",
            "calibrate-prefilter",
            "formal-prefilter",
            "replication-prefilter",
            "physical-formal",
            "physical-replication",
        ],
    )
    args = parser.parse_args()
    dispatch = {
        "r0": analyze_r0,
        "calibrate-detector": calibrate_detector,
        "formal-detector": lambda: evaluate_detector("evaluation"),
        "replication-detector": lambda: evaluate_detector("replication"),
        "calibrate-refresh": lambda: summarize_refresh(
            "calibration", select=True
        ),
        "formal-refresh": lambda: summarize_refresh("evaluation"),
        "replication-refresh": lambda: summarize_refresh("replication"),
        "calibrate-prefilter": lambda: analyze_prefilter(
            "calibration", select=True
        ),
        "formal-prefilter": lambda: analyze_prefilter("evaluation"),
        "replication-prefilter": lambda: analyze_prefilter(
            "replication"
        ),
        "physical-formal": lambda: analyze_physical(
            "physical_evaluation"
        ),
        "physical-replication": lambda: analyze_physical(
            "physical_replication"
        ),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
