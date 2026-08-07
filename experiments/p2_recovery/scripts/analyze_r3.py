#!/usr/bin/env python3
"""Mechanically analyze R3 calibration, formal, and replication stages."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (P0_DIR, P2_DIR, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import ranking_metrics  # noqa: E402
from p2_core import atomic_frame, atomic_json, sha256_file  # noqa: E402
from recovery_core import sequence_gate  # noqa: E402


PRIMARY = ["H1", "H2", "H3"]
UNIT = ["sample_id", "task", "anchor", "layer", "history_id"]
RANK_METRICS = [
    "spearman",
    "pairwise_sign_accuracy",
    "top1_accuracy",
    "topk_overlap",
    "normalized_regret",
    "symmetric_scale_ratio",
]


def load_config() -> Dict[str, Any]:
    path = (
        ROOT
        / "experiments/p2_recovery/"
        "r3_path_integrated_readout/r3_config.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def stage_dir(stage: str) -> Path:
    return (
        ROOT
        / "experiments/p2_recovery/"
        f"r3_path_integrated_readout/results/{stage}"
    )


def load_stage(stage: str) -> Dict[str, Any]:
    directory = stage_dir(stage)
    metadata = json.loads(
        (directory / "stage_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if not metadata.get("completed"):
        raise RuntimeError(f"R3 {stage} is not complete")
    return {
        "directory": directory,
        "metadata": metadata,
        "response": pd.read_parquet(
            directory / "path_response_rows.parquet"
        ),
        "directions": pd.read_parquet(
            directory / "direction_rows.parquet"
        ),
        "vectors": pd.read_parquet(
            directory / "sequence_vector_metrics.parquet"
        ),
    }


def vector_gates(
    response: pd.DataFrame,
    vectors: pd.DataFrame,
    rule: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for method in sorted(response["method"].unique()):
        output[str(method)] = sequence_gate(
            vectors[vectors["method"] == method],
            response[
                (response["method"] == method)
                & response["history_id"].isin(PRIMARY)
            ],
            rule,
        )
    return output


def response_breakdown(response: pd.DataFrame) -> pd.DataFrame:
    primary = response[
        response["history_id"].isin(PRIMARY)
    ].copy()
    specifications: Sequence[tuple[str, Sequence[str]]] = [
        ("overall", []),
        ("task", ["task"]),
        ("sequence", ["sample_id", "task"]),
        ("layer", ["layer"]),
        ("history", ["history_id"]),
        ("task_layer", ["task", "layer"]),
        ("task_history", ["task", "history_id"]),
    ]
    rows: List[Dict[str, Any]] = []
    for stratum_type, columns in specifications:
        groups = (
            [((), primary)]
            if not columns
            else list(
                primary.groupby(
                    columns[0]
                    if len(columns) == 1
                    else list(columns),
                    sort=True,
                )
            )
        )
        for key, frame in groups:
            keys = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(columns, keys))
            for method, group in frame.groupby("method", sort=True):
                large_threshold = float(
                    primary["action_r_norm"].quantile(0.75)
                )
                large = group[
                    group["action_r_norm"] >= large_threshold
                ]
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum_json": json.dumps(
                            labels, sort_keys=True, default=str
                        ),
                        **labels,
                        "method": method,
                        "jvp_cost": int(group["jvp_cost"].iloc[0]),
                        "row_count": len(group),
                        "median_cosine": float(group["cosine"].median()),
                        "median_relative_l2": float(
                            group["relative_l2"].median()
                        ),
                        "median_fisher_relative_error": float(
                            group["fisher_relative_error"].median()
                        ),
                        "row_cosine_pass_fraction": float(
                            group["cosine"].ge(0.99).mean()
                        ),
                        "large_action_row_count": len(large),
                        "large_action_median_cosine": (
                            float(large["cosine"].median())
                            if len(large)
                            else float("nan")
                        ),
                        "large_action_median_relative_l2": (
                            float(large["relative_l2"].median())
                            if len(large)
                            else float("nan")
                        ),
                        "all_finite": bool(group["finite"].all()),
                        "large_action_threshold": large_threshold,
                    }
                )
    return pd.DataFrame(rows)


def score_rows(
    response: pd.DataFrame, directions: pd.DataFrame
) -> pd.DataFrame:
    method = response[
        response["history_id"].isin(PRIMARY)
    ][
        UNIT
        + [
            "candidate_id",
            "candidate_source",
            "mask_hash",
            "method",
            "jvp_cost",
            "score",
            "controlled_exact_kl",
        ]
    ].rename(columns={"method": "score_type"})
    baseline = directions[
        directions["history_id"].isin(PRIMARY)
    ][
        UNIT
        + [
            "candidate_id",
            "candidate_source",
            "mask_hash",
            "reference_action_fisher_score",
            "controlled_exact_kl",
        ]
    ].copy()
    baseline["score_type"] = "reference_action_fisher"
    baseline["jvp_cost"] = 0
    baseline = baseline.rename(
        columns={"reference_action_fisher_score": "score"}
    )
    return pd.concat([method, baseline], ignore_index=True)


def make_rankings(scores: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, unit in scores.groupby(UNIT, sort=False):
        common = dict(zip(UNIT, key))
        for score_type, group in unit.groupby(
            "score_type", sort=True
        ):
            if len(group) != 8 or group["mask_hash"].nunique() != 8:
                raise RuntimeError(
                    f"R3 ranking unit is not eight-distinct: "
                    f"{key}/{score_type}"
                )
            ordered = group.sort_values(
                ["score", "candidate_id"], kind="mergesort"
            )
            truth = group.sort_values(
                ["controlled_exact_kl", "candidate_id"],
                kind="mergesort",
            )
            chosen = ordered.iloc[0]
            best = truth.iloc[0]
            rows.append(
                {
                    **common,
                    "score_type": score_type,
                    "jvp_cost": int(group["jvp_cost"].iloc[0]),
                    "candidate_count": len(group),
                    "chosen_candidate_id": chosen["candidate_id"],
                    "chosen_exact_kl": float(
                        chosen["controlled_exact_kl"]
                    ),
                    "best_candidate_id": best["candidate_id"],
                    "best_exact_kl": float(
                        best["controlled_exact_kl"]
                    ),
                    **ranking_metrics(
                        group["score"].to_numpy(dtype=np.float64),
                        group["controlled_exact_kl"].to_numpy(
                            dtype=np.float64
                        ),
                        2,
                    ),
                }
            )
    return pd.DataFrame(rows)


def sequence_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    return (
        rankings.groupby(
            ["sample_id", "task", "score_type"],
            as_index=False,
        )[RANK_METRICS]
        .median()
        .sort_values(["task", "sample_id", "score_type"])
    )


def score_value(
    sequence: pd.DataFrame,
    score_type: str,
    metric: str,
    task: str | None = None,
) -> float:
    source = sequence[sequence["score_type"] == score_type]
    if task is not None:
        source = source[source["task"] == task]
    return float(source[metric].median())


def decision_gate(
    sequence: pd.DataFrame,
    method: str,
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = "reference_action_fisher"
    tasks = sorted(str(value) for value in sequence["task"].unique())
    rho = score_value(sequence, method, "spearman")
    action_rho = score_value(sequence, baseline, "spearman")
    task_rho = {
        task: score_value(sequence, method, "spearman", task)
        for task in tasks
    }
    task_gain = {
        task: task_rho[task]
        - score_value(sequence, baseline, "spearman", task)
        for task in tasks
    }
    pivot = sequence.pivot(
        index=["sample_id", "task"],
        columns="score_type",
        values="spearman",
    )
    positive_fraction = float(
        (pivot[method] - pivot[baseline]).gt(0.0).mean()
    )
    pairwise_gain = (
        score_value(sequence, method, "pairwise_sign_accuracy")
        - score_value(
            sequence, baseline, "pairwise_sign_accuracy"
        )
    )
    top1_gain = (
        score_value(sequence, method, "top1_accuracy")
        - score_value(sequence, baseline, "top1_accuracy")
    )
    regret_gain = (
        score_value(sequence, baseline, "normalized_regret")
        - score_value(sequence, method, "normalized_regret")
    )
    maximum_degradation = float(
        rule["secondary_max_degradation"]
    )
    metrics = {
        "overall_spearman": rho,
        "action_spearman": action_rho,
        "overall_action_gain": rho - action_rho,
        "task_spearman": task_rho,
        "task_action_gain": task_gain,
        "positive_sequence_fraction": positive_fraction,
        "pairwise_gain": pairwise_gain,
        "top1_gain": top1_gain,
        "normalized_regret_gain": regret_gain,
    }
    checks = {
        "overall_spearman": rho
        >= float(rule["overall_sequence_first_spearman_min"]),
        "each_task_spearman": all(
            value >= float(rule["each_task_spearman_min"])
            for value in task_rho.values()
        ),
        "each_task_action_gain": all(
            value
            > float(rule["each_task_action_gain_strict_min"])
            for value in task_gain.values()
        ),
        "positive_sequence_fraction": positive_fraction
        >= float(rule["positive_sequence_fraction_min"]),
        "pairwise_gain": pairwise_gain
        >= float(rule["pairwise_gain_min"]),
        "top1_or_regret_improves": (
            top1_gain > 0.0 or regret_gain > 0.0
        ),
        "secondary_not_degraded": (
            top1_gain >= -maximum_degradation
            and regret_gain >= -maximum_degradation
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
    }


def reduced_gate(
    sequence: pd.DataFrame,
    selected: str,
    oracle: str,
    selected_cost: int,
    rule: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = "reference_action_fisher"
    selected_rho = score_value(sequence, selected, "spearman")
    oracle_rho = score_value(sequence, oracle, "spearman")
    baseline_rho = score_value(sequence, baseline, "spearman")
    selected_gain = selected_rho - baseline_rho
    oracle_gain = oracle_rho - baseline_rho
    retention = (
        selected_gain / oracle_gain
        if oracle_gain > 0.0
        else float("-inf")
    )
    metrics = {
        "selected_method": selected,
        "oracle_method": oracle,
        "selected_jvp_cost": int(selected_cost),
        "selected_spearman": selected_rho,
        "oracle_spearman": oracle_rho,
        "baseline_spearman": baseline_rho,
        "selected_gain": selected_gain,
        "oracle_gain": oracle_gain,
        "gain_retention": retention,
        "oracle_spearman_gap": oracle_rho - selected_rho,
    }
    checks = {
        "cost": selected_cost
        <= int(rule["maximum_jvp_cost"]),
        "gain_retention": retention
        >= float(rule["gain_retention_min"]),
        "oracle_gap": oracle_rho - selected_rho
        <= float(rule["oracle_spearman_gap_max"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "metrics": metrics,
    }


def calibration_analysis(
    config: Mapping[str, Any], data: Mapping[str, Any]
) -> Dict[str, Any]:
    gates = vector_gates(
        data["response"],
        data["vectors"],
        config["calibration_selection"]["vector_gate"],
    )
    order = [
        str(value)
        for value in config["methods"]["selection_order"]
    ]
    eligible = [
        method
        for method in order
        if gates[method]["passed"]
        and bool(config["methods"][method]["selectable"])
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda method: (
                int(config["methods"][method]["jvp_cost"]),
                float(
                    data["response"][
                        (data["response"]["method"] == method)
                        & data["response"]["history_id"].isin(PRIMARY)
                    ]["relative_l2"].median()
                ),
                order.index(method),
            ),
        )
        cost: int | None = int(
            config["methods"][selected]["jvp_cost"]
        )
    else:
        selected = None
        cost = None
    return {
        "stage": "calibration",
        "selection_rule": config["calibration_selection"]["rule"],
        "method_gates": gates,
        "eligible_methods": eligible,
        "selected_method": selected,
        "selected_jvp_cost": cost,
    }


def formal_analysis(
    stage: str,
    config: Mapping[str, Any],
    data: Mapping[str, Any],
) -> Dict[str, Any]:
    selected = str(
        config["calibration_selection"]["selected_method"]
    )
    vector = vector_gates(
        data["response"],
        data["vectors"],
        config["formal_gates"]["mechanism"],
    )
    scores = score_rows(data["response"], data["directions"])
    rankings = make_rankings(scores)
    sequence = sequence_rankings(rankings)
    directory = data["directory"]
    atomic_frame(directory / "score_rows.parquet", scores)
    atomic_frame(directory / "ranking_rows.parquet", rankings)
    atomic_frame(
        directory / "sequence_first_rankings.parquet", sequence
    )
    sequence.to_csv(
        directory / "sequence_first_rankings.csv", index=False
    )
    decision = decision_gate(
        sequence, selected, config["formal_gates"]["decision"]
    )
    if stage == "evaluation":
        reduced = reduced_gate(
            sequence,
            selected,
            str(config["formal_gates"]["reduced"]["oracle_method"]),
            int(
                config["calibration_selection"][
                    "selected_jvp_cost"
                ]
            ),
            config["formal_gates"]["reduced"],
        )
    else:
        reduced = {
            "passed": True,
            "not_recomputed": (
                "Replication runs the frozen selected method only; "
                "formal reduced evidence remains applicable."
            ),
        }
    applicable_pass = bool(
        vector[selected]["passed"]
        and decision["passed"]
        and reduced["passed"]
    )
    return {
        "stage": stage,
        "selected_method": selected,
        "mechanism": vector[selected],
        "all_method_mechanism_gates": vector,
        "decision": decision,
        "reduced": reduced,
        "passed": applicable_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["calibration", "evaluation", "replication"],
    )
    args = parser.parse_args()
    config = load_config()
    data = load_stage(args.stage)
    if args.stage == "calibration":
        result = calibration_analysis(config, data)
    else:
        result = formal_analysis(args.stage, config, data)
    direction_norms = data["directions"][
        UNIT + ["candidate_id", "action_r_norm"]
    ]
    enriched_response = data["response"].merge(
        direction_norms,
        on=UNIT + ["candidate_id"],
        validate="many_to_one",
    )
    breakdown = response_breakdown(enriched_response)
    atomic_frame(
        data["directory"] / "response_breakdown.parquet",
        breakdown,
    )
    breakdown.to_csv(
        data["directory"] / "response_breakdown.csv", index=False
    )
    result["stage_metadata_sha256"] = sha256_file(
        data["directory"] / "stage_metadata.json"
    )
    result["row_counts"] = {
        "directions": len(data["directions"]),
        "response": len(data["response"]),
        "vectors": len(data["vectors"]),
    }
    atomic_json(data["directory"] / "analysis_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
