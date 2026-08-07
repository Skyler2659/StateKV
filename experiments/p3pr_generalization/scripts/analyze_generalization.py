#!/usr/bin/env python3
"""Analyze cross-model/task P3PR results with sequence-first gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3pr_generalization"
P3PR_SCRIPTS = ROOT / "experiments/p3_physical_recovery/scripts"
for value in (ROOT, ROOT / "benchmarks/torch", P3PR_SCRIPTS):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p3pr_core import (  # noqa: E402
    atomic_frame,
    atomic_json,
    ranking_metrics,
)


SCORES = (
    "action_only_risk",
    "dense_all_layer_mechanistic_risk",
    "relative_penultimate_exact_map_risk",
    "relative_penultimate_path_k1_risk",
)
STAGES = ("calibration", "formal", "replication")


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(
        (EXPERIMENT / "p3pr_generalization_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def load_results(
    config: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_frames = []
    unit_frames = []
    for stage in STAGES:
        for model_key in config["runtime"]["model_order"]:
            directory = EXPERIMENT / "results" / stage / str(model_key)
            candidate_frames.append(
                pd.read_parquet(directory / "candidate_rows.parquet")
            )
            unit_frames.append(
                pd.read_parquet(directory / "unit_rows.parquet")
            )
    return (
        pd.concat(candidate_frames, ignore_index=True),
        pd.concat(unit_frames, ignore_index=True),
    )


def sequence_metrics(
    rows: pd.DataFrame, score: str
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    keys = ("stage", "model_key", "model_family", "sample_id", "task")
    for key, group in rows.groupby(list(keys), sort=True):
        metrics = ranking_metrics(
            group["exact_physical_kl"].to_numpy(),
            group[score].to_numpy(),
        )
        records.append(
            {
                **dict(zip(keys, key)),
                "score": score,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def _aggregate(group: pd.DataFrame) -> Dict[str, Any]:
    values = group["spearman"].to_numpy(dtype=np.float64)
    generator = np.random.default_rng(2026072817 + len(values))
    bootstrap = np.mean(
        values[
            generator.integers(
                0,
                len(values),
                size=(20000, len(values)),
            )
        ],
        axis=1,
    )
    return {
        "sequence_count": int(len(group)),
        "spearman": float(group["spearman"].mean()),
        "spearman_bootstrap_ci_low": float(
            np.quantile(bootstrap, 0.025)
        ),
        "spearman_bootstrap_ci_high": float(
            np.quantile(bootstrap, 0.975)
        ),
        "pairwise_accuracy": float(group["pairwise_accuracy"].mean()),
        "top1_accuracy": float(group["top1_correct"].mean()),
        "positive_sequence_fraction": float(
            (group["spearman"] > 0.0).mean()
        ),
        "normalized_regret": float(group["normalized_regret"].mean()),
        "median_exact_range": float(group["exact_range"].median()),
    }


def exact_paired_sign_flip(
    left: np.ndarray, right: np.ndarray
) -> Dict[str, Any]:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    observed = float(np.mean(differences))
    count = len(differences)
    null_values = []
    for mask in range(1 << count):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(count)]
        )
        null_values.append(float(np.mean(signs * differences)))
    p_value = float(
        np.mean(np.asarray(null_values, dtype=np.float64) >= observed - 1.0e-15)
    )
    return {
        "paired_sequence_count": count,
        "mean_spearman_gain": observed,
        "exact_one_sided_sign_flip_p": p_value,
        "all_sequence_gains_positive": bool((differences > 0.0).all()),
        "minimum_sequence_gain": float(differences.min()),
        "maximum_sequence_gain": float(differences.max()),
    }


def summaries(sequence: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for (stage, score), group in sequence.groupby(
        ["stage", "score"], sort=True
    ):
        records.append(
            {
                "stage": stage,
                "score": score,
                "stratum_type": "overall",
                "stratum": "all",
                **_aggregate(group),
            }
        )
        for model_key, subgroup in group.groupby("model_key", sort=True):
            records.append(
                {
                    "stage": stage,
                    "score": score,
                    "stratum_type": "model",
                    "stratum": str(model_key),
                    **_aggregate(subgroup),
                }
            )
        for task, subgroup in group.groupby("task", sort=True):
            records.append(
                {
                    "stage": stage,
                    "score": score,
                    "stratum_type": "task",
                    "stratum": str(task),
                    **_aggregate(subgroup),
                }
            )
        for (model_key, task), subgroup in group.groupby(
            ["model_key", "task"], sort=True
        ):
            records.append(
                {
                    "stage": stage,
                    "score": score,
                    "stratum_type": "model_task",
                    "stratum": f"{model_key}|{task}",
                    **_aggregate(subgroup),
                }
            )
    return pd.DataFrame(records)


def score_gate(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
    stage: str,
    score: str,
) -> Dict[str, Any]:
    gates = config["gates"]["formal_and_replication"]
    current = summary[
        (summary["stage"] == stage) & (summary["score"] == score)
    ].copy()
    overall = current[current["stratum_type"] == "overall"].iloc[0]
    models = current[current["stratum_type"] == "model"]
    tasks = current[current["stratum_type"] == "task"]
    model_tasks = current[current["stratum_type"] == "model_task"]
    checks = {
        "overall_spearman": float(overall["spearman"])
        >= float(gates["overall_spearman_min"]),
        "each_model_spearman": bool(
            (
                models["spearman"]
                >= float(gates["each_model_spearman_min"])
            ).all()
        ),
        "each_task_spearman": bool(
            (
                tasks["spearman"]
                >= float(gates["each_task_spearman_min"])
            ).all()
        ),
        "each_model_task_spearman": bool(
            (
                model_tasks["spearman"]
                >= float(gates["each_model_task_spearman_min"])
            ).all()
        ),
        "pairwise_accuracy": float(overall["pairwise_accuracy"])
        >= float(gates["pairwise_accuracy_min"]),
        "top1_accuracy": float(overall["top1_accuracy"])
        >= float(gates["top1_accuracy_min"]),
        "positive_sequence_fraction": float(
            overall["positive_sequence_fraction"]
        )
        >= float(gates["positive_sequence_fraction_min"]),
        "normalized_regret": float(overall["normalized_regret"])
        <= float(gates["normalized_regret_max"]),
    }
    return {
        "stage": stage,
        "score": score,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "overall": {
            key: (
                int(overall[key])
                if key == "sequence_count"
                else float(overall[key])
            )
            for key in (
                "sequence_count",
                "spearman",
                "pairwise_accuracy",
                "top1_accuracy",
                "positive_sequence_fraction",
                "normalized_regret",
                "median_exact_range",
                "spearman_bootstrap_ci_low",
                "spearman_bootstrap_ci_high",
            )
        },
        "model_spearman": {
            str(row["stratum"]): float(row["spearman"])
            for _, row in models.iterrows()
        },
        "task_spearman": {
            str(row["stratum"]): float(row["spearman"])
            for _, row in tasks.iterrows()
        },
        "model_task_spearman": {
            str(row["stratum"]): float(row["spearman"])
            for _, row in model_tasks.iterrows()
        },
    }


def integrity_gate(
    units: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, Any]:
    gates = config["gates"]["integrity"]
    stage_metadata = []
    for stage in STAGES:
        for model_key in config["runtime"]["model_order"]:
            path = (
                EXPERIMENT
                / "results"
                / stage
                / str(model_key)
                / "stage_metadata.json"
            )
            stage_metadata.append(json.loads(path.read_text()))
    hook_fractions = [
        float(item["model_info"]["attention_hooked_layers"])
        / max(float(item["model_info"]["num_layers"]), 1.0)
        for item in stage_metadata
    ]
    checks = {
        "hook_coverage": min(hook_fractions)
        >= float(gates["hook_coverage_fraction_min"]),
        "no_op_kl": float(units["no_op_exact_kl"].max())
        <= float(gates["no_op_kl_max"]),
        "baseline_replay": float(
            units["baseline_repeat_max_abs_error"].max()
        )
        <= float(gates["replay_logit_max_abs_error"]),
        "candidate_replay": float(
            units["candidate_replay_max_abs_error"].max()
        )
        <= float(gates["replay_logit_max_abs_error"]),
        "deletion_identity": float(
            units["identity_stable_relative_l2_tau_1e8_max"].max()
        )
        <= float(gates["identity_relative_l2_max"]),
        "finite_candidates": int(units["finite_candidate_count"].min())
        >= int(gates["finite_candidates_min"]),
        "clone_isolation": bool(units["prequery_clone_isolated"].all()),
        "generator_no_exact_kl": bool(
            (~units["candidate_generator_exact_kl_used"]).all()
        ),
        "generator_no_endpoint_logits": bool(
            (~units["candidate_generator_endpoint_logits_used"]).all()
        ),
        "generator_no_task_id": bool(
            (~units["candidate_generator_task_id_used"]).all()
        ),
        "query_alignment": bool(
            (units["query_position"] == units["expected_query_position"]).all()
        ),
        "token_alignment": bool(
            (units["token_id"] == units["expected_token_id"]).all()
        ),
    }
    return {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "max_no_op_kl": float(units["no_op_exact_kl"].max()),
        "max_baseline_replay_logit_error": float(
            units["baseline_repeat_max_abs_error"].max()
        ),
        "max_candidate_replay_logit_error": float(
            units["candidate_replay_max_abs_error"].max()
        ),
        "max_identity_relative_l2": float(
            units["identity_stable_relative_l2_tau_1e8_max"].max()
        ),
        "minimum_hook_coverage": min(hook_fractions),
        "minimum_finite_candidates": int(
            units["finite_candidate_count"].min()
        ),
    }


def calibration_boundary_scan(
    candidates: pd.DataFrame,
) -> List[Dict[str, Any]]:
    rows = candidates[candidates["stage"] == "calibration"]
    output: List[Dict[str, Any]] = []
    for model_key, group in rows.groupby("model_key", sort=True):
        boundaries = sorted(
            {
                int(column.split("_", 1)[0][1:])
                for column in group.columns
                if column.startswith("b") and column.endswith("_path_k1_risk")
                and group[column].notna().all()
            }
        )
        for boundary in boundaries:
            score = f"b{boundary}_path_k1_risk"
            metrics = sequence_metrics(group.assign(stage="calibration"), score)
            aggregate = _aggregate(metrics)
            output.append(
                {
                    "model_key": str(model_key),
                    "num_layers": int(group["num_layers"].iloc[0]),
                    "boundary": boundary,
                    "relative_depth": float(
                        boundary / int(group["num_layers"].iloc[0])
                    ),
                    **aggregate,
                }
            )
    return output


def verify_role_isolation(
    candidates: pd.DataFrame, config: Mapping[str, Any]
) -> Dict[str, Any]:
    ids_by_stage = {
        stage: set(
            candidates[candidates["stage"] == stage]["sample_id"]
            .astype(str)
            .unique()
        )
        for stage in STAGES
    }
    pairwise_disjoint = all(
        not (ids_by_stage[left] & ids_by_stage[right])
        for left_index, left in enumerate(STAGES)
        for right in STAGES[left_index + 1 :]
    )
    paired_models = True
    for stage in STAGES:
        sets = [
            set(group["sample_id"].astype(str).unique())
            for _, group in candidates[
                candidates["stage"] == stage
            ].groupby("model_key")
        ]
        paired_models = paired_models and bool(
            sets and all(current == sets[0] for current in sets)
        )
    return {
        "stage_sample_ids": {
            stage: sorted(values) for stage, values in ids_by_stage.items()
        },
        "roles_pairwise_disjoint": pairwise_disjoint,
        "models_paired_within_role": paired_models,
        "passed": bool(pairwise_disjoint and paired_models),
    }


def main() -> None:
    config = load_config()
    candidates, units = load_results(config)
    sequence = pd.concat(
        [sequence_metrics(candidates, score) for score in SCORES],
        ignore_index=True,
    )
    summary = summaries(sequence)
    atomic_frame(
        EXPERIMENT / "results/analysis/sequence_metrics.parquet",
        sequence,
    )
    sequence.to_csv(
        EXPERIMENT / "results/analysis/sequence_metrics.csv", index=False
    )
    summary.to_csv(
        EXPERIMENT / "results/analysis/score_summary.csv", index=False
    )

    integrity = integrity_gate(units, config)
    isolation = verify_role_isolation(candidates, config)
    gates = {
        stage: {
            score: score_gate(summary, config, stage, score)
            for score in SCORES
        }
        for stage in ("formal", "replication")
    }
    mechanism_gates = config["gates"]["mechanism"]
    gains = {}
    for stage in ("formal", "replication"):
        dense = gates[stage]["dense_all_layer_mechanistic_risk"]["overall"][
            "spearman"
        ]
        primary = gates[stage]["relative_penultimate_path_k1_risk"][
            "overall"
        ]["spearman"]
        action = gates[stage]["action_only_risk"]["overall"]["spearman"]
        stage_sequence = sequence[sequence["stage"] == stage]
        by_score = {
            score: group.sort_values(
                ["model_key", "sample_id"], kind="mergesort"
            )["spearman"].to_numpy(dtype=np.float64)
            for score, group in stage_sequence.groupby("score", sort=True)
        }
        gains[stage] = {
            "dense_over_action": float(dense - action),
            "penultimate_over_action": float(primary - action),
            "dense_over_action_pass": bool(
                dense - action
                > float(
                    mechanism_gates[
                        "dense_over_action_spearman_gain_strict_min"
                    ]
                )
            ),
            "penultimate_over_action_pass": bool(
                primary - action
                > float(
                    mechanism_gates[
                        "penultimate_over_action_spearman_gain_strict_min"
                    ]
                )
            ),
            "dense_over_action_paired_test": exact_paired_sign_flip(
                by_score["dense_all_layer_mechanistic_risk"],
                by_score["action_only_risk"],
            ),
            "penultimate_over_action_paired_test": exact_paired_sign_flip(
                by_score["relative_penultimate_path_k1_risk"],
                by_score["action_only_risk"],
            ),
        }

    primary_pass = all(
        gates[stage]["relative_penultimate_path_k1_risk"]["passed"]
        for stage in ("formal", "replication")
    )
    dense_pass = all(
        gates[stage]["dense_all_layer_mechanistic_risk"]["passed"]
        for stage in ("formal", "replication")
    )
    mechanism_gain_pass = all(
        gains[stage]["dense_over_action_pass"]
        and gains[stage]["penultimate_over_action_pass"]
        for stage in ("formal", "replication")
    )
    if (
        integrity["passed"]
        and isolation["passed"]
        and primary_pass
        and dense_pass
        and mechanism_gain_pass
    ):
        outcome = "G-A"
        interpretation = (
            "cross_family_cross_task_dense_and_relative_penultimate_closure"
        )
    elif integrity["passed"] and isolation["passed"] and dense_pass:
        outcome = "G-B"
        interpretation = (
            "dense_mechanism_generalizes_but_relative_boundary_does_not"
        )
    else:
        outcome = "G-C"
        interpretation = "generalization_not_closed_under_frozen_gates"

    result = {
        "outcome": outcome,
        "interpretation": interpretation,
        "integrity": integrity,
        "role_isolation": isolation,
        "gates": gates,
        "mechanism_gains": gains,
        "calibration_boundary_scan": calibration_boundary_scan(candidates),
        "row_counts": {
            "candidate_rows": int(len(candidates)),
            "unit_rows": int(len(units)),
            "sequence_metric_rows": int(len(sequence)),
            "summary_rows": int(len(summary)),
        },
        "models": {
            str(model_key): {
                "family": str(group["model_family"].iloc[0]),
                "source": str(group["model_source"].iloc[0]),
                "num_layers": int(group["num_layers"].iloc[0]),
                "hidden_size": int(group["hidden_size"].iloc[0]),
                "primary_boundary": int(group["primary_boundary"].iloc[0]),
            }
            for model_key, group in candidates.groupby(
                "model_key", sort=True
            )
        },
        "tasks": sorted(candidates["task"].astype(str).unique()),
        "stages": list(STAGES),
    }
    atomic_json(EXPERIMENT / "results/analysis/analysis_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
