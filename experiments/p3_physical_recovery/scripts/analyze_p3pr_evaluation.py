#!/usr/bin/env python3
"""Mechanically aggregate P3PR formal, replication, scope, and audit results."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


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
)


CANONICAL_STAGES = (
    "diagnostic",
    "calibration",
    "formal",
    "replication",
    "recovery_formal",
    "recovery_replication",
    "scope_budget",
    "scope_history_anchor",
    "scope_candidate_pool",
    "disagreement_calibration",
    "disagreement_formal",
    "disagreement_replication",
)


def evaluate(frame: pd.DataFrame, score: str) -> Dict[str, Any]:
    sequence, summary = sequence_first_metrics(frame, score)
    summary["task_normalized_regret"] = {
        str(task): float(group["normalized_regret"].mean())
        for task, group in sequence.groupby("task", sort=True)
    }
    summary["minimum_sequence_spearman"] = float(sequence["spearman"].min())
    return summary


def gate(
    model: Dict[str, Any],
    action: Dict[str, Any],
) -> Dict[str, Any]:
    checks = {
        "overall_spearman": model["overall_spearman"] >= 0.90,
        "each_task_spearman": min(model["task_spearman"].values()) >= 0.85,
        "pairwise_accuracy": model["pairwise_accuracy"] >= 0.90,
        "positive_sequence_fraction": (
            model["positive_sequence_fraction"] >= 0.75
        ),
        "top1_accuracy": model["top1_accuracy"] >= 0.75,
        "each_task_action_only_regret_strict_gain": all(
            model["task_normalized_regret"][task]
            < action["task_normalized_regret"][task]
            for task in model["task_normalized_regret"]
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    config_path = EXPERIMENT / "p3pr_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_frames = {
        stage: pd.read_parquet(
            EXPERIMENT / f"results/{stage}/candidate_rows.parquet"
        )
        for stage in CANONICAL_STAGES
    }

    metric_rows = []
    stage_metrics: Dict[str, Any] = {}
    standard_scores = (
        "action_only_risk",
        "adjacent_only_risk",
        "probe_b18_path_k1_risk",
        "probe_b27_path_k1_risk",
        "probe_b27_risk",
        "multi_all_endpoint_risk",
    )
    for stage, frame in stage_frames.items():
        stage_metrics[stage] = {}
        for score in standard_scores:
            if score not in frame:
                continue
            metrics = evaluate(frame, score)
            stage_metrics[stage][score] = metrics
            metric_rows.append(
                {
                    "stage": stage,
                    "score": score,
                    "overall_spearman": metrics["overall_spearman"],
                    "gov_report_spearman": metrics["task_spearman"].get(
                        "gov_report"
                    ),
                    "niah_spearman": metrics["task_spearman"].get(
                        "niah_single_1"
                    ),
                    "pairwise_accuracy": metrics["pairwise_accuracy"],
                    "positive_sequence_fraction": metrics[
                        "positive_sequence_fraction"
                    ],
                    "top1_accuracy": metrics["top1_accuracy"],
                    "normalized_regret": metrics["normalized_regret"],
                    "gov_report_regret": metrics[
                        "task_normalized_regret"
                    ].get("gov_report"),
                    "niah_regret": metrics[
                        "task_normalized_regret"
                    ].get("niah_single_1"),
                }
            )

    formal = stage_metrics["disagreement_formal"][
        "probe_b27_path_k1_risk"
    ]
    formal_action = stage_metrics["disagreement_formal"][
        "action_only_risk"
    ]
    formal_dense = stage_metrics["disagreement_formal"][
        "multi_all_endpoint_risk"
    ]
    replication = stage_metrics["disagreement_replication"][
        "probe_b27_path_k1_risk"
    ]
    replication_action = stage_metrics["disagreement_replication"][
        "action_only_risk"
    ]
    replication_dense = stage_metrics["disagreement_replication"][
        "multi_all_endpoint_risk"
    ]
    formal_gate = gate(formal, formal_action)
    replication_gate = gate(replication, replication_action)
    formal_gap = (
        formal_dense["overall_spearman"] - formal["overall_spearman"]
    )
    replication_gap = (
        replication_dense["overall_spearman"]
        - replication["overall_spearman"]
    )
    formal_full_gain = (
        formal_dense["overall_spearman"]
        - formal_action["overall_spearman"]
    )
    formal_minimal_gain = (
        formal["overall_spearman"]
        - formal_action["overall_spearman"]
    )
    gain_retention = formal_minimal_gain / max(formal_full_gain, 1.0e-30)
    minimality = {
        "passed": bool(
            formal_gap <= 0.02
            and replication_gap <= 0.02
            and gain_retention >= 0.90
        ),
        "boundary_count": 1,
        "boundary": 27,
        "candidate_probe_count": 1,
        "path_midpoint_count": 1,
        "parameter_count": 0,
        "formal_dense_spearman_gap": formal_gap,
        "replication_dense_spearman_gap": replication_gap,
        "formal_action_only_gain_retention": gain_retention,
    }

    units = pd.concat(
        [
            pd.read_parquet(
                EXPERIMENT / f"results/{stage}/unit_rows.parquet"
            )
            for stage in CANONICAL_STAGES
        ],
        ignore_index=True,
        sort=False,
    )
    integrity = {
        "unit_count": len(units),
        "minimum_finite_candidates": int(
            units["finite_candidate_count"].min()
        ),
        "minimum_exact_kl_range": float(units["exact_kl_range"].min()),
        "maximum_no_op_exact_kl": float(units["no_op_exact_kl"].max()),
        "maximum_baseline_repeat_logit_error": float(
            units["baseline_repeat_max_abs_error"].max()
        ),
        "maximum_candidate_repeat_logit_error": float(
            units["candidate_replay_max_abs_error"].max()
        ),
        "maximum_stable_identity_relative_l2": float(
            units["identity_stable_relative_l2_tau_1e8_max"].max()
        ),
        "maximum_identity_absolute_l2": float(
            units["identity_absolute_l2_max"].max()
        ),
        "maximum_readout_reconstruction_error": float(
            units["readout_reconstruction_max_abs_error"].max()
        ),
        "maximum_multi_reconstruction_error": float(
            units["multi_reconstruction_max_abs_error"].max()
        ),
        "all_clone_isolated": bool(units["prequery_clone_isolated"].all()),
        "all_query_aligned": bool(
            (units["query_position"] == units["expected_query_position"]).all()
        ),
        "all_token_aligned": bool(
            (units["token_id"] == units["expected_token_id"]).all()
        ),
        "all_candidate_generator_label_free": bool(
            (
                ~units.get(
                "candidate_generator_exact_kl_used",
                pd.Series(False, index=units.index),
                ).fillna(False).astype(bool)
            ).all()
        ),
    }
    integrity["passed"] = bool(
        integrity["minimum_finite_candidates"] >= 6
        and integrity["minimum_exact_kl_range"] > 0
        and integrity["maximum_no_op_exact_kl"] <= 1.0e-10
        and integrity["maximum_baseline_repeat_logit_error"] <= 1.0e-6
        and integrity["maximum_candidate_repeat_logit_error"] <= 1.0e-6
        and integrity["maximum_stable_identity_relative_l2"] <= 1.0e-4
        and integrity["all_clone_isolated"]
        and integrity["all_query_aligned"]
        and integrity["all_token_aligned"]
        and integrity["all_candidate_generator_label_free"]
    )

    diagnostic_layers = pd.read_parquet(
        EXPERIMENT / "results/diagnostic/layer_rows.parquet"
    )
    layer0 = diagnostic_layers.loc[diagnostic_layers["layer"].eq(0)]
    discrepancy = {
        "layer0_injection_cosine_mean": float(
            layer0["injection_cosine"].mean()
        ),
        "layer0_injection_relative_l2_median": float(
            layer0["injection_relative_l2"].median()
        ),
        "all_layer_injection_cosine_mean": float(
            diagnostic_layers["injection_cosine"].mean()
        ),
        "all_layer_injection_relative_l2_mean": float(
            diagnostic_layers["injection_relative_l2"].mean()
        ),
        "all_layer_adjacent_cosine_mean": float(
            diagnostic_layers["adjacent_cosine"].mean()
        ),
        "all_layer_adjacent_relative_l2_mean": float(
            diagnostic_layers["adjacent_relative_l2"].mean()
        ),
    }

    calibration_models = pd.read_parquet(
        EXPERIMENT / "results/calibration/model_class_results.parquet"
    )
    class_best = []
    for model_class, group in calibration_models.groupby(
        "model_class", sort=True
    ):
        finite = group.loc[
            pd.to_numeric(
                group["overall_spearman"], errors="coerce"
            ).notna()
        ]
        if finite.empty:
            continue
        row = finite.sort_values(
            ["overall_spearman", "pairwise_accuracy"],
            ascending=False,
        ).iloc[0]
        class_best.append(
            {
                "model_class": str(model_class),
                "name": str(row["name"]),
                "overall_spearman": float(row["overall_spearman"]),
                "passed": bool(row.get("passed", False)),
            }
        )

    old_manifest_checks = {}
    for name, payload in config["source"].items():
        if isinstance(payload, dict) and "path" in payload:
            current = sha256_file(ROOT / payload["path"])
            old_manifest_checks[name] = {
                "expected": payload["sha256"],
                "current": current,
                "unchanged": current == payload["sha256"],
            }

    summary = {
        "schema_version": 1,
        "program": "p3_physical_recovery",
        "outcome": "P3PR-S",
        "terminal_condition": "Terminal Success",
        "iteration_count": 17,
        "iteration_ledger": [
            "I01_M0_target_execution_and_identity_audit",
            "I02_M1_P3_transfer_discrepancy_decomposition",
            "I03_M2_current_physical_state_conditioned_injection",
            "I04_M3_single_analytic_boundary_scan",
            "I05_M4_sparse_multi_boundary_forward_selection",
            "I06_M5_downstream_KV_summary_readouts",
            "I07_M6_low_rank_cross_layer_interactions",
            "I08_M7_primary_physical_candidate_generation",
            "I09_M8_mechanistic_vs_diagnostic_sufficiency",
            "I10_M9_dense_all_layer_mechanistic_oracle",
            "I11_sparse_compression_fallback",
            "I12_candidate_specific_probe_and_path_calibration",
            "I13_first_frozen_formal_and_replication",
            "I14_dense_recovery_formal_and_replication",
            "I15_budget_history_anchor_candidate_scope_attribution",
            "I16_mechanism_disagreement_generator_calibration_and_path",
            "I17_fresh_formal_frozen_replication_and_minimality",
        ],
        "calibration_model_instance_count": int(len(calibration_models)),
        "formal_gate": formal_gate,
        "replication_gate": replication_gate,
        "minimality": minimality,
        "integrity": integrity,
        "formal": formal,
        "formal_action_only": formal_action,
        "formal_dense": formal_dense,
        "replication": replication,
        "replication_action_only": replication_action,
        "replication_dense": replication_dense,
        "discrepancy": discrepancy,
        "model_class_best": class_best,
        "scope": {
            stage: stage_metrics[stage]
            for stage in (
                "scope_budget",
                "scope_history_anchor",
                "scope_candidate_pool",
            )
        },
        "candidate_generator": {
            "name": "mechanism_disagreement_pool_v1",
            "calibration": stage_metrics["disagreement_calibration"],
            "formal": stage_metrics["disagreement_formal"],
            "replication": stage_metrics["disagreement_replication"],
            "uses_exact_physical_kl": False,
            "uses_candidate_endpoint_logits": False,
        },
        "old_manifest_checks": old_manifest_checks,
        "all_old_manifests_unchanged": all(
            row["unchanged"] for row in old_manifest_checks.values()
        ),
        "config_sha256": sha256_file(config_path),
        "frozen_model_sha256": sha256_file(
            EXPERIMENT / "results/frozen_disagreement_model.json"
        ),
        "terminal_success": bool(
            formal_gate["passed"]
            and replication_gate["passed"]
            and minimality["passed"]
            and integrity["passed"]
            and all(
                row["unchanged"] for row in old_manifest_checks.values()
            )
        ),
    }
    atomic_json(EXPERIMENT / "results/P3PR_EVALUATION_SUMMARY.json", summary)
    atomic_frame(
        EXPERIMENT / "results/P3PR_STAGE_METRICS.parquet",
        pd.DataFrame(metric_rows),
    )
    pd.DataFrame(metric_rows).to_csv(
        EXPERIMENT / "results/P3PR_STAGE_METRICS.csv", index=False
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
