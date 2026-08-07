#!/usr/bin/env python3
"""Sequence-first analysis and mechanical P1 outcome adjudication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
for value in (SCRIPT_DIR, P0_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import ranking_metrics
from p1_core import atomic_frame, atomic_json, sha256_file


SCORE_COLUMNS = {
    "direct": "direct_score",
    "local": "local_score",
    "action_fisher": "action_fisher_score",
    "state_fisher": "state_fisher_score",
    "midpoint_oracle": "midpoint_fisher_oracle",
}
UNIT = ["sample_id", "task", "anchor", "layer", "history_id"]


def make_rankings(
    response: pd.DataFrame, top_k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    reversals: List[Dict[str, Any]] = []
    for key, group in response.groupby(UNIT, sort=False):
        if len(group) != 8 or group["mask_hash"].nunique() != 8:
            raise RuntimeError(f"unit {key} is not eight-distinct")
        common = dict(zip(UNIT, key))
        for score_type, column in SCORE_COLUMNS.items():
            rows.append(
                {
                    **common,
                    "score_type": score_type,
                    "candidate_count": len(group),
                    **ranking_metrics(
                        group[column].to_numpy(dtype=np.float64),
                        group["controlled_exact_kl"].to_numpy(
                            dtype=np.float64
                        ),
                        top_k,
                    ),
                }
            )
        action_index = group["action_fisher_score"].idxmin()
        state_index = group["state_fisher_score"].idxmin()
        exact_min = float(group["controlled_exact_kl"].min())
        scale = max(
            float(group["controlled_exact_kl"].max()) - exact_min,
            1.0e-12,
        )
        reversals.append(
            {
                **common,
                "action_candidate_id": group.loc[
                    action_index, "candidate_id"
                ],
                "state_candidate_id": group.loc[
                    state_index, "candidate_id"
                ],
                "ranking_reversal": action_index != state_index,
                "action_normalized_regret": (
                    float(
                        group.loc[
                            action_index, "controlled_exact_kl"
                        ]
                    )
                    - exact_min
                )
                / scale,
                "state_normalized_regret": (
                    float(
                        group.loc[
                            state_index, "controlled_exact_kl"
                        ]
                    )
                    - exact_min
                )
                / scale,
            }
        )
    reversal = pd.DataFrame(reversals)
    reversal["state_lower_regret"] = (
        reversal["state_normalized_regret"]
        < reversal["action_normalized_regret"]
    )
    return pd.DataFrame(rows), reversal


def sequence_rankings(
    rankings: pd.DataFrame, include_h0: bool
) -> pd.DataFrame:
    source = (
        rankings
        if include_h0
        else rankings[rankings["history_id"] != "H0"]
    )
    metrics = [
        "spearman",
        "pairwise_sign_accuracy",
        "top1_accuracy",
        "topk_overlap",
        "normalized_regret",
        "symmetric_scale_ratio",
    ]
    return (
        source.groupby(
            ["sample_id", "task", "score_type"], as_index=False
        )[metrics]
        .median()
        .sort_values(["task", "sample_id", "score_type"])
    )


def history_sequence_rankings(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "spearman",
        "pairwise_sign_accuracy",
        "top1_accuracy",
        "topk_overlap",
        "normalized_regret",
        "symmetric_scale_ratio",
    ]
    return (
        rankings.groupby(
            ["sample_id", "task", "history_id", "score_type"],
            as_index=False,
        )[metrics]
        .median()
        .sort_values(
            ["task", "sample_id", "history_id", "score_type"]
        )
    )


def state_action_delta(
    sequence: pd.DataFrame,
) -> pd.DataFrame:
    pivot = sequence.pivot(
        index=["sample_id", "task"],
        columns="score_type",
        values="spearman",
    ).reset_index()
    pivot["delta_state_action"] = (
        pivot["state_fisher"] - pivot["action_fisher"]
    )
    return pivot


def gate_outcome(
    protocol: Mapping[str, Any],
    response: pd.DataFrame,
    state: pd.DataFrame,
    identity: pd.DataFrame,
    registry: pd.DataFrame,
    audit: pd.DataFrame,
    sequence_vectors: pd.DataFrame,
    rankings: pd.DataFrame,
    sequence: pd.DataFrame,
    reversals: pd.DataFrame,
    regression: Mapping[str, Any],
    calibration: Mapping[str, Any],
    metadata: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
) -> Dict[str, Any]:
    h0 = response[response["history_id"] == "H0"]
    h0_sequence = (
        history_sequence_rankings(rankings)
        .query("history_id == 'H0' and score_type == 'action_fisher'")
    )
    h0_vectors = sequence_vectors[
        sequence_vectors["history_id"] == "H0"
    ]
    gate0_rule = protocol["gates"]["p0_regression"]
    gate0_checks = {
        "p0_regression_artifact": bool(regression["passed"]),
        "p0_manifest_all_match": bool(
            regression["checks"]["manifest_all_match"]
        ),
        "p0_outcome_A": bool(
            regression["checks"]["p0_outcome_is_A"]
        ),
        "h0_delta_exact_zero": float(h0["state_norm"].max())
        <= float(gate0_rule["h0_delta_x_max_norm"]),
        "h0_cross_zero": float(
            h0["cross_fisher_score"].abs().max()
        )
        <= float(gate0_rule["h0_cross_max_abs"]),
        "h0_state_equals_action": float(
            (
                h0["state_fisher_score"]
                - h0["action_fisher_score"]
            )
            .abs()
            .max()
        )
        <= float(gate0_rule["h0_state_action_score_max_abs"]),
        "h0_action_ranking": float(h0_sequence["spearman"].median())
        >= float(gate0_rule["h0_sequence_first_action_spearman_min"]),
        "h0_combined_cosine": float(h0_vectors["cosine"].median())
        >= float(
            gate0_rule["h0_sequence_first_combined_cosine_min"]
        ),
        "h0_combined_relative_l2": float(
            h0_vectors["relative_l2"].median()
        )
        <= float(
            gate0_rule[
                "h0_sequence_first_combined_relative_l2_max"
            ]
        ),
        "config_hash_unchanged": sha256_file(
            ROOT / "configs/frozen/p1_state_conditioned_config.yaml"
        )
        == str(metadata["config_sha256"]),
    }
    gate0 = all(gate0_checks.values())

    finite_columns = [
        column
        for column in response.columns
        if column.endswith("_finite")
    ]
    state_unit_sizes = state.groupby(UNIT).size()
    state_key_sizes = state.groupby(UNIT)["state_hash"].nunique()
    candidate_shared = response.groupby(UNIT)["state_hash"].nunique()
    candidate_sets = (
        response.groupby(UNIT)["mask_hash"]
        .apply(lambda values: tuple(sorted(set(values))))
        .reset_index()
    )
    set_equal = (
        candidate_sets.groupby(
            ["sample_id", "task", "anchor", "layer"]
        )["mask_hash"]
        .nunique()
        .eq(1)
        .all()
    )
    state_pivot = state.pivot(
        index=["sample_id", "task", "anchor", "layer"],
        columns="history_id",
        values="state_hash",
    )
    gate1_rule = protocol["gates"]["history_validity"]
    gate1_checks = {
        "response_finite": bool(
            response[finite_columns].all().all()
        ),
        "identity_finite": bool(identity["finite"].all()),
        "state_finite": bool(
            state["state_finite"].all()
            and state["state_jvp_vs_manual_finite"].all()
            and np.isfinite(state["physical_history_kl"]).all()
        ),
        "stale_state_nonzero": float(
            state.loc[
                state["history_id"] != "H0", "state_norm"
            ].min()
        )
        > float(gate1_rule["nonzero_state_norm_strict_min"]),
        "one_unique_state_per_unit": bool(
            state_unit_sizes.eq(1).all()
            and state_key_sizes.eq(1).all()
        ),
        "candidate_shared_state_hash": bool(
            candidate_shared.eq(1).all()
            and audit["candidate_shared_state_hash"].all()
        ),
        "candidate_sets_equal_across_histories": bool(set_equal),
        "h1_h2_differ": bool((state_pivot["H1"] != state_pivot["H2"]).all()),
        "h2_h3_differ": bool((state_pivot["H2"] != state_pivot["H3"]).all()),
        "split_isolation": bool(
            all(metadata["split_audit"]["checks"].values())
        ),
        "calibration_pass": bool(calibration["passed"]),
        "candidate_count_and_budget": bool(
            registry.groupby(["sample_id", "anchor"])
            .size()
            .eq(8)
            .all()
            and registry["active_budget"]
            .eq(int(protocol["cache"]["total_budget"]))
            .all()
        ),
        "cache_and_boundary_invariants": bool(
            audit["cache_fingerprint_invariant"].all()
            and audit["anchor_cache_all_fp32"].all()
            and audit["anchor_cache_shapes_valid"].all()
            and audit["boundary_map_baseline_relative_l2"]
            .eq(0.0)
            .all()
            and audit["repeat_max_absolute_error"].eq(0.0).all()
        ),
    }
    gate1 = all(gate1_checks.values())

    all_vectors = sequence_vectors[
        sequence_vectors["history_id"] == "all"
    ]
    gate2_rule = protocol["gates"]["combined_readout"]
    gate2_metrics = {
        "overall_sequence_first_cosine": float(
            all_vectors["cosine"].median()
        ),
        "overall_sequence_first_relative_l2": float(
            all_vectors["relative_l2"].median()
        ),
        "each_task_median_cosine": (
            all_vectors.groupby("task")["cosine"].median().to_dict()
        ),
        "row_pass_fraction": float(
            response["combined_jvp_vs_manual_cosine"]
            .ge(float(gate2_rule["row_cosine_threshold"]))
            .mean()
        ),
    }
    gate2_checks = {
        "overall_sequence_first_cosine": gate2_metrics[
            "overall_sequence_first_cosine"
        ]
        >= float(gate2_rule["overall_sequence_first_cosine_min"]),
        "each_task_median_cosine": all(
            float(value)
            >= float(gate2_rule["each_task_median_cosine_min"])
            for value in gate2_metrics[
                "each_task_median_cosine"
            ].values()
        ),
        "overall_sequence_first_relative_l2": gate2_metrics[
            "overall_sequence_first_relative_l2"
        ]
        <= float(
            gate2_rule["overall_sequence_first_relative_l2_max"]
        ),
        "row_pass_fraction": gate2_metrics["row_pass_fraction"]
        >= float(gate2_rule["row_pass_fraction_min"]),
    }
    gate2 = all(gate2_checks.values())

    delta = state_action_delta(sequence)
    main_reversals = reversals[reversals["history_id"] != "H0"]
    score_medians = sequence.groupby("score_type")[
        [
            "pairwise_sign_accuracy",
            "top1_accuracy",
            "normalized_regret",
        ]
    ].median()
    pair_gain = float(
        score_medians.loc[
            "state_fisher", "pairwise_sign_accuracy"
        ]
        - score_medians.loc[
            "action_fisher", "pairwise_sign_accuracy"
        ]
    )
    top1_gain = float(
        score_medians.loc["state_fisher", "top1_accuracy"]
        - score_medians.loc["action_fisher", "top1_accuracy"]
    )
    regret_gain = float(
        score_medians.loc[
            "action_fisher", "normalized_regret"
        ]
        - score_medians.loc[
            "state_fisher", "normalized_regret"
        ]
    )
    reversal_only = main_reversals[
        main_reversals["ranking_reversal"]
    ]
    gate3_rule = protocol["gates"]["decision_gain"]
    task_delta = (
        delta.groupby("task")["delta_state_action"].median().to_dict()
    )
    gate3_metrics = {
        "overall_median_delta_spearman": float(
            delta["delta_state_action"].median()
        ),
        "task_median_delta_spearman": task_delta,
        "positive_sequence_fraction": float(
            delta["delta_state_action"].gt(0.0).mean()
        ),
        "pairwise_accuracy_gain": pair_gain,
        "top1_accuracy_gain": top1_gain,
        "normalized_regret_gain": regret_gain,
        "top1_reversal_rate": float(
            main_reversals["ranking_reversal"].mean()
        ),
        "reversal_state_lower_regret_fraction": float(
            reversal_only["state_lower_regret"].mean()
        ),
        "reversal_unit_count": len(reversal_only),
    }
    secondary_limit = float(
        gate3_rule["secondary_metric_max_degradation"]
    )
    gate3_checks = {
        "overall_delta_spearman": gate3_metrics[
            "overall_median_delta_spearman"
        ]
        >= float(
            gate3_rule["overall_median_delta_spearman_min"]
        ),
        "each_task_delta_positive": all(
            float(value)
            > float(
                gate3_rule[
                    "each_task_median_delta_spearman_strict_min"
                ]
            )
            for value in task_delta.values()
        ),
        "positive_sequence_fraction": gate3_metrics[
            "positive_sequence_fraction"
        ]
        >= float(gate3_rule["positive_sequence_fraction_min"]),
        "pairwise_accuracy_gain": pair_gain
        >= float(gate3_rule["pairwise_accuracy_gain_min"]),
        "top1_or_regret_improves": (
            (top1_gain > 0.0 and regret_gain >= -secondary_limit)
            or (
                regret_gain > 0.0
                and top1_gain >= -secondary_limit
            )
        ),
        "ranking_reversal_nonzero": gate3_metrics[
            "top1_reversal_rate"
        ]
        > float(gate3_rule["top1_reversal_rate_strict_min"]),
        "reversal_more_often_better": gate3_metrics[
            "reversal_state_lower_regret_fraction"
        ]
        > float(
            gate3_rule[
                "reversal_state_lower_regret_fraction_strict_min"
            ]
        ),
        "state_total_ranking_identity": bool(
            all(
                np.array_equal(
                    np.argsort(
                        group["state_fisher_score"].to_numpy()
                    ),
                    np.argsort(
                        group["total_fisher_score"].to_numpy()
                    ),
                )
                for _key, group in response.groupby(UNIT)
            )
        ),
    }
    gate3 = all(gate3_checks.values())

    diagnostic_pass = bool(
        diagnostic.get("passed_same_readout_thresholds", False)
    )
    if not gate0 or not gate1:
        outcome = "N"
    elif gate2:
        outcome = "A" if gate3 else "B"
    elif diagnostic_pass:
        outcome = "C"
    else:
        outcome = "D"
    return {
        "outcome": outcome,
        "outcome_definition": protocol["outcomes"][outcome],
        "gate0_p0_regression": {
            "passed": gate0,
            "checks": gate0_checks,
            "metrics": {
                "h0_action_sequence_median_spearman": float(
                    h0_sequence["spearman"].median()
                ),
                "h0_combined_sequence_median_cosine": float(
                    h0_vectors["cosine"].median()
                ),
                "h0_combined_sequence_median_relative_l2": float(
                    h0_vectors["relative_l2"].median()
                ),
            },
        },
        "gate1_history_validity": {
            "passed": gate1,
            "checks": gate1_checks,
            "metrics": {
                "minimum_stale_state_norm": float(
                    state.loc[
                        state["history_id"] != "H0", "state_norm"
                    ].min()
                ),
                "maximum_state_norm": float(state["state_norm"].max()),
            },
        },
        "gate2_combined_readout": {
            "passed": gate2,
            "checks": gate2_checks,
            "metrics": gate2_metrics,
        },
        "gate3_decision_gain": {
            "passed": gate3,
            "checks": gate3_checks,
            "metrics": gate3_metrics,
        },
        "state_operating_point_diagnostic": {
            "passed": diagnostic_pass,
            "metrics": diagnostic.get("metrics", {}),
        },
    }


def severity_table(
    response: pd.DataFrame,
    rankings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    main = response[response["history_id"] != "H0"]
    state_unit = (
        main.groupby(UNIT, as_index=False)
        .agg(
            state_norm=("state_norm", "first"),
            state_fisher_energy=("state_fisher_energy", "first"),
            physical_history_kl=("physical_history_kl", "first"),
            median_combined_cosine=(
                "combined_jvp_vs_manual_cosine",
                "median",
            ),
            median_combined_relative_l2=(
                "combined_jvp_vs_manual_relative_l2",
                "median",
            ),
            median_action_norm=("action_r_norm", "median"),
        )
    )
    pivot = rankings[rankings["history_id"] != "H0"].pivot(
        index=UNIT,
        columns="score_type",
        values="spearman",
    ).reset_index()
    pivot["delta_state_action"] = (
        pivot["state_fisher"] - pivot["action_fisher"]
    )
    unit = state_unit.merge(pivot, on=UNIT, validate="one_to_one")
    rows = []
    for outcome in (
        "median_combined_cosine",
        "median_combined_relative_l2",
        "delta_state_action",
    ):
        for predictor in (
            "state_norm",
            "state_fisher_energy",
            "physical_history_kl",
            "median_action_norm",
            "layer",
            "history_length",
        ):
            if predictor == "history_length":
                values = unit["history_id"].map(
                    {"H1": 8, "H2": 32, "H3": 32}
                )
            else:
                values = unit[predictor]
            correlation, pvalue = spearmanr(values, unit[outcome])
            rows.append(
                {
                    "outcome_metric": outcome,
                    "predictor": predictor,
                    "spearman": float(correlation),
                    "pvalue": float(pvalue),
                    "unit_count": len(unit),
                }
            )
    return unit, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/frozen/p1_state_conditioned_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/p1_state_conditioned/results",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protocol = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    response = pd.read_parquet(output / "response_rows.parquet")
    state = pd.read_parquet(output / "state_registry.parquet")
    identity = pd.read_parquet(output / "identity_rows.parquet")
    registry = pd.read_parquet(output / "candidate_registry.parquet")
    audit = pd.read_parquet(output / "unit_audit.parquet")
    vectors = pd.read_parquet(output / "sequence_vector_metrics.parquet")
    regression = json.loads(
        (output / "p0_regression_summary.json").read_text()
    )
    calibration = json.loads(
        (output / "calibration_summary.json").read_text()
    )
    metadata = json.loads(
        (output / "evaluation_metadata.json").read_text()
    )
    diagnostic = json.loads(
        (output / "state_operating_point_summary.json").read_text()
    )
    rankings, reversals = make_rankings(
        response, int(protocol["metrics"]["top_k"])
    )
    sequence = sequence_rankings(rankings, include_h0=False)
    history_sequence = history_sequence_rankings(rankings)
    delta = state_action_delta(sequence)
    unit_severity, severity = severity_table(response, rankings)
    gate = gate_outcome(
        protocol,
        response,
        state,
        identity,
        registry,
        audit,
        vectors,
        rankings,
        sequence,
        reversals,
        regression,
        calibration,
        metadata,
        diagnostic,
    )
    artifacts = {
        "ranking_rows": rankings,
        "reversal_rows": reversals,
        "sequence_first_ranking": sequence,
        "history_sequence_first_ranking": history_sequence,
        "sequence_state_action_delta": delta,
        "unit_severity": unit_severity,
        "severity_correlations": severity,
    }
    for name, frame in artifacts.items():
        atomic_frame(output / f"{name}.parquet", frame)
        if name in {
            "sequence_first_ranking",
            "sequence_state_action_delta",
            "severity_correlations",
        }:
            frame.to_csv(output / f"{name}.csv", index=False)
    summary = {
        "outcome": gate["outcome"],
        "row_counts": {
            "response": len(response),
            "state": len(state),
            "ranking": len(rankings),
            "sequence_ranking": len(sequence),
            "reversal_units": len(reversals),
        },
        "gates": gate,
        "score_sequence_medians": (
            sequence.groupby("score_type")[
                [
                    "spearman",
                    "pairwise_sign_accuracy",
                    "top1_accuracy",
                    "topk_overlap",
                    "normalized_regret",
                ]
            ]
            .median()
            .reset_index()
            .to_dict("records")
        ),
        "history_score_medians": (
            history_sequence.groupby(["history_id", "score_type"])[
                [
                    "spearman",
                    "pairwise_sign_accuracy",
                    "top1_accuracy",
                    "topk_overlap",
                    "normalized_regret",
                ]
            ]
            .median()
            .reset_index()
            .to_dict("records")
        ),
    }
    atomic_json(output / "p1_gate_outcome.json", gate)
    atomic_json(output / "p1_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
