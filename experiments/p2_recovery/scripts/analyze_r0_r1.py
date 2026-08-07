#!/usr/bin/env python3
"""Analyze retrospective R0 failure map and R1 scaling mechanism."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
for value in (SCRIPT_DIR, P2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p2_core import atomic_frame, atomic_json, sha256_file  # noqa: E402
from recovery_core import log_log_slope, sequence_gate  # noqa: E402


UNIT = ["sample_id", "task", "anchor", "layer", "history_id"]
PRIMARY = ["H1", "H2", "H3"]


def breakdown(rows: pd.DataFrame) -> pd.DataFrame:
    specs: List[tuple[str, Sequence[str]]] = [
        ("overall", []),
        ("task", ["task"]),
        ("sequence", ["sample_id", "task"]),
        ("layer", ["layer"]),
        ("anchor", ["anchor"]),
        ("history", ["history_id"]),
        ("candidate", ["candidate_source"]),
        ("task_layer", ["task", "layer"]),
        ("task_candidate", ["task", "candidate_source"]),
        ("interaction_sign", ["state_action_interaction_sign"]),
    ]
    output = []
    for label, columns in specs:
        groups = (
            [((), rows)]
            if not columns
            else list(
                rows.groupby(
                    columns[0]
                    if len(columns) == 1
                    else list(columns),
                    sort=True,
                )
            )
        )
        for key, group in groups:
            keys = key if isinstance(key, tuple) else (key,)
            values = dict(zip(columns, keys))
            output.append(
                {
                    "stratum_type": label,
                    "stratum_json": json.dumps(
                        values, sort_keys=True, default=str
                    ),
                    **values,
                    "row_count": len(group),
                    "median_cosine": float(group["cosine"].median()),
                    "median_relative_l2": float(
                        group["relative_l2"].median()
                    ),
                    "median_symmetric_norm_ratio": float(
                        group["symmetric_norm_ratio"].median()
                    ),
                    "median_parallel_relative_error": float(
                        group["parallel_relative_error"].median()
                    ),
                    "median_orthogonal_relative_error": float(
                        group["orthogonal_relative_error"].median()
                    ),
                    "median_fisher_relative_error": float(
                        group["fisher_relative_error"].median()
                    ),
                    "row_cosine_pass_fraction": float(
                        group["cosine"].ge(0.99).mean()
                    ),
                    "direction_error_fraction": float(
                        group["failure_type"]
                        .eq("direction")
                        .mean()
                    ),
                    "amplitude_error_fraction": float(
                        group["failure_type"]
                        .eq("amplitude")
                        .mean()
                    ),
                }
            )
    return pd.DataFrame(output)


def correlations(
    rows: pd.DataFrame,
    targets: Sequence[str],
) -> pd.DataFrame:
    predictors = [
        "action_r_norm",
        "state_norm",
        "action_to_state_workpoint_ratio",
        "controlled_exact_kl",
        "retained_mass_median",
        "nonlinear_increment_norm",
        "jacobian_drift_relative_l2",
        "state_action_interaction",
    ]
    output = []
    for predictor in predictors:
        for target in targets:
            clean = rows[[predictor, target]].dropna()
            output.append(
                {
                    "predictor": predictor,
                    "target": target,
                    "spearman": float(
                        spearmanr(
                            clean[predictor], clean[target]
                        ).statistic
                    ),
                    "row_count": len(clean),
                }
            )
    return pd.DataFrame(output)


def ranking_source(
    r0: pd.DataFrame,
) -> pd.DataFrame:
    p2_sequence = pd.read_parquet(
        ROOT
        / "experiments/p2_state_local_risk/results/"
        "sequence_first_summary.parquet"
    )
    p2_geometry = pd.read_parquet(
        ROOT
        / "experiments/p2_state_local_risk/results/"
        "geometry_score_rows.parquet"
    )
    rows = []
    for key, group in r0[
        r0["history_id"].isin(PRIMARY)
    ].groupby(UNIT, sort=False):
        common = dict(zip(UNIT, key))
        full = p2_geometry[
            (p2_geometry["score_type"] == "full_state_local")
        ]
        for column, value in common.items():
            full = full[full[column] == value]
        if len(full) != 8:
            raise RuntimeError("P2 full score unit mismatch")
        merged = group.merge(
            full[
                ["candidate_id", "score", "controlled_exact_kl"]
            ],
            on="candidate_id",
            suffixes=("", "_p2"),
            validate="one_to_one",
        )
        exact = merged["controlled_exact_kl"].to_numpy()
        rows.append(
            {
                **common,
                "exact_kl_unique_count": int(
                    merged["controlled_exact_kl"].nunique()
                ),
                "full_score_unique_count": int(
                    merged["score"].nunique()
                ),
                "action_norm_exact_kl_spearman": float(
                    spearmanr(
                        merged["action_r_norm"], exact
                    ).statistic
                ),
                "truth_norm_exact_kl_spearman": float(
                    spearmanr(
                        merged["nonlinear_increment_norm"], exact
                    ).statistic
                ),
                "projection_exact_kl_spearman": float(
                    spearmanr(
                        merged["projection_coefficient"], exact
                    ).statistic
                ),
                "full_score_exact_kl_spearman": float(
                    spearmanr(merged["score"], exact).statistic
                ),
            }
        )
    output = pd.DataFrame(rows)
    sequence_full = p2_sequence[
        p2_sequence["score_type"] == "full_state_local"
    ][["sample_id", "task", "spearman"]].copy()
    atomic_frame(
        ROOT
        / "experiments/p2_recovery/r0_failure_map/results/"
        "p2_full_sequence_spearman.parquet",
        sequence_full,
    )
    return output


def analyze_r0(r0_config: Dict[str, Any]) -> Dict[str, Any]:
    output = (
        ROOT
        / "experiments/p2_recovery/r0_failure_map/results"
    )
    rows = pd.read_parquet(output / "r0_rows.parquet")
    primary = rows[rows["history_id"].isin(PRIMARY)].copy()
    rules = r0_config["classification"]
    primary["failure_type"] = np.select(
        [
            primary["cosine"]
            < float(rules["direction_error_cosine_below"]),
            (
                primary["cosine"]
                >= float(rules["amplitude_error_cosine_min"])
            )
            & (
                primary["relative_l2"]
                > float(
                    rules["amplitude_error_relative_l2_above"]
                )
            ),
        ],
        ["direction", "amplitude"],
        default="passes_or_other",
    )
    action_threshold = float(
        primary["action_r_norm"].quantile(
            float(rules["large_action_quantile"])
        )
    )
    primary["large_action"] = (
        primary["action_r_norm"] >= action_threshold
    )
    break_frame = breakdown(primary)
    corr = correlations(
        primary,
        [
            "relative_l2",
            "orthogonal_relative_error",
            "fisher_relative_error",
            "cosine",
        ],
    )
    rank_source = ranking_source(rows)
    atomic_frame(output / "r0_classified_rows.parquet", primary)
    atomic_frame(output / "r0_breakdown.parquet", break_frame)
    break_frame.to_csv(output / "r0_breakdown.csv", index=False)
    atomic_frame(output / "r0_correlations.parquet", corr)
    corr.to_csv(output / "r0_correlations.csv", index=False)
    atomic_frame(output / "r0_ranking_source.parquet", rank_source)
    worst = primary.sort_values(
        ["cosine", "relative_l2"], ascending=[True, False]
    ).head(30)
    worst.to_csv(output / "r0_worst_rows.csv", index=False)
    full_sequence = pd.read_parquet(
        output / "p2_full_sequence_spearman.parquet"
    )
    summary = {
        "iteration": "r0_failure_map",
        "claim_scope": "retrospective_only",
        "row_count": len(primary),
        "direction_error_fraction": float(
            primary["failure_type"].eq("direction").mean()
        ),
        "amplitude_error_fraction": float(
            primary["failure_type"].eq("amplitude").mean()
        ),
        "pass_or_other_fraction": float(
            primary["failure_type"]
            .eq("passes_or_other")
            .mean()
        ),
        "large_action_threshold": action_threshold,
        "large_action_direction_error_fraction": float(
            primary.loc[
                primary["large_action"], "failure_type"
            ]
            .eq("direction")
            .mean()
        ),
        "non_large_action_direction_error_fraction": float(
            primary.loc[
                ~primary["large_action"], "failure_type"
            ]
            .eq("direction")
            .mean()
        ),
        "p2_full_sequence_spearman": full_sequence.to_dict(
            "records"
        ),
        "p2_full_all_sequences_equal_one": bool(
            (full_sequence["spearman"] == 1.0).all()
        ),
        "units_with_exact_kl_ties": int(
            (rank_source["exact_kl_unique_count"] < 8).sum()
        ),
        "units_with_full_score_ties": int(
            (rank_source["full_score_unique_count"] < 8).sum()
        ),
        "median_action_norm_exact_kl_spearman": float(
            rank_source[
                "action_norm_exact_kl_spearman"
            ].median()
        ),
        "median_truth_norm_exact_kl_spearman": float(
            rank_source[
                "truth_norm_exact_kl_spearman"
            ].median()
        ),
        "median_full_score_exact_kl_spearman": float(
            rank_source[
                "full_score_exact_kl_spearman"
            ].median()
        ),
        "config_sha256": sha256_file(
            ROOT
            / "experiments/p2_recovery/r0_failure_map/"
            "r0_config.yaml"
        ),
    }
    atomic_json(output / "r0_summary.json", summary)
    return summary


def analyze_r1(r1_config: Dict[str, Any]) -> Dict[str, Any]:
    output = (
        ROOT
        / "experiments/p2_recovery/"
        "r1_amplitude_trust_region/results"
    )
    rows = pd.read_parquet(output / "scaling_rows.parquet")
    primary = rows[rows["history_id"].isin(PRIMARY)].copy()
    sequence = pd.read_parquet(
        output / "sequence_scaling_metrics.parquet"
    )
    gate_rows = []
    gate_results: Dict[str, Any] = {}
    for gamma in sorted(primary["gamma"].unique()):
        gate = sequence_gate(
            sequence[sequence["gamma"] == gamma],
            primary[primary["gamma"] == gamma],
            r1_config["gates"],
        )
        gate_results[str(float(gamma))] = gate
        gate_rows.append(
            {
                "gamma": float(gamma),
                "passed": gate["passed"],
                **gate["metrics"],
                **{
                    f"check_{key}": value
                    for key, value in gate["checks"].items()
                },
            }
        )
    gate_frame = pd.DataFrame(gate_rows)
    slope_rows = []
    direction = [
        "sample_id",
        "task",
        "anchor",
        "layer",
        "history_id",
        "candidate_id",
        "candidate_source",
    ]
    for key, group in primary.groupby(direction, sort=False):
        slope_rows.append(
            {
                **dict(zip(direction, key)),
                "residual_log_log_slope": log_log_slope(
                    group["gamma"], group["residual_norm"]
                ),
                "relative_error_log_log_slope": log_log_slope(
                    group["gamma"], group["relative_l2"]
                ),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    smallest = min(float(value) for value in primary["gamma"])
    smallest_gate = gate_results[str(smallest)]
    full_gate = gate_results["1.0"]
    failures = primary[
        (primary["gamma"] == smallest)
        & (primary["cosine"] < r1_config["gates"]["row_cosine_min"])
    ].copy()
    if len(failures):
        concentration = (
            failures.groupby(
                ["task", "layer", "candidate_source"]
            )
            .size()
            .sort_values(ascending=False)
        )
        top_groups = int(
            r1_config["branch"][
                "r1_c_max_task_layer_candidate_groups"
            ]
        )
        concentration_fraction = float(
            concentration.head(top_groups).sum() / len(failures)
        )
    else:
        concentration_fraction = 0.0
    r1_c = bool(
        not smallest_gate["passed"]
        and smallest_gate["metrics"]["row_pass_fraction"]
        >= float(r1_config["branch"]["r1_c_row_pass_floor"])
        and concentration_fraction
        >= float(
            r1_config["branch"][
                "r1_c_failure_concentration_min"
            ]
        )
    )
    if smallest_gate["passed"] and not full_gate["passed"]:
        branch = "R1-A"
        next_iteration = "R3"
        outcome = "D2"
    elif not smallest_gate["passed"]:
        branch = "R1-B"
        next_iteration = "R2"
        outcome = "D1"
    else:
        branch = "R1-no-natural-amplitude-failure"
        next_iteration = "R4"
        outcome = "unresolved"
    scaling_rule = r1_config["quadratic_scaling"]
    median_slope = float(
        slopes["residual_log_log_slope"].median()
    )
    summary = {
        "iteration": "r1_amplitude_trust_region",
        "claim_scope": "retrospective_only",
        "row_count": len(rows),
        "direction_count": len(slopes),
        "gates_by_gamma": gate_results,
        "median_residual_log_log_slope": median_slope,
        "quadratic_scaling_consistent": bool(
            median_slope
            >= float(scaling_rule["median_log_log_slope_min"])
            and median_slope
            <= float(scaling_rule["median_log_log_slope_max"])
        ),
        "branch": branch,
        "r1_c_candidate_specific_flag": r1_c,
        "small_gamma_failure_concentration": (
            concentration_fraction
        ),
        "next_iteration": next_iteration,
        "cumulative_mechanism_outcome": outcome,
        "config_sha256": sha256_file(
            ROOT
            / "experiments/p2_recovery/"
            "r1_amplitude_trust_region/r1_config.yaml"
        ),
    }
    atomic_frame(output / "r1_gate_by_gamma.parquet", gate_frame)
    gate_frame.to_csv(
        output / "r1_gate_by_gamma.csv", index=False
    )
    atomic_frame(output / "r1_scaling_slopes.parquet", slopes)
    slopes.to_csv(output / "r1_scaling_slopes.csv", index=False)
    breakdown_frame = (
        primary.groupby(
            [
                "gamma",
                "task",
                "layer",
                "history_id",
                "candidate_source",
            ],
            as_index=False,
        )
        .agg(
            row_count=("cosine", "size"),
            median_cosine=("cosine", "median"),
            median_relative_l2=("relative_l2", "median"),
            median_residual_norm=("residual_norm", "median"),
            row_pass_fraction=("cosine", lambda value: (value >= 0.99).mean()),
        )
    )
    atomic_frame(
        output / "r1_stratified_scaling.parquet",
        breakdown_frame,
    )
    breakdown_frame.to_csv(
        output / "r1_stratified_scaling.csv", index=False
    )
    atomic_json(output / "r1_summary.json", summary)
    return summary


def main() -> None:
    r0_config = yaml.safe_load(
        (
            ROOT
            / "experiments/p2_recovery/r0_failure_map/"
            "r0_config.yaml"
        ).read_text(encoding="utf-8")
    )
    r1_config = yaml.safe_load(
        (
            ROOT
            / "experiments/p2_recovery/"
            "r1_amplitude_trust_region/r1_config.yaml"
        ).read_text(encoding="utf-8")
    )
    result = {
        "r0": analyze_r0(r0_config),
        "r1": analyze_r1(r1_config),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
