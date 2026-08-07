#!/usr/bin/env python3
"""Sequence-first P2 analysis and mechanical gate adjudication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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

from p0_v2_core import ranking_metrics  # noqa: E402
from p2_core import (  # noqa: E402
    FACTORIAL_REGISTRY,
    atomic_frame,
    atomic_json,
    factorial_effects,
    score_registry_rows,
    sha256_file,
)


UNIT = ["sample_id", "task", "anchor", "layer", "history_id"]
PRIMARY_HISTORIES = ["H1", "H2", "H3"]
RANK_METRICS = [
    "spearman",
    "pairwise_sign_accuracy",
    "top1_accuracy",
    "topk_overlap",
    "normalized_regret",
    "symmetric_scale_ratio",
]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_rankings(
    geometry: pd.DataFrame, top_k: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranking_rows: List[Dict[str, Any]] = []
    reversal_rows: List[Dict[str, Any]] = []
    for key, unit in geometry.groupby(UNIT, sort=False):
        common = dict(zip(UNIT, key))
        selected: Dict[str, Dict[str, Any]] = {}
        for score_type, group in unit.groupby("score_type", sort=False):
            if len(group) != 8 or group["mask_hash"].nunique() != 8:
                raise RuntimeError(
                    f"unit {key}/{score_type} is not eight-distinct"
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
            metrics = ranking_metrics(
                group["score"].to_numpy(dtype=np.float64),
                group["controlled_exact_kl"].to_numpy(
                    dtype=np.float64
                ),
                top_k,
            )
            row = {
                **common,
                "score_type": score_type,
                "candidate_count": len(group),
                "chosen_candidate_id": chosen["candidate_id"],
                "chosen_candidate_source": chosen[
                    "candidate_source"
                ],
                "chosen_exact_kl": float(
                    chosen["controlled_exact_kl"]
                ),
                "best_candidate_id": best["candidate_id"],
                "best_candidate_source": best["candidate_source"],
                "best_exact_kl": float(best["controlled_exact_kl"]),
                **metrics,
            }
            ranking_rows.append(row)
            selected[str(score_type)] = row
        action = selected["reference_action_fisher"]
        p1 = selected["p1_reference_state_fisher"]
        full = selected["full_state_local"]
        reversal_rows.append(
            {
                **common,
                "action_candidate_id": action[
                    "chosen_candidate_id"
                ],
                "p1_candidate_id": p1["chosen_candidate_id"],
                "full_candidate_id": full["chosen_candidate_id"],
                "full_reverses_action": full[
                    "chosen_candidate_id"
                ]
                != action["chosen_candidate_id"],
                "full_reverses_p1": full["chosen_candidate_id"]
                != p1["chosen_candidate_id"],
                "full_improves_action_regret": full[
                    "normalized_regret"
                ]
                < action["normalized_regret"],
                "full_improves_p1_regret": full[
                    "normalized_regret"
                ]
                < p1["normalized_regret"],
            }
        )
    return pd.DataFrame(ranking_rows), pd.DataFrame(reversal_rows)


def sequence_rankings(
    rankings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    primary = rankings[
        rankings["history_id"].isin(PRIMARY_HISTORIES)
    ]
    sequence = (
        primary.groupby(
            ["sample_id", "task", "score_type"], as_index=False
        )[RANK_METRICS]
        .median()
        .sort_values(["task", "sample_id", "score_type"])
    )
    by_history = (
        rankings.groupby(
            ["sample_id", "task", "history_id", "score_type"],
            as_index=False,
        )[RANK_METRICS]
        .median()
        .sort_values(
            ["task", "sample_id", "history_id", "score_type"]
        )
    )
    return sequence, by_history


def _score_value(
    sequence: pd.DataFrame,
    score_type: str,
    metric: str,
    task: str | None = None,
) -> float:
    source = sequence[sequence["score_type"] == score_type]
    if task is not None:
        source = source[source["task"] == task]
    return float(source[metric].median())


def component_attribution(
    sequence: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scopes: List[Tuple[str, str, pd.DataFrame]] = [
        ("overall", "overall", sequence)
    ]
    scopes.extend(
        (
            "task",
            str(task),
            group,
        )
        for task, group in sequence.groupby("task", sort=True)
    )
    scopes.extend(
        (
            "sequence",
            str(sample_id),
            group,
        )
        for sample_id, group in sequence.groupby(
            "sample_id", sort=True
        )
    )
    for scope_type, scope_value, frame in scopes:
        action = frame[
            frame["score_type"] == "reference_action_fisher"
        ][RANK_METRICS].median()
        p1 = frame[
            frame["score_type"] == "p1_reference_state_fisher"
        ][RANK_METRICS].median()
        full = frame[
            frame["score_type"] == "full_state_local"
        ][RANK_METRICS].median()
        for score_type, group in frame.groupby(
            "score_type", sort=True
        ):
            values = group[RANK_METRICS].median()
            rows.append(
                {
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "score_type": score_type,
                    **{
                        metric: float(values[metric])
                        for metric in RANK_METRICS
                    },
                    "spearman_delta_vs_action": float(
                        values["spearman"] - action["spearman"]
                    ),
                    "spearman_delta_vs_p1": float(
                        values["spearman"] - p1["spearman"]
                    ),
                    "spearman_gap_from_full": float(
                        full["spearman"] - values["spearman"]
                    ),
                    "pairwise_gain_vs_action": float(
                        values["pairwise_sign_accuracy"]
                        - action["pairwise_sign_accuracy"]
                    ),
                    "top1_gain_vs_action": float(
                        values["top1_accuracy"]
                        - action["top1_accuracy"]
                    ),
                    "normalized_regret_gain_vs_action": float(
                        action["normalized_regret"]
                        - values["normalized_regret"]
                    ),
                    "normalized_regret_degradation_vs_full": float(
                        values["normalized_regret"]
                        - full["normalized_regret"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def stratified_component_attribution(
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """Describe score performance by frozen task/layer/history strata."""
    primary = rankings[
        rankings["history_id"].isin(PRIMARY_HISTORIES)
    ]
    specifications: List[Tuple[str, Sequence[str]]] = [
        ("task", ["task"]),
        ("layer", ["layer"]),
        ("history", ["history_id"]),
        ("task_layer", ["task", "layer"]),
        ("task_history", ["task", "history_id"]),
    ]
    rows: List[Dict[str, Any]] = []
    for stratum_type, columns in specifications:
        grouper: Any = (
            columns[0] if len(columns) == 1 else list(columns)
        )
        for key, frame in primary.groupby(grouper, sort=True):
            keys = key if isinstance(key, tuple) else (key,)
            labels = dict(zip(columns, keys))
            action = frame[
                frame["score_type"] == "reference_action_fisher"
            ][RANK_METRICS].median()
            p1 = frame[
                frame["score_type"]
                == "p1_reference_state_fisher"
            ][RANK_METRICS].median()
            full = frame[
                frame["score_type"] == "full_state_local"
            ][RANK_METRICS].median()
            for score_type, score_frame in frame.groupby(
                "score_type", sort=True
            ):
                values = score_frame[RANK_METRICS].median()
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum_json": json.dumps(
                            labels, sort_keys=True, default=str
                        ),
                        **labels,
                        "score_type": score_type,
                        **{
                            metric: float(values[metric])
                            for metric in RANK_METRICS
                        },
                        "spearman_delta_vs_action": float(
                            values["spearman"]
                            - action["spearman"]
                        ),
                        "spearman_delta_vs_p1": float(
                            values["spearman"] - p1["spearman"]
                        ),
                        "spearman_gap_from_full": float(
                            full["spearman"] - values["spearman"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def readout_breakdown(response: pd.DataFrame) -> pd.DataFrame:
    primary = response[
        response["history_id"].isin(PRIMARY_HISTORIES)
    ].copy()
    specifications: List[Tuple[str, Sequence[str]]] = [
        ("overall", []),
        ("sequence", ["sample_id", "task"]),
        ("task", ["task"]),
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
            keys = (
                key
                if isinstance(key, tuple)
                else (key,)
            )
            labels = dict(zip(columns, keys))
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum_json": json.dumps(
                        labels, sort_keys=True, default=str
                    ),
                    **labels,
                    "row_count": len(frame),
                    "median_cosine": float(
                        frame[
                            "state_local_readout_cosine"
                        ].median()
                    ),
                    "median_relative_l2": float(
                        frame[
                            "state_local_readout_relative_l2"
                        ].median()
                    ),
                    "row_cosine_pass_fraction": float(
                        frame["state_local_readout_cosine"]
                        .ge(0.99)
                        .mean()
                    ),
                    "median_state_norm": float(
                        frame["state_norm"].median()
                    ),
                    "median_action_norm": float(
                        frame["action_r_norm"].median()
                    ),
                    "median_state_fisher_energy": float(
                        frame["state_fisher_energy"].median()
                    ),
                    "median_physical_history_kl": float(
                        frame["physical_history_kl"].median()
                    ),
                    "low_norm_truth_fraction": float(
                        frame[
                            "nonlinear_action_logit_norm"
                        ]
                        .lt(1.0e-8)
                        .mean()
                    ),
                    "all_finite": bool(
                        frame["state_local_readout_finite"].all()
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_factorial_effect_rows(
    sequence: pd.DataFrame,
) -> pd.DataFrame:
    registry = score_registry_rows()
    name_to_cell = {}
    for name, values in registry.items():
        if not values["factorial"]:
            continue
        name_to_cell[name] = (
            int(values["gradient"] == "state_local"),
            int(values["jacobian"] == "state_local"),
            int(values["fisher"] == "state_local"),
        )
    scopes: List[Tuple[str, str, pd.DataFrame]] = [
        ("overall", "overall", sequence)
    ]
    scopes.extend(
        ("task", str(task), group)
        for task, group in sequence.groupby("task", sort=True)
    )
    scopes.extend(
        ("sequence", str(sample_id), group)
        for sample_id, group in sequence.groupby(
            "sample_id", sort=True
        )
    )
    rows = []
    for scope_type, scope_value, frame in scopes:
        values = {
            name_to_cell[name]: float(
                frame.loc[
                    frame["score_type"] == name, "spearman"
                ].median()
            )
            for name in name_to_cell
        }
        rows.append(
            {
                "scope_type": scope_type,
                "scope_value": scope_value,
                **factorial_effects(values),
            }
        )
    return pd.DataFrame(rows)


def make_diagnostics(
    response: pd.DataFrame,
    state: pd.DataFrame,
    rankings: pd.DataFrame,
    reversals: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_rank = rankings[
        rankings["history_id"].isin(PRIMARY_HISTORIES)
    ]
    pivot = primary_rank.pivot_table(
        index=UNIT,
        columns="score_type",
        values="spearman",
        aggfunc="first",
    ).reset_index()
    pivot["full_action_spearman_gain"] = (
        pivot["full_state_local"]
        - pivot["reference_action_fisher"]
    )
    pivot["full_p1_spearman_gain"] = (
        pivot["full_state_local"]
        - pivot["p1_reference_state_fisher"]
    )
    gradient = state[
        state["history_id"].isin(PRIMARY_HISTORIES)
    ].merge(
        pivot[
            UNIT
            + [
                "full_action_spearman_gain",
                "full_p1_spearman_gain",
            ]
        ],
        on=UNIT,
        how="left",
        validate="one_to_one",
    )
    gradient["gradient_norm_ratio_reference_to_state"] = (
        gradient["reference_linear_gradient_norm"]
        / gradient["state_gradient_norm"].clip(lower=1.0e-12)
    )
    primary_response = response[
        response["history_id"].isin(PRIMARY_HISTORIES)
    ].copy()
    jacobian = primary_response.merge(
        reversals[
            UNIT
            + [
                "full_reverses_action",
                "full_reverses_p1",
                "full_improves_action_regret",
            ]
        ],
        on=UNIT,
        how="left",
        validate="many_to_one",
    )
    unit_jacobian = (
        jacobian.groupby(UNIT, as_index=False)
        .agg(
            median_jacobian_cosine=(
                "jacobian_reference_vs_state_cosine",
                "median",
            ),
            median_jacobian_relative_l2=(
                "jacobian_reference_vs_state_relative_l2",
                "median",
            ),
            median_fisher_weighted_discrepancy_reference=(
                "jacobian_fisher_weighted_discrepancy_reference",
                "median",
            ),
            median_fisher_weighted_discrepancy_state=(
                "jacobian_fisher_weighted_discrepancy_state",
                "median",
            ),
            probability_total_variation=(
                "probability_total_variation",
                "first",
            ),
            full_reverses_action=(
                "full_reverses_action",
                "first",
            ),
            full_reverses_p1=("full_reverses_p1", "first"),
            full_improves_action_regret=(
                "full_improves_action_regret",
                "first",
            ),
        )
        .merge(
            pivot[
                UNIT
                + [
                    "full_action_spearman_gain",
                    "full_p1_spearman_gain",
                ]
            ],
            on=UNIT,
            how="left",
            validate="one_to_one",
        )
    )
    correlations: List[Dict[str, Any]] = []

    def add_correlations(
        diagnostic: str,
        frame: pd.DataFrame,
        left_columns: Sequence[str],
        right_columns: Sequence[str],
    ) -> None:
        for left in left_columns:
            for right in right_columns:
                clean = frame[[left, right]].dropna()
                correlation = (
                    float(
                        spearmanr(
                            clean[left], clean[right]
                        ).statistic
                    )
                    if len(clean) >= 3
                    else float("nan")
                )
                correlations.append(
                    {
                        "diagnostic": diagnostic,
                        "left": left,
                        "right": right,
                        "spearman": correlation,
                        "row_count": len(clean),
                    }
                )

    add_correlations(
        "gradient",
        gradient,
        [
            "gradient_reference_vs_state_relative_l2",
            "gradient_reference_vs_state_cosine",
            "gradient_norm_ratio_reference_to_state",
        ],
        [
            "state_norm",
            "controlled_history_kl",
            "layer",
            "full_action_spearman_gain",
        ],
    )
    add_correlations(
        "jacobian",
        unit_jacobian,
        [
            "median_jacobian_relative_l2",
            "median_jacobian_cosine",
            "median_fisher_weighted_discrepancy_state",
        ],
        [
            "probability_total_variation",
            "full_action_spearman_gain",
            "full_reverses_action",
        ],
    )
    return gradient, jacobian, pd.DataFrame(correlations)


def gate_outcome(
    protocol: Mapping[str, Any],
    response: pd.DataFrame,
    identity: pd.DataFrame,
    audit: pd.DataFrame,
    sequence_vectors: pd.DataFrame,
    sequence: pd.DataFrame,
    attribution: pd.DataFrame,
    integrity: Mapping[str, Any],
    smoke: Mapping[str, Any],
    calibration: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    h0 = response[response["history_id"] == "H0"]
    gate0_checks = {
        "integrity_artifact_passed": bool(integrity["passed"]),
        "p0_manifest_all_match": bool(
            integrity["checks"]["p0_manifest_all_match"]
        ),
        "p1_manifest_all_match": bool(
            integrity["checks"]["p1_manifest_all_match"]
        ),
        "p0_p1_tests_pass": bool(
            integrity["checks"]["p0_p1_tests_pass"]
        ),
        "p0_p1_outcomes_unchanged": bool(
            integrity["checks"]["outcomes_unchanged"]
        ),
        "smoke_passed": bool(smoke["passed"]),
        "p1_diagnostic_exact_regression": float(
            smoke["maximum_p1_regression_absolute_error"]
        )
        <= float(
            protocol["gates"]["integrity"][
                "p1_smoke_metric_absolute_tolerance"
            ]
        ),
        "calibration_passed": bool(calibration["passed"]),
        "split_isolation": all(
            metadata["split_audit"]["checks"].values()
        ),
        "h0_state_zero": float(h0["state_norm"].max())
        <= float(
            protocol["gates"]["integrity"]["h0_delta_norm_max"]
        ),
        "h0_probability_identity": float(
            h0["probability_total_variation"].max()
        )
        <= float(
            protocol["gates"]["integrity"][
                "h0_probability_max_absolute_error"
            ]
        ),
        "h0_gradient_zero": float(
            h0["state_gradient_norm"].max()
        )
        <= float(
            protocol["gates"]["integrity"][
                "h0_gradient_norm_max"
            ]
        ),
        "h0_score_identity": float(
            h0["h0_full_action_score_absolute_error"].max()
        )
        <= float(
            protocol["gates"]["integrity"][
                "h0_score_max_absolute_error"
            ]
        ),
        "identity_rows_finite": bool(identity["finite"].all()),
        "formal_config_hash_unchanged": sha256_file(
            ROOT / "configs/frozen/p2_state_local_config.yaml"
        )
        == str(metadata["config_sha256"]),
    }
    gate0 = all(gate0_checks.values())

    primary_vectors = sequence_vectors[
        sequence_vectors["history_id"] == "primary"
    ]
    primary_response = response[
        response["history_id"].isin(PRIMARY_HISTORIES)
    ]
    gate1_rule = protocol["gates"]["state_local_readout"]
    gate1_metrics = {
        "overall_sequence_first_median_cosine": float(
            primary_vectors["cosine"].median()
        ),
        "overall_sequence_first_median_relative_l2": float(
            primary_vectors["relative_l2"].median()
        ),
        "task_median_cosine": {
            str(key): float(value)
            for key, value in primary_vectors.groupby("task")[
                "cosine"
            ]
            .median()
            .items()
        },
        "task_median_relative_l2": {
            str(key): float(value)
            for key, value in primary_vectors.groupby("task")[
                "relative_l2"
            ]
            .median()
            .items()
        },
        "row_cosine_pass_fraction": float(
            primary_response["state_local_readout_cosine"]
            .ge(float(gate1_rule["row_cosine_threshold"]))
            .mean()
        ),
        "row_count": len(primary_response),
        "all_vectors_finite": bool(
            primary_response["state_local_readout_finite"].all()
            and primary_vectors["finite"].all()
        ),
        "maximum_operating_point_output_error": float(
            audit[
                "state_operating_point_output_max_error"
            ].max()
        ),
    }
    gate1_checks = {
        "overall_cosine": gate1_metrics[
            "overall_sequence_first_median_cosine"
        ]
        >= float(
            gate1_rule["overall_sequence_first_cosine_min"]
        ),
        "each_task_cosine": all(
            value >= float(gate1_rule["each_task_median_cosine_min"])
            for value in gate1_metrics["task_median_cosine"].values()
        ),
        "overall_relative_l2": gate1_metrics[
            "overall_sequence_first_median_relative_l2"
        ]
        <= float(
            gate1_rule["overall_sequence_first_relative_l2_max"]
        ),
        "row_pass_fraction": gate1_metrics[
            "row_cosine_pass_fraction"
        ]
        >= float(gate1_rule["row_pass_fraction_min"]),
        "all_vectors_finite": gate1_metrics["all_vectors_finite"],
        "baseline_identity": gate1_metrics[
            "maximum_operating_point_output_error"
        ]
        <= float(
            protocol["numeric"][
                "baseline_max_absolute_error_max"
            ]
        ),
    }
    gate1 = all(gate1_checks.values())

    tasks = sorted(str(value) for value in sequence["task"].unique())
    full = "full_state_local"
    action = "reference_action_fisher"
    p1 = "p1_reference_state_fisher"
    full_rho = _score_value(sequence, full, "spearman")
    action_rho = _score_value(sequence, action, "spearman")
    p1_rho = _score_value(sequence, p1, "spearman")
    pivot_spearman = sequence.pivot(
        index=["sample_id", "task"],
        columns="score_type",
        values="spearman",
    ).reset_index()
    pivot_spearman["full_action_gain"] = (
        pivot_spearman[full] - pivot_spearman[action]
    )
    pair_gain = _score_value(
        sequence, full, "pairwise_sign_accuracy"
    ) - _score_value(sequence, action, "pairwise_sign_accuracy")
    top1_gain = _score_value(
        sequence, full, "top1_accuracy"
    ) - _score_value(sequence, action, "top1_accuracy")
    regret_gain = _score_value(
        sequence, action, "normalized_regret"
    ) - _score_value(sequence, full, "normalized_regret")
    secondary_improves = top1_gain > 0.0 or regret_gain > 0.0
    max_degradation = float(
        protocol["gates"]["full_state_local_risk"][
            "secondary_metric_max_degradation"
        ]
    )
    secondary_not_degraded = bool(
        (top1_gain >= -max_degradation)
        and (regret_gain >= -max_degradation)
    )
    gate2_metrics = {
        "overall_full_spearman": full_rho,
        "overall_action_spearman": action_rho,
        "overall_p1_spearman": p1_rho,
        "overall_full_action_spearman_gain": full_rho
        - action_rho,
        "overall_full_p1_spearman_gain": full_rho - p1_rho,
        "task_full_spearman": {
            task: _score_value(
                sequence, full, "spearman", task
            )
            for task in tasks
        },
        "task_full_action_spearman_gain": {
            task: _score_value(
                sequence, full, "spearman", task
            )
            - _score_value(sequence, action, "spearman", task)
            for task in tasks
        },
        "positive_sequence_fraction": float(
            (pivot_spearman["full_action_gain"] > 0.0).mean()
        ),
        "pairwise_accuracy_gain": pair_gain,
        "top1_accuracy_gain": top1_gain,
        "normalized_regret_gain": regret_gain,
    }
    gate2_rule = protocol["gates"]["full_state_local_risk"]
    gate2_checks = {
        "overall_full_spearman": full_rho
        >= float(gate2_rule["overall_sequence_first_spearman_min"]),
        "each_task_full_spearman": all(
            value
            >= float(gate2_rule["each_task_median_spearman_min"])
            for value in gate2_metrics["task_full_spearman"].values()
        ),
        "overall_action_delta": full_rho - action_rho
        >= float(
            gate2_rule["overall_action_delta_spearman_min"]
        ),
        "each_task_action_delta_positive": all(
            value
            > float(
                gate2_rule["each_task_action_delta_strict_min"]
            )
            for value in gate2_metrics[
                "task_full_action_spearman_gain"
            ].values()
        ),
        "positive_sequence_fraction": gate2_metrics[
            "positive_sequence_fraction"
        ]
        >= float(gate2_rule["positive_sequence_fraction_min"]),
        "pairwise_gain": pair_gain
        >= float(gate2_rule["pairwise_accuracy_gain_min"]),
        "top1_or_regret_strictly_improves": secondary_improves,
        "other_secondary_not_degraded": secondary_not_degraded,
        "strict_p1_spearman_improvement": full_rho > p1_rho,
        "both_tasks_contribute": all(
            value > 0.0
            for value in gate2_metrics[
                "task_full_action_spearman_gain"
            ].values()
        ),
    }
    gate2_raw = all(gate2_checks.values())
    gate2 = bool(gate1 and gate2_raw)

    full_gain = full_rho - action_rho
    reduced_rows: List[Dict[str, Any]] = []
    for name in FACTORIAL_REGISTRY:
        if name == full:
            continue
        reduced_rho = _score_value(sequence, name, "spearman")
        reduced_gain = reduced_rho - action_rho
        task_deltas = {
            task: _score_value(
                sequence, name, "spearman", task
            )
            - _score_value(sequence, action, "spearman", task)
            for task in tasks
        }
        regret_degradation = _score_value(
            sequence, name, "normalized_regret"
        ) - _score_value(sequence, full, "normalized_regret")
        row = {
            "score_type": name,
            "overall_spearman": reduced_rho,
            "full_spearman_gap": full_rho - reduced_rho,
            "action_spearman_gain": reduced_gain,
            "full_gain_retention": (
                reduced_gain / full_gain
                if full_gain > 0.0
                else float("-inf")
            ),
            "task_action_gains": task_deltas,
            "normalized_regret_degradation_vs_full": (
                regret_degradation
            ),
        }
        rule = protocol["gates"]["reduced_geometry"]
        checks = {
            "full_gap": row["full_spearman_gap"]
            <= float(rule["full_spearman_gap_max"]),
            "gain_retention": row["full_gain_retention"]
            >= float(rule["full_gain_retention_min"]),
            "each_task_action_delta_positive": all(
                value
                > float(
                    rule["each_task_action_delta_strict_min"]
                )
                for value in task_deltas.values()
            ),
            "normalized_regret": regret_degradation
            <= float(
                rule["normalized_regret_max_degradation"]
            ),
        }
        row["checks"] = checks
        row["qualifies"] = all(checks.values())
        reduced_rows.append(row)
    qualifying = [
        row["score_type"]
        for row in reduced_rows
        if row["qualifies"]
    ]
    gate3_raw = len(qualifying) > 0
    gate3 = bool(gate2 and gate3_raw)

    if not gate0 or not calibration["passed"]:
        outcome = "N"
    elif not gate1:
        outcome = "D"
    elif not gate2:
        outcome = "C"
    elif not gate3:
        outcome = "B"
    else:
        outcome = "A"
    return {
        "outcome": outcome,
        "outcome_definition": protocol["outcomes"][outcome],
        "gate0": {
            "passed": gate0,
            "checks": gate0_checks,
        },
        "gate1": {
            "passed": gate1,
            "checks": gate1_checks,
            "metrics": gate1_metrics,
        },
        "gate2": {
            "eligible": gate1,
            "passed": gate2,
            "raw_checks_passed": gate2_raw,
            "checks": gate2_checks,
            "metrics": gate2_metrics,
        },
        "gate3": {
            "eligible": gate2,
            "passed": gate3,
            "raw_reduced_geometry_exists": gate3_raw,
            "qualifying_reduced_geometries": qualifying,
            "candidates": reduced_rows,
        },
        "config_sha256": metadata["config_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/frozen/p2_state_local_config.yaml",
    )
    args = parser.parse_args()
    protocol = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    output_dir = ROOT / protocol["runtime"]["output_dir"]
    response = pd.read_parquet(output_dir / "response_rows.parquet")
    geometry = pd.read_parquet(
        output_dir / "geometry_score_rows.parquet"
    )
    identity = pd.read_parquet(output_dir / "identity_rows.parquet")
    state = pd.read_parquet(output_dir / "state_registry.parquet")
    audit = pd.read_parquet(output_dir / "unit_audit.parquet")
    sequence_vectors = pd.read_parquet(
        output_dir / "sequence_vector_metrics.parquet"
    )
    rankings, reversals = make_rankings(
        geometry, int(protocol["metrics"]["top_k"])
    )
    sequence, by_history = sequence_rankings(rankings)
    attribution = component_attribution(sequence)
    stratified = stratified_component_attribution(rankings)
    readout = readout_breakdown(response)
    effects = make_factorial_effect_rows(sequence)
    gradient, jacobian, correlations = make_diagnostics(
        response, state, rankings, reversals
    )
    outcome = gate_outcome(
        protocol,
        response,
        identity,
        audit,
        sequence_vectors,
        sequence,
        attribution,
        _read_json(output_dir / "integrity_summary.json"),
        _read_json(output_dir / "smoke_summary.json"),
        _read_json(output_dir / "calibration_summary.json"),
        _read_json(output_dir / "evaluation_metadata.json"),
    )
    atomic_frame(output_dir / "unit_ranking_rows.parquet", rankings)
    atomic_frame(
        output_dir / "sequence_first_summary.parquet", sequence
    )
    sequence.to_csv(
        output_dir / "sequence_first_summary.csv", index=False
    )
    atomic_frame(
        output_dir / "history_sequence_first_summary.parquet",
        by_history,
    )
    atomic_frame(
        output_dir / "component_attribution.parquet",
        attribution,
    )
    attribution.to_csv(
        output_dir / "component_attribution.csv", index=False
    )
    atomic_frame(
        output_dir / "stratified_component_attribution.parquet",
        stratified,
    )
    stratified.to_csv(
        output_dir / "stratified_component_attribution.csv",
        index=False,
    )
    atomic_frame(
        output_dir / "readout_breakdown.parquet", readout
    )
    readout.to_csv(
        output_dir / "readout_breakdown.csv", index=False
    )
    atomic_frame(
        output_dir / "factorial_effects.parquet", effects
    )
    effects.to_csv(
        output_dir / "factorial_effects.csv", index=False
    )
    atomic_frame(
        output_dir / "candidate_reversals.parquet", reversals
    )
    atomic_frame(
        output_dir / "gradient_diagnostics.parquet", gradient
    )
    atomic_frame(
        output_dir / "jacobian_diagnostics.parquet", jacobian
    )
    atomic_frame(
        output_dir / "diagnostic_correlations.parquet",
        correlations,
    )
    correlations.to_csv(
        output_dir / "diagnostic_correlations.csv", index=False
    )
    worst = (
        response[
            response["history_id"].isin(PRIMARY_HISTORIES)
        ]
        .sort_values(
            [
                "state_local_readout_cosine",
                "state_local_readout_relative_l2",
            ],
            ascending=[True, False],
        )
        .head(25)
    )
    worst.to_csv(output_dir / "worst_readout_rows.csv", index=False)
    atomic_json(output_dir / "p2_gate_outcome.json", outcome)
    summary = {
        "outcome": outcome["outcome"],
        "gates": {
            name: outcome[name]["passed"]
            for name in ("gate0", "gate1", "gate2", "gate3")
        },
        "formal_row_counts": _read_json(
            output_dir / "evaluation_metadata.json"
        )["row_counts"],
        "calibration": {
            "selected_relative_radius": _read_json(
                output_dir / "calibration_summary.json"
            )["selected_relative_radius"],
            "direction_count": _read_json(
                output_dir / "calibration_summary.json"
            )["direction_count"],
            "row_count": _read_json(
                output_dir / "calibration_summary.json"
            )["row_count"],
        },
        "gate1_metrics": outcome["gate1"]["metrics"],
        "gate2_descriptive_metrics": outcome["gate2"]["metrics"],
        "qualifying_reduced_geometries": outcome["gate3"][
            "qualifying_reduced_geometries"
        ],
        "table_counts": {
            "unit_ranking_rows": len(rankings),
            "sequence_first_summary": len(sequence),
            "history_sequence_first_summary": len(by_history),
            "component_attribution": len(attribution),
            "stratified_component_attribution": len(stratified),
            "readout_breakdown": len(readout),
            "factorial_effects": len(effects),
            "gradient_diagnostics": len(gradient),
            "jacobian_diagnostics": len(jacobian),
            "diagnostic_correlations": len(correlations),
        },
        "config_sha256": sha256_file(args.config),
    }
    atomic_json(output_dir / "p2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
