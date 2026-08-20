"""Gates, fresh-test ledger, leaderboard, and Pareto reporting."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from statekv.storage import atomic_frame, atomic_json
from statekv.trajectory_analysis import cluster_bootstrap_interval


def register_fresh_test_component(output_root: Path, component: str) -> None:
    """Register components inside one frozen, non-interactive test opening."""

    frozen = output_root / "frozen_validation_selection.json"
    if not frozen.exists():
        raise RuntimeError("fresh test cannot open before validation selection is frozen")
    ledger_path = output_root / "test_open_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    count = int(ledger.get("fresh_test_evaluations", 0))
    status = str(ledger.get("status", "sealed"))
    components = list(ledger.get("components") or [])
    if count == 0:
        count = 1
        status = "open"
        ledger["opened_at"] = datetime.now(timezone.utc).isoformat()
    elif status != "open" and str(component) not in components:
        raise RuntimeError("fresh-test opening is closed")
    if count > int(ledger["fresh_test_open_limit"]):
        raise RuntimeError("fresh-test opening limit exceeded")
    if str(component) not in components:
        components.append(str(component))
    ledger.update(
        {
            "fresh_test_evaluations": count,
            "status": status,
            "components": components,
        }
    )
    atomic_json(ledger_path, ledger)


def close_fresh_test_opening(output_root: Path) -> None:
    ledger_path = output_root / "test_open_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if int(ledger.get("fresh_test_evaluations", 0)) != 1:
        raise RuntimeError("fresh-test opening was not registered")
    ledger["status"] = "closed"
    ledger["closed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(ledger_path, ledger)


def summarize_sequence_metrics(
    sequence_path: Path,
    output_path: Path,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    sequence = pd.read_csv(sequence_path)
    rows: List[Dict[str, Any]] = []
    for (method, horizon), group in sequence.groupby(
        ["method", "future_horizon"], sort=True
    ):
        ci = cluster_bootstrap_interval(
            group,
            "oracle_gap_recovery",
            cluster="sample_id",
            samples=int(bootstrap_repetitions),
            seed=int(seed) + int(horizon),
            statistic="mean",
        )
        rows.append(
            {
                "method": method,
                "future_horizon": int(horizon),
                "future_recall": float(group["future_topk_recall"].mean()),
                "spearman": float(group["spearman"].mean()),
                "pairwise_accuracy": float(group.get("pairwise_accuracy", pd.Series([np.nan])).mean()),
                "ndcg": float(group["ndcg"].mean()),
                "oracle_gap_recovery": float(ci["estimate"]),
                "recovery_ci_low": float(ci["ci_low"]),
                "recovery_ci_high": float(ci["ci_high"]),
                "sequence_win_rate": float(group["beats_baseline"].mean()),
                "sequences": int(group["sample_id"].nunique()),
            }
        )
    summary = pd.DataFrame(rows)
    atomic_frame(summary, output_path)
    return summary


def freeze_validation_selection(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    sequence_path = output_root / "evaluation" / "validation" / "sequence_metrics.csv"
    summary = summarize_sequence_metrics(
        sequence_path,
        output_root / "evaluation" / "validation" / "summary_with_ci.csv",
        int(config["gate_a"]["bootstrap_repetitions"]),
        int(config["gate_a"]["random_seed"]),
    )
    primary_horizon = int(config["gate_a"]["primary_horizon"])
    candidates = summary[
        (summary["future_horizon"] == primary_horizon)
        & (summary["method"] != "best_per_head_fixed_ema")
        & (summary["method"] != "feature_gbdt_I_global_full")
    ].sort_values(
        ["oracle_gap_recovery", "future_recall", "method"],
        ascending=[False, False, True],
    )
    if candidates.empty:
        raise RuntimeError("validation produced no causal learned candidate")
    selected = candidates.iloc[0]
    path = output_root / "frozen_validation_selection.json"
    atomic_json(
        path,
        {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "selection_metric": "validation oracle_gap_recovery at H=32",
            "selected_method": str(selected["method"]),
            "selected_horizon": primary_horizon,
            "validation_recovery": float(selected["oracle_gap_recovery"]),
            "validation_recovery_ci": [
                float(selected["recovery_ci_low"]),
                float(selected["recovery_ci_high"]),
            ],
            "test_results_seen": False,
            "all_preregistered_methods_evaluated_in_one_test_opening": True,
        },
    )
    return path


def _method_metadata(method: str) -> Dict[str, Any]:
    if method == "best_per_head_fixed_ema":
        return {"method_family": "fixed", "learned": False, "rollout": False, "counterfactual": False}
    if method.startswith("CAUSAL_EXPENSIVE_ROLLOUT"):
        return {"method_family": "rollout", "learned": False, "rollout": True, "counterfactual": False}
    if method.startswith("COUNTERFACTUAL"):
        return {"method_family": "counterfactual", "learned": False, "rollout": True, "counterfactual": True}
    if method == "FULL_CACHE_REFERENCE":
        return {
            "method_family": "full_cache_reference",
            "learned": False,
            "rollout": False,
            "counterfactual": False,
        }
    if method.startswith("STRICT_"):
        return {
            "method_family": "strict_closed_loop",
            "learned": False,
            "rollout": "CAUSAL_ROLLOUT" in method,
            "counterfactual": False,
        }
    if method.startswith("rollout_distilled_mlp"):
        return {
            "method_family": "rollout_distillation",
            "learned": True,
            "rollout": False,
            "counterfactual": False,
        }
    if method.startswith("deepsets"):
        family = "set_predictor"
    elif method.startswith("temporal_gru"):
        family = "temporal_predictor"
    elif method.startswith("query_conditioned_mlp"):
        family = "tokenwise_neural"
    elif method.startswith("feature_ridge") or method.startswith("feature_gbdt"):
        family = "feature_ablation"
    else:
        family = "classical_predictor"
    return {"method_family": family, "learned": True, "rollout": False, "counterfactual": False}


def build_existence_leaderboard(
    config_path: Path,
    repository_root: Path,
    split: str,
    close_test: bool = False,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    result_root = repository_root / "results" / "statekv_existence"
    learned_sequence = output_root / "evaluation" / str(split) / "sequence_metrics.csv"
    learned = summarize_sequence_metrics(
        learned_sequence,
        output_root / "evaluation" / str(split) / "summary_with_ci.csv",
        int(config["gate_a"]["bootstrap_repetitions"]),
        int(config["gate_a"]["random_seed"]),
    )
    learned_cost_path = output_root / "evaluation" / str(split) / "inference_costs.csv"
    learned_cost = (
        pd.read_csv(learned_cost_path).set_index("method")["runtime_multiplier"].to_dict()
        if learned_cost_path.exists()
        else {}
    )
    rows: List[Dict[str, Any]] = []
    for record in learned.to_dict("records"):
        metadata = _method_metadata(str(record["method"]))
        rows.append(
            {
                **record,
                **metadata,
                "causal": True,
                "closed_loop_kl": np.nan,
                "task_score": np.nan,
                "runtime_multiplier": float(learned_cost.get(record["method"], np.nan)),
                "peak_memory": np.nan,
                "test_split": str(split),
            }
        )
    rollout_sequence = output_root / "rollout" / str(split) / "sequence_metrics.csv"
    if rollout_sequence.exists():
        rollout = summarize_sequence_metrics(
            rollout_sequence,
            output_root / "rollout" / str(split) / "summary_with_ci.csv",
            int(config["gate_a"]["bootstrap_repetitions"]),
            int(config["gate_a"]["random_seed"]) + 17,
        )
        costs = pd.read_csv(output_root / "rollout" / str(split) / "costs.csv")
        cost_summary = costs.groupby(["method", "future_horizon"]).agg(
            runtime_multiplier=("runtime_multiplier", "mean"),
            peak_memory=("peak_memory_bytes", "max"),
        )
        for record in rollout.to_dict("records"):
            metadata = _method_metadata(str(record["method"]))
            rows.append(
                {
                    **record,
                    **metadata,
                    "causal": True,
                    "closed_loop_kl": np.nan,
                    "task_score": np.nan,
                    "runtime_multiplier": float(
                        cost_summary.loc[
                            (record["method"], int(record["future_horizon"])),
                            "runtime_multiplier",
                        ]
                    ),
                    "peak_memory": float(
                        cost_summary.loc[
                            (record["method"], int(record["future_horizon"])),
                            "peak_memory",
                        ]
                    ),
                    "test_split": str(split),
                }
            )
    closed_loop_path = output_root / "closed_loop" / "closed_loop_test" / "sample_summary.csv"
    if closed_loop_path.exists():
        closed = pd.read_csv(closed_loop_path)
        wall_baseline = (
            closed[closed["policy"] == str(config["closed_loop"]["primary_baseline"])]
            .groupby("budget")["wall_time_s"]
            .mean()
            .to_dict()
        )
        for (policy, budget), group in closed.groupby(["policy", "budget"], sort=True):
            method = f"{policy}_B{int(budget)}"
            metadata = _method_metadata(str(policy))
            rows.append(
                {
                    "method": method,
                    "future_horizon": int(config["closed_loop"]["rollout_horizon"]),
                    "future_recall": np.nan,
                    "spearman": np.nan,
                    "pairwise_accuracy": np.nan,
                    "ndcg": np.nan,
                    "oracle_gap_recovery": np.nan,
                    "recovery_ci_low": np.nan,
                    "recovery_ci_high": np.nan,
                    "sequence_win_rate": np.nan,
                    **metadata,
                    "causal": True,
                    "closed_loop_kl": float(group["mean_trajectory_exact_kl"].mean()),
                    "task_score": float(group["official_score"].mean()),
                    "runtime_multiplier": float(
                        group["wall_time_s"].mean()
                        / max(
                            float(
                                wall_baseline.get(
                                    int(budget), group["wall_time_s"].mean()
                                )
                            ),
                            1.0e-9,
                        )
                    ),
                    "peak_memory": np.nan,
                    "test_split": "closed_loop_test",
                }
            )
    columns = [
        "method",
        "method_family",
        "causal",
        "learned",
        "rollout",
        "counterfactual",
        "future_horizon",
        "future_recall",
        "oracle_gap_recovery",
        "recovery_ci_low",
        "recovery_ci_high",
        "spearman",
        "pairwise_accuracy",
        "ndcg",
        "sequence_win_rate",
        "closed_loop_kl",
        "task_score",
        "runtime_multiplier",
        "peak_memory",
        "test_split",
    ]
    leaderboard = pd.DataFrame(rows)[columns].sort_values(
        ["future_horizon", "oracle_gap_recovery"], ascending=[True, False]
    )
    path = result_root / "leaderboard.csv"
    atomic_frame(leaderboard, path)

    recovery_plot = leaderboard[
        np.isfinite(leaderboard["runtime_multiplier"])
        & np.isfinite(leaderboard["oracle_gap_recovery"])
    ].copy()
    closed_loop_plot = leaderboard[
        np.isfinite(leaderboard["runtime_multiplier"])
        & np.isfinite(leaderboard["closed_loop_kl"])
    ].copy()
    figure_path = result_root / "pareto_compute_vs_recovery.png"
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for family, group in recovery_plot.groupby("method_family"):
        axes[0].scatter(
            group["runtime_multiplier"],
            group["oracle_gap_recovery"],
            label=family,
            alpha=0.8,
        )
    if not recovery_plot.empty:
        best = recovery_plot.sort_values("oracle_gap_recovery", ascending=False).head(8)
        for _, row in best.iterrows():
            axes[0].annotate(
                f"{row['method']} H={int(row['future_horizon'])}",
                (row["runtime_multiplier"], row["oracle_gap_recovery"]),
                fontsize=7,
            )
    axes[0].axhline(0.25, color="tab:orange", linestyle="--", label="Gate A/B 25%")
    axes[0].axhline(0.50, color="tab:green", linestyle=":", label="strong 50%")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Runtime multiplier (model scoring + selection)")
    axes[0].set_ylabel("Oracle-gap recovery (higher is better)")
    axes[0].set_title(f"Intermediate predictability ({split})")
    axes[0].legend(fontsize=7)

    for family, group in closed_loop_plot.groupby("method_family"):
        axes[1].scatter(
            group["runtime_multiplier"],
            group["closed_loop_kl"],
            label=family,
            alpha=0.8,
        )
    if not closed_loop_plot.empty:
        best = closed_loop_plot.sort_values("closed_loop_kl").head(10)
        for _, row in best.iterrows():
            axes[1].annotate(
                str(row["method"]),
                (row["runtime_multiplier"], row["closed_loop_kl"]),
                fontsize=7,
            )
        axes[1].set_xscale("log")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(
            0.5,
            0.5,
            "Closed-loop evidence pending Gate B",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("Runtime multiplier vs fixed EMA")
    axes[1].set_ylabel("Mean trajectory exact KL (lower is better)")
    axes[1].set_title("Strict physical-eviction closed loop")
    figure.suptitle("StateKV causal existence frontier")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    if close_test and str(split) == "fresh_test":
        close_fresh_test_opening(output_root)
    return path


def evaluate_existence_gates(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    leaderboard_path = repository_root / "results" / "statekv_existence" / "leaderboard.csv"
    leaderboard = pd.read_csv(leaderboard_path)
    test = leaderboard[leaderboard["test_split"] == "fresh_test"].copy()
    if test.empty:
        raise RuntimeError("fresh-test leaderboard is unavailable")
    frozen = json.loads(
        (output_root / "frozen_validation_selection.json").read_text(encoding="utf-8")
    )
    learned = test[
        (test["method"] == str(frozen["selected_method"]))
        & (test["future_horizon"] == int(frozen["selected_horizon"]))
    ]
    if len(learned) != 1:
        raise RuntimeError("frozen learned candidate is missing from fresh test")
    learned_row = learned.iloc[0]

    threshold = float(config["gate_b"]["minimum_recovery"])
    gate_b_ci_floor = float(config["gate_b"].get("require_ci_lower_above", 0.0))
    gate_b_require_majority = bool(
        config["gate_b"].get("require_majority_sequence_wins", True)
    )
    candidates = test[
        (test["method"] != "best_per_head_fixed_ema")
        & (test["causal"].astype(str).str.lower() == "true")
    ].copy()
    candidates["passes"] = (
        (candidates["oracle_gap_recovery"] >= threshold)
        & (candidates["recovery_ci_low"] > gate_b_ci_floor)
        & (
            (candidates["sequence_win_rate"] > 0.5)
            if gate_b_require_majority
            else True
        )
    )
    passing = candidates[candidates["passes"]].sort_values(
        ["oracle_gap_recovery", "future_recall"], ascending=False
    )
    gate_a_ci_floor = float(config["gate_a"].get("require_ci_lower_above", 0.0))
    gate_a_require_majority = bool(
        config["gate_a"].get("require_majority_sequence_wins", True)
    )
    gate_a_pass = bool(
        float(learned_row["oracle_gap_recovery"]) >= float(config["gate_a"]["minimum_recovery"])
        and float(learned_row["recovery_ci_low"]) > gate_a_ci_floor
        and (
            float(learned_row["sequence_win_rate"]) > 0.5
            if gate_a_require_majority
            else True
        )
    )
    gate_b_pass = not passing.empty
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "fresh_test_openings": 1,
        "gate_a_learned_predictability": gate_a_pass,
        "gate_a_selected_method": str(learned_row["method"]),
        "gate_a_selected_horizon": int(learned_row["future_horizon"]),
        "gate_a_recovery": float(learned_row["oracle_gap_recovery"]),
        "gate_a_ci": [
            float(learned_row["recovery_ci_low"]),
            float(learned_row["recovery_ci_high"]),
        ],
        "gate_a_sequence_win_rate": float(learned_row["sequence_win_rate"]),
        "gate_b_any_causal_teacher": gate_b_pass,
        "gate_b_threshold": threshold,
        "gate_b_ci_lower_threshold": gate_b_ci_floor,
        "gate_b_requires_majority_sequence_wins": gate_b_require_majority,
        "passing_methods": passing[
            [
                "method",
                "future_horizon",
                "oracle_gap_recovery",
                "recovery_ci_low",
                "recovery_ci_high",
                "sequence_win_rate",
            ]
        ].to_dict("records"),
        "closed_loop_authorized": gate_b_pass,
    }
    path = output_root / (
        "gate_b_passed.json" if gate_b_pass else "gate_b_failed.json"
    )
    atomic_json(path, result)
    return path


def evaluate_closed_loop_gate(
    config_path: Path, repository_root: Path
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = repository_root / str(config["output_run"])
    comparison_path = (
        output_root / "closed_loop" / "closed_loop_test" / "paired_comparison.csv"
    )
    if not comparison_path.exists():
        raise RuntimeError("closed-loop paired comparison is unavailable")
    comparison = pd.read_csv(comparison_path)
    primary = comparison[
        comparison["primary_comparison"].astype(str).str.lower() == "true"
    ].copy()
    expected_budgets = {int(value) for value in config["closed_loop"]["budgets"]}
    if set(primary["budget"].astype(int)) != expected_budgets:
        raise RuntimeError("closed-loop primary comparison does not cover every budget")
    primary["passes"] = (
        (primary["mean_kl_improvement"] > 0.0)
        & (
            primary["ci_low"]
            > float(config["gate_c"].get("require_ci_lower_above", 0.0))
        )
        & (primary["sequence_win_rate"] > 0.5)
    )
    passed = bool(
        primary["passes"].all()
        if bool(config["gate_c"].get("require_all_budgets", True))
        else primary["passes"].any()
    )
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "gate_c_closed_loop_usefulness": passed,
        "strict_pure_eviction": True,
        "primary_baseline": str(config["closed_loop"]["primary_baseline"]),
        "require_all_budgets": bool(config["gate_c"].get("require_all_budgets", True)),
        "budget_results": primary.to_dict("records"),
    }
    path = output_root / ("gate_c_passed.json" if passed else "gate_c_failed.json")
    atomic_json(path, result)
    return path


__all__ = [
    "build_existence_leaderboard",
    "close_fresh_test_opening",
    "evaluate_existence_gates",
    "evaluate_closed_loop_gate",
    "freeze_validation_selection",
    "register_fresh_test_component",
    "summarize_sequence_metrics",
]
