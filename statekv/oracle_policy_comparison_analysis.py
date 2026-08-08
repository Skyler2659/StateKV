"""Consolidate StateKV teacher-versus-policy closed-loop evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from statekv.storage import atomic_frame, atomic_json


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_oracle_policy_comparison(
    repository_root: Path, output_dir: Path
) -> Path:
    results = repository_root / "results" / "temporal_cache_discovery"
    teacher_dir = (
        results / "statekv_oracle_policy_comparison_independent_p28_v1"
    )
    horizon_dirs = {
        1: results / "statekv_oracle_policy_freegen_h1_p29b_v1",
        4: results / "statekv_oracle_policy_freegen_h4_p29c_v1",
        8: results / "statekv_oracle_policy_freegen_p29_v1",
    }
    independent_dir = (
        results / "statekv_oracle_policy_freegen_independent_p30_v1"
    )
    teacher = _read_json(teacher_dir / "summary.json")
    independent = _read_json(independent_dir / "summary.json")
    horizon_rows = []
    for horizon, run_dir in sorted(horizon_dirs.items()):
        payload = _read_json(run_dir / "summary.json")
        aggregate = {
            row["policy"]: row for row in payload["policy_aggregates"]
        }["statekv_exact_mean"]
        horizon_rows.append(
            {
                "control_horizon": int(horizon),
                "passed_joint_gate": bool(payload["passed"]),
                "statekv_mean_trajectory_exact_kl": float(
                    aggregate["mean_trajectory_exact_kl"]
                ),
                "statekv_govreport_rouge_l": float(
                    aggregate["mean_govreport_rouge_l"]
                ),
                "statekv_niah_retrieval": float(
                    aggregate["mean_niah_retrieval"]
                ),
                "elapsed_s": float(payload["collection_elapsed_s"]),
            }
        )
    sample_frame = pd.read_csv(independent_dir / "sample_results.csv")
    pivot_kl = sample_frame.pivot(
        index="sample_id",
        columns="policy",
        values="mean_trajectory_exact_kl",
    )
    paired_rows = []
    for baseline in ("attention", "snapkv", "h2o"):
        delta = pivot_kl[baseline] - pivot_kl["statekv_exact_mean"]
        for sample_id, value in delta.items():
            paired_rows.append(
                {
                    "baseline": baseline,
                    "sample_id": str(sample_id),
                    "baseline_minus_statekv_trajectory_kl": float(value),
                    "statekv_wins": bool(value > 0.0),
                }
            )
    paired = pd.DataFrame(paired_rows)
    policy = {
        row["policy"]: row for row in independent["policy_aggregates"]
    }
    relative_reductions = {}
    for baseline in ("attention", "snapkv", "h2o"):
        baseline_kl = float(policy[baseline]["mean_trajectory_exact_kl"])
        statekv_kl = float(
            policy["statekv_exact_mean"]["mean_trajectory_exact_kl"]
        )
        relative_reductions[baseline] = float(
            (baseline_kl - statekv_kl) / baseline_kl
        )
    result = {
        "teacher_forced_independent": {
            "passed": bool(teacher["passed"]),
            "policy_aggregates": teacher["policy_aggregates"],
            "paired_comparisons": teacher["paired_comparisons"],
        },
        "development_horizon_ablation": horizon_rows,
        "free_generation_independent": {
            "passed_joint_gate": bool(independent["passed"]),
            "lower_trajectory_kl_than_each_fixed_policy": bool(
                independent[
                    "statekv_lower_trajectory_kl_than_each_fixed_policy"
                ]
            ),
            "task_metrics_nonworse_than_each_fixed_policy": bool(
                independent[
                    "statekv_task_metrics_nonworse_than_each_fixed_policy"
                ]
            ),
            "policy_aggregates": independent["policy_aggregates"],
            "paired_comparisons": independent["paired_comparisons"],
            "relative_trajectory_kl_reduction": relative_reductions,
            "sample_wins": {
                baseline: int(
                    paired.loc[paired["baseline"] == baseline, "statekv_wins"].sum()
                )
                for baseline in ("attention", "snapkv", "h2o")
            },
            "sample_count": int(sample_frame["sample_id"].nunique()),
        },
        "verdict": (
            "The exact StateKV teacher improves distributional trajectory risk "
            "overall, but does not uniformly improve downstream generation "
            "quality or retrieval over fixed policies."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_frame(pd.DataFrame(horizon_rows), output_dir / "horizon_ablation.csv")
    atomic_frame(paired, output_dir / "paired_sample_trajectory_kl.csv")
    atomic_json(output_dir / "analysis.json", result)
    return output_dir / "analysis.json"


__all__ = ["analyze_oracle_policy_comparison"]
