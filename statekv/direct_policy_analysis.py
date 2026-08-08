"""Paired sequence-level uncertainty for direct-policy physical replay."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from statekv.storage import atomic_frame, atomic_json


DEVELOPMENT_CHECKS = (
    "mean_exact_kl_improves",
    "p95_exact_kl_nonworse",
    "cvar95_exact_kl_nonworse",
    "maximum_exact_kl_nonworse",
    "large_loss_rate_nonworse",
    "all_task_means_improve",
)


def select_direct_policy_candidate(
    metrics: pd.DataFrame,
    stratified: pd.DataFrame,
    baseline: str,
    candidates: Sequence[str],
    tolerance: float = 1e-12,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Apply a common development gate to heterogeneous direct policies."""

    indexed = metrics.set_index("policy", drop=False)
    if baseline not in indexed.index:
        raise ValueError(f"baseline policy is missing from metrics: {baseline}")
    baseline_row = indexed.loc[baseline]
    task_rows = stratified[stratified["stratum"] == "task"]
    task_table = task_rows.pivot(index="value", columns="policy", values="mean_exact_kl")
    if baseline not in task_table.columns:
        raise ValueError(f"baseline policy is missing from task strata: {baseline}")

    audit_rows = []
    for candidate in candidates:
        if candidate not in indexed.index:
            raise ValueError(f"candidate policy is missing from metrics: {candidate}")
        if candidate not in task_table.columns:
            raise ValueError(f"candidate policy is missing from task strata: {candidate}")
        candidate_row = indexed.loc[candidate]
        checks = {
            "mean_exact_kl_improves": bool(
                candidate_row["mean_exact_kl"] < baseline_row["mean_exact_kl"] - tolerance
            ),
            "p95_exact_kl_nonworse": bool(
                candidate_row["p95_exact_kl"] <= baseline_row["p95_exact_kl"] + tolerance
            ),
            "cvar95_exact_kl_nonworse": bool(
                candidate_row["cvar95_exact_kl"]
                <= baseline_row["cvar95_exact_kl"] + tolerance
            ),
            "maximum_exact_kl_nonworse": bool(
                candidate_row["maximum_exact_kl"]
                <= baseline_row["maximum_exact_kl"] + tolerance
            ),
            "large_loss_rate_nonworse": bool(
                candidate_row["large_loss_rate"]
                <= baseline_row["large_loss_rate"] + tolerance
            ),
            "all_task_means_improve": bool(
                (
                    task_table[candidate]
                    < task_table[baseline] - tolerance
                ).all()
            ),
        }
        audit_rows.append(
            {
                "policy": candidate,
                "mean_exact_kl": float(candidate_row["mean_exact_kl"]),
                "mean_exact_kl_reduction": float(
                    baseline_row["mean_exact_kl"] - candidate_row["mean_exact_kl"]
                ),
                "p95_exact_kl": float(candidate_row["p95_exact_kl"]),
                "cvar95_exact_kl": float(candidate_row["cvar95_exact_kl"]),
                "maximum_exact_kl": float(candidate_row["maximum_exact_kl"]),
                "large_loss_rate": float(candidate_row["large_loss_rate"]),
                **checks,
                "eligible": bool(all(checks.values())),
            }
        )

    audit = pd.DataFrame(audit_rows).reset_index(drop=True)
    eligible = audit[audit["eligible"]].sort_values(
        ["mean_exact_kl_reduction", "policy"],
        ascending=[False, True],
    )
    selected = None if eligible.empty else str(eligible.iloc[0]["policy"])
    result = {
        "status": "direct_policy_development_selection",
        "confirmatory_evidence": False,
        "baseline": str(baseline),
        "objective": "maximum_mean_exact_kl_reduction",
        "tie_break": "policy_name_for_exact_numeric_ties",
        "required_development_checks": list(DEVELOPMENT_CHECKS),
        "eligible_policies": eligible["policy"].astype(str).tolist(),
        "selected_policy": selected,
        "independent_run_authorized": selected is not None,
        "decision": (
            "No candidate satisfies every preregistered development constraint; "
            "keep the independent sample IDs untouched."
            if selected is None
            else f"Freeze {selected} before evaluating the independent sample IDs."
        ),
    }
    return result, audit


def select_protected_rescue_candidate(
    metrics: pd.DataFrame,
    stratified: pd.DataFrame,
    inventory: pd.DataFrame,
    baseline: str,
    candidates: Sequence[str],
    rescue_slots: Mapping[str, int],
    tolerance: float = 1e-12,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Add the protected-rescue action-radius check to the common gate."""

    result, audit = select_direct_policy_candidate(
        metrics, stratified, baseline, candidates, tolerance
    )
    audit["rescue_slots"] = [int(rescue_slots[policy]) for policy in audit["policy"]]
    maximum_changes = []
    for policy in audit["policy"]:
        current = inventory[inventory["policy"] == policy]
        if current.empty:
            raise ValueError(f"candidate policy is missing from inventory: {policy}")
        maximum_changes.append(int(current["core_changes_vs_attention"].max()))
    audit["maximum_core_changes_vs_attention"] = maximum_changes
    audit["action_radius_respected"] = (
        audit["maximum_core_changes_vs_attention"] <= audit["rescue_slots"]
    )
    audit["eligible"] = audit["eligible"] & audit["action_radius_respected"]
    audit = audit.sort_values("rescue_slots").reset_index(drop=True)
    eligible = audit[audit["eligible"]].sort_values(
        ["mean_exact_kl_reduction", "rescue_slots"],
        ascending=[False, True],
    )
    selected = None if eligible.empty else str(eligible.iloc[0]["policy"])
    result.update(
        {
            "status": "protected_rescue_development_selection",
            "tie_break": "smaller_rescue_slot_count",
            "structural_checks": ["action_radius_respected"],
            "eligible_policies": eligible["policy"].astype(str).tolist(),
            "selected_policy": selected,
            "independent_run_authorized": selected is not None,
            "decision": (
                "No candidate satisfies every preregistered development "
                "constraint; keep the independent sample IDs untouched."
                if selected is None
                else f"Freeze {selected} before evaluating the independent sample IDs."
            ),
        }
    )
    return result, audit


def paired_bootstrap_mean(
    differences: np.ndarray, samples: int, seed: int
) -> Tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if values.size == 0 or int(samples) <= 0:
        raise ValueError("differences and bootstrap samples must be nonempty")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_tail_migration(
    rows: pd.DataFrame,
    baseline: str,
    primary: str,
    quantile: float = 0.95,
    large_loss_threshold: float = 1.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Describe whether a policy removes or creates paired tail-loss steps."""

    keys = ["sample_id", "task", "anchor", "horizon_offset"]
    paired = rows[rows["policy"].isin([baseline, primary])].pivot(
        index=keys, columns="policy", values="exact_kl"
    ).reset_index()
    if baseline not in paired or primary not in paired:
        raise RuntimeError("baseline or primary is missing from tail analysis")
    baseline_threshold = float(paired[baseline].quantile(float(quantile)))
    primary_threshold = float(paired[primary].quantile(float(quantile)))
    paired["baseline_tail"] = paired[baseline] >= baseline_threshold
    paired["primary_tail"] = paired[primary] >= primary_threshold
    paired["exact_kl_reduction"] = paired[baseline] - paired[primary]
    paired["tail_category"] = np.select(
        [
            paired["baseline_tail"] & paired["primary_tail"],
            paired["baseline_tail"] & ~paired["primary_tail"],
            ~paired["baseline_tail"] & paired["primary_tail"],
        ],
        ["shared_tail", "escaped_tail", "new_primary_tail"],
        default="non_tail",
    )
    baseline_large = paired[baseline] >= float(large_loss_threshold)
    primary_large = paired[primary] >= float(large_loss_threshold)
    paired["large_loss_transition"] = np.select(
        [baseline_large & ~primary_large, ~baseline_large & primary_large],
        ["removed_large_loss", "created_large_loss"],
        default="unchanged",
    )
    baseline_tail = paired[paired["baseline_tail"]]
    primary_tail = paired[paired["primary_tail"]]
    shared = paired[paired["tail_category"] == "shared_tail"]
    escaped = paired[paired["tail_category"] == "escaped_tail"]
    created = paired[paired["tail_category"] == "new_primary_tail"]
    union = int((paired["baseline_tail"] | paired["primary_tail"]).sum())
    summary = {
        "quantile": float(quantile),
        "steps": int(len(paired)),
        "baseline_tail_threshold": baseline_threshold,
        "primary_tail_threshold": primary_threshold,
        "baseline_tail_steps": int(len(baseline_tail)),
        "primary_tail_steps": int(len(primary_tail)),
        "shared_tail_steps": int(len(shared)),
        "escaped_tail_steps": int(len(escaped)),
        "new_primary_tail_steps": int(len(created)),
        "tail_jaccard": float(len(shared) / max(union, 1)),
        "paired_reduction_on_baseline_tail": float(
            baseline_tail["exact_kl_reduction"].mean()
        ),
        "paired_reduction_on_primary_tail": float(
            primary_tail["exact_kl_reduction"].mean()
        ),
        "paired_reduction_on_new_primary_tail": float(
            created["exact_kl_reduction"].mean()
        )
        if len(created)
        else 0.0,
        "removed_large_loss_steps": int(
            (paired["large_loss_transition"] == "removed_large_loss").sum()
        ),
        "created_large_loss_steps": int(
            (paired["large_loss_transition"] == "created_large_loss").sum()
        ),
        "largest_step_harm": float(
            np.maximum(-paired["exact_kl_reduction"], 0.0).max()
        ),
    }
    return summary, paired.sort_values("exact_kl_reduction")


def analyze_direct_policy_replay(
    run_dir: Path,
    baseline: str,
    primary: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    rows = pd.read_parquet(run_dir / "physical_replay_rows.parquet")
    analysis_root = run_dir / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    selected = rows[rows["policy"].isin([str(baseline), str(primary)])]
    sequence = selected.pivot_table(
        index=["sample_id", "task"],
        columns="policy",
        values="exact_kl",
        aggfunc="mean",
    ).reset_index()
    if baseline not in sequence or primary not in sequence:
        raise RuntimeError("baseline or primary is missing from replay rows")
    sequence["mean_exact_kl_reduction"] = (
        sequence[baseline] - sequence[primary]
    )
    task_rows = []
    groups = [("all", sequence)] + list(sequence.groupby("task", sort=True))
    for index, (task, current) in enumerate(groups):
        differences = current["mean_exact_kl_reduction"].to_numpy(dtype=np.float64)
        lower, upper = paired_bootstrap_mean(
            differences, int(bootstrap_samples), int(seed) + index
        )
        wins = int((differences > 0.0).sum())
        task_rows.append(
            {
                "task": str(task),
                "sequences": int(len(current)),
                "mean_exact_kl_reduction": float(differences.mean()),
                "median_exact_kl_reduction": float(np.median(differences)),
                "bootstrap_mean_ci_lower": lower,
                "bootstrap_mean_ci_upper": upper,
                "sequence_wins": wins,
                "sequence_win_rate": float(wins / len(current)),
                "two_sided_sign_test_p": float(
                    binomtest(wins, len(current), 0.5).pvalue
                ),
            }
        )
    task_summary = pd.DataFrame(task_rows)
    overall = task_summary[task_summary["task"] == "all"].iloc[0]
    tail_summary, tail_steps = paired_tail_migration(
        rows, baseline, primary
    )
    overall_lower = float(overall["bootstrap_mean_ci_lower"])
    if overall_lower > 0.0:
        interpretation = (
            "The paired sequence bootstrap interval is positive on this run; "
            "external replication and the preregistered gate still determine scope."
        )
    elif float(overall["mean_exact_kl_reduction"]) > 0.0:
        interpretation = (
            "The mean paired reduction is positive, but its sequence-bootstrap "
            "interval crosses zero; the run does not establish a stable mean gain."
        )
    else:
        interpretation = (
            "The mean paired reduction is non-positive and the run does not "
            "support an average-risk improvement."
        )
    result = {
        "status": "paired_sequence_bootstrap_development_analysis",
        "confirmatory_evidence": False,
        "baseline": str(baseline),
        "primary": str(primary),
        "bootstrap_samples": int(bootstrap_samples),
        "seed": int(seed),
        "sequence_count": int(len(sequence)),
        "mean_exact_kl_reduction": float(overall["mean_exact_kl_reduction"]),
        "median_exact_kl_reduction": float(
            overall["median_exact_kl_reduction"]
        ),
        "bootstrap_mean_ci_95": [
            float(overall["bootstrap_mean_ci_lower"]),
            float(overall["bootstrap_mean_ci_upper"]),
        ],
        "sequence_wins": int(overall["sequence_wins"]),
        "sequence_win_rate": float(overall["sequence_win_rate"]),
        "two_sided_sign_test_p": float(overall["two_sided_sign_test_p"]),
        "tail_migration": tail_summary,
        "interpretation": interpretation,
    }
    atomic_frame(sequence, analysis_root / "sequence_differences.csv")
    atomic_frame(task_summary, analysis_root / "task_uncertainty.csv")
    atomic_frame(tail_steps, analysis_root / "tail_step_diagnostics.csv")
    atomic_json(analysis_root / "tail_migration.json", tail_summary)
    atomic_json(analysis_root / "uncertainty.json", result)
    return result


__all__ = [
    "analyze_direct_policy_replay",
    "paired_bootstrap_mean",
    "paired_tail_migration",
    "select_direct_policy_candidate",
    "select_protected_rescue_candidate",
]
