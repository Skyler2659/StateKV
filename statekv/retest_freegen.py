"""No-gate multi-policy re-evaluation runner (retrospective retest program).

This runner re-tests policies that earlier phases rejected through
preregistered joint gates.  It deliberately contains **no pass/fail logic**:
the protocol declares task scores as the primary endpoint and KL/NLL as
diagnostics, and every result is reported continuously (point estimate,
paired bootstrap CI, win/tie/loss) against every other policy in the panel.

Semantics match the recoverable (CPU backing store) line: every policy runs
through ``_run_free_policy`` with the shared recoverable backing pool.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.cheap_policy import (
    CHEAP_POLICIES,
    CheapPolicyContext,
    HistoricalCandidateRanker,
)
from statekv.config import apply_named_overrides, load_discovery_config
from statekv.oracle_policy_freegen import (
    _check_prompt_truncation,
    _metric_row,
    _paired_bootstrap_interval,
    _run_free_policy,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks

TIERED_POLICY = "qk_tiered_v"


def classify_policies(policies: Sequence[str]) -> Dict[str, List[str]]:
    """Split the policy list into cheap / tiered / plain-panel groups."""
    groups: Dict[str, List[str]] = {"cheap": [], "tiered": [], "panel": []}
    for policy in policies:
        name = str(policy)
        if name in CHEAP_POLICIES:
            groups["cheap"].append(name)
        elif name == TIERED_POLICY:
            groups["tiered"].append(name)
        else:
            groups["panel"].append(name)
    return groups


def _bucket_means(current: pd.DataFrame) -> Dict[str, Optional[float]]:
    gov = current["task_bucket"] == "GovReport"
    niah = current["task_bucket"] == "NIAH"
    reasoning = current["task_bucket"] == "Reasoning"
    return {
        "mean_govreport_rouge_l": (
            float(current.loc[gov, "rouge_l"].mean()) if gov.any() else None
        ),
        "mean_govreport_official": (
            float(current.loc[gov, "official_score"].mean()) if gov.any() else None
        ),
        "mean_niah_retrieval": (
            float(current.loc[niah, "needle_retrieval_accuracy"].mean())
            if niah.any()
            else None
        ),
        "mean_reasoning_official": (
            float(current.loc[reasoning, "official_score"].mean())
            if reasoning.any()
            else None
        ),
    }


def _aggregates(
    frame: pd.DataFrame, step_frame: pd.DataFrame
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    nll = (
        step_frame.groupby("policy")["delta_nll"].mean()
        if not step_frame.empty
        else pd.Series(dtype=np.float64)
    )
    for policy, current in frame.groupby("policy", sort=True):
        rows.append(
            {
                "policy": str(policy),
                "samples": int(len(current)),
                "mean_official_score": float(current["official_score"].mean()),
                "mean_trajectory_exact_kl": float(
                    current["mean_trajectory_exact_kl"].mean()
                ),
                "mean_delta_nll": (
                    float(nll[policy]) if policy in nll.index else None
                ),
                "mean_repetition_4gram_rate": float(
                    current["repetition_4gram_rate"].mean()
                ),
                **_bucket_means(current),
            }
        )
    return rows


def _paired_metric(
    frame: pd.DataFrame,
    policy: str,
    baseline: str,
    column: str,
    bucket: Optional[str],
    seed: int,
    bootstrap_samples: int,
) -> Optional[Dict[str, Any]]:
    left = frame.loc[frame["policy"] == policy]
    right = frame.loc[frame["policy"] == baseline]
    if bucket is not None:
        left = left.loc[left["task_bucket"] == bucket]
        right = right.loc[right["task_bucket"] == bucket]
    paired = left.set_index("sample_id")[[column]].join(
        right.set_index("sample_id")[[column]],
        how="inner",
        lsuffix="_policy",
        rsuffix="_baseline",
    ).dropna()
    if paired.empty:
        return None
    delta = (
        paired["%s_policy" % column] - paired["%s_baseline" % column]
    ).to_numpy(np.float64)
    ci = _paired_bootstrap_interval(delta, int(seed), int(bootstrap_samples))
    tolerance = 1.0e-12
    return {
        "policy": str(policy),
        "baseline": str(baseline),
        "metric": str(column),
        "task_bucket": bucket or "all",
        "paired_samples": int(len(paired)),
        "mean_delta_policy_minus_baseline": float(np.mean(delta)),
        "delta_ci95_low": float(ci[0]),
        "delta_ci95_high": float(ci[1]),
        "wins": int(np.sum(delta > tolerance)),
        "ties": int(np.sum(np.abs(delta) <= tolerance)),
        "losses": int(np.sum(delta < -tolerance)),
    }


def _paired_comparisons(
    frame: pd.DataFrame,
    seed: int,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    policies = sorted(str(policy) for policy in frame["policy"].unique())
    metrics: Tuple[Tuple[str, Optional[str]], ...] = (
        ("official_score", None),
        ("mean_trajectory_exact_kl", None),
        ("rouge_l", "GovReport"),
        ("needle_retrieval_accuracy", "NIAH"),
        ("official_score", "Reasoning"),
    )
    rows: List[Dict[str, Any]] = []
    pair_index = 0
    for index, policy in enumerate(policies):
        for baseline in policies[index + 1 :]:
            for column, bucket in metrics:
                row = _paired_metric(
                    frame,
                    policy,
                    baseline,
                    column,
                    bucket,
                    int(seed) + pair_index,
                    bootstrap_samples,
                )
                if row is not None:
                    rows.append(row)
            pair_index += 1
    return rows


def _report(
    aggregates: Sequence[Dict[str, Any]],
    comparisons: pd.DataFrame,
    references: Sequence[str],
    elapsed_s: float,
) -> str:
    lines = [
        "# StateKV gate-retest panel (no verdicts)",
        "",
        "Primary endpoint: task scores.  KL / delta-NLL are diagnostics.  "
        "All comparisons are continuous (point estimate + paired bootstrap "
        "CI + win/tie/loss); this report contains no pass/fail judgement.",
        "",
        "| policy | n | official ↑ | GovReport ROUGE-L ↑ | NIAH ↑ | "
        "Reasoning ↑ | mean KL ↓ | mean ΔNLL ↓ | repetition ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        aggregates,
        key=lambda value: -float(value["mean_official_score"]),
    ):
        def fmt(value: Any, digits: int = 4) -> str:
            return "—" if value is None else ("%%.%df" % digits) % float(value)

        lines.append(
            "| {policy} | {n} | {score} | {gov} | {niah} | {reasoning} | "
            "{kl} | {nll} | {rep} |".format(
                policy=row["policy"],
                n=int(row["samples"]),
                score=fmt(row["mean_official_score"]),
                gov=fmt(row["mean_govreport_rouge_l"], 3),
                niah=fmt(row["mean_niah_retrieval"], 3),
                reasoning=fmt(row["mean_reasoning_official"], 3),
                kl=fmt(row["mean_trajectory_exact_kl"], 6),
                nll=fmt(row["mean_delta_nll"], 5),
                rep=fmt(row["mean_repetition_4gram_rate"], 4),
            )
        )
    lines.extend(["", "## Paired comparisons vs references", ""])
    for reference in references:
        current = comparisons.loc[
            (comparisons["baseline"] == reference)
            & (comparisons["metric"] == "official_score")
            & (comparisons["task_bucket"] == "all")
        ]
        if current.empty:
            continue
        lines.extend(
            [
                "### vs %s" % reference,
                "",
                "| policy | Δ official | CI95 | wins/ties/losses |",
                "|---|---:|---|---|",
            ]
        )
        for row in current.sort_values(
            "mean_delta_policy_minus_baseline", ascending=False
        ).itertuples(index=False):
            lines.append(
                "| %s | %+.4f | [%+.4f, %+.4f] | %d/%d/%d |"
                % (
                    row.policy,
                    row.mean_delta_policy_minus_baseline,
                    row.delta_ci95_low,
                    row.delta_ci95_high,
                    row.wins,
                    row.ties,
                    row.losses,
                )
            )
        lines.append("")
    lines.append(
        "Full all-pairs results (all metrics, all buckets) live in "
        "`paired_comparisons.csv`.  Collection took %.1f s." % elapsed_s
    )
    return "\n".join(lines)


def run_retest_freegen(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        if not hasattr(cfg.model, str(key)):
            raise ValueError("unknown model override: %s" % key)
        setattr(cfg.model, str(key), value)
    apply_named_overrides(cfg.runtime, config.get("runtime_overrides"), "runtime")
    allow_prompt_truncation = bool(config.get("allow_prompt_truncation", False))
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(value) for value in config["sample_ids"])
    expected_sample_count = int(
        config.get("expected_sample_count", len(sample_ids))
    )
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    policies = [str(value) for value in config["policies"]]
    candidates = [str(value) for value in config["candidate_panel"]]
    groups = classify_policies(policies)
    missing = [name for name in groups["panel"] if name not in candidates]
    if missing:
        raise ValueError(
            "panel policies missing from candidate_panel: %s" % missing
        )
    if groups["tiered"] and "qk_pool" not in candidates:
        raise ValueError("qk_tiered_v requires qk_pool in candidate_panel")

    start_anchor = int(config["start_anchor"])
    cfg.anchor_steps = [start_anchor]
    cycles = int(config["control_cycles"])
    horizon = int(config["control_horizon"])
    total_budget = int(config["total_budget"])
    sink_size = int(config["sink_size"])
    recent_size = int(config["recent_size"])
    core_budget = int(config["core_budget"])
    cfg.cache.total_budget = total_budget
    cfg.cache.sink_size = sink_size
    cfg.cache.recent_size = recent_size
    cfg.cache.selected_core_budget = core_budget
    cfg.validate()

    context: Optional[CheapPolicyContext] = None
    if groups["cheap"]:
        ranker = None
        if "b1_historical_tiny_ranker" in groups["cheap"]:
            ranker = HistoricalCandidateRanker.fit(
                repository_root / str(config["b1_historical_trace"]),
                ridge=float(config.get("b1_ridge", 1.0e-3)),
            )
        context = CheapPolicyContext(
            core_budget=core_budget,
            sink_size=sink_size,
            recent_size=recent_size,
            pooling_kernel=int(config["snapkv_pooling_kernel"]),
            pooling_method=str(config["snapkv_pooling"]),
            cascade_margin=float(config.get("a4_cascade_margin", 0.15)),
            adaptive_budget_delta=int(config.get("b3_adaptive_budget_delta", 44)),
            output_diagnostic_layers=tuple(
                int(value)
                for value in config.get(
                    "a3_output_diagnostic_layers", [0, 9, 18, 27, 35]
                )
            ),
            ranker=ranker,
        )
    value_tier = config.get("value_tier")

    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if {str(sample.sample_id) for sample in selected_samples} != sample_ids:
        raise RuntimeError("configured retest samples were not loaded")
    if len(selected_samples) != expected_sample_count:
        raise RuntimeError(
            "expected %d retest samples, loaded %d"
            % (expected_sample_count, len(selected_samples))
        )

    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    model_info = runner.model.load()
    print(
        "[retest] loaded %s; policies=%d (cheap=%d tiered=%d panel=%d)"
        % (
            model_info.get("model_name"),
            len(policies),
            len(groups["cheap"]),
            len(groups["tiered"]),
            len(groups["panel"]),
        ),
        flush=True,
    )
    try:
        for sample_index, sample in enumerate(selected_samples, start=1):
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            _check_prompt_truncation(
                reference, str(sample.sample_id), allow_prompt_truncation
            )
            try:
                prompt_tokens = int(len(reference.prompt_token_ids))
                print(
                    "[retest] sample %d/%d %s prompt_tokens=%d"
                    % (
                        sample_index,
                        len(selected_samples),
                        sample.sample_id,
                        prompt_tokens,
                    ),
                    flush=True,
                )
                reference_tokens = [
                    int(value)
                    for value in reference.generated_token_ids[
                        : start_anchor + cycles * horizon
                    ]
                ]
                summaries.append(
                    {
                        "policy": "full_cache",
                        "cycles_completed": cycles,
                        "generated_tokens": len(reference_tokens),
                        "mean_trajectory_exact_kl": 0.0,
                        "refresh_events": 0,
                        "recovery_events": 0,
                        "all_budgets_respected": True,
                        "prompt_tokens": prompt_tokens,
                        "cache_budget": None,
                        "retained_prompt_fraction": 1.0,
                        **_metric_row(
                            runner, sample, "full_cache", reference_tokens, 0.0
                        ),
                    }
                )
                for policy in policies:
                    cycles_current, steps_current, summary = _run_free_policy(
                        runner,
                        reference,
                        sample,
                        policy,
                        candidates,
                        start_anchor,
                        cycles,
                        horizon,
                        total_budget,
                        sink_size,
                        recent_size,
                        core_budget,
                        int(config["snapkv_observation_window"]),
                        int(config["snapkv_pooling_kernel"]),
                        str(config["snapkv_pooling"]),
                        cheap_policy_context=(
                            context if policy in groups["cheap"] else None
                        ),
                        quest_page_size=int(config.get("quest_page_size", 16)),
                        value_tier=value_tier if policy in groups["tiered"] else None,
                        obswin_size=int(config.get("obswin_size", 32)),
                    )
                    summary["prompt_tokens"] = prompt_tokens
                    summary["cache_budget"] = total_budget
                    summary["retained_prompt_fraction"] = float(
                        min(1.0, total_budget / max(1, prompt_tokens))
                    )
                    base = {
                        "sample_id": str(sample.sample_id),
                        "task": str(sample.task),
                    }
                    cycle_rows.extend({**base, **row} for row in cycles_current)
                    step_rows.extend({**base, **row} for row in steps_current)
                    summaries.append(summary)
                    print(
                        "[retest] sample %d/%d policy=%s kl=%.6f score=%.4f"
                        % (
                            sample_index,
                            len(selected_samples),
                            policy,
                            float(summary["mean_trajectory_exact_kl"]),
                            float(summary["official_score"]),
                        ),
                        flush=True,
                    )
                atomic_frame(
                    pd.DataFrame(summaries),
                    output_root / "partial_sample_results.csv",
                )
                atomic_frame(
                    pd.DataFrame(cycle_rows),
                    output_root / "partial_cycle_rows.parquet",
                )
                atomic_frame(
                    pd.DataFrame(step_rows),
                    output_root / "partial_step_rows.parquet",
                )
                atomic_json(
                    output_root / "progress.json",
                    {
                        "completed_samples": sample_index,
                        "expected_samples": expected_sample_count,
                        "last_sample_id": str(sample.sample_id),
                        "elapsed_s": float(time.perf_counter() - started),
                    },
                )
            finally:
                runner.model.release(reference)
    finally:
        runner.model.close()

    summary_frame = pd.DataFrame(summaries)
    step_frame = pd.DataFrame(step_rows)
    aggregates = _aggregates(summary_frame, step_frame)
    comparison_rows = _paired_comparisons(
        summary_frame,
        int(config["data_seed"]),
        int(config.get("bootstrap_samples", 20000)),
    )
    comparison_frame = pd.DataFrame(comparison_rows)
    elapsed = float(time.perf_counter() - started)
    references = tuple(
        str(value)
        for value in config.get("report_references", ["attention", "qk_pool"])
    )
    result = {
        "experiment": str(config["experiment_name"]),
        "status": "no_gate_multi_policy_retrospective_retest",
        "analysis_contract": {
            "primary_endpoint": "task_scores",
            "diagnostics": ["mean_trajectory_exact_kl", "mean_delta_nll"],
            "reporting": "continuous_effect_sizes_with_paired_bootstrap_ci",
            "hard_gates": False,
            "semantics": "recoverable_shared_backing_pool",
        },
        "samples": sorted(sample_ids),
        "policies": policies,
        "candidate_panel": candidates,
        "control_cycles": cycles,
        "control_horizon": horizon,
        "total_budget": total_budget,
        "model_info": model_info,
        "sample_results": summaries,
        "policy_aggregates": aggregates,
        "paired_comparisons": comparison_rows,
        "all_budgets_respected": bool(
            summary_frame.loc[
                summary_frame["policy"] != "full_cache", "all_budgets_respected"
            ].all()
        ),
        "collection_elapsed_s": elapsed,
    }
    atomic_frame(pd.DataFrame(cycle_rows), output_root / "cycle_rows.parquet")
    atomic_frame(step_frame, output_root / "step_rows.parquet")
    atomic_frame(summary_frame, output_root / "sample_results.csv")
    atomic_frame(pd.DataFrame(aggregates), output_root / "policy_aggregates.csv")
    atomic_frame(comparison_frame, output_root / "paired_comparisons.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "report.md",
        _report(aggregates, comparison_frame, references, elapsed),
    )
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = ["classify_policies", "run_retest_freegen"]
