"""Qwen closed-loop comparison for cheap StateKV controller paths."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from statekv.candidate_pullback import CandidatePullbackRunner
from statekv.cheap_policy import (
    CHEAP_POLICIES,
    CheapPolicyContext,
    HistoricalCandidateRanker,
)
from statekv.config import load_discovery_config
from statekv.oracle_policy_freegen import (
    _paired_bootstrap_interval,
    _run_free_policy,
)
from statekv.storage import atomic_frame, atomic_json, atomic_text
from statekv.tasks import load_discovery_tasks


def _aggregate(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for policy, current in frame.groupby("policy", sort=True):
        gov = current["task_bucket"] == "GovReport"
        niah = current["task_bucket"] == "NIAH"
        rows.append(
            {
                "policy": str(policy),
                "samples": int(len(current)),
                "mean_trajectory_exact_kl": float(
                    current["mean_trajectory_exact_kl"].mean()
                ),
                "mean_official_score": float(current["official_score"].mean()),
                "mean_govreport_rouge_l": (
                    float(current.loc[gov, "rouge_l"].mean())
                    if gov.any()
                    else None
                ),
                "mean_niah_retrieval": (
                    float(current.loc[niah, "needle_retrieval_accuracy"].mean())
                    if niah.any()
                    else None
                ),
            }
        )
    return rows


def _comparison_rows(
    new_frame: pd.DataFrame,
    old_frame: pd.DataFrame,
    seed: int,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    old_policies = (
        "full_cache",
        "statekv_exact_mean",
        "attention",
        "snapkv",
        "h2o",
    )
    for new_index, new_policy in enumerate(CHEAP_POLICIES):
        current = new_frame.loc[
            new_frame["policy"] == new_policy,
            ["sample_id", "official_score", "mean_trajectory_exact_kl"],
        ].set_index("sample_id")
        for old_index, old_policy in enumerate(old_policies):
            baseline = old_frame.loc[
                old_frame["policy"] == old_policy,
                ["sample_id", "official_score", "mean_trajectory_exact_kl"],
            ].set_index("sample_id")
            paired = current.join(
                baseline,
                how="inner",
                lsuffix="_new",
                rsuffix="_old",
            )
            if len(paired) != len(current):
                raise RuntimeError("new and reused sample sets are not paired")
            quality = (
                paired["official_score_new"]
                - paired["official_score_old"]
            ).to_numpy(np.float64)
            kl = (
                paired["mean_trajectory_exact_kl_new"]
                - paired["mean_trajectory_exact_kl_old"]
            ).to_numpy(np.float64)
            quality_ci = _paired_bootstrap_interval(
                quality,
                int(seed) + 10 * new_index + old_index,
                bootstrap_samples,
            )
            rows.append(
                {
                    "policy": new_policy,
                    "baseline": old_policy,
                    "paired_samples": int(len(paired)),
                    "mean_official_score_delta": float(np.mean(quality)),
                    "official_score_delta_ci95_low": quality_ci[0],
                    "official_score_delta_ci95_high": quality_ci[1],
                    "official_score_wins": int(np.sum(quality > 1.0e-12)),
                    "official_score_ties": int(np.sum(np.abs(quality) <= 1.0e-12)),
                    "official_score_losses": int(np.sum(quality < -1.0e-12)),
                    "mean_exact_kl_delta": float(np.mean(kl)),
                    "exact_kl_wins": int(np.sum(kl < -1.0e-12)),
                    "exact_kl_ties": int(np.sum(np.abs(kl) <= 1.0e-12)),
                    "exact_kl_losses": int(np.sum(kl > 1.0e-12)),
                }
            )
    return rows


def _report(
    aggregates: Sequence[Dict[str, Any]],
    comparisons: pd.DataFrame,
    elapsed_s: float,
    old_run: str,
) -> str:
    header = (
        "| 方法 | 来源 | 平均 KL ↓ | 平均任务分 ↑ | GovReport ROUGE-L ↑ | "
        "NIAH 准确率 ↑ |\n|---|---|---:|---:|---:|---:|"
    )
    table = [header]
    for row in sorted(
        aggregates, key=lambda value: float(value["mean_trajectory_exact_kl"])
    ):
        table.append(
            "| {policy} | {source} | {kl:.6f} | {score:.4f} | {gov:.6f} | {niah:.3f} |".format(
                policy=row["policy"],
                source=row["source"],
                kl=float(row["mean_trajectory_exact_kl"]),
                score=float(row["mean_official_score"]),
                gov=float(row["mean_govreport_rouge_l"]),
                niah=float(row["mean_niah_retrieval"]),
            )
        )
    statekv = comparisons[comparisons["baseline"] == "statekv_exact_mean"]
    lines = [
        "# A1–B3 Qwen3-8B 闭环实验",
        "",
        "本次只运行 A1–B3。Full KV、StateKV exact teacher、Attention、SnapKV、H2O 的数值直接复用 "
        f"`{old_run}`，没有重新运行旧策略。每个新策略仍逐 token 用 Full KV 计算评测 KL；这部分是评测仪器，不属于控制器部署开销。",
        "",
        *table,
        "",
        "## 公平性与实现边界",
        "",
        "A1/A3/A4/B1 只排序既有的七个合法动作；A2/B2/B3 直接生成物理缓存集合。所有新控制器的候选模型 rollout 数均为 0。B3 满足",
        "",
        r"$$\sum_{l=1}^{L}|C_l|=L\,B_{\mathrm{core}},$$",
        "",
        "因此它只在层间重分配缓存，而没有增加总 KV 数。A3 的集合扰动分数使用删除整个集合后的注意力输出变化，而不是把单 token 分数简单相加。",
        "",
        "## 相对昂贵教师",
        "",
    ]
    for row in statekv.sort_values("mean_exact_kl_delta").itertuples(index=False):
        lines.append(
            "- {policy}: KL 差 {kl:+.6f}，任务分差 {score:+.4f}，任务样本胜/平/负 {wins}/{ties}/{losses}。".format(
                policy=row.policy,
                kl=float(row.mean_exact_kl_delta),
                score=float(row.mean_official_score_delta),
                wins=int(row.official_score_wins),
                ties=int(row.official_score_ties),
                losses=int(row.official_score_losses),
            )
        )
    lines.extend(
        [
            "",
            "完整实验收集耗时（包含 Full-KV KL 评测仪器）为 %.1f 秒。" % elapsed_s,
            "",
        ]
    )
    return "\n".join(lines)


def run_cheap_policy_freegen(config_path: Path, repository_root: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(repository_root / str(config["base_config"])))
    cfg.experiment_name = str(config["experiment_name"])
    for key, value in dict(config.get("model_overrides") or {}).items():
        if not hasattr(cfg.model, str(key)):
            raise ValueError("unknown model override: %s" % key)
        setattr(cfg.model, str(key), value)
    cfg.tasks = dict(config["task_overrides"])
    cfg.runtime.seed = int(config["data_seed"])
    cfg.runtime.run_id = str(config["runtime_run_id"])
    sample_ids = set(str(value) for value in config["sample_ids"])
    expected_sample_count = int(config["expected_sample_count"])
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
    policies = tuple(str(value) for value in config["policies"])
    if policies != CHEAP_POLICIES:
        raise ValueError("cheap policy protocol must run A1 through B3 in order")
    candidates = tuple(str(value) for value in config["candidate_panel"])
    historical_path = repository_root / str(config["b1_historical_trace"])
    ranker = HistoricalCandidateRanker.fit(
        historical_path, ridge=float(config.get("b1_ridge", 1.0e-3))
    )
    context = CheapPolicyContext(
        core_budget=core_budget,
        sink_size=sink_size,
        recent_size=recent_size,
        pooling_kernel=int(config["snapkv_pooling_kernel"]),
        pooling_method=str(config["snapkv_pooling"]),
        cascade_margin=float(config["a4_cascade_margin"]),
        adaptive_budget_delta=int(config["b3_adaptive_budget_delta"]),
        output_diagnostic_layers=tuple(
            int(value) for value in config["a3_output_diagnostic_layers"]
        ),
        ranker=ranker,
    )
    old_run = repository_root / str(config["reused_baseline_run"])
    old_summary = json.loads((old_run / "summary.json").read_text(encoding="utf-8"))
    old_frame = pd.read_csv(old_run / "sample_results.csv")
    if set(str(value) for value in old_summary["samples"]) != sample_ids:
        raise RuntimeError("reused baseline sample set differs from cheap run")
    if str(old_summary["model_info"].get("model_name")) != str(cfg.model.name):
        raise RuntimeError("reused baseline model differs from cheap run")
    output_root = repository_root / str(config["output_run"])
    output_root.mkdir(parents=True, exist_ok=True)
    runner = CandidatePullbackRunner(cfg, repository_root)
    samples, _ = load_discovery_tasks(cfg)
    selected_samples = [
        sample for sample in samples if str(sample.sample_id) in sample_ids
    ]
    if len(selected_samples) != expected_sample_count:
        raise RuntimeError("configured cheap-policy samples were not loaded")
    cycle_rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    started = time.perf_counter()
    model_info = runner.model.load()
    print(
        "[cheap-freegen] loaded %s; old baselines will not be rerun"
        % model_info.get("model_name"),
        flush=True,
    )
    try:
        for sample_index, sample in enumerate(selected_samples, start=1):
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            try:
                prompt_tokens = int(len(reference.prompt_token_ids))
                print(
                    "[cheap-freegen] sample %d/%d %s prompt_tokens=%d"
                    % (
                        sample_index,
                        len(selected_samples),
                        sample.sample_id,
                        prompt_tokens,
                    ),
                    flush=True,
                )
                for policy in policies:
                    current_cycles, current_steps, summary = _run_free_policy(
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
                        cheap_policy_context=context,
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
                    cycle_rows.extend({**base, **row} for row in current_cycles)
                    step_rows.extend({**base, **row} for row in current_steps)
                    summaries.append(summary)
                    print(
                        "[cheap-freegen] sample %d/%d policy=%s kl=%.6f score=%.4f"
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
                    pd.DataFrame(summaries), output_root / "partial_sample_results.csv"
                )
                atomic_frame(
                    pd.DataFrame(cycle_rows), output_root / "partial_cycle_rows.parquet"
                )
                atomic_frame(
                    pd.DataFrame(step_rows), output_root / "partial_step_rows.parquet"
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
    new_frame = pd.DataFrame(summaries)
    new_aggregates = _aggregate(new_frame)
    old_aggregates = [
        {**row, "source": "reused_p31"}
        for row in old_summary["policy_aggregates"]
    ]
    all_aggregates = old_aggregates + [
        {**row, "source": "new_a1_b3"} for row in new_aggregates
    ]
    comparison_rows = _comparison_rows(
        new_frame,
        old_frame,
        int(config["data_seed"]),
        int(config.get("bootstrap_samples", 20000)),
    )
    comparison_frame = pd.DataFrame(comparison_rows)
    cycles_frame = pd.DataFrame(cycle_rows)
    elapsed = float(time.perf_counter() - started)
    result = {
        "experiment": str(config["experiment_name"]),
        "status": "cheap_controller_physical_closed_loop_generation",
        "samples": sorted(sample_ids),
        "policies": list(policies),
        "model_info": model_info,
        "control_cycles": cycles,
        "control_horizon": horizon,
        "nominal_total_budget_per_layer": total_budget,
        "candidate_model_rollouts_per_decision": 0,
        "reused_baseline_run": str(config["reused_baseline_run"]),
        "reused_baselines_rerun": False,
        "b1_ranker": ranker.metadata(),
        "policy_aggregates": all_aggregates,
        "sample_results": summaries,
        "comparisons": comparison_rows,
        "selection_time_s_total": float(cycles_frame["selection_time_s"].sum()),
        "selection_time_ms_mean": float(
            1000.0 * cycles_frame["selection_time_s"].mean()
        ),
        "all_budgets_respected": bool(
            new_frame["all_budgets_respected"].all()
        ),
        "b3_global_core_budget_exact": bool(
            cycles_frame.loc[
                cycles_frame["policy"] == "b3_layer_adaptive_budget",
                "selected_core_tokens_total",
            ].eq(int(model_info["num_layers"]) * core_budget).all()
        ),
        "collection_elapsed_s": elapsed,
        "execution_valid": bool(
            len(new_frame) == expected_sample_count * len(policies)
            and new_frame["all_budgets_respected"].all()
        ),
    }
    atomic_frame(cycles_frame, output_root / "cycle_rows.parquet")
    atomic_frame(pd.DataFrame(step_rows), output_root / "step_rows.parquet")
    atomic_frame(new_frame, output_root / "sample_results.csv")
    atomic_frame(pd.DataFrame(all_aggregates), output_root / "comparison_table.csv")
    atomic_frame(comparison_frame, output_root / "paired_comparisons.csv")
    atomic_json(output_root / "summary.json", result)
    atomic_text(
        output_root / "report.md",
        _report(
            all_aggregates,
            comparison_frame,
            elapsed,
            str(config["reused_baseline_run"]),
        ),
    )
    atomic_text(
        output_root / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    return output_root


__all__ = ["run_cheap_policy_freegen"]
