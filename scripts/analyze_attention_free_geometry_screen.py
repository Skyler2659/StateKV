#!/usr/bin/env python
"""Analyze the preregistered P19 attention-free geometry screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.storage import atomic_frame, atomic_json, atomic_text

CONTROL = "sink_recent_random"
REFERENCE = "full"
METRICS = (
    "official_score",
    "mean_nll",
    "ppl",
    "answer_f1",
    "contains_ground_truth",
    "end_to_end_decode_tokens_per_second",
    "peak_memory_bytes",
    "prefill_time_s",
    "decode_time_s",
    "eviction_time_s",
    "score_time_s",
    "max_kv_len_observed",
    "attention_hook_errors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _mean(values: Iterable[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if numeric.isna().any():
        raise ValueError("required metric contains missing/non-numeric values")
    return float(numeric.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/attention_free_geometry_screen_protocol.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/temporal_cache_discovery/"
            "statekv_attention_free_geometry_screen_p19_v1"
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = [str(item["name"]) for item in config["methods"]]
    candidates = [
        str(item["name"])
        for item in config["methods"]
        if item["role"] == "candidate"
    ]
    frames = []
    source_files = []
    runner_errors = 0
    for workload in config["workloads"]:
        run_dir = Path(workload["run_dir"])
        results_path = run_dir / "results.jsonl"
        summary_path = run_dir / "summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete workload output: {run_dir}")
        frame = pd.DataFrame(_read_jsonl(results_path))
        frame = frame.loc[frame["method"].isin(method_names)].copy()
        workload_name = str(workload["sample_ids"][0]).split(":", 1)[0]
        frame["workload"] = workload_name
        expected = len(workload["sample_ids"])
        for method in method_names:
            selected = frame.loc[frame["method"] == method]
            if len(selected) != expected:
                raise ValueError(
                    f"{workload_name}/{method}: expected {expected}, found {len(selected)}"
                )
            if selected["sample_idx"].duplicated().any():
                raise ValueError(f"duplicate sample_idx: {workload_name}/{method}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runner_errors += int(summary.get("num_errors", 0))
        frames.append(frame)
        source_files.extend((results_path, summary_path, run_dir / "config.yaml"))
    results = pd.concat(frames, ignore_index=True)
    missing = [metric for metric in METRICS if metric not in results]
    if missing:
        raise ValueError(f"results missing metrics: {missing}")

    aggregate_rows = []
    for (workload, method), frame in results.groupby(
        ["workload", "method"], sort=False
    ):
        aggregate_rows.append(
            {
                "workload": workload,
                "method": method,
                "n": len(frame),
                **{f"mean_{metric}": _mean(frame[metric]) for metric in METRICS},
                "max_peak_memory_bytes": int(frame["peak_memory_bytes"].max()),
                "max_max_kv_len_observed": int(frame["max_kv_len_observed"].max()),
                "sum_attention_hook_errors": int(frame["attention_hook_errors"].sum()),
            }
        )
    aggregates = pd.DataFrame(aggregate_rows)

    control = results.loc[results["method"] == CONTROL]
    reference = results.loc[results["method"] == REFERENCE]
    control_throughput = _mean(control["end_to_end_decode_tokens_per_second"])
    reference_peak = int(reference["peak_memory_bytes"].max())
    budget = int(config["fixed_budget"]["total_tokens_per_layer"])
    candidate_summaries = {}
    ranking_rows = []
    for candidate in candidates:
        frame = results.loc[results["method"] == candidate]
        quality_by_workload = {}
        strict_task_wins = 0
        task_losses = 0
        for workload in results["workload"].drop_duplicates():
            control_task = control.loc[control["workload"] == workload]
            candidate_task = frame.loc[frame["workload"] == workload]
            control_score = _mean(control_task["official_score"])
            candidate_score = _mean(candidate_task["official_score"])
            delta = candidate_score - control_score
            strict_task_wins += int(delta > 0.0)
            task_losses += int(delta < 0.0)
            quality_by_workload[workload] = {
                "control": control_score,
                "candidate": candidate_score,
                "delta": delta,
            }
        candidate_nll = _mean(frame["mean_nll"])
        control_nll = _mean(control["mean_nll"])
        candidate_throughput = _mean(
            frame["end_to_end_decode_tokens_per_second"]
        )
        candidate_peak = int(frame["peak_memory_bytes"].max())
        hook_errors = int(frame["attention_hook_errors"].sum())
        max_cache = int(frame["max_kv_len_observed"].max())
        checks = {
            "not_worse_on_both_task_quality_means": task_losses < 2,
            "throughput_at_least_90_percent_of_control": (
                candidate_throughput >= 0.90 * control_throughput
            ),
            "peak_memory_no_more_than_125_percent_of_full": (
                candidate_peak <= 1.25 * reference_peak
            ),
            "zero_attention_hook_and_runner_errors": (
                hook_errors == 0 and runner_errors == 0
            ),
            "cache_budget_respected": max_cache <= budget,
        }
        eligible = all(checks.values())
        candidate_summaries[candidate] = {
            "quality_by_workload": quality_by_workload,
            "strict_task_wins": strict_task_wins,
            "overall_mean_nll": candidate_nll,
            "control_overall_mean_nll": control_nll,
            "overall_mean_nll_delta": candidate_nll - control_nll,
            "decode_throughput_tokens_per_second": candidate_throughput,
            "throughput_ratio_vs_control": (
                candidate_throughput / control_throughput
            ),
            "peak_memory_bytes": candidate_peak,
            "peak_memory_ratio_vs_full": candidate_peak / reference_peak,
            "attention_hook_errors": hook_errors,
            "maximum_observed_cache": max_cache,
            "checks": checks,
            "eligible_for_replication": eligible,
        }
        ranking_rows.append(
            {
                "candidate": candidate,
                "eligible_for_replication": eligible,
                "strict_task_wins": strict_task_wins,
                "task_losses": task_losses,
                "overall_mean_nll": candidate_nll,
                "overall_mean_nll_delta_vs_control": candidate_nll - control_nll,
                "throughput_ratio_vs_control": (
                    candidate_throughput / control_throughput
                ),
                "peak_memory_ratio_vs_full": candidate_peak / reference_peak,
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        [
            "eligible_for_replication",
            "strict_task_wins",
            "overall_mean_nll",
            "throughput_ratio_vs_control",
        ],
        ascending=[False, False, True, False],
    )
    eligible = ranking.loc[ranking["eligible_for_replication"]]
    selected = None if eligible.empty else str(eligible.iloc[0]["candidate"])
    summary = {
        "experiment": config["experiment_name"],
        "status": "completed_development_screen",
        "attention_capture_required": False,
        "candidate_algorithms_run_per_deployment_decision": 0,
        "control": CONTROL,
        "upper_reference": REFERENCE,
        "samples_per_method": len(control),
        "runner_errors": runner_errors,
        "control_values": {
            "overall_mean_nll": _mean(control["mean_nll"]),
            "decode_throughput_tokens_per_second": control_throughput,
            "maximum_peak_memory_bytes": int(control["peak_memory_bytes"].max()),
        },
        "upper_reference_values": {
            "maximum_peak_memory_bytes": reference_peak,
            "overall_mean_nll": _mean(reference["mean_nll"]),
        },
        "candidates": candidate_summaries,
        "selected_for_independent_replication": selected,
        "screen_passed": selected is not None,
        "scope": {
            "development_only": True,
            "claims_excluded": config["claims_excluded"],
            "note": (
                "Four samples per method support route triage only, not a "
                "population-level superiority claim."
            ),
        },
        "source_files": {
            str(path): _sha256(path) for path in source_files if path.is_file()
        },
    }
    atomic_frame(aggregates, output_dir / "workload_metrics.csv")
    atomic_frame(ranking, output_dir / "candidate_ranking.csv")
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
