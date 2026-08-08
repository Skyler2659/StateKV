#!/usr/bin/env python
"""Evaluate the frozen P18 temporal-volatility free-generation gate."""
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

BASELINE = "latest_attention_shared"
CANDIDATE = "temporal_volatility_shared"
REFERENCE = "full"
METHODS = (REFERENCE, BASELINE, CANDIDATE)
PAIR_METRICS = (
    "official_score",
    "mean_nll",
    "ppl",
    "answer_f1",
    "exact_match",
    "contains_ground_truth",
    "end_to_end_decode_tokens_per_second",
    "peak_memory_bytes",
    "prefill_time_s",
    "decode_time_s",
    "eviction_time_s",
    "score_time_s",
    "topk_time_s",
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
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not an object")
                rows.append(value)
    return rows


def _finite_mean(values: Iterable[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    if numeric.isna().any():
        raise ValueError("required metric contains a missing or non-numeric value")
    return float(numeric.mean())


def _metric_mean(frame: pd.DataFrame, metric: str) -> float:
    return _finite_mean(frame[metric].tolist())


def _relative_delta(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return (candidate - baseline) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stages/temporal_volatility_freegen_protocol.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/temporal_cache_discovery/"
            "statekv_temporal_volatility_freegen_p18_v1"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    source_files = []
    runner_errors = 0
    for workload in config["workloads"]:
        run_dir = Path(workload["run_dir"])
        results_path = run_dir / "results.jsonl"
        summary_path = run_dir / "summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete workload output: {run_dir}")
        rows = _read_jsonl(results_path)
        frame = pd.DataFrame(rows)
        frame = frame.loc[frame["method"].isin(METHODS)].copy()
        workload_name = str(workload["sample_ids"][0]).split(":", 1)[0]
        frame["workload"] = workload_name
        expected_samples = len(workload["sample_ids"])
        for method in METHODS:
            method_frame = frame.loc[frame["method"] == method]
            if len(method_frame) != expected_samples:
                raise ValueError(
                    f"{workload_name}/{method}: expected {expected_samples} rows, "
                    f"found {len(method_frame)}"
                )
            if method_frame["sample_idx"].duplicated().any():
                raise ValueError(f"duplicate sample_idx for {workload_name}/{method}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runner_errors += int(summary.get("num_errors", 0))
        all_frames.append(frame)
        source_files.extend((results_path, summary_path, run_dir / "config.yaml"))

    results = pd.concat(all_frames, ignore_index=True)
    missing_metrics = [metric for metric in PAIR_METRICS if metric not in results]
    if missing_metrics:
        raise ValueError(f"results are missing metrics: {missing_metrics}")

    paired_rows = []
    for workload_name, workload_frame in results.groupby("workload", sort=False):
        baseline = workload_frame.loc[workload_frame["method"] == BASELINE]
        candidate = workload_frame.loc[workload_frame["method"] == CANDIDATE]
        joined = baseline.merge(
            candidate,
            on="sample_idx",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        for row in joined.to_dict(orient="records"):
            output = {
                "workload": workload_name,
                "sample_idx": int(row["sample_idx"]),
            }
            for metric in PAIR_METRICS:
                baseline_value = row[f"{metric}_baseline"]
                candidate_value = row[f"{metric}_candidate"]
                output[f"baseline_{metric}"] = baseline_value
                output[f"candidate_{metric}"] = candidate_value
                output[f"delta_{metric}"] = candidate_value - baseline_value
            paired_rows.append(output)
    paired = pd.DataFrame(paired_rows)

    aggregate_rows = []
    for (workload_name, method), frame in results.groupby(
        ["workload", "method"], sort=False
    ):
        aggregate_rows.append(
            {
                "workload": workload_name,
                "method": method,
                "n": len(frame),
                **{f"mean_{metric}": _metric_mean(frame, metric) for metric in PAIR_METRICS},
                "max_peak_memory_bytes": int(frame["peak_memory_bytes"].max()),
                "max_max_kv_len_observed": int(frame["max_kv_len_observed"].max()),
                "sum_attention_hook_errors": int(frame["attention_hook_errors"].sum()),
            }
        )
    aggregates = pd.DataFrame(aggregate_rows)

    baseline_all = results.loc[results["method"] == BASELINE]
    candidate_all = results.loc[results["method"] == CANDIDATE]
    reference_all = results.loc[results["method"] == REFERENCE]
    official_gates = {}
    for workload_name in results["workload"].drop_duplicates():
        baseline_workload = baseline_all.loc[baseline_all["workload"] == workload_name]
        candidate_workload = candidate_all.loc[candidate_all["workload"] == workload_name]
        baseline_score = _metric_mean(baseline_workload, "official_score")
        candidate_score = _metric_mean(candidate_workload, "official_score")
        official_gates[workload_name] = {
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
            "pass": candidate_score >= baseline_score,
        }

    baseline_nll = _metric_mean(baseline_all, "mean_nll")
    candidate_nll = _metric_mean(candidate_all, "mean_nll")
    baseline_throughput = _metric_mean(
        baseline_all, "end_to_end_decode_tokens_per_second"
    )
    candidate_throughput = _metric_mean(
        candidate_all, "end_to_end_decode_tokens_per_second"
    )
    baseline_peak_memory = int(baseline_all["peak_memory_bytes"].max())
    candidate_peak_memory = int(candidate_all["peak_memory_bytes"].max())
    hook_errors = int(
        baseline_all["attention_hook_errors"].sum()
        + candidate_all["attention_hook_errors"].sum()
    )
    cache_budget = int(config["frozen_policy"]["total_budget"])
    maximum_observed_cache = int(
        max(
            baseline_all["max_kv_len_observed"].max(),
            candidate_all["max_kv_len_observed"].max(),
        )
    )
    gates = {
        "each_task_mean_official_score_nonworse": {
            "by_workload": official_gates,
            "pass": all(item["pass"] for item in official_gates.values()),
        },
        "overall_mean_teacher_forced_nll_nonworse": {
            "baseline": baseline_nll,
            "candidate": candidate_nll,
            "delta": candidate_nll - baseline_nll,
            "pass": candidate_nll <= baseline_nll,
        },
        "overall_end_to_end_decode_throughput_at_least_90_percent_of_baseline": {
            "baseline_tokens_per_second": baseline_throughput,
            "candidate_tokens_per_second": candidate_throughput,
            "ratio": candidate_throughput / baseline_throughput,
            "threshold": 0.90,
            "pass": candidate_throughput >= 0.90 * baseline_throughput,
        },
        "overall_peak_memory_no_more_than_105_percent_of_baseline": {
            "baseline_max_bytes": baseline_peak_memory,
            "candidate_max_bytes": candidate_peak_memory,
            "ratio": candidate_peak_memory / baseline_peak_memory,
            "threshold": 1.05,
            "pass": candidate_peak_memory <= 1.05 * baseline_peak_memory,
        },
        "no_attention_hook_errors": {
            "matched_methods_hook_errors": hook_errors,
            "runner_errors": runner_errors,
            "pass": hook_errors == 0 and runner_errors == 0,
        },
        "cache_budget_respected": {
            "budget": cache_budget,
            "maximum_observed": maximum_observed_cache,
            "pass": maximum_observed_cache <= cache_budget,
        },
    }
    primary_gate_pass = all(item["pass"] for item in gates.values())
    summary = {
        "experiment": config["experiment_name"],
        "status": "completed",
        "candidate": CANDIDATE,
        "matched_baseline": BASELINE,
        "upper_reference": REFERENCE,
        "sample_count_per_method": len(candidate_all),
        "primary_gate_pass": primary_gate_pass,
        "gates": gates,
        "paired_secondary": {
            "official_score_wins": int((paired["delta_official_score"] > 0).sum()),
            "official_score_ties": int((paired["delta_official_score"] == 0).sum()),
            "teacher_forced_nll_wins": int((paired["delta_mean_nll"] < 0).sum()),
            "teacher_forced_nll_ties": int((paired["delta_mean_nll"] == 0).sum()),
            "mean_answer_f1_delta": float(paired["delta_answer_f1"].mean()),
            "mean_official_score_delta": float(paired["delta_official_score"].mean()),
            "mean_nll_delta": float(paired["delta_mean_nll"].mean()),
            "throughput_relative_delta": _relative_delta(
                candidate_throughput, baseline_throughput
            ),
        },
        "upper_reference_diagnostic_not_a_primary_gate": {
            "reference_max_peak_memory_bytes": int(
                reference_all["peak_memory_bytes"].max()
            ),
            "candidate_max_peak_memory_bytes": candidate_peak_memory,
            "candidate_to_reference_peak_memory_ratio": (
                candidate_peak_memory
                / int(reference_all["peak_memory_bytes"].max())
            ),
            "reference_mean_prefill_time_s": _metric_mean(
                reference_all, "prefill_time_s"
            ),
            "candidate_mean_prefill_time_s": _metric_mean(
                candidate_all, "prefill_time_s"
            ),
            "warning": (
                "The matched comparison is fair, but current MLX attention "
                "capture has a large absolute 16K prefill memory footprint; "
                "the policy is not yet a low-memory deployment path."
            ),
        },
        "interpretation_scope": {
            "frozen_before_execution": True,
            "govreport": "three 700-word project-workload samples",
            "ruler": "three official 16K NIAH samples",
            "excluded": config["claims_excluded"],
            "statistical_note": (
                "Six paired samples are a decision gate for the next experiment, "
                "not evidence of population-level superiority."
            ),
        },
        "source_files": {
            str(path): _sha256(path) for path in source_files if path.is_file()
        },
    }

    atomic_frame(paired, output_dir / "paired_results.csv")
    atomic_frame(aggregates, output_dir / "workload_metrics.csv")
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
