#!/usr/bin/env python
"""Evaluate the preregistered P21 token-rarity independent replication."""
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

CANDIDATE = "token_rarity_shared"
BASELINE = "latest_attention_shared"
CONTROL = "position_coverage_shared"
REFERENCE = "full"
METHODS = (REFERENCE, BASELINE, CONTROL, CANDIDATE)
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
        default="configs/stages/token_rarity_replication_protocol.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/temporal_cache_discovery/"
            "statekv_token_rarity_replication_p21_v1"
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    sources = []
    runner_errors = 0
    for workload in config["workloads"]:
        run_dir = Path(workload["run_dir"])
        results_path = run_dir / "results.jsonl"
        summary_path = run_dir / "summary.json"
        if not results_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"incomplete workload output: {run_dir}")
        frame = pd.DataFrame(_read_jsonl(results_path))
        frame = frame.loc[frame["method"].isin(METHODS)].copy()
        if "error" in frame and frame["error"].notna().any():
            errors = frame.loc[frame["error"].notna(), ["method", "error"]]
            raise RuntimeError(f"workload contains failed rows: {errors.to_dict('records')}")
        workload_name = str(workload["sample_ids"][0]).split(":", 1)[0]
        frame["workload"] = workload_name
        expected = len(workload["sample_ids"])
        for method in METHODS:
            selected = frame.loc[frame["method"] == method]
            if len(selected) != expected:
                raise ValueError(
                    f"{workload_name}/{method}: expected {expected}, found {len(selected)}"
                )
            if selected["sample_idx"].duplicated().any():
                raise ValueError(f"duplicate sample_idx: {workload_name}/{method}")
        runner_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runner_errors += int(runner_summary.get("num_errors", 0))
        frames.append(frame)
        sources.extend((results_path, summary_path, run_dir / "config.yaml"))
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

    baseline = results.loc[results["method"] == BASELINE]
    candidate = results.loc[results["method"] == CANDIDATE]
    control = results.loc[results["method"] == CONTROL]
    reference = results.loc[results["method"] == REFERENCE]
    paired_rows = []
    for workload in results["workload"].drop_duplicates():
        baseline_task = baseline.loc[baseline["workload"] == workload]
        candidate_task = candidate.loc[candidate["workload"] == workload]
        joined = baseline_task.merge(
            candidate_task,
            on="sample_idx",
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        for row in joined.to_dict(orient="records"):
            output = {"workload": workload, "sample_idx": int(row["sample_idx"])}
            for metric in METRICS:
                baseline_value = row[f"{metric}_baseline"]
                candidate_value = row[f"{metric}_candidate"]
                output[f"baseline_{metric}"] = baseline_value
                output[f"candidate_{metric}"] = candidate_value
                output[f"delta_{metric}"] = candidate_value - baseline_value
            paired_rows.append(output)
    paired = pd.DataFrame(paired_rows)

    official_by_workload = {}
    for workload in results["workload"].drop_duplicates():
        baseline_score = _mean(
            baseline.loc[baseline["workload"] == workload, "official_score"]
        )
        candidate_score = _mean(
            candidate.loc[candidate["workload"] == workload, "official_score"]
        )
        official_by_workload[workload] = {
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
            "pass": candidate_score >= baseline_score,
        }
    baseline_nll = _mean(baseline["mean_nll"])
    candidate_nll = _mean(candidate["mean_nll"])
    baseline_throughput = _mean(
        baseline["end_to_end_decode_tokens_per_second"]
    )
    candidate_throughput = _mean(
        candidate["end_to_end_decode_tokens_per_second"]
    )
    baseline_peak = int(baseline["peak_memory_bytes"].max())
    candidate_peak = int(candidate["peak_memory_bytes"].max())
    hook_errors = int(
        baseline["attention_hook_errors"].sum()
        + candidate["attention_hook_errors"].sum()
    )
    budget = int(config["frozen_candidate"]["total_budget"])
    maximum_cache = int(
        max(
            baseline["max_kv_len_observed"].max(),
            candidate["max_kv_len_observed"].max(),
        )
    )
    gates = {
        "each_task_mean_official_score_nonworse_than_latest_attention": {
            "by_workload": official_by_workload,
            "pass": all(item["pass"] for item in official_by_workload.values()),
        },
        "overall_mean_teacher_forced_nll_nonworse_than_latest_attention": {
            "baseline": baseline_nll,
            "candidate": candidate_nll,
            "delta": candidate_nll - baseline_nll,
            "pass": candidate_nll <= baseline_nll,
        },
        "decode_throughput_nonworse_than_latest_attention": {
            "baseline": baseline_throughput,
            "candidate": candidate_throughput,
            "ratio": candidate_throughput / baseline_throughput,
            "pass": candidate_throughput >= baseline_throughput,
        },
        "maximum_peak_memory_lower_than_latest_attention": {
            "baseline_bytes": baseline_peak,
            "candidate_bytes": candidate_peak,
            "ratio": candidate_peak / baseline_peak,
            "pass": candidate_peak < baseline_peak,
        },
        "zero_attention_hook_errors": {
            "matched_method_hook_errors": hook_errors,
            "runner_errors": runner_errors,
            "pass": hook_errors == 0 and runner_errors == 0,
        },
        "cache_budget_respected": {
            "budget": budget,
            "maximum_observed": maximum_cache,
            "pass": maximum_cache <= budget,
        },
    }
    primary_pass = all(item["pass"] for item in gates.values())
    summary = {
        "experiment": config["experiment_name"],
        "status": "completed_independent_replication",
        "candidate": CANDIDATE,
        "matched_strong_baseline": BASELINE,
        "matched_low_cost_control": CONTROL,
        "upper_reference": REFERENCE,
        "samples_per_method": len(candidate),
        "primary_gate_pass": primary_pass,
        "gates": gates,
        "paired_secondary": {
            "official_score_wins": int((paired["delta_official_score"] > 0).sum()),
            "official_score_ties": int((paired["delta_official_score"] == 0).sum()),
            "nll_wins": int((paired["delta_mean_nll"] < 0).sum()),
            "nll_ties": int((paired["delta_mean_nll"] == 0).sum()),
            "mean_official_score_delta": float(paired["delta_official_score"].mean()),
            "mean_nll_delta": float(paired["delta_mean_nll"].mean()),
        },
        "control_diagnostic": {
            "position_coverage_overall_mean_nll": _mean(control["mean_nll"]),
            "position_coverage_mean_throughput": _mean(
                control["end_to_end_decode_tokens_per_second"]
            ),
            "full_overall_mean_nll": _mean(reference["mean_nll"]),
            "full_max_peak_memory_bytes": int(reference["peak_memory_bytes"].max()),
        },
        "scope": {
            "independent_samples": True,
            "claims_excluded": config["claims_excluded"],
            "note": (
                "Six paired samples test replication of the development signal; "
                "they do not establish population-level superiority."
            ),
        },
        "source_files": {
            str(path): _sha256(path) for path in sources if path.is_file()
        },
    }
    atomic_frame(aggregates, output_dir / "workload_metrics.csv")
    atomic_frame(paired, output_dir / "paired_results.csv")
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
