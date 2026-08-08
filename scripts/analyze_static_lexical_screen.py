#!/usr/bin/env python
"""Analyze the preregistered P20 attention-free static lexical screen."""
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

POSITION_CONTROL = "position_coverage_shared"
RANDOM_CONTROL = "sink_recent_random"
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
        default="configs/stages/static_lexical_screen_protocol.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/temporal_cache_discovery/"
            "statekv_static_lexical_screen_p20_v1"
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
        if "error" in frame and frame["error"].notna().any():
            errors = frame.loc[frame["error"].notna(), ["method", "error"]]
            raise RuntimeError(f"workload contains failed rows: {errors.to_dict('records')}")
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
        runner_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runner_errors += int(runner_summary.get("num_errors", 0))
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

    position = results.loc[results["method"] == POSITION_CONTROL]
    random = results.loc[results["method"] == RANDOM_CONTROL]
    reference = results.loc[results["method"] == REFERENCE]
    position_throughput = _mean(
        position["end_to_end_decode_tokens_per_second"]
    )
    reference_peak = int(reference["peak_memory_bytes"].max())
    budget = int(config["fixed_policy"]["total_budget"])
    candidate_summaries = {}
    ranking_rows = []
    for candidate in candidates:
        frame = results.loc[results["method"] == candidate]
        quality_by_workload = {}
        for workload in results["workload"].drop_duplicates():
            candidate_task = frame.loc[frame["workload"] == workload]
            position_task = position.loc[position["workload"] == workload]
            random_task = random.loc[random["workload"] == workload]
            candidate_score = _mean(candidate_task["official_score"])
            position_score = _mean(position_task["official_score"])
            random_score = _mean(random_task["official_score"])
            quality_by_workload[workload] = {
                "candidate": candidate_score,
                "position_control": position_score,
                "random_control": random_score,
                "delta_vs_position": candidate_score - position_score,
                "delta_vs_random": candidate_score - random_score,
            }
        candidate_nll = _mean(frame["mean_nll"])
        position_nll = _mean(position["mean_nll"])
        random_nll = _mean(random["mean_nll"])
        throughput = _mean(frame["end_to_end_decode_tokens_per_second"])
        peak_memory = int(frame["peak_memory_bytes"].max())
        hook_errors = int(frame["attention_hook_errors"].sum())
        maximum_cache = int(frame["max_kv_len_observed"].max())
        ruler = frame.loc[frame["workload"] == "niah_single_1"]
        ruler_recovered = int((ruler["official_score"] > 0.0).sum())
        system_checks = {
            "throughput_at_least_90_percent_of_position_control": (
                throughput >= 0.90 * position_throughput
            ),
            "peak_memory_no_more_than_105_percent_of_full": (
                peak_memory <= 1.05 * reference_peak
            ),
            "zero_attention_hook_and_runner_errors": (
                hook_errors == 0 and runner_errors == 0
            ),
            "cache_budget_respected": maximum_cache <= budget,
        }
        system_eligible = all(system_checks.values())
        candidate_summaries[candidate] = {
            "quality_by_workload": quality_by_workload,
            "ruler_needles_recovered": ruler_recovered,
            "overall_mean_nll": candidate_nll,
            "nll_delta_vs_position": candidate_nll - position_nll,
            "nll_delta_vs_random": candidate_nll - random_nll,
            "decode_throughput_tokens_per_second": throughput,
            "throughput_ratio_vs_position": throughput / position_throughput,
            "peak_memory_bytes": peak_memory,
            "peak_memory_ratio_vs_full": peak_memory / reference_peak,
            "attention_hook_errors": hook_errors,
            "maximum_observed_cache": maximum_cache,
            "system_checks": system_checks,
            "system_eligible": system_eligible,
        }
        ranking_rows.append(
            {
                "candidate": candidate,
                "system_eligible": system_eligible,
                "ruler_needles_recovered": ruler_recovered,
                "gov_report_official_score": quality_by_workload["gov_report"][
                    "candidate"
                ],
                "overall_mean_nll": candidate_nll,
                "throughput_ratio_vs_position": throughput / position_throughput,
                "peak_memory_ratio_vs_full": peak_memory / reference_peak,
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        [
            "system_eligible",
            "ruler_needles_recovered",
            "gov_report_official_score",
            "overall_mean_nll",
            "throughput_ratio_vs_position",
        ],
        ascending=[False, False, False, True, False],
    )
    replication_pool = ranking.loc[
        ranking["system_eligible"] & (ranking["ruler_needles_recovered"] > 0)
    ]
    selected = (
        None if replication_pool.empty else str(replication_pool.iloc[0]["candidate"])
    )
    summary = {
        "experiment": config["experiment_name"],
        "status": "completed_development_screen",
        "attention_capture_required": False,
        "candidate_algorithms_run_per_deployment_decision": 0,
        "position_control": POSITION_CONTROL,
        "random_control": RANDOM_CONTROL,
        "upper_reference": REFERENCE,
        "samples_per_method": len(position),
        "runner_errors": runner_errors,
        "controls": {
            POSITION_CONTROL: {
                "overall_mean_nll": _mean(position["mean_nll"]),
                "decode_throughput_tokens_per_second": position_throughput,
                "maximum_peak_memory_bytes": int(
                    position["peak_memory_bytes"].max()
                ),
            },
            RANDOM_CONTROL: {
                "overall_mean_nll": _mean(random["mean_nll"]),
                "decode_throughput_tokens_per_second": _mean(
                    random["end_to_end_decode_tokens_per_second"]
                ),
                "maximum_peak_memory_bytes": int(random["peak_memory_bytes"].max()),
            },
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
