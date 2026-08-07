"""Aggregate atomic runs into CSV, Parquet, Markdown, and LaTeX tables."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        columns = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append(
                "| "
                + " | ".join(str(value).replace("|", "\\|") for value in row)
                + " |"
            )
        return "\n".join(lines) + "\n"


def _bootstrap_ci(values: List[float], seed: int = 0, draws: int = 2000) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if len(finite) == 1:
        return {
            "mean": finite[0], "ci95_low": finite[0], "ci95_high": finite[0], "n": 1
        }
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(finite) for _ in finite) / len(finite)
        for _ in range(max(100, int(draws)))
    )
    return {
        "mean": sum(finite) / len(finite),
        "ci95_low": boot[int(0.025 * (len(boot) - 1))],
        "ci95_high": boot[int(0.975 * (len(boot) - 1))],
        "n": len(finite),
    }


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("predictions.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["run_dir"] = str(path.parent)
                rows.append(row)
    return rows


def _diagnostic_tables(rows: List[Dict[str, Any]]) -> tuple:
    diagnostics: List[Dict[str, Any]] = []
    quadrants: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata", {})
        base = {
            "run_dir": row.get("run_dir"),
            "sample_id": row.get("sample_id"),
            "task": row.get("task"),
            "benchmark": metadata.get("benchmark"),
            "method": metadata.get("method"),
            "method_variant": metadata.get("method_variant"),
            "protocol_visibility": metadata.get("protocol_visibility"),
            "cache_mode": metadata.get("cache_mode"),
            "context_length": metadata.get("context_length"),
            "cache_budget": metadata.get("cache_budget"),
            "seed": metadata.get("seed"),
            "correct": row.get("correct"),
        }
        for event in row.get("diagnostics", {}).get("events", []):
            for layer, values in event.get("layers", {}).items():
                overlap = values.get("overlap", {})
                correlation = values.get("rank_correlation", {})
                attention = values.get("attention", {})
                reconstruction = values.get("reconstruction", {})
                record = {
                    **base,
                    "snapshot_id": event.get("snapshot_id"),
                    "phase": event.get("phase"),
                    "decode_step": event.get("decode_step"),
                    "layer": int(layer),
                    "head": None,
                    "actual_retained_count": values.get("actual_retained_count"),
                    "mandatory_count": values.get("mandatory_count"),
                    "selectable_budget": values.get("selectable_budget"),
                    "evidence_total": values.get("evidence_total"),
                    "evidence_retained": values.get("evidence_retained"),
                    "evidence_recall": values.get("evidence_recall"),
                    "any_evidence_recall": values.get("any_evidence_recall"),
                    "complete_evidence_recall": values.get("complete_evidence_recall"),
                    "selection_precision_evidence": values.get(
                        "selection_precision_evidence"
                    ),
                    "selection_source_counts": json.dumps(
                        values.get("selection_source_counts", {}), sort_keys=True
                    ),
                    "overlap_at_k": overlap.get("overlap_at_k"),
                    "jaccard": overlap.get("jaccard"),
                    "overlap_k": overlap.get("k"),
                    "spearman": correlation.get("spearman"),
                    "kendall_tau_b": correlation.get("kendall_tau_b"),
                    "attention_entropy": attention.get("entropy"),
                    "attention_gini": attention.get("gini"),
                    "attention_effective_support": attention.get("effective_support"),
                    "attention_topk_mass": attention.get("topk_mass"),
                    "relative_frobenius_error": reconstruction.get(
                        "relative_frobenius_error"
                    ),
                    "effective_rank_preserved": reconstruction.get(
                        "effective_rank_preserved"
                    ),
                }
                diagnostics.append(record)
                for head_values in values.get("per_head_complementarity", []):
                    diagnostics.append(
                        {
                            **base,
                            "snapshot_id": event.get("snapshot_id"),
                            "phase": event.get("phase"),
                            "decode_step": event.get("decode_step"),
                            "layer": int(layer),
                            "head": head_values.get("head"),
                            "overlap_at_k": head_values.get("overlap_at_k"),
                            "jaccard": head_values.get("jaccard"),
                            "spearman": head_values.get("spearman"),
                            "kendall_tau_b": head_values.get("kendall_tau_b"),
                        }
                    )
                for threshold, by_quadrant in values.get("quadrants", {}).items():
                    for quadrant, quadrant_values in by_quadrant.items():
                        quadrants.append(
                            {
                                **base,
                                "phase": event.get("phase"),
                                "decode_step": event.get("decode_step"),
                                "layer": int(layer),
                                "threshold": threshold,
                                "quadrant": quadrant,
                                "token_count": quadrant_values.get("token_count"),
                                "evidence_count": quadrant_values.get("evidence_count"),
                                "selected_count": quadrant_values.get("selected_count"),
                                "evidence_fraction": quadrant_values.get(
                                    "evidence_fraction"
                                ),
                                "selection_rate": quadrant_values.get("selection_rate"),
                            }
                        )
    return pd.DataFrame(diagnostics), pd.DataFrame(quadrants)


def _paired_statistics(differences: List[float], seed: int = 0) -> Dict[str, Any]:
    finite = [float(value) for value in differences if math.isfinite(float(value))]
    if not finite:
        return {
            "paired_mean_difference": None,
            "paired_ci95_low": None,
            "paired_ci95_high": None,
            "sign_flip_p_value": None,
            "paired_n": 0,
        }
    ci = _bootstrap_ci(finite, seed=seed)
    observed = abs(sum(finite) / len(finite))
    rng = random.Random(seed + 7919)
    draws = 5000
    extreme = 1
    for _ in range(draws):
        permuted = abs(
            sum(value if rng.random() < 0.5 else -value for value in finite)
            / len(finite)
        )
        extreme += int(permuted >= observed)
    return {
        "paired_mean_difference": ci["mean"],
        "paired_ci95_low": ci["ci95_low"],
        "paired_ci95_high": ci["ci95_high"],
        "sign_flip_p_value": extreme / (draws + 1),
        "paired_n": len(finite),
    }


def _scbench_turn_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata", {})
        if metadata.get("benchmark") != "scbench":
            continue
        predictions = row.get("predictions") or [row.get("prediction", "")]
        references = row.get("references") or []
        evaluations = row.get("diagnostics", {}).get("query_evaluations", [])
        for index, evaluation in enumerate(evaluations):
            records.append(
                {
                    "run_dir": row.get("run_dir"),
                    "sample_id": row.get("sample_id"),
                    "task": evaluation.get("query_task", row.get("task")),
                    "query_index": evaluation.get("query_index", index),
                    "prediction": predictions[index] if index < len(predictions) else None,
                    "reference": references[index] if index < len(references) else None,
                    "score": evaluation.get("score"),
                    "correct": evaluation.get("correct"),
                    "metric_name": evaluation.get("metric_name"),
                    "method": metadata.get("method"),
                    "method_variant": metadata.get("method_variant"),
                    "protocol_visibility": metadata.get("protocol_visibility"),
                    "reuse_mode": metadata.get("reuse_mode"),
                    "cache_mode": metadata.get("cache_mode"),
                    "cache_budget": metadata.get("cache_budget"),
                    "seed": metadata.get("seed"),
                }
            )
    return pd.DataFrame(records)


def aggregate(root: Path, output: Path) -> Dict[str, str]:
    rows = load_rows(root)
    if not rows:
        raise RuntimeError("no predictions.jsonl files found under %s" % root)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.json_normalize(rows, sep=".")
    frame.to_csv(output / "samples.csv", index=False)
    frame.to_parquet(output / "samples.parquet", index=False)

    group_columns = [
        "metadata.benchmark",
        "metadata.task" if "metadata.task" in frame.columns else "task",
        "metadata.method_variant" if "metadata.method_variant" in frame.columns else "metadata.method",
        "metadata.protocol_visibility",
        "metadata.cache_mode",
        "metadata.context_length",
        "metadata.cache_budget",
    ]
    group_columns = [column for column in group_columns if column in frame.columns]
    summaries = []
    for key, group in frame.groupby(group_columns, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_values))
        ci = _bootstrap_ci(group["score"].dropna().astype(float).tolist())
        record.update(ci)
        record["score_std"] = float(group["score"].dropna().std(ddof=1)) if group["score"].notna().sum() > 1 else 0.0
        record["mean_prefill_s"] = float(group["timing.prefill_s"].mean())
        record["mean_scoring_s"] = float(group["timing.scoring_s"].mean())
        record["mean_compression_s"] = float(group["timing.compression_s"].mean())
        record["mean_decode_s"] = float(group["timing.decode_s"].mean())
        record["mean_end_to_end_s"] = (
            float(group["timing.end_to_end_s"].mean())
            if "timing.end_to_end_s" in group
            else None
        )
        if "timing.decode_tokens_per_s" in group:
            record["mean_decode_tokens_per_s"] = float(
                group["timing.decode_tokens_per_s"].dropna().mean()
            )
        if "timing.score_computation_count" in group:
            record["mean_score_computation_count"] = float(
                group["timing.score_computation_count"].mean()
            )
        record["mean_peak_gpu_memory_bytes"] = float(group["cache.peak_gpu_memory_bytes"].mean())
        if "cache.peak_cpu_rss_bytes" in group:
            record["mean_peak_cpu_rss_bytes"] = float(
                group["cache.peak_cpu_rss_bytes"].mean()
            )
        if "cache.max_physical_cache_bytes" in group:
            record["mean_max_physical_cache_bytes"] = float(
                group["cache.max_physical_cache_bytes"].mean()
            )
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    (output / "summary.md").write_text(_markdown_table(summary), encoding="utf-8")
    (output / "summary.tex").write_text(
        summary.to_latex(index=False, float_format="%.4f"), encoding="utf-8"
    )

    failure_rows = []
    identity = [
        column
        for column in (
            "sample_id",
            "task",
            "metadata.context_length",
            "metadata.cache_budget",
            "metadata.protocol_visibility",
            "metadata.cache_mode",
            "metadata.seed",
        )
        if column in frame.columns
    ]
    if "metadata.method" in frame.columns:
        for key, group in frame.groupby(identity, dropna=False):
            outcome = {
                str(row["metadata.method"]): row.get("correct")
                for _, row in group.iterrows()
            }
            details = {
                str(row["metadata.method"]): {
                    "score": row.get("score"),
                    "prediction": row.get("prediction"),
                    "predictions": row.get("predictions"),
                    "references": row.get("references"),
                    "run_dir": row.get("run_dir"),
                }
                for _, row in group.iterrows()
            }
            category = None
            if outcome.get("full") is False:
                category = "full_cache_fails"
            elif outcome.get("residual_v") is True and outcome.get("attention") is False and outcome.get("v_leverage") is False:
                category = "residual_wins_both_singles_fail"
            elif outcome.get("attention") is True and outcome.get("v_leverage") is False:
                category = "attention_wins"
            elif outcome.get("v_leverage") is True and outcome.get("attention") is False:
                category = "v_leverage_wins"
            elif outcome.get("full") is True and all(
                value is False for name, value in outcome.items() if name != "full"
            ):
                category = "all_compressed_methods_fail"
            if category:
                values = key if isinstance(key, tuple) else (key,)
                failure_rows.append(
                    {
                        **dict(zip(identity, values)),
                        "category": category,
                        "outcomes": outcome,
                        "details": details,
                    }
                )
    with open(output / "failure_cases.jsonl", "w", encoding="utf-8") as handle:
        for row in failure_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            )

    broad_group = [
        column
        for column in (
            "metadata.benchmark",
            "metadata.category",
            "metadata.method_variant",
            "metadata.protocol_visibility",
            "metadata.cache_mode",
            "metadata.cache_budget",
        )
        if column in frame.columns
    ]
    category_records: List[Dict[str, Any]] = []
    if broad_group:
        for key, group in frame.groupby(broad_group, dropna=False):
            key_values = key if isinstance(key, tuple) else (key,)
            record = dict(zip(broad_group, key_values))
            record.update(_bootstrap_ci(group["score"].dropna().astype(float).tolist()))
            category_records.append(record)
    category = pd.DataFrame(category_records)
    category.to_csv(output / "category_summary.csv", index=False)
    category.to_parquet(output / "category_summary.parquet", index=False)

    diagnostic_frame, quadrant_frame = _diagnostic_tables(rows)
    diagnostic_frame.to_csv(output / "diagnostics.csv", index=False)
    diagnostic_frame.to_parquet(output / "diagnostics.parquet", index=False)
    quadrant_frame.to_csv(output / "quadrant_diagnostics.csv", index=False)
    quadrant_frame.to_parquet(output / "quadrant_diagnostics.parquet", index=False)

    comparison_columns = [
        column
        for column in (
            "metadata.benchmark",
            "task",
            "metadata.protocol_visibility",
            "metadata.cache_mode",
            "metadata.context_length",
            "metadata.cache_budget",
        )
        if column in frame.columns
    ]
    paired_records: List[Dict[str, Any]] = []
    if "metadata.method" in frame.columns:
        sample_keys = [
            column
            for column in ("sample_id", "metadata.seed")
            if column in frame.columns
        ]
        for group_key, group in frame.groupby(comparison_columns, dropna=False):
            if not (group["metadata.method"] == "residual_v").any():
                continue
            values = group_key if isinstance(group_key, tuple) else (group_key,)
            identity_record = dict(zip(comparison_columns, values))
            ours = group[group["metadata.method"] == "residual_v"]
            for baseline in sorted(set(group["metadata.method"]) - {"residual_v"}):
                other = group[group["metadata.method"] == baseline]
                paired = ours[sample_keys + ["score"]].merge(
                    other[sample_keys + ["score"]],
                    on=sample_keys,
                    suffixes=("_ours", "_baseline"),
                ).dropna()
                differences = (
                    paired["score_ours"].astype(float)
                    - paired["score_baseline"].astype(float)
                ).tolist()
                paired_records.append(
                    {
                        **identity_record,
                        "ours": "residual_v",
                        "baseline": baseline,
                        **_paired_statistics(differences),
                    }
                )
    paired_frame = pd.DataFrame(paired_records)
    paired_frame.to_csv(output / "paired_comparisons.csv", index=False)
    paired_frame.to_parquet(output / "paired_comparisons.parquet", index=False)

    scbench_turns = _scbench_turn_table(rows)
    scbench_turns.to_csv(output / "scbench_turns.csv", index=False)
    scbench_turns.to_parquet(output / "scbench_turns.parquet", index=False)

    failed_runs: List[Dict[str, Any]] = []
    for status_path in sorted(root.rglob("status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        for sample_id, value in status.get("samples", {}).items():
            if value.get("state") != "complete":
                failed_runs.append(
                    {
                        "run_dir": str(status_path.parent),
                        "sample_id": sample_id,
                        "state": value.get("state"),
                        "error": value.get("error"),
                    }
                )
    with open(output / "failed_samples.jsonl", "w", encoding="utf-8") as handle:
        for row in failed_runs:
            handle.write(
                json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            )
    return {
        "samples_csv": str(output / "samples.csv"),
        "samples_parquet": str(output / "samples.parquet"),
        "summary_csv": str(output / "summary.csv"),
        "summary_markdown": str(output / "summary.md"),
        "summary_latex": str(output / "summary.tex"),
        "failure_cases": str(output / "failure_cases.jsonl"),
        "category_summary_csv": str(output / "category_summary.csv"),
        "diagnostics_parquet": str(output / "diagnostics.parquet"),
        "quadrant_diagnostics_parquet": str(output / "quadrant_diagnostics.parquet"),
        "paired_comparisons_csv": str(output / "paired_comparisons.csv"),
        "scbench_turns_csv": str(output / "scbench_turns.csv"),
        "failed_samples": str(output / "failed_samples.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = aggregate(Path(args.results_root), Path(args.output_dir))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
