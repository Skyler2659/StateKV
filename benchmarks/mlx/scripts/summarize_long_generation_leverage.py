#!/usr/bin/env python3
"""Summarize the small GovReport leverage-update sweep.

The sweep intentionally stores each update condition in a separate run
directory so the standard benchmark runner can remain unchanged.  This script
joins those runs by sample id and reports both quality and end-to-end decode
speed (model time plus eviction time).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


EXPERIMENT_NAME = "long_generation_govreport_l2_update_sweep"
VARIANT_ORDER = {
    "full": 0,
    "prefill_only": 1,
    "every_64": 2,
    "every_16": 3,
    "every_1": 4,
}


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = [_number(row.get(key)) for row in rows]
    finite = [value for value in values if value is not None]
    return mean(finite) if finite else None


def _variant(row: Dict[str, Any]) -> str:
    method = str(row.get("canonical_method") or row.get("method") or "")
    if method == "full":
        return "full"
    if method == "l2_prefill_only" or row.get("update_policy") == "prefill_only":
        return "prefill_only"
    interval = int(row.get("update_interval") or 0)
    return f"every_{interval}"


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"warning: cannot read {path}: {exc}")
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict) or row.get("experiment_name") != EXPERIMENT_NAME:
                continue
            copied = dict(row)
            copied["source_results"] = str(path)
            copied["variant"] = _variant(copied)
            rows.append(copied)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _fmt(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results/long_generation_leverage"))
    args = parser.parse_args()

    rows = _load_rows(args.results_root)
    ok = [row for row in rows if not row.get("error") and not row.get("skipped")]
    errors = [row for row in rows if row.get("error") or row.get("skipped")]
    if not ok:
        raise SystemExit(f"No completed {EXPERIMENT_NAME} rows found under {args.results_root}")

    detail_fields = [
        "variant",
        "sample_id",
        "context_length",
        "generated_token_count",
        "n_eval_tokens",
        "official_score",
        "mean_nll",
        "ppl",
        "generation_time_s",
        "total_time_s",
        "end_to_end_decode_tokens_per_second",
        "eviction_overhead_fraction",
        "ppl_end_to_end_decode_tokens_per_second",
        "ppl_eviction_overhead_fraction",
        "score_time_s",
        "ppl_score_time_s",
        "score_refit_count",
        "ppl_score_refit_count",
        "update_policy",
        "update_interval",
        "sample_manifest_hash",
        "source_results",
    ]
    detail_rows = sorted(
        ok,
        key=lambda row: (VARIANT_ORDER.get(row["variant"], 99), int(row.get("sample_id") or 0)),
    )

    summaries: List[Dict[str, Any]] = []
    for variant in sorted({row["variant"] for row in ok}, key=lambda item: VARIANT_ORDER.get(item, 99)):
        group = [row for row in ok if row["variant"] == variant]
        mean_nll = _mean(group, "mean_nll")
        summaries.append(
            {
                "variant": variant,
                "n": len(group),
                "avg_context_tokens": _mean(group, "context_length"),
                "avg_generated_tokens": _mean(group, "generated_token_count"),
                "avg_teacher_forced_tokens": _mean(group, "n_eval_tokens"),
                "avg_rouge_l": _mean(group, "official_score"),
                "avg_mean_nll": mean_nll,
                "aggregate_ppl": math.exp(mean_nll) if mean_nll is not None else None,
                "avg_generation_time_s": _mean(group, "generation_time_s"),
                "avg_total_time_s": _mean(group, "total_time_s"),
                "avg_free_decode_tok_s": _mean(group, "end_to_end_decode_tokens_per_second"),
                "avg_free_eviction_fraction": _mean(group, "eviction_overhead_fraction"),
                "avg_teacher_forced_tok_s": _mean(
                    group, "ppl_end_to_end_decode_tokens_per_second"
                ),
                "avg_teacher_eviction_fraction": _mean(
                    group, "ppl_eviction_overhead_fraction"
                ),
                "avg_score_time_s": _mean(group, "score_time_s"),
                "avg_teacher_score_time_s": _mean(group, "ppl_score_time_s"),
                "avg_generation_refits": _mean(group, "score_refit_count"),
                "avg_teacher_refits": _mean(group, "ppl_score_refit_count"),
            }
        )

    prefill = next((row for row in summaries if row["variant"] == "prefill_only"), None)
    for row in summaries:
        if prefill is None or row["variant"] == "prefill_only":
            row["rouge_delta_vs_prefill"] = 0.0 if row["variant"] == "prefill_only" else None
            row["nll_delta_vs_prefill"] = 0.0 if row["variant"] == "prefill_only" else None
            row["free_speed_ratio_vs_prefill"] = 1.0 if row["variant"] == "prefill_only" else None
            continue
        rouge = _number(row.get("avg_rouge_l"))
        base_rouge = _number(prefill.get("avg_rouge_l"))
        nll = _number(row.get("avg_mean_nll"))
        base_nll = _number(prefill.get("avg_mean_nll"))
        speed = _number(row.get("avg_free_decode_tok_s"))
        base_speed = _number(prefill.get("avg_free_decode_tok_s"))
        row["rouge_delta_vs_prefill"] = (
            rouge - base_rouge if rouge is not None and base_rouge is not None else None
        )
        row["nll_delta_vs_prefill"] = (
            nll - base_nll if nll is not None and base_nll is not None else None
        )
        row["free_speed_ratio_vs_prefill"] = (
            speed / base_speed if speed is not None and base_speed not in (None, 0.0) else None
        )

    summary_fields = list(summaries[0].keys())
    out_dir = args.results_root / "sweep_summary"
    _write_csv(out_dir / "per_sample.csv", detail_rows, detail_fields)
    _write_csv(out_dir / "summary.csv", summaries, summary_fields)

    manifests = {row.get("sample_manifest_hash") for row in ok if row.get("sample_manifest_hash")}
    lines = [
        "# GovReport L2 leverage update sweep",
        "",
        f"Completed rows: {len(ok)}; error/skipped rows: {len(errors)}.",
        f"Matched sample manifests: {'yes' if len(manifests) == 1 else 'NO'} ({len(manifests)} hashes).",
        "",
        "| variant | n | ROUGE-L | mean NLL | PPL | free tok/s | teacher tok/s | eviction fraction | unit refits |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {variant} | {n} | {rouge} | {nll} | {ppl} | {free} | {teacher} | {overhead} | {refits} |".format(
                variant=row["variant"],
                n=row["n"],
                rouge=_fmt(row.get("avg_rouge_l")),
                nll=_fmt(row.get("avg_mean_nll")),
                ppl=_fmt(row.get("aggregate_ppl")),
                free=_fmt(row.get("avg_free_decode_tok_s")),
                teacher=_fmt(row.get("avg_teacher_forced_tok_s")),
                overhead=_fmt(row.get("avg_free_eviction_fraction")),
                refits=_fmt(row.get("avg_generation_refits"), 1),
            )
        )
    rules = [
        "- Treat this small-sample run as a directional pilot, not a statistical benchmark result.",
        "- Lower mean NLL/PPL and higher ROUGE-L are better; free/teacher tok/s already include eviction time.",
    ]
    present_variants = {str(row.get("variant")) for row in summaries}
    if "every_1" in present_variants:
        rules.extend(
            [
                "- If every-1 does not improve either ROUGE-L or mean NLL over prefill-only, do not pay its speed cost.",
                "- If every-16/64 matches every-1 within roughly 1 ROUGE-L and 0.02 mean NLL, prefer the largest interval with that quality.",
            ]
        )
    rules.append(
        "- Inspect per-sample rows: an average driven by one sample is only a reason for a larger confirmation run."
    )
    lines.extend(["", "## Interpretation rules", "", *rules])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'summary.csv'} and {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
