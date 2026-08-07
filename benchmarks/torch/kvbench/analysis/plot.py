"""Generate paper diagnostic figures directly from atomic result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from kvbench.analysis.aggregate import load_rows


def _method(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata", {})
    return str(metadata.get("method_variant", metadata.get("method", "unknown")))


def _base_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        metadata = row.get("metadata", {})
        timing = row.get("timing", {})
        records.append(
            {
                "benchmark": metadata.get("benchmark"),
                "task": row.get("task"),
                "method": _method(row),
                "budget": metadata.get("cache_budget"),
                "length": metadata.get("context_length"),
                "score": row.get("score"),
                "overhead_s": float(timing.get("scoring_s", 0.0) or 0.0)
                + float(timing.get("compression_s", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(records)


def _curve(frame: pd.DataFrame, x: str, path: Path) -> bool:
    usable = frame.dropna(subset=[x, "score", "method"])
    if usable.empty:
        return False
    grouped = usable.groupby(["method", x], as_index=False)["score"].mean()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for method, values in grouped.groupby("method"):
        values = values.sort_values(x)
        axis.plot(values[x], values["score"], marker="o", label=method)
    axis.set_xlabel("Cache budget" if x == "budget" else "Context length")
    axis.set_ylabel("Benchmark score")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _quality_overhead(frame: pd.DataFrame, path: Path) -> bool:
    usable = frame.dropna(subset=["score", "overhead_s", "method"])
    if usable.empty:
        return False
    grouped = usable.groupby("method", as_index=False)[["score", "overhead_s"]].mean()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.scatter(grouped["overhead_s"], grouped["score"])
    for row in grouped.itertuples(index=False):
        axis.annotate(row.method, (row.overhead_s, row.score), fontsize=7)
    axis.set_xlabel("Mean scoring + compression time (s)")
    axis.set_ylabel("Benchmark score")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _diagnostic_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in rows:
        for event in row.get("diagnostics", {}).get("events", []):
            for layer, values in event.get("layers", {}).items():
                correlation = values.get("rank_correlation", {})
                overlap = values.get("overlap", {})
                records.append(
                    {
                        "method": _method(row),
                        "layer": int(layer),
                        "overlap_at_k": overlap.get("overlap_at_k"),
                        "spearman": correlation.get("spearman"),
                    }
                )
    return records


def _overlap_figure(rows: List[Dict[str, Any]], path: Path) -> bool:
    frame = pd.DataFrame(_diagnostic_rows(rows))
    if frame.empty or frame[["overlap_at_k", "spearman"]].notna().sum().sum() == 0:
        return False
    grouped = frame.groupby("layer", as_index=False)[["overlap_at_k", "spearman"]].mean()
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    if grouped["overlap_at_k"].notna().any():
        axis.plot(grouped["layer"], grouped["overlap_at_k"], label="overlap@k")
    if grouped["spearman"].notna().any():
        axis.plot(grouped["layer"], grouped["spearman"], label="Spearman")
    axis.set_xlabel("Layer")
    axis.set_ylabel("Mean diagnostic value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def _quadrant_figure(rows: List[Dict[str, Any]], path: Path) -> bool:
    records: List[Dict[str, Any]] = []
    for row in rows:
        for event in row.get("diagnostics", {}).get("events", []):
            for values in event.get("layers", {}).values():
                quadrants = values.get("quadrants", {}).get("top10pct", {})
                for name, statistics in quadrants.items():
                    records.append(
                        {
                            "quadrant": name,
                            "evidence_fraction": statistics.get("evidence_fraction"),
                            "selection_rate": statistics.get("selection_rate"),
                        }
                    )
    frame = pd.DataFrame(records)
    if frame.empty or "evidence_fraction" not in frame:
        return False
    grouped = frame.groupby("quadrant", as_index=False)[
        ["evidence_fraction", "selection_rate"]
    ].mean()
    short_names = {
        "high_attention_high_leverage": "high-A/high-V",
        "high_attention_low_leverage": "high-A/low-V",
        "low_attention_high_leverage": "low-A/high-V",
        "low_attention_low_leverage": "low-A/low-V",
    }
    grouped["label"] = grouped["quadrant"].map(short_names).fillna(grouped["quadrant"])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    positions = list(range(len(grouped)))
    width = 0.38
    axis.bar(
        [value - width / 2 for value in positions],
        grouped["evidence_fraction"],
        width=width,
        label="evidence fraction",
    )
    if grouped["selection_rate"].notna().any():
        axis.bar(
            [value + width / 2 for value in positions],
            grouped["selection_rate"],
            width=width,
            label="selection rate",
        )
    axis.set_xticks(positions, grouped["label"], rotation=18, ha="right")
    axis.set_ylabel("Mean fraction")
    axis.set_title("Attention / V-leverage top-10% quadrants")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def generate_figures(results_root: Path, output_dir: Path) -> Dict[str, Any]:
    rows = load_rows(results_root)
    if not rows:
        raise RuntimeError("no prediction artifacts found under %s" % results_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _base_frame(rows)
    candidates = {
        "budget_curve": (_curve(frame, "budget", output_dir / "budget_curve.png"), "budget_curve.png"),
        "length_curve": (_curve(frame, "length", output_dir / "length_curve.png"), "length_curve.png"),
        "quality_overhead": (_quality_overhead(frame, output_dir / "quality_overhead.png"), "quality_overhead.png"),
        "overlap_correlation": (_overlap_figure(rows, output_dir / "overlap_correlation.png"), "overlap_correlation.png"),
        "quadrant_evidence": (_quadrant_figure(rows, output_dir / "quadrant_evidence.png"), "quadrant_evidence.png"),
    }
    report = {
        name: str(output_dir / filename) if generated else None
        for name, (generated, filename) in candidates.items()
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = generate_figures(Path(args.results_root), Path(args.output_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
