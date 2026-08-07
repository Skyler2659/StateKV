#!/usr/bin/env python3
"""Post-hoc analysis runner for saved benchmark results."""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig, load_analysis_config
from src.artifacts.schema import ScoreArtifact, SelectionArtifact, load_artifact
from src.utils.io import load_results, load_scores, save_results
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("run_analysis")


def _load_selected(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("selected_tokens_path")
    if path:
        p = Path(path)
        if not p.exists():
            return None
        data = load_results(p) if p.suffix == ".json" else load_scores(p)
    else:
        data = result.get("selected_tokens") or result.get("selected_tokens_by_layer")
        if not data:
            return None
    return {int(k): torch.tensor(v, dtype=torch.long) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _load_score_dict(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("scores_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = load_results(p) if p.suffix == ".json" else load_scores(p)
    return {int(k): torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in data.items()}


def _load_selection_artifact(result: Dict[str, Any]) -> tuple:
    """Load a schema-v2 selection artifact without coercing legacy files."""

    path = result.get("selection_artifact_path") or result.get("selected_tokens_path")
    if not path:
        return None, "missing_selection_artifact"
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None, "missing_selection_artifact_file"
    if artifact_path.suffix != ".json":
        return None, "legacy_selection_artifact"
    try:
        artifact = load_artifact(artifact_path)
    except (KeyError, TypeError, ValueError):
        return None, "legacy_or_invalid_selection_artifact"
    if not isinstance(artifact, SelectionArtifact):
        return None, "wrong_selection_artifact_type"
    return artifact, None


def _load_score_artifact(result: Dict[str, Any]) -> tuple:
    """Load a schema-v2 score artifact without guessing token positions."""

    path = result.get("score_artifact_path") or result.get("scores_path")
    if not path:
        return None, "missing_score_artifact"
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None, "missing_score_artifact_file"
    if artifact_path.suffix != ".json":
        return None, "legacy_score_artifact"
    try:
        artifact = load_artifact(artifact_path)
    except (KeyError, TypeError, ValueError):
        return None, "legacy_or_invalid_score_artifact"
    if not isinstance(artifact, ScoreArtifact):
        return None, "wrong_score_artifact_type"
    return artifact, None


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_ci(values: List[float], seed: int = 0, draws: int = 2000) -> Dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "n": 0}
    if len(finite) == 1:
        return {"mean": finite[0], "ci95_low": finite[0], "ci95_high": finite[0], "n": 1}
    rng = random.Random(seed)
    boot = sorted(
        sum(rng.choice(finite) for _ in finite) / len(finite)
        for _ in range(max(100, int(draws)))
    )
    low = boot[int(0.025 * (len(boot) - 1))]
    high = boot[int(0.975 * (len(boot) - 1))]
    return {"mean": sum(finite) / len(finite), "ci95_low": low, "ci95_high": high, "n": len(finite)}


def _write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_matrix(path: Path, labels: List[str], matrix: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + labels)
        for left in labels:
            writer.writerow([left] + [matrix.get(left, {}).get(right, 0.0) for right in labels])


def run_analysis(results: List[Dict], cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    if cfg.analysis.counterfactual_deletion or cfg.analysis.restoration:
        raise ValueError(
            "text-level deletion/restoration is quarantined because it changes "
            "token adjacency, RoPE positions, and contextual states; a fixed-snapshot "
            "KV intervention runner is required"
        )
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_results: Dict[str, Any] = {}
    filtered_results = [
        r
        for r in results
        if "error" not in r
        and not r.get("skipped")
        and (cfg.analysis.include_oracle or not r.get("oracle"))
    ]
    skipped = [
        {
            "method": r.get("method"),
            "sample_id": r.get("sample_id", r.get("sample_idx")),
            "budget": r.get("budget"),
            "skipped_reason": r.get("skipped_reason") or r.get("unsupported_reason"),
        }
        for r in results
        if r.get("skipped")
    ]
    if skipped:
        save_results(skipped, analysis_dir / "skipped_methods.json")
        analysis_results["skipped"] = skipped

    analysis_results["tables"] = analyze_metric_tables(filtered_results, analysis_dir)

    if cfg.analysis.overlap:
        logger.info("Running overlap analysis")
        analysis_results["overlap"] = analyze_overlap(filtered_results, analysis_dir)

    if cfg.analysis.rank_correlation:
        logger.info("Running rank correlation analysis")
        analysis_results["rank_correlation"] = analyze_rank_correlation(filtered_results, analysis_dir)

    if cfg.analysis.evidence_recall:
        logger.info("Running evidence recall analysis")
        analysis_results["evidence_recall"] = analyze_evidence_recall(filtered_results, analysis_dir)

    if cfg.analysis.case_study:
        logger.info("Exporting case studies")
        analysis_results["case_study"] = analyze_case_studies(
            filtered_results, analysis_dir, cfg.analysis.case_study_count
        )

    save_results(analysis_results, analysis_dir / "analysis_summary.json")
    return analysis_results


def analyze_overlap(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.overlap import OverlapAnalyzer

    analyzer = OverlapAnalyzer()
    grouped = defaultdict(dict)
    skipped = defaultdict(int)
    for result in results:
        selected, reason = _load_selection_artifact(result)
        if selected is None:
            skipped[reason] += 1
            continue
        key = (
            result.get("sample_id", result.get("sample_idx")),
            result.get("budget"),
            selected.snapshot.snapshot_id,
        )
        grouped[key][result["method"]] = selected

    pair_values = defaultdict(lambda: defaultdict(list))
    budget_pair_values = defaultdict(lambda: defaultdict(list))
    unit_details = []
    frequency_details = []
    mismatches = []
    method_names = set()
    for group_key, methods in grouped.items():
        sample_id, budget, snapshot_id = group_key
        names = sorted(methods)
        method_names.update(names)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                try:
                    overlap = analyzer.artifact_pairwise(methods[left], methods[right])
                    frequency = analyzer.frequency_correlation(methods[left], methods[right])
                except ValueError as exc:
                    mismatches.append({"left": left, "right": right, "reason": str(exc)})
                    continue
                pair = pair_values[f"{left}_vs_{right}"]
                budget_pair = budget_pair_values[(f"{left}_vs_{right}", int(budget))]
                pair["jaccard"].append(overlap.jaccard)
                pair["micro_jaccard"].append(overlap.micro_jaccard)
                pair["intersection_ratio"].append(overlap.intersection_ratio)
                pair["overlap_coefficient"].append(overlap.overlap_coefficient)
                pair["recall_a_by_b"].append(overlap.recall_a_by_b)
                pair["recall_b_by_a"].append(overlap.recall_b_by_a)
                pair["selection_frequency_spearman"].append(frequency.spearman)
                for metric in (
                    "jaccard", "micro_jaccard", "intersection_ratio",
                    "overlap_coefficient", "recall_a_by_b", "recall_b_by_a",
                ):
                    budget_pair[metric].append(float(getattr(overlap, metric)))
                for label, unit in overlap.unit_results.items():
                    unit_details.append(
                        {
                            "sample_id": sample_id,
                            "budget": budget,
                            "snapshot_id": snapshot_id,
                            "left": left,
                            "right": right,
                            "unit": label,
                            **unit.__dict__,
                        }
                    )
                positions = sorted(frequency.frequency_a)
                frequency_details.append(
                    {
                        "sample_id": sample_id,
                        "budget": budget,
                        "snapshot_id": snapshot_id,
                        "left": left,
                        "right": right,
                        "frequency_left": frequency.frequency_a,
                        "frequency_right": frequency.frequency_b,
                        "common_core_frequency": {
                            str(position): min(
                                frequency.frequency_a[position], frequency.frequency_b[position]
                            )
                            for position in positions
                        },
                        "left_only_frequency": {
                            str(position): max(
                                0.0, frequency.frequency_a[position] - frequency.frequency_b[position]
                            )
                            for position in positions
                        },
                        "right_only_frequency": {
                            str(position): max(
                                0.0, frequency.frequency_b[position] - frequency.frequency_a[position]
                            )
                            for position in positions
                        },
                    }
                )

    def finite_mean(values):
        finite = [value for value in values if math.isfinite(value)]
        return sum(finite) / len(finite) if finite else None

    pairs = {
        pair: {metric: finite_mean(values) for metric, values in metrics.items()}
        for pair, metrics in pair_values.items()
    }
    confidence_intervals = {
        pair: {
            metric: _mean_ci(values, seed=17 + metric_index)
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for pair, metrics in pair_values.items()
    }
    budget_curve = {
        f"{pair}|budget={budget}": {
            metric: _mean_ci(values, seed=31 + metric_index)
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for (pair, budget), metrics in budget_pair_values.items()
    }
    labels = sorted(method_names)
    matrix = {m: {n: (1.0 if m == n else 0.0) for n in labels} for m in labels}
    for pair, metrics in pairs.items():
        left, right = pair.split("_vs_", 1)
        matrix.setdefault(left, {})[right] = metrics["jaccard"]
        matrix.setdefault(right, {})[left] = metrics["jaccard"]
    _write_matrix(analysis_dir / "overlap_matrix.csv", labels, matrix)
    summary = {
        "schema_version": 2,
        "aggregation": "macro_over_layer_head_units_then_samples",
        "pairs": pairs,
        "confidence_intervals": confidence_intervals,
        "overlap_budget_curve": budget_curve,
        "unit_detail_path": str(analysis_dir / "overlap_per_sample_layer_head.csv"),
        "selection_frequency_detail_path": str(analysis_dir / "selection_frequency.json"),
        "skipped_artifacts": dict(skipped),
        "mismatched_pairs": mismatches,
    }
    _write_rows(analysis_dir / "overlap_per_sample_layer_head.csv", unit_details)
    save_results(frequency_details, analysis_dir / "selection_frequency.json")
    if not pairs:
        summary["note"] = "schema-v2 matched selection artifacts are required"
    save_results(summary, analysis_dir / "overlap_summary.json")
    return summary


def analyze_rank_correlation(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.rank_correlation import RankCorrelationAnalyzer

    analyzer = RankCorrelationAnalyzer()
    grouped = defaultdict(dict)
    skipped = defaultdict(int)
    for result in results:
        scores, reason = _load_score_artifact(result)
        if scores is None:
            skipped[reason] += 1
            continue
        key = (
            result.get("sample_id", result.get("sample_idx")),
            result.get("budget"),
            scores.snapshot.snapshot_id,
        )
        grouped[key][result["method"]] = scores

    pair_values = defaultdict(lambda: defaultdict(list))
    budget_pair_values = defaultdict(lambda: defaultdict(list))
    unit_details = []
    mismatches = []
    method_names = set()
    for group_key, methods in grouped.items():
        sample_id, budget, snapshot_id = group_key
        names = sorted(methods)
        method_names.update(names)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                try:
                    corr = analyzer.artifact_pairwise(methods[left], methods[right])
                except ValueError as exc:
                    mismatches.append({"left": left, "right": right, "reason": str(exc)})
                    continue
                pair = pair_values[f"{left}_vs_{right}"]
                budget_pair = budget_pair_values[(f"{left}_vs_{right}", int(budget))]
                pair["pearson"].append(corr.pearson)
                pair["spearman"].append(corr.spearman)
                pair["kendall_tau_b"].append(corr.kendall_tau)
                budget_pair["pearson"].append(corr.pearson)
                budget_pair["spearman"].append(corr.spearman)
                budget_pair["kendall_tau_b"].append(corr.kendall_tau)
                for label, unit in corr.unit_results.items():
                    unit_details.append(
                        {
                            "sample_id": sample_id,
                            "budget": budget,
                            "snapshot_id": snapshot_id,
                            "left": left,
                            "right": right,
                            "unit": label,
                            **unit.__dict__,
                        }
                    )

    def finite_mean(values):
        finite = [value for value in values if math.isfinite(value)]
        return sum(finite) / len(finite) if finite else None

    pairs = {
        pair: {metric: finite_mean(values) for metric, values in metrics.items()}
        for pair, metrics in pair_values.items()
    }
    confidence_intervals = {
        pair: {
            metric: _mean_ci(values, seed=71 + metric_index)
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for pair, metrics in pair_values.items()
    }
    budget_curve = {
        f"{pair}|budget={budget}": {
            metric: _mean_ci(values, seed=89 + metric_index)
            for metric_index, (metric, values) in enumerate(metrics.items())
        }
        for (pair, budget), metrics in budget_pair_values.items()
    }
    labels = sorted(method_names)
    matrix = {m: {n: (1.0 if m == n else 0.0) for n in labels} for m in labels}
    for pair, metrics in pairs.items():
        left, right = pair.split("_vs_", 1)
        matrix.setdefault(left, {})[right] = metrics["spearman"]
        matrix.setdefault(right, {})[left] = metrics["spearman"]
    _write_matrix(analysis_dir / "rank_correlation.csv", labels, matrix)
    summary = {
        "schema_version": 2,
        "aggregation": "macro_over_layer_head_units_then_samples",
        "pairs": pairs,
        "confidence_intervals": confidence_intervals,
        "correlation_budget_curve": budget_curve,
        "unit_detail_path": str(analysis_dir / "rank_per_sample_layer_head.csv"),
        "skipped_artifacts": dict(skipped),
        "mismatched_pairs": mismatches,
    }
    _write_rows(analysis_dir / "rank_per_sample_layer_head.csv", unit_details)
    if not pairs:
        summary["note"] = "schema-v2 matched score artifacts are required"
    save_results(summary, analysis_dir / "rank_correlation_summary.json")
    return summary


def analyze_evidence_recall(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    values = defaultdict(list)
    details = []
    for result in results:
        evidence = result.get("evidence_positions") or []
        recall = result.get("evidence_recall")
        if not evidence or recall is None:
            continue
        values[result["method"]].append(float(recall))
        details.append(
            {
                "method": result["method"],
                "sample_id": result.get("sample_id"),
                "budget": result.get("budget"),
                "aggregation": "macro_over_layer_head_units",
                "evidence_recall": recall,
                "evidence_any_unit_recall": result.get("evidence_any_unit_recall"),
                "evidence_precision": result.get("evidence_precision"),
                "evidence_recall_by_unit": result.get("evidence_recall_by_unit"),
                "assignment_node_recall": result.get("assignment_node_recall"),
                "dependency_edge_recall": result.get("dependency_edge_recall"),
                "complete_path_rate": result.get("complete_path_rate"),
                "distractor_retention": result.get("distractor_retention"),
            }
        )

    summary = {method: _mean(vals) for method, vals in values.items()}
    _write_rows(analysis_dir / "evidence_recall.csv", details)
    save_results(details, analysis_dir / "evidence_recall_details.json")
    save_results(summary, analysis_dir / "evidence_recall_summary.json")
    return summary or {"note": "no evidence positions or selected token files"}


def analyze_metric_tables(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    rows = [r for r in results if "error" not in r and not r.get("skipped")]
    by_method_budget = defaultdict(list)
    by_method_context = defaultdict(list)
    by_method_budget_model = defaultdict(list)
    by_method_model = defaultdict(list)
    for r in rows:
        model = r.get("model_name") or r.get("model")
        by_method_budget[(r.get("method"), r.get("budget"))].append(r)
        by_method_context[(r.get("method"), r.get("context_length"))].append(r)
        by_method_budget_model[(model, r.get("method"), r.get("budget"))].append(r)
        by_method_model[(model, r.get("method"))].append(r)

    def summarize(grouped):
        out = []
        for key, vals in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
            row = {
                    "value": key[-1],
                    "n": len(vals),
                    "accuracy": _mean([1.0 if v.get("correct") else 0.0 for v in vals]),
                    "contains_ground_truth": _mean(
                        [1.0 if v.get("contains_ground_truth") else 0.0 for v in vals]
                    ),
                    "avg_official_score": _mean(
                        [
                            float(v.get("official_score"))
                            for v in vals
                            if v.get("official_score") is not None
                        ]
                    ),
                    "official_accuracy": _mean(
                        [
                            1.0 if v.get("official_correct") else 0.0
                            for v in vals
                            if v.get("official_correct") is not None
                        ]
                    ),
                    "avg_primary_score": _mean(
                        [
                            float(v.get("primary_score"))
                            for v in vals
                            if v.get("primary_score") is not None
                        ]
                    ),
                    "avg_ppl": _mean(
                        [float(v.get("ppl")) for v in vals if v.get("ppl") is not None]
                    ),
                    "avg_evidence_recall": _mean(
                        [
                            float(v.get("evidence_recall"))
                            for v in vals
                            if v.get("evidence_recall") is not None
                        ]
                    ),
                    "avg_tokens_per_second": _mean(
                        [
                            float(v.get("tokens_per_second"))
                            for v in vals
                            if v.get("tokens_per_second") is not None
                        ]
                    ),
            }
            if len(key) == 2:
                row["method"] = key[0]
            elif len(key) == 3:
                row["model_name"] = key[0]
                row["method"] = key[1]
            out.append(row)
        return out

    mb = summarize(by_method_budget)
    mc = summarize(by_method_context)
    mbm = summarize(by_method_budget_model)
    mm = []
    for (model, method), vals in sorted(by_method_model.items(), key=lambda item: tuple(str(x) for x in item[0])):
        mm.append(
            {
                "model_name": model,
                "method": method,
                "n": len(vals),
                "accuracy": _mean([1.0 if v.get("correct") else 0.0 for v in vals]),
                "avg_official_score": _mean(
                    [
                        float(v.get("official_score"))
                        for v in vals
                        if v.get("official_score") is not None
                    ]
                ),
                "official_accuracy": _mean(
                    [
                        1.0 if v.get("official_correct") else 0.0
                        for v in vals
                        if v.get("official_correct") is not None
                    ]
                ),
                "avg_primary_score": _mean(
                    [
                        float(v.get("primary_score"))
                        for v in vals
                        if v.get("primary_score") is not None
                    ]
                ),
                "avg_evidence_recall": _mean(
                    [
                        float(v.get("evidence_recall"))
                        for v in vals
                        if v.get("evidence_recall") is not None
                    ]
                ),
                "avg_tokens_per_second": _mean(
                    [
                        float(v.get("tokens_per_second"))
                        for v in vals
                        if v.get("tokens_per_second") is not None
                    ]
                ),
            }
        )
    for row in mb:
        row["budget"] = row.pop("value")
    for row in mc:
        row["context_length"] = row.pop("value")
    for row in mbm:
        row["budget"] = row.pop("value")
    _write_rows(analysis_dir / "method_budget_accuracy.csv", mb)
    _write_rows(analysis_dir / "method_context_accuracy.csv", mc)
    _write_rows(analysis_dir / "model_method_budget_metrics.csv", mbm)
    _write_rows(analysis_dir / "model_method_metrics.csv", mm)
    return {
        "method_budget_accuracy_csv": str(analysis_dir / "method_budget_accuracy.csv"),
        "method_context_accuracy_csv": str(analysis_dir / "method_context_accuracy.csv"),
        "model_method_budget_metrics_csv": str(analysis_dir / "model_method_budget_metrics.csv"),
        "model_method_metrics_csv": str(analysis_dir / "model_method_metrics.csv"),
    }


def analyze_case_studies(
    results: List[Dict], analysis_dir: Path, count: int = 5
) -> Dict[str, Any]:
    by_sample = defaultdict(list)
    for result in results:
        if "error" not in result and not result.get("skipped"):
            by_sample[result.get("sample_id", result.get("sample_idx"))].append(result)

    cases = []
    for sample_id, sample_results in by_sample.items():
        if len(sample_results) < 2:
            continue
        ranked = sorted(
            sample_results,
            key=lambda r: float(r.get("ppl")) if r.get("ppl") is not None else float("inf"),
        )
        best = ranked[0]
        worst = ranked[-1]
        cases.append(
            {
                "sample_id": sample_id,
                "category": "best_vs_worst",
                "best_method": best.get("method"),
                "best_ppl": best.get("ppl"),
                "worst_method": worst.get("method"),
                "worst_ppl": worst.get("ppl"),
                "ground_truth": best.get("ground_truth"),
                "evidence_positions": best.get("evidence_positions"),
                "best_selected_tokens_path": best.get("selected_tokens_path"),
                "worst_selected_tokens_path": worst.get("selected_tokens_path"),
            }
        )

    cases = cases[:count]
    save_results(cases, analysis_dir / "case_studies.json")
    return {"num_cases": len(cases), "path": str(analysis_dir / "case_studies.json")}


def _apply_analysis_filter(cfg: ExperimentConfig, selected: str) -> None:
    if selected == "all":
        return
    enabled = {name.strip() for name in selected.split(",") if name.strip()}
    cfg.analysis.overlap = "overlap" in enabled
    cfg.analysis.rank_correlation = "rank_correlation" in enabled or "rank_corr" in enabled
    cfg.analysis.evidence_recall = "evidence_recall" in enabled
    cfg.analysis.case_study = "case_study" in enabled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--results_dir", "--run-dir", dest="input", type=str, required=True)
    parser.add_argument("--config", type=str, default=None, help="Analysis fragment or full config")
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        help="Comma-separated: overlap,rank_correlation,evidence_recall,case_study",
    )
    args = parser.parse_args()
    setup_logging()

    results_dir = Path(args.input)
    results = load_results(results_dir / "results.json")
    cfg = ExperimentConfig.from_yaml(results_dir / "config.yaml")
    if args.config:
        cfg.analysis = load_analysis_config(args.config)
    _apply_analysis_filter(cfg, args.analysis)

    summary = run_analysis(results, cfg, results_dir)
    logger.info(
        "Analysis complete: overlap_pairs=%d rank_pairs=%d evidence_methods=%d",
        len((summary.get("overlap") or {}).get("pairs", {})),
        len((summary.get("rank_correlation") or {}).get("pairs", {})),
        len(summary.get("evidence_recall") or {}),
    )
    logger.info("Analysis outputs saved in %s", results_dir / "analysis")


if __name__ == "__main__":
    main()
