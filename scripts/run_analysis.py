#!/usr/bin/env python3
"""Post-hoc analysis runner for saved benchmark results."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig, load_analysis_config
from src.utils.io import load_results, load_scores, save_results
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("run_analysis")


def _load_selected(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("selected_tokens_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = load_scores(p)
    return {int(k): v for k, v in data.items()}


def _load_score_dict(result: Dict[str, Any]) -> Optional[Dict[int, torch.Tensor]]:
    path = result.get("scores_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = load_scores(p)
    return {int(k): v for k, v in data.items()}


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_analysis(results: List[Dict], cfg: ExperimentConfig, output_dir: Path) -> Dict[str, Any]:
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_results: Dict[str, Any] = {}

    if cfg.analysis.overlap:
        logger.info("Running overlap analysis")
        analysis_results["overlap"] = analyze_overlap(results, analysis_dir)

    if cfg.analysis.rank_correlation:
        logger.info("Running rank correlation analysis")
        analysis_results["rank_correlation"] = analyze_rank_correlation(results, analysis_dir)

    if cfg.analysis.evidence_recall:
        logger.info("Running evidence recall analysis")
        analysis_results["evidence_recall"] = analyze_evidence_recall(results, analysis_dir)

    if cfg.analysis.case_study:
        logger.info("Exporting case studies")
        analysis_results["case_study"] = analyze_case_studies(
            results, analysis_dir, cfg.analysis.case_study_count
        )

    save_results(analysis_results, analysis_dir / "analysis_summary.json")
    return analysis_results


def analyze_overlap(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.overlap import OverlapAnalyzer

    analyzer = OverlapAnalyzer()
    grouped = defaultdict(dict)
    for result in results:
        selected = _load_selected(result)
        if selected is None:
            continue
        key = (result.get("sample_id", result.get("sample_idx")), result.get("budget"))
        grouped[key][result["method"]] = selected

    pair_values = defaultdict(list)
    for methods in grouped.values():
        names = sorted(methods)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = analyzer.pairwise_overlap(methods[left], methods[right], left, right)
                pair_values[f"{left}_vs_{right}"].append(overlap.jaccard)

    summary = {pair: _mean(vals) for pair, vals in pair_values.items()}
    save_results(summary, analysis_dir / "overlap_summary.json")
    return summary or {"note": "no selected token files"}


def analyze_rank_correlation(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.rank_correlation import RankCorrelationAnalyzer

    analyzer = RankCorrelationAnalyzer()
    grouped = defaultdict(dict)
    for result in results:
        scores = _load_score_dict(result)
        if scores is None:
            continue
        key = (result.get("sample_id", result.get("sample_idx")), result.get("budget"))
        grouped[key][result["method"]] = scores

    pair_values = defaultdict(list)
    for methods in grouped.values():
        names = sorted(methods)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                corr = analyzer.layer_wise(methods[left], methods[right], left, right)
                pair_values[f"{left}_vs_{right}"].append(corr.spearman)

    summary = {pair: _mean(vals) for pair, vals in pair_values.items()}
    save_results(summary, analysis_dir / "rank_correlation_summary.json")
    return summary or {"note": "no score files"}


def analyze_evidence_recall(results: List[Dict], analysis_dir: Path) -> Dict[str, Any]:
    from src.analysis.evidence_recall import EvidenceRecallAnalyzer

    analyzer = EvidenceRecallAnalyzer()
    values = defaultdict(list)
    details = []
    for result in results:
        selected = _load_selected(result)
        evidence = result.get("evidence_positions") or []
        if selected is None or not evidence:
            continue
        recall = analyzer.analyze(
            selected_indices=selected,
            evidence_positions=torch.tensor(evidence, dtype=torch.long),
            seq_len=int(result.get("context_length", 0)),
            method_name=result["method"],
        )
        values[result["method"]].append(recall.evidence_recall)
        details.append(asdict(recall))

    summary = {method: _mean(vals) for method, vals in values.items()}
    save_results(details, analysis_dir / "evidence_recall_details.json")
    save_results(summary, analysis_dir / "evidence_recall_summary.json")
    return summary or {"note": "no evidence positions or selected token files"}


def analyze_case_studies(
    results: List[Dict], analysis_dir: Path, count: int = 5
) -> Dict[str, Any]:
    by_sample = defaultdict(list)
    for result in results:
        if "error" not in result:
            by_sample[result.get("sample_id", result.get("sample_idx"))].append(result)

    cases = []
    for sample_id, sample_results in by_sample.items():
        attention = next((r for r in sample_results if r.get("method") == "attention"), None)
        l1 = next((r for r in sample_results if r.get("method") == "l1_leverage"), None)
        if not attention or not l1:
            continue
        attn_ppl = attention.get("ppl", float("inf"))
        l1_ppl = l1.get("ppl", float("inf"))
        category = "l1_lower_ppl" if l1_ppl < attn_ppl else "attention_lower_or_tie"
        cases.append(
            {
                "sample_id": sample_id,
                "category": category,
                "attention_ppl": attn_ppl,
                "l1_ppl": l1_ppl,
                "ground_truth": attention.get("ground_truth"),
                "evidence_positions": attention.get("evidence_positions"),
                "attention_selected_tokens_path": attention.get("selected_tokens_path"),
                "l1_selected_tokens_path": l1.get("selected_tokens_path"),
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
    parser.add_argument("--input", "--results_dir", dest="input", type=str, required=True)
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
    logger.info("Analysis complete: %s", summary)
    logger.info("Analysis outputs saved in %s", results_dir / "analysis")


if __name__ == "__main__":
    main()
