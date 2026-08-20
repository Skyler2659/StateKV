#!/usr/bin/env python3
"""Reproduce one matched StateKV baseline aggregate in the new namespace."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from statekv.storage import atomic_frame, atomic_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "results/temporal_cache_discovery/statekv_pure_eviction_qwen3_8b_p35_v1"
)
OUTPUT = ROOT / "results/adaptive_temporal/reproduction/p35_attention_budget256"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(SOURCE / "sample_results.csv")
    steps = pd.read_parquet(SOURCE / "step_rows.parquet")
    rows = samples[
        samples["total_budget"].eq(256) & samples["policy"].eq("attention")
    ].copy()
    step_rows = steps[
        steps["total_budget"].eq(256) & steps["policy"].eq("attention")
    ].copy()
    if len(rows) != 10 or len(step_rows) != 640:
        raise RuntimeError("matched P35 attention substrate is incomplete")
    reproduced = {
        "source_run": str(SOURCE.relative_to(ROOT)),
        "method": "attention",
        "samples": int(len(rows)),
        "steps": int(len(step_rows)),
        "total_budget": 256,
        "core_budget": 220,
        "mean_trajectory_exact_kl": float(rows["mean_trajectory_exact_kl"].mean()),
        "mean_official_score": float(rows["official_score"].mean()),
        "mean_niah_retrieval": float(rows["needle_retrieval_accuracy"].mean()),
        "mean_govreport_rouge_l": float(rows["rouge_l"].mean()),
        "all_irreversible_inclusions_hold": bool(
            rows["irreversible_set_inclusion_all_cycles"].all()
        ),
        "all_budgets_respected": bool(
            rows["global_kv_budget_respected_all_cycles"].all()
        ),
    }
    canonical = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    expected = next(
        row
        for row in canonical["policy_aggregates"]
        if int(row["total_budget"]) == 256 and row["policy"] == "attention"
    )
    checks = {
        "mean_trajectory_exact_kl": abs(
            reproduced["mean_trajectory_exact_kl"] - expected["mean_exact_kl"]
        )
        < 1.0e-12,
        "mean_official_score": abs(
            reproduced["mean_official_score"] - expected["mean_official_score"]
        )
        < 1.0e-12,
        "mean_niah_retrieval": abs(
            reproduced["mean_niah_retrieval"] - expected["mean_niah_retrieval"]
        )
        < 1.0e-12,
    }
    result = {
        "status": "reproduced" if all(checks.values()) else "mismatch",
        "reproduced": reproduced,
        "canonical": expected,
        "checks": checks,
    }
    atomic_frame(rows, OUTPUT / "matched_sample_rows.csv")
    atomic_json(OUTPUT / "summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

