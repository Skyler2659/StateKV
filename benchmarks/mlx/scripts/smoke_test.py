#!/usr/bin/env python3
"""No-pytest smoke tests for P0 framework repairs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DEV_CONFIG = "configs/experiments/dev/tiny_niah_cpu.yaml"
TMP_OUTPUT = Path("/tmp/l1_robust_kv_cache_smoke")


def run(cmd):
    print("$", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr)
        raise SystemExit(completed.returncode)
    return completed


def core_assertions():
    from scripts.run_benchmark import instantiate_benchmark
    from src.config import ExperimentConfig
    from src.eviction.base import validate_selected_indices
    from src.eviction.l2_leverage import l2_row_leverage_scores
    from src.eviction.registry import create_eviction

    cfg = ExperimentConfig.from_yaml(ROOT / DEV_CONFIG)
    assert instantiate_benchmark(cfg).name == "niah"

    for method in ["recency", "sink_recent", "attention", "l1_leverage", "l2_leverage", "attention+l1"]:
        eviction = create_eviction(
            method,
            cache_size=8,
            k_seq_dim=2,
            v_seq_dim=2,
            sink_size=2,
            recent_size=3,
            sketch_dim=16,
            debug_budget=True,
        )
        k = torch.randn(1, 2, 20, 4)
        v = torch.randn(1, 2, 20, 4)
        if method == "attention":
            eviction.update_attention(0, torch.softmax(torch.randn(1, 2, 1, 20), dim=-1))
        eviction(((k, v),))
        validate_selected_indices(eviction.last_selected[0], 20, 8)

    rows = torch.randn(12, 4)
    scores = l2_row_leverage_scores(rows)
    assert abs(float(scores.sum()) - torch.linalg.matrix_rank(rows.float()).item()) < 1e-4
    print("core assertions ok")


def assert_run_results(run_dir: Path) -> None:
    results = json.loads((run_dir / "results.json").read_text())
    assert results, "CPU smoke produced no result rows"
    assert not [row for row in results if row.get("error")], results
    assert not [row for row in results if row.get("sanity_check_failed")], results
    score_methods = {"attention", "l1_leverage", "l2_leverage", "attention_l1"}
    for row in results:
        if row.get("method") not in score_methods:
            continue
        assert row.get("has_scores"), row
        raw = (row.get("score_stats") or {}).get("raw_score_stats") or {}
        assert int(raw.get("finite_numel") or 0) > 0, row
        assert not raw.get("all_non_finite"), row


def main():
    core_assertions()
    py = sys.executable
    run([py, "scripts/run_benchmark.py", "--config", DEV_CONFIG, "--num_samples", "1", "--progress_every", "120", "--skip_analysis"])
    latest = sorted((TMP_OUTPUT / "tiny_niah_cpu").glob("20*"))[-1]
    assert_run_results(latest)
    run([py, "scripts/run_analysis.py", "--input", str(latest), "--config", "configs/analysis/basic.yaml"])
    run([
        py,
        "scripts/run_profile.py",
        "--config",
        DEV_CONFIG,
        "--max_steps",
        "32",
        "--warmup",
        "1",
        "--repeats",
        "1",
        "--output_dir",
        str(TMP_OUTPUT / "profile"),
    ])
    print("smoke test ok")


if __name__ == "__main__":
    main()
