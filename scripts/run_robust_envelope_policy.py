#!/usr/bin/env python
"""Run matched-count stateful robust-envelope refresh policies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.config import load_discovery_config
from statekv.robust_envelope_analysis import atomic_json
from statekv.robust_envelope_policy import (
    RobustEnvelopePolicyRunner,
    summarize_policy_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--threshold-only", action="store_true")
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = Path(args.run_dir).resolve()
    path = RobustEnvelopePolicyRunner(
        cfg,
        REPOSITORY_ROOT,
        run_dir,
        oracle_only=args.oracle_only,
        threshold_only=args.threshold_only,
    ).run()
    import pandas as pd

    summary = summarize_policy_rows(pd.read_parquet(path), cfg)
    summary_path = run_dir / "envelope_refresh_policy_summary.json"
    atomic_json(summary_path, summary)
    status_path = run_dir / "status.json"
    if status_path.exists():
        with status_path.open() as handle:
            status = json.load(handle)
        status["state"] = "complete"
        status["analysis_complete"] = True
        status["robust_envelope_policy_complete"] = True
        atomic_json(status_path, status)
    print(path)
    print(summary_path)


if __name__ == "__main__":
    main()
