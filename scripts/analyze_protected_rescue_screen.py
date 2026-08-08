#!/usr/bin/env python
"""Apply the frozen development gate to a protected-rescue replay run."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.direct_policy_analysis import select_protected_rescue_candidate
from statekv.storage import atomic_frame, atomic_json


def _rescue_slots(policy: str) -> int:
    match = re.search(r"_m(\d+)_", policy)
    if match is None:
        raise ValueError(f"cannot infer rescue slot count from policy: {policy}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    run_dir = Path(config["output_run"])
    baseline = str(config["baseline"])
    candidates = [
        policy
        for policy in config["policies"]
        if policy != baseline
    ]
    result, audit = select_protected_rescue_candidate(
        pd.read_csv(run_dir / "metrics.csv"),
        pd.read_csv(run_dir / "analysis" / "stratified_metrics.csv"),
        pd.read_parquet(run_dir / "selection_inventory.parquet"),
        baseline,
        candidates,
        {policy: _rescue_slots(policy) for policy in candidates},
    )
    analysis_dir = run_dir / "analysis"
    atomic_frame(audit, analysis_dir / "protected_rescue_selection.csv")
    atomic_json(analysis_dir / "protected_rescue_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
