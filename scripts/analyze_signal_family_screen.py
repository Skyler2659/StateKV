#!/usr/bin/env python
"""Select among frozen training-free signal families on a development run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.direct_policy_analysis import select_direct_policy_candidate
from statekv.storage import atomic_frame, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(config["output_run"])
    baseline = str(config["baseline"])
    candidates = [str(value) for value in config["development_selection"]["candidates"]]
    result, audit = select_direct_policy_candidate(
        pd.read_csv(run_dir / "metrics.csv"),
        pd.read_csv(run_dir / "analysis" / "stratified_metrics.csv"),
        baseline,
        candidates,
    )
    result["signal_families"] = dict(config["signal_families"])
    result["development_sample_ids"] = sorted(str(value) for value in config["sample_ids"])
    result["reserved_independent_sample_ids"] = sorted(
        str(value) for value in config["reserved_independent_sample_ids"]
    )
    analysis_dir = run_dir / "analysis"
    atomic_frame(audit, analysis_dir / "signal_family_selection.csv")
    atomic_json(analysis_dir / "signal_family_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
