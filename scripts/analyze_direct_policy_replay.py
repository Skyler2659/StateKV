#!/usr/bin/env python
"""Analyze paired sequence uncertainty in a direct-policy replay run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.direct_policy_analysis import analyze_direct_policy_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    result = analyze_direct_policy_replay(
        Path(args.run_dir),
        args.baseline,
        args.primary,
        args.bootstrap_samples,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
