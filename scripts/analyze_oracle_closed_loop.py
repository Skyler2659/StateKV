#!/usr/bin/env python3
"""Analyze a completed StateKV physical-oracle closed-loop run."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.oracle_closed_loop_analysis import analyze_oracle_closed_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPOSITORY_ROOT / run_dir
    print(analyze_oracle_closed_loop(run_dir))


if __name__ == "__main__":
    main()
