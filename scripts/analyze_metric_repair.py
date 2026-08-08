#!/usr/bin/env python
"""Run the frozen-vector low-cost metric-repair screen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.metric_repair_analysis import analyze_metric_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(analyze_metric_repair(Path(args.config), REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
