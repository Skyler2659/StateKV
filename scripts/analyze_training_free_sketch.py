#!/usr/bin/env python3
"""Run the StateKV-TF retrospective state-sketch feasibility experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.training_free_analysis import analyze_training_free_sketch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = analyze_training_free_sketch(
        Path(args.config).resolve(), REPOSITORY_ROOT
    )
    print(output)


if __name__ == "__main__":
    main()
