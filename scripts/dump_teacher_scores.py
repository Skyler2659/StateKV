#!/usr/bin/env python3
"""Repackage per-token R2 teacher scores for student training.

The causal rollout study already persists per-token R2 scores under
``<source-run>/rollout/<split>/teacher_scores/``.  This CPU-only script
copies them into the counterfactual results tree, validates their shape and
causal contract, and records the (sample_id, cycle, position) join to the
full-cache feature artifacts.  The source tree is never modified.
"""
from pathlib import Path
import argparse

from statekv.causal_student import dump_teacher_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        default="results/statekv_existence/causal_existence_qwen3_8b_v1",
    )
    parser.add_argument(
        "--dest",
        default="results/statekv_counterfactual/teacher_scores",
    )
    parser.add_argument("--split", action="append", dest="splits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        dump_teacher_scores(
            root / args.source_run,
            root / args.dest,
            splits=args.splits or ("train", "validation"),
        )
    )


if __name__ == "__main__":
    main()
