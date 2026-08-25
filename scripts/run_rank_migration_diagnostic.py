#!/usr/bin/env python3
"""Rank-migration diagnostic runner (read-only instrumentation).

Runs the strict closed loop with the frozen R2 policy only and writes one
npz per (sample, budget, policy) arm under
``<output_run>/rank_migration/_shards/<output_tag>/<split>/``. No
closed_loop CSV/parquet artifacts are produced; decisions are identical to
the publication run.
"""
from pathlib import Path
import argparse

import yaml

from statekv.causal_closed_loop import run_strict_causal_closed_loop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    rank_migration_dir = (
        root
        / str(config["output_run"])
        / "rank_migration"
        / "_shards"
        / str(args.output_tag)
    )
    print(
        run_strict_causal_closed_loop(
            root / args.config,
            root,
            split=args.split,
            max_samples=args.max_samples,
            budgets=[256],
            sample_ids=args.sample_ids,
            policies=["STRICT_CAUSAL_ROLLOUT_R2"],
            output_tag=args.output_tag,
            rank_migration_dir=str(rank_migration_dir),
        )
    )


if __name__ == "__main__":
    main()
