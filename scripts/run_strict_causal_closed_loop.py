#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.causal_closed_loop import (
    merge_closed_loop_shards,
    run_strict_causal_closed_loop,
    select_validation_refresh_frequency,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--split", default="closed_loop_test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--budget", action="append", type=int, dest="budgets")
    parser.add_argument("--cycle-limit", type=int)
    parser.add_argument("--refresh-frequency", type=int)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--policy", action="append", dest="policies")
    parser.add_argument("--output-tag")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--select-validation-refresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.select_validation_refresh:
        print(select_validation_refresh_frequency(root / args.config, root))
        return
    if args.merge_shards:
        print(merge_closed_loop_shards(root / args.config, root, args.split))
        return
    print(
        run_strict_causal_closed_loop(
            root / args.config,
            root,
            split=args.split,
            max_samples=args.max_samples,
            budgets=args.budgets,
            cycle_limit=args.cycle_limit,
            refresh_frequency=args.refresh_frequency,
            sample_ids=args.sample_ids,
            policies=args.policies,
            output_tag=args.output_tag,
        )
    )


if __name__ == "__main__":
    main()
