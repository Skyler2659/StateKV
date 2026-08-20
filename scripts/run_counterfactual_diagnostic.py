#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.counterfactual_diagnostic import run_counterfactual_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_counterfactual/cf_diagnostic_dev_v1.yaml",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(
        run_counterfactual_diagnostic(
            root / args.config,
            root,
            max_samples=args.max_samples,
            sample_ids=args.sample_ids,
        )
    )


if __name__ == "__main__":
    main()
