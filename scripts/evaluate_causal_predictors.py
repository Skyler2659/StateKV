#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.causal_predictors import evaluate_causal_predictors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(evaluate_causal_predictors(root / args.config, root, split=args.split))


if __name__ == "__main__":
    main()

