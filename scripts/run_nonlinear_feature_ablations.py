#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.causal_feature_ablation import (
    evaluate_nonlinear_feature_ablations,
    train_nonlinear_feature_ablations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--split")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / args.config
    if args.train:
        print(train_nonlinear_feature_ablations(config, root))
    if args.split:
        print(evaluate_nonlinear_feature_ablations(config, root, args.split))
    if not args.train and not args.split:
        raise ValueError("select --train and/or --split")


if __name__ == "__main__":
    main()
