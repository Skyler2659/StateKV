#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.causal_distillation import train_rollout_distilled_predictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(train_rollout_distilled_predictor(root / args.config, root))


if __name__ == "__main__":
    main()
