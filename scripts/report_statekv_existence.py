#!/usr/bin/env python3
from pathlib import Path
import argparse

from statekv.existence_reporting import (
    build_existence_leaderboard,
    evaluate_closed_loop_gate,
    evaluate_existence_gates,
    freeze_validation_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--freeze-validation", action="store_true")
    parser.add_argument("--close-test", action="store_true")
    parser.add_argument("--evaluate-gates", action="store_true")
    parser.add_argument("--evaluate-closed-loop", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = root / args.config
    if args.freeze_validation:
        print(freeze_validation_selection(config, root))
    print(
        build_existence_leaderboard(
            config, root, split=args.split, close_test=args.close_test
        )
    )
    if args.evaluate_gates:
        print(evaluate_existence_gates(config, root))
    if args.evaluate_closed_loop:
        print(evaluate_closed_loop_gate(config, root))


if __name__ == "__main__":
    main()
