#!/usr/bin/env python3
from pathlib import Path
import argparse
import yaml

from statekv.causal_existence import sample_id_for
from statekv.causal_rollout import run_causal_rollout_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/statekv_existence/causal_existence_qwen3_8b.yaml",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cycle", action="append", type=int, dest="cycles")
    parser.add_argument("--counterfactual", action="store_true")
    parser.add_argument(
        "--implementation", action="append", dest="implementations"
    )
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--distillation-train-subset", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    sample_ids = args.sample_ids
    if args.distillation_train_subset:
        if sample_ids:
            raise ValueError("choose explicit sample IDs or the distillation subset")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        sample_ids = [
            sample_id_for(str(family), int(index))
            for family in config["task_families"]
            for index in config["distillation"]["train_indices"]
        ]
    print(
        run_causal_rollout_study(
            config_path,
            root,
            split=args.split,
            max_samples=args.max_samples,
            cycles=args.cycles or (0, 8, 16, 24),
            counterfactual=args.counterfactual,
            implementations=args.implementations,
            sample_ids=sample_ids,
        )
    )


if __name__ == "__main__":
    main()
