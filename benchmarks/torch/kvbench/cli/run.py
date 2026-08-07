"""Run one atomic Torch/CUDA experiment."""
from __future__ import annotations

import argparse
import json
import sys

from kvbench.config import load_experiment
from kvbench.methods.policy import list_methods
from kvbench.runners.experiment import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch KV-cache paper benchmark")
    parser.add_argument("--config", required=False)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Dotted override, for example method.name=residual_v",
    )
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()
    if args.list_methods:
        print(json.dumps(list_methods(), indent=2))
        return
    if not args.config:
        parser.error("--config is required unless --list-methods is used")
    cfg = load_experiment(args.config, args.overrides)
    runner = ExperimentRunner(cfg, command=sys.argv)
    run_dir = runner.run()
    print(json.dumps({"run_dir": run_dir}, indent=2))


if __name__ == "__main__":
    main()

