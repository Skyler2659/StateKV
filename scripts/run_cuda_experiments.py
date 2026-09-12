#!/usr/bin/env python3
"""Build a queue, run one GPU worker, or resume one CUDA experiment job."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks/torch"), str(ROOT / "benchmarks/mlx")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "worker", "job", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/cuda/validation.yaml")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--gpu", type=int, choices=(3, 4, 5, 6))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "job":
        from statekv.cuda_experiments import run_job

        run_job(args.root, json.loads((args.output / "config.json").read_text()), args.output)
    else:
        from statekv.cuda_queue import build_queue, status, worker

        if args.mode == "build":
            print(build_queue(args.plan, args.root))
        elif args.mode == "status":
            if args.run is None:
                parser.error("status needs --run")
            print(json.dumps(status(args.run, json.loads((args.run / "queue.json").read_text())["jobs"])))
        else:
            if args.gpu is None or args.run is None:
                parser.error("worker needs --gpu and --run")
            worker(args.root, args.run, args.gpu)


if __name__ == "__main__":
    main()
