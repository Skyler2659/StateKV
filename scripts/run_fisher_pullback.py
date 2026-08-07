#!/usr/bin/env python
"""Apply the Stage-A gate before any Fisher-pullback computation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.config import load_discovery_config
from statekv.fisher_pullback import load_gate_and_apply_skips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    run_dir = (
        REPOSITORY_ROOT
        / cfg.runtime.output_root
        / str(cfg.runtime.run_id)
    )
    decision = load_gate_and_apply_skips(run_dir)
    if decision["status"] == "stage_b_authorized":
        raise RuntimeError(
            "Stage B is authorized; run the model-backed pullback collector."
        )
    print(decision)


if __name__ == "__main__":
    main()
