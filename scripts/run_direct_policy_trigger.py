#!/usr/bin/env python
"""Extract direct-policy trigger features and screen or validate one rule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for root in (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from statekv.direct_policy_trigger import run_direct_policy_trigger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_direct_policy_trigger(Path(args.config), REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
