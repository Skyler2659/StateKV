#!/usr/bin/env python3
"""Fail-fast check for the pinned Linux CUDA paper environment."""
from __future__ import annotations

import argparse
import json
import platform
import sys


EXPECTED = {
    "python": "3.9.6",
    "torch": "2.4.1",
    "transformers": "4.46.3",
    "accelerate": "1.1.1",
    "cuda_runtime": "12.1",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    import accelerate
    import torch
    import transformers

    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__.split("+")[0],
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
    }
    errors = []
    for name in ("python", "torch", "transformers", "accelerate"):
        if actual[name] != EXPECTED[name]:
            errors.append("%s=%s expected=%s" % (name, actual[name], EXPECTED[name]))
    if not args.allow_cpu:
        if not actual["cuda_available"]:
            errors.append("CUDA is not available")
        if actual["cuda_runtime"] != EXPECTED["cuda_runtime"]:
            errors.append(
                "torch CUDA runtime=%s expected=%s"
                % (actual["cuda_runtime"], EXPECTED["cuda_runtime"])
            )
    report = {"valid": not errors, "expected": EXPECTED, "actual": actual, "errors": errors}
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

