#!/usr/bin/env python3
"""Mechanically construct the next unused Recovery sequence IDs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
for value in (P1_DIR, P2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p1_core import atomic_json  # noqa: E402
from run_p1 import load_fp32_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=86)
    parser.add_argument("--end", type=int, default=91)
    args = parser.parse_args()
    if args.end < args.start:
        raise ValueError("--end must be at least --start")
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    values = list(range(args.start, args.end + 1))
    protocol["data"]["evaluation"]["gov_report_indices"] = values
    protocol["data"]["evaluation"]["niah_offsets"] = values
    model, model_info, samples, events = load_fp32_model(
        protocol, "evaluation"
    )
    rows = []
    required = [16, 24, 32, 40, 48, 56, 64]
    try:
        for sample in samples:
            try:
                reference = model.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                missing = [
                    anchor
                    for anchor in required
                    if anchor not in reference.anchors
                ]
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "success": (
                            len(reference.generated_token_ids) == 66
                            and not missing
                            and not reference.prompt_truncated
                        ),
                        "generated_tokens": len(
                            reference.generated_token_ids
                        ),
                        "prompt_length": reference.prompt_length,
                        "prompt_truncated": reference.prompt_truncated,
                        "missing_anchors": missing,
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "success": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        result = {
            "rule": (
                f"all requested IDs from {args.start} through "
                f"{args.end} must pass before allocation"
            ),
            "scanned_ids": values,
            "required_generated_tokens": 66,
            "required_anchors": required,
            "rows": rows,
            "all_pass": all(row["success"] for row in rows),
            "model_info": model_info,
            "dataset_events": events,
        }
        destination = (
            ROOT
            / "experiments/p2_recovery/results/"
            f"data_scan_{args.start}_{args.end}.json"
        )
        atomic_json(destination, result)
        if not result["all_pass"]:
            raise RuntimeError("recovery ID scan requires extension")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
