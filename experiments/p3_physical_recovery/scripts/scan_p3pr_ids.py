#!/usr/bin/env python3
"""Verify automatic exclusion and construction of every frozen P3PR ID."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (
    ROOT / "experiments/p1_state_conditioned/scripts",
    SCRIPT_DIR,
):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p3pr_core import atomic_json, discover_old_ids, source_integrity  # noqa: E402
from run_p1 import load_fp32_model  # noqa: E402
from run_p3pr import model_protocol  # noqa: E402


EXPERIMENT = ROOT / "experiments/p3_physical_recovery"


def all_allocated(config: Dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(value)
            for stage, payload in config["data"].items()
            if isinstance(payload, dict) and "gov_report_indices" in payload
            for key in ("gov_report_indices", "niah_offsets")
            for value in payload[key]
        }
    )


def main() -> None:
    config_path = EXPERIMENT / "p3pr_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_checks = source_integrity(config)
    exclusions = discover_old_ids()
    planned = all_allocated(config)
    overlap = sorted(set(planned) & set(exclusions["ids"]))
    if overlap:
        raise RuntimeError(f"new/old ID overlap: {overlap}")

    # Reuse already verified rows and construct only newly appended frozen
    # roles.  This preserves the original scan while avoiding needless
    # regeneration of IDs 110--129.
    previous_path = EXPERIMENT / "results/data_scan_110_129.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else {"rows": []}
    )
    previous_rows = {
        str(row["sample_id"]): row
        for row in previous.get("rows", [])
        if bool(row.get("success"))
    }
    planned_sample_ids = {
        *(f"gov_report:{value}" for value in planned),
        *(f"synthetic_niah_{value}" for value in planned),
    }
    missing_ids = planned_sample_ids - set(previous_rows)
    missing_numbers = sorted(
        {
            int(sample_id.split(":")[-1].split("_")[-1])
            for sample_id in missing_ids
        }
    )

    # Use one merged evaluation section so every newly appended role is
    # constructed before any of those result roles is opened.
    protocol = model_protocol(config, "diagnostic")
    protocol["data"]["evaluation"]["gov_report_indices"] = missing_numbers
    protocol["data"]["evaluation"]["niah_offsets"] = missing_numbers
    protocol["data"]["evaluation"]["target_anchors"] = [40, 48, 64]
    backend, model_info, samples, events = load_fp32_model(
        protocol, "evaluation"
    )
    rows = list(previous_rows.values())
    required = [32, 40, 48, 64]
    try:
        for sample in samples:
            try:
                reference = backend.generate_reference(
                    sample.sample_id, sample.task, sample.prompt
                )
                missing = [
                    value for value in required if value not in reference.anchors
                ]
                row = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "success": bool(
                        len(reference.generated_token_ids)
                        == int(config["runtime"]["max_new_tokens"])
                        and not missing
                        and not reference.prompt_truncated
                    ),
                    "generated_tokens": len(reference.generated_token_ids),
                    "prompt_length": int(reference.prompt_length),
                    "prompt_truncated": bool(reference.prompt_truncated),
                    "missing_anchors": missing,
                }
            except Exception as error:
                row = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        result = {
            "rule": (
                "all frozen P3PR IDs must be absent from recursively "
                "discovered old IDs and construct 66 teacher-forced tokens "
                "with anchors 32, 40, 48, and 64"
            ),
            "source_checks": source_checks,
            "automatic_exclusions": exclusions,
            "planned_ids": planned,
            "new_old_overlap": overlap,
            "required_generated_tokens": int(
                config["runtime"]["max_new_tokens"]
            ),
            "required_anchors": required,
            "rows": rows,
            "all_pass": bool(rows)
            and not overlap
            and all(bool(row["success"]) for row in rows),
            "model_info": model_info,
            "dataset_events": events,
        }
        destination = EXPERIMENT / "results/data_scan_all_allocated.json"
        atomic_json(destination, result)
        if not result["all_pass"]:
            raise RuntimeError("P3PR frozen data scan failed")
        print(
            json.dumps(
                {
                    "all_pass": True,
                    "sequence_count": len(rows),
                    "excluded_id_count": exclusions["count"],
                    "planned_id_count": len(planned),
                },
                indent=2,
            )
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
