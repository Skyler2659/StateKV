#!/usr/bin/env python3
"""Construct every preregistered P3 sequence before role allocation is used."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[3]
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
P3_DIR = Path(__file__).resolve().parent
for value in (P1_DIR, P3_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p3_core import atomic_json  # noqa: E402
from run_p1 import load_fp32_model  # noqa: E402


def model_protocol(config: Dict[str, Any]) -> Dict[str, Any]:
    protocol = yaml.safe_load(
        (ROOT / "configs/frozen/p2_state_local_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    values = sorted(
        {
            int(value)
            for role in (
                "diagnostic",
                "calibration",
                "evaluation",
                "replication",
                "physical_evaluation",
                "physical_replication",
            )
            for key in ("gov_report_indices", "niah_offsets")
            for value in config["data"][role][key]
        }
    )
    section = copy.deepcopy(protocol["data"]["evaluation"])
    section["gov_report_indices"] = values
    section["niah_offsets"] = values
    section["target_anchors"] = list(
        config["trajectory"]["target_anchors"]
    )
    section["layers"] = list(config["trajectory"]["layers"])
    protocol["data"]["evaluation"] = section
    protocol["runtime"]["run_id"] = config["runtime"]["run_id"]
    return protocol


def main() -> None:
    config_path = (
        ROOT / "experiments/p3_decision_validity/p3_config.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = model_protocol(config)
    model, model_info, samples, events = load_fp32_model(
        protocol, "evaluation"
    )
    required = [
        int(value) for value in config["trajectory"]["target_anchors"]
    ]
    rows = []
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
                        "task": sample.task,
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
                print(
                    json.dumps(rows[-1], ensure_ascii=False),
                    flush=True,
                )
            except Exception as error:
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "success": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
        result = {
            "rule": (
                "all preregistered IDs 98--109 must construct with "
                "66 teacher-forced tokens and every P3 target anchor"
            ),
            "required_generated_tokens": 66,
            "required_anchors": required,
            "rows": rows,
            "all_pass": bool(rows) and all(row["success"] for row in rows),
            "model_info": model_info,
            "dataset_events": events,
        }
        destination = (
            ROOT
            / "experiments/p3_decision_validity/results/"
            "data_scan_98_109.json"
        )
        atomic_json(destination, result)
        if not result["all_pass"]:
            raise RuntimeError("P3 ID scan requires an allowed extension")
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        model.close()


if __name__ == "__main__":
    main()
