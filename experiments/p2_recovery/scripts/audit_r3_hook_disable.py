#!/usr/bin/env python3
"""Verify disabling diagnostic capture leaves fixed-boundary outputs unchanged."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P0_DIR = ROOT / "experiments/p0_v2_fixed_boundary/scripts"
P1_DIR = ROOT / "experiments/p1_state_conditioned/scripts"
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
for value in (SCRIPT_DIR, P0_DIR, P1_DIR, P2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import FixedBoundaryReadoutMap, full_replay  # noqa: E402
from p2_core import atomic_json, state_local_symmetric_fd  # noqa: E402
from run_p1 import (  # noqa: E402
    _history_bundle,
    _state_delta,
    load_fp32_model,
)
from run_r3 import (  # noqa: E402
    disable_readout_recording,
    load_config,
    model_protocol,
)


def main() -> None:
    config = load_config()
    protocol, stage = model_protocol(config, "calibration")
    model, model_info, samples, _events = load_fp32_model(
        protocol, stage
    )
    try:
        sample = samples[0]
        reference = model.generate_reference(
            sample.sample_id, sample.task, sample.prompt
        )
        target = int(
            config["data"]["calibration"]["target_anchors"][0]
        )
        base_logits, base_record, base_positions, _dtypes = (
            full_replay(model, reference, target)
        )
        histories = _history_bundle(
            model,
            reference,
            protocol,
            target,
            base_logits,
            base_record,
            base_positions,
        )
        boundary = 15
        downstream = FixedBoundaryReadoutMap(
            model, reference.anchors[target], base_record, boundary
        )
        delta = _state_delta(
            histories["H1"], base_record, boundary
        )
        direction = np.linspace(
            -1.0, 1.0, int(model_info["hidden_size"])
        ).astype(np.float64)
        state = model.runner.attention_state
        state["enabled"] = True
        state["temporal_record_diagnostics"] = True
        output_recording = downstream.evaluate(delta)
        derivative_recording = state_local_symmetric_fd(
            downstream,
            delta,
            direction,
            float(config["numeric_backend"]["relative_radius"]),
        )["derivative"]
        previous = disable_readout_recording(model)
        output_disabled = downstream.evaluate(delta)
        derivative_disabled = state_local_symmetric_fd(
            downstream,
            delta,
            direction,
            float(config["numeric_backend"]["relative_radius"]),
        )["derivative"]
        output_error = float(
            np.max(np.abs(output_recording - output_disabled))
        )
        derivative_error = float(
            np.max(
                np.abs(
                    derivative_recording - derivative_disabled
                )
            )
        )
        result = {
            "passed": bool(
                output_error == 0.0 and derivative_error == 0.0
            ),
            "sample_id": sample.sample_id,
            "target": target,
            "boundary": boundary,
            "previous_recording_state": previous,
            "output_max_absolute_error": output_error,
            "directional_derivative_max_absolute_error": (
                derivative_error
            ),
            "comparison": (
                "diagnostic capture enabled versus disabled"
            ),
        }
        atomic_json(
            ROOT
            / "experiments/p2_recovery/"
            "r3_path_integrated_readout/results/"
            "hook_disable_equivalence_audit.json",
            result,
        )
        if not result["passed"]:
            raise SystemExit(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
