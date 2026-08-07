#!/usr/bin/env python3
"""Calibrate k=1/2/4/8 path rules at the frozen minimum probe boundary."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (
    ROOT,
    ROOT / "benchmarks/torch",
    ROOT / "experiments/p0_v2_fixed_boundary/scripts",
    ROOT / "experiments/p1_state_conditioned/scripts",
    SCRIPT_DIR,
):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from p0_v2_core import FixedBoundaryReadoutMap  # noqa: E402
from p1_core import clear_runtime_controls  # noqa: E402
from run_p1 import load_fp32_model  # noqa: E402
from p3pr_core import atomic_frame, atomic_json, clone_mlx_state, exact_kl, state_to_anchor  # noqa: E402
from run_p3pr import (  # noqa: E402
    _path_delta,
    _tensor,
    model_protocol,
    prequery_physical_state,
)


def main() -> None:
    import mlx.core as mx

    config = yaml.safe_load(
        (EXPERIMENT / "p3pr_config.yaml").read_text(encoding="utf-8")
    )
    request = json.loads(
        (EXPERIMENT / "results/path_calibration_request.json").read_text(
            encoding="utf-8"
        )
    )
    layer = int(request["layer"])
    boundary = int(request["boundary"])
    midpoint_grid = [int(value) for value in request["midpoint_grid"]]
    existing = pd.read_parquet(
        EXPERIMENT / "results/calibration/candidate_rows.parquet"
    )
    registry = pd.read_parquet(
        EXPERIMENT / "results/calibration/candidate_registry.parquet"
    )
    protocol = model_protocol(config, "calibration")
    backend, model_info, samples, events = load_fp32_model(
        protocol, "evaluation"
    )
    rows = []
    try:
        for sample in samples:
            reference = backend.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            for target in sorted(
                existing.loc[
                    existing["sample_id"].eq(sample.sample_id),
                    "target_anchor",
                ].unique()
            ):
                state, _fixed, _trace = prequery_physical_state(
                    backend,
                    reference,
                    protocol,
                    int(config["physical_state"]["history_start_anchor"]),
                    int(target),
                )
                token = int(reference.anchors[int(target)].query_token_id)
                baseline_state = clone_mlx_state(state)
                clear_runtime_controls(backend)
                base_logits_tensor, base_record, _elapsed = backend.forward_one(
                    baseline_state, token, capture_attention=True
                )
                base_logits = base_logits_tensor.double().numpy()
                anchor = state_to_anchor(
                    backend, baseline_state, token, int(target)
                )
                readout = FixedBoundaryReadoutMap(
                    backend, anchor, base_record, boundary
                )
                selected = registry[
                    registry["sample_id"].eq(sample.sample_id)
                    & registry["target_anchor"].eq(int(target))
                ].sort_values("candidate_id")
                for candidate in selected.itertuples(index=False):
                    from run_p3pr import _candidate_branch

                    _candidate_logits, candidate_record, branch = (
                        _candidate_branch(
                            backend,
                            state,
                            int(candidate.deleted_position),
                            token,
                        )
                    )
                    direction = (
                        candidate_record.residual_inputs[boundary]
                        - base_record.residual_inputs[boundary]
                    ).double().numpy()
                    record = {
                        "sample_id": sample.sample_id,
                        "task": sample.task,
                        "stage": "calibration",
                        "target_anchor": int(target),
                        "candidate_id": candidate.candidate_id,
                        "candidate_source": candidate.candidate_source,
                        "boundary": boundary,
                    }
                    for count in midpoint_grid:
                        delta = _path_delta(
                            readout,
                            direction,
                            count,
                            float(
                                config["representations"][
                                    "finite_difference_relative_radius"
                                ]
                            ),
                        )
                        record[f"probe_b{boundary}_path_k{count}_risk"] = (
                            exact_kl(base_logits, base_logits + delta)
                        )
                    rows.append(record)
                    backend.release(branch)
                    mx.synchronize()
                    gc.collect()
                    mx.clear_cache()
                backend.release(baseline_state, state)
                print(
                    json.dumps(
                        {
                            "event": "p3pr_path_unit_complete",
                            "sample_id": sample.sample_id,
                            "target": int(target),
                            "boundary": boundary,
                        }
                    ),
                    flush=True,
                )
        path = pd.DataFrame(rows)
        keys = [
            "sample_id",
            "task",
            "target_anchor",
            "candidate_id",
            "candidate_source",
        ]
        score_columns = [
            f"probe_b{boundary}_path_k{count}_risk"
            for count in midpoint_grid
        ]
        merged = existing.merge(
            path[keys + score_columns],
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if merged[score_columns].isna().any().any():
            raise RuntimeError("path calibration merge is incomplete")
        atomic_frame(
            EXPERIMENT / "results/calibration/path_calibration_rows.parquet",
            path,
        )
        path.to_csv(
            EXPERIMENT / "results/calibration/path_calibration_rows.csv",
            index=False,
        )
        atomic_frame(
            EXPERIMENT / "results/calibration/candidate_rows.parquet",
            merged,
        )
        merged.to_csv(
            EXPERIMENT / "results/calibration/candidate_rows.csv",
            index=False,
        )
        summary = {
            "completed": True,
            "boundary": boundary,
            "layer": layer,
            "midpoint_grid": midpoint_grid,
            "row_count": len(path),
            "sample_ids": [sample.sample_id for sample in samples],
            "model_info": model_info,
            "dataset_events": events,
        }
        atomic_json(
            EXPERIMENT / "results/calibration/path_calibration_metadata.json",
            summary,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        backend.close()


if __name__ == "__main__":
    main()

