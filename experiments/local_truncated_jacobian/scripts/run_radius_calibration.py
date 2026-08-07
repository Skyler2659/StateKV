#!/usr/bin/env python3
"""Run the frozen train-only radius calibration before formal L0."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREDICTIVE_SCRIPTS = (
    ROOT / "experiments/predictive_closure/scripts"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))
sys.path.insert(0, str(PREDICTIVE_SCRIPTS))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config
from kvbench.temporal.tasks import load_discovery_tasks

from mlx_predictive_core import single_layer_selection

from local_core import (
    FP32LocalBlock,
    atomic_frame,
    atomic_json,
    choose_radius,
    cosine,
    load_candidates,
    max_pairwise_noise,
    relative_l2,
    replay_record,
    symmetric_norm_ratio,
    to_physical_candidate,
)


REGISTRY = (
    ROOT
    / "experiments/predictive_closure/raw/p0_alignment"
    / "formal_4bit_retry1/candidate_registry_rows.parquet"
)
REGISTRY_SHA256 = (
    "f2d06b2732a2a0bf8baac6694ef35aa2ed4393a19e75400564a545786d787307"
)
RADII = (
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
)
LAYERS = (0, 7, 14, 21, 26)
SOURCES = {"attention_only", "random_reference"}


def run(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    import mlx.core as mx

    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "radius_calibration",
        "state": "running",
        "formal_matrix_touched": False,
        "split": "train",
        "calibration_or_test_loaded": False,
        "sample_id": "synthetic_niah_24",
        "errors": [],
    }
    atomic_json(output_dir / "status.json", status)
    cfg = load_discovery_config(str(config_path))
    cfg.tasks = {
        "ruler_niah": {
            "num_samples": 1,
            "context_length": 768,
            "sample_offset": 24,
        }
    }
    cfg.anchor_steps = [16]
    cfg.horizons = [1]
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = [16]
    cfg.independent_fisher.segment_horizon = 1
    samples, dataset_events = load_discovery_tasks(cfg)
    if len(samples) != 1 or samples[0].sample_id != "synthetic_niah_24":
        raise RuntimeError("radius calibration sequence isolation failed")
    model = MLXTemporalModel(cfg)
    rows: List[Dict[str, Any]] = []
    unit_rows: List[Dict[str, Any]] = []
    block_metadata: Dict[str, Any] = {}
    try:
        model_info = model.load()
        sample = samples[0]
        reference = model.generate_reference(
            sample_id=sample.sample_id,
            task=sample.task,
            prompt=sample.prompt,
        )
        anchor_step = 16
        anchor = reference.anchors[anchor_step]
        full_record = reference.query_records[anchor_step]
        candidates = load_candidates(
            REGISTRY,
            sample.sample_id,
            anchor_step,
            int(anchor.logical_length - 1),
            REGISTRY_SHA256,
        )
        candidates = [
            candidate
            for candidate in candidates
            if candidate.source in SOURCES
        ]
        if len(candidates) != 2:
            raise RuntimeError("calibration candidate subset mismatch")
        for candidate in candidates:
            physical_candidate = to_physical_candidate(candidate)
            for layer in LAYERS:
                selection, cache_cfg = single_layer_selection(
                    reference,
                    anchor_step,
                    physical_candidate,
                    layer,
                )
                _logits, physical_record, native_seconds = replay_record(
                    model,
                    reference,
                    anchor_step,
                    selection,
                    cache_cfg,
                )
                u_phys = (
                    physical_record.projected_attention_outputs[layer]
                    .float()
                    - full_record.projected_attention_outputs[layer].float()
                ).numpy()
                if not np.isfinite(u_phys).all():
                    raise FloatingPointError(
                        "calibration physical direction is non-finite"
                    )
                local = FP32LocalBlock(
                    model,
                    layer,
                    full_record.post_attention_residuals[layer].numpy(),
                )
                block_metadata[str(layer)] = local.metadata
                repeated = [local.baseline() for _ in range(5)]
                noise_norm = max_pairwise_noise(repeated)
                baseline = repeated[0]
                native_next = (
                    full_record.layer_outputs[layer].double().numpy()
                )
                base_jvp, derivative, method = local.jvp(u_phys)
                if relative_l2(base_jvp, baseline) > 1.0e-12:
                    raise RuntimeError("JVP primal/local baseline mismatch")
                common = {
                    "sample_id": sample.sample_id,
                    "task": sample.task,
                    "split": "train",
                    "anchor": anchor_step,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "mask_hash": candidate.mask_hash,
                    "layer": layer,
                    "direction": "u_phys",
                    "u_norm": float(np.linalg.norm(u_phys)),
                    "b_norm": float(
                        np.linalg.norm(
                            full_record.post_attention_residuals[
                                layer
                            ].double().numpy()
                        )
                    ),
                    "noise_norm": float(noise_norm),
                    "jvp_method": method,
                    "jvp_norm": float(np.linalg.norm(derivative)),
                    "native_replay_seconds": native_seconds,
                    "fp32_base_vs_native_cosine": cosine(
                        baseline, native_next
                    ),
                    "fp32_base_vs_native_relative_l2": relative_l2(
                        baseline, native_next
                    ),
                    "fp32_base_vs_native_symmetric_norm_ratio": (
                        symmetric_norm_ratio(baseline, native_next)
                    ),
                }
                unit_rows.append(dict(common))
                for radius in RADII:
                    finite_difference = local.symmetric_fd(
                        u_phys, radius
                    )
                    fd_derivative = finite_difference["derivative"]
                    rows.append(
                        {
                            **common,
                            "radius": radius,
                            "epsilon_absolute": finite_difference[
                                "epsilon_absolute"
                            ],
                            "fd_norm": float(
                                np.linalg.norm(fd_derivative)
                            ),
                            "fd_to_noise_ratio": float(
                                np.linalg.norm(fd_derivative)
                                / max(noise_norm, 1.0e-30)
                            ),
                            "jvp_fd_cosine": cosine(
                                derivative, fd_derivative
                            ),
                            "jvp_fd_relative_l2": relative_l2(
                                derivative, fd_derivative
                            ),
                            "jvp_fd_symmetric_norm_ratio": (
                                symmetric_norm_ratio(
                                    derivative, fd_derivative
                                )
                            ),
                            "finite": bool(
                                np.isfinite(derivative).all()
                                and np.isfinite(fd_derivative).all()
                            ),
                        }
                    )
        frame = pd.DataFrame(rows)
        units = pd.DataFrame(unit_rows)
        decision = choose_radius(frame)
        summary = {
            "stage": "radius_calibration",
            "calibration_passed": decision["calibration_passed"],
            "selected_radius": decision["selected_radius"],
            "stable_plateau": decision["stable_plateau"],
            "selection_rule": decision["selection_rule"],
            "aggregates": decision["aggregates"],
            "row_count": len(frame),
            "unit_count": len(units),
            "sample_ids": [sample.sample_id],
            "formal_matrix_touched": False,
            "calibration_or_test_loaded": False,
            "dataset_events": dataset_events,
            "model_info": model_info,
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": int(
                    resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss
                ),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            },
        }
        atomic_frame(output_dir / "radius_rows.parquet", frame)
        atomic_frame(output_dir / "calibration_units.parquet", units)
        atomic_json(output_dir / "local_block_metadata.json", block_metadata)
        atomic_json(output_dir / "radius_calibration_summary.json", summary)
        status.update(
            {
                "state": "complete",
                "calibration_passed": decision["calibration_passed"],
                "selected_radius": decision["selected_radius"],
            }
        )
        atomic_json(output_dir / "status.json", status)
        return summary
    except Exception as exc:
        status["state"] = "failed"
        status["errors"].append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        atomic_json(output_dir / "status.json", status)
        raise
    finally:
        if model.runner is not None:
            model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/configs/p0_formal_4bit.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/local_truncated_jacobian/raw"
        / "radius_calibration",
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
