#!/usr/bin/env python3
"""Diagnose native-4bit P0 failures without touching formal split sequences."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config

from mlx_predictive_core import (
    PhysicalCandidate,
    PureMultiBoundaryMap,
    direct_injections,
    make_smoke_candidates,
    single_layer_selection,
)
from run_p0_smoke import cosine, prompt_text, relative_l2


def replay_with_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
    candidate: PhysicalCandidate,
    layer: int,
) -> Tuple[np.ndarray, Any]:
    selection, cache_cfg = single_layer_selection(
        reference, anchor_step, candidate, layer
    )
    state, _fixed = backend.state_from_anchor(
        reference.anchors[anchor_step],
        selection,
        cache_config=cache_cfg,
    )
    try:
        logits, record, _elapsed = backend.forward_one(
            state,
            int(reference.anchors[anchor_step].query_token_id),
            capture_attention=True,
        )
        return logits.double().numpy(), record
    finally:
        backend.release(state)


def theoretical_head_delta(
    backend: Any,
    reference: Any,
    anchor_step: int,
    retained_positions: Sequence[int],
    layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = reference.anchors[anchor_step]
    full_record = reference.query_records[anchor_step]
    positions = [
        int(value) for value in anchor.position_maps[layer].tolist()
    ]
    row_by_position = {
        position: row for row, position in enumerate(positions)
    }
    rows = torch.tensor(
        [row_by_position[int(position)] for position in retained_positions],
        dtype=torch.long,
    )
    attention = full_record.all_head_attention_distributions[layer].float()
    attention = attention / attention.sum(dim=1, keepdim=True)
    values = anchor.values[layer][0].float()
    repeats = (
        int(backend.model_info["num_attention_heads"])
        // int(backend.model_info["num_key_value_heads"])
    )
    repeated_values = values.repeat_interleave(repeats, dim=0)
    kept_attention = attention.index_select(1, rows)
    masked = (
        kept_attention[:, :, None]
        * repeated_values.index_select(1, rows)
    ).sum(dim=1) / kept_attention.sum(dim=1, keepdim=True)
    recomputed_full = (
        attention[:, :, None] * repeated_values
    ).sum(dim=1)
    recorded_full = full_record.all_head_attention_outputs[layer].float()
    return masked - recorded_full, recomputed_full, recorded_full


def run(config_path: Path) -> Dict[str, Any]:
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = [16]
    cfg.independent_fisher.segment_horizon = 1
    model = MLXTemporalModel(cfg)
    output: Dict[str, Any] = {
        "formal_p0": False,
        "heldout_touched": False,
        "purpose": "native_4bit_numeric_diagnosis",
    }
    try:
        model.load()
        reference = model.generate_reference(
            sample_id="predictive_closure_p0_numeric_synthetic",
            task="p0_synthetic",
            prompt=prompt_text(),
        )
        anchor_step = 16
        candidates = make_smoke_candidates(
            model,
            reference,
            anchor_step,
            cfg.cache,
            cfg.runtime.seed,
        )
        pure_map = PureMultiBoundaryMap(
            model, reference.anchors[anchor_step]
        )
        zeros = [
            np.zeros(int(model.model_info["hidden_size"]), dtype=np.float32)
            for _ in range(int(model.model_info["num_layers"]))
        ]
        base = pure_map.evaluate(zeros)
        radius_rows: List[Dict[str, Any]] = []
        kernel_rows: List[Dict[str, Any]] = []
        for candidate in candidates:
            blocks, _identity, _projection = direct_injections(
                model,
                reference,
                anchor_step,
                candidate.retained_positions,
                torch.float32,
            )
            _base_jvp, derivative, method = pure_map.jvp(blocks)
            for radius in (
                1.0e-4,
                3.0e-4,
                1.0e-3,
                3.0e-3,
                1.0e-2,
                3.0e-2,
                1.0e-1,
                2.5e-1,
                5.0e-1,
                1.0,
            ):
                finite_difference = pure_map.symmetric_fd(blocks, radius)[
                    "symmetric_delta"
                ]
                scaled_jvp = float(radius) * derivative
                radius_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "radius": radius,
                        "jvp_method": method,
                        "cosine": cosine(scaled_jvp, finite_difference),
                        "relative_l2": relative_l2(
                            scaled_jvp, finite_difference
                        ),
                        "scaled_jvp_norm": float(
                            np.linalg.norm(scaled_jvp)
                        ),
                        "finite_difference_norm": float(
                            np.linalg.norm(finite_difference)
                        ),
                    }
                )
            for layer in (7, 14, 21, 27):
                physical_logits, physical_record = replay_with_record(
                    model, reference, anchor_step, candidate, layer
                )
                theoretical, recomputed_full, recorded_full = (
                    theoretical_head_delta(
                        model,
                        reference,
                        anchor_step,
                        candidate.retained_positions,
                        layer,
                    )
                )
                physical_delta = (
                    physical_record.all_head_attention_outputs[layer].float()
                    - recorded_full
                )
                projected_theoretical = model.project_features(
                    layer, theoretical.reshape(1, -1)
                )[0]
                projected_physical = model.project_features(
                    layer, physical_delta.reshape(1, -1)
                )[0]
                actual_projected_delta = (
                    physical_record.projected_attention_outputs[layer].float()
                    - reference.query_records[
                        anchor_step
                    ].projected_attention_outputs[layer].float()
                )
                theoretical_blocks = [
                    np.zeros_like(value) for value in blocks
                ]
                theoretical_blocks[layer] = (
                    projected_theoretical.numpy().astype(np.float32)
                )
                kernel_blocks = [
                    np.zeros_like(value) for value in blocks
                ]
                kernel_blocks[layer] = (
                    projected_physical.numpy().astype(np.float32)
                )
                theoretical_logits = pure_map.evaluate(theoretical_blocks)
                kernel_logits = pure_map.evaluate(kernel_blocks)
                actual_projected_blocks = [
                    np.zeros_like(value) for value in blocks
                ]
                actual_projected_blocks[layer] = (
                    actual_projected_delta.numpy().astype(np.float32)
                )
                actual_projected_logits = pure_map.evaluate(
                    actual_projected_blocks
                )
                actual_logit_delta = physical_logits - base
                kernel_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "layer": layer,
                        "full_kernel_vs_recomputed_head_cosine": cosine(
                            recorded_full, recomputed_full
                        ),
                        "full_kernel_vs_recomputed_head_relative_l2": (
                            relative_l2(recomputed_full, recorded_full)
                        ),
                        "theory_vs_physical_head_delta_cosine": cosine(
                            theoretical, physical_delta
                        ),
                        "theory_vs_physical_head_delta_relative_l2": (
                            relative_l2(theoretical, physical_delta)
                        ),
                        "theory_vs_physical_projected_cosine": cosine(
                            projected_theoretical, projected_physical
                        ),
                        "theory_vs_physical_projected_relative_l2": (
                            relative_l2(
                                projected_theoretical, projected_physical
                            )
                        ),
                        "linear_vs_actual_projected_delta_cosine": cosine(
                            projected_physical, actual_projected_delta
                        ),
                        "linear_vs_actual_projected_delta_relative_l2": (
                            relative_l2(
                                projected_physical,
                                actual_projected_delta,
                            )
                        ),
                        "theory_vs_physical_logit_cosine": cosine(
                            theoretical_logits - base, actual_logit_delta
                        ),
                        "kernel_vs_physical_logit_cosine": cosine(
                            kernel_logits - base, actual_logit_delta
                        ),
                        "kernel_vs_physical_logit_relative_l2": relative_l2(
                            kernel_logits - base, actual_logit_delta
                        ),
                        "actual_projected_vs_physical_logit_cosine": cosine(
                            actual_projected_logits - base,
                            actual_logit_delta,
                        ),
                        "actual_projected_vs_physical_logit_relative_l2": (
                            relative_l2(
                                actual_projected_logits - base,
                                actual_logit_delta,
                            )
                        ),
                    }
                )
        output["radius_rows"] = radius_rows
        output["kernel_rows"] = kernel_rows
        output["radius_median_cosine"] = {
            str(radius): float(
                np.median(
                    [
                        row["cosine"]
                        for row in radius_rows
                        if row["radius"] == radius
                    ]
                )
            )
            for radius in sorted({row["radius"] for row in radius_rows})
        }
        output["kernel_medians"] = {
            key: float(np.median([row[key] for row in kernel_rows]))
            for key in (
                "full_kernel_vs_recomputed_head_cosine",
                "theory_vs_physical_head_delta_cosine",
                "theory_vs_physical_projected_cosine",
                "theory_vs_physical_logit_cosine",
                "kernel_vs_physical_logit_cosine",
                "kernel_vs_physical_logit_relative_l2",
                "actual_projected_vs_physical_logit_cosine",
                "actual_projected_vs_physical_logit_relative_l2",
            )
        }
        return output
    finally:
        if model.runner is not None:
            model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/configs/p0_smoke_4bit.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/raw/p0_alignment/smoke_4bit"
        / "numeric_diagnosis.json",
    )
    args = parser.parse_args()
    result = run(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
