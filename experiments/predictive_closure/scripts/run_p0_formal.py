#!/usr/bin/env python3
"""Execute the preregistered train-only native-4bit formal P0 gate."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config
from kvbench.temporal.tasks import load_discovery_tasks

from mlx_predictive_core import (
    PureMultiBoundaryMap,
    direct_injections,
    full_selection,
    make_selector_candidates,
    replay_physical,
    single_layer_selection,
)
from run_p0_smoke import cosine, relative_l2


TABLE_NAMES = (
    "candidate_registry",
    "deletion_identity",
    "projection_block",
    "single_layer",
    "jvp_fd",
    "anchor_audit",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".parquet", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_id(sample_id: str) -> str:
    return "".join(
        value if value.isalnum() or value in "._-" else "_"
        for value in sample_id
    )


def expected_sample_ids() -> List[str]:
    return [
        "gov_report:24",
        "gov_report:25",
        "synthetic_niah_24",
        "synthetic_niah_25",
    ]


def write_sequence_tables(
    sequence_dir: Path, tables: Mapping[str, List[Dict[str, Any]]]
) -> None:
    for name in TABLE_NAMES:
        atomic_frame(
            sequence_dir / f"{name}.parquet",
            pd.DataFrame(tables[name]),
        )


def load_sequence_tables(
    sequence_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        name: pd.read_parquet(
            sequence_dir / f"{name}.parquet"
        ).to_dict("records")
        for name in TABLE_NAMES
    }


def run_sequence(
    model: MLXTemporalModel,
    sample: Any,
    cfg: Any,
    sequence_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    started = time.perf_counter()
    tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLE_NAMES
    }
    status = {
        "formal_p0": True,
        "split": "train",
        "heldout_touched": False,
        "sample_id": sample.sample_id,
        "task": sample.task,
        "state": "running",
    }
    atomic_json(sequence_dir / "status.json", status)
    reference = model.generate_reference(
        sample_id=sample.sample_id,
        task=sample.task,
        prompt=sample.prompt,
    )
    anchors = [16, 32, 48, 64]
    missing = [
        anchor
        for anchor in anchors
        if anchor not in reference.anchors
        or anchor not in reference.probe_logits
    ]
    if missing:
        raise RuntimeError(f"missing preregistered anchors: {missing}")
    previous_attention_core = None
    hidden_size = int(model.model_info["hidden_size"])
    layers = int(model.model_info["num_layers"])
    rng = np.random.default_rng(
        int(cfg.runtime.seed)
        + int.from_bytes(
            sample.sample_id.encode("utf-8")[:8].ljust(8, b"\0"), "big"
        )
    )
    for anchor_step in anchors:
        anchor_started = time.perf_counter()
        candidates, registry = make_selector_candidates(
            model,
            reference,
            anchor_step,
            cfg.cache,
            cfg.runtime.run_id,
            previous_attention_core=previous_attention_core,
        )
        previous_attention_core = registry["attention_core"]
        pure_map = PureMultiBoundaryMap(
            model, reference.anchors[anchor_step]
        )
        zero_blocks = [
            np.zeros(hidden_size, dtype=np.float32)
            for _ in range(layers)
        ]
        fingerprint_before = pure_map.cache_fingerprint()
        base_first = pure_map.evaluate(zero_blocks)
        base_second = pure_map.evaluate(zero_blocks)
        reference_logits = (
            reference.probe_logits[anchor_step].double().numpy()
        )
        full_core, full_cache_cfg = full_selection(reference, anchor_step)
        full_physical = replay_physical(
            model,
            reference,
            anchor_step,
            full_core,
            full_cache_cfg,
        )
        for candidate in candidates:
            common = {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "split": "train",
                "anchor": anchor_step,
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "mask_hash": candidate.mask_hash,
            }
            tables["candidate_registry"].append(
                {
                    **common,
                    "candidate_seed": candidate.seed,
                    "active_budget": len(candidate.retained_positions),
                    "core_budget": len(candidate.core_positions),
                    "retained_positions_json": json.dumps(
                        candidate.retained_positions,
                        separators=(",", ":"),
                    ),
                    "core_positions_json": json.dumps(
                        candidate.core_positions,
                        separators=(",", ":"),
                    ),
                    "dedup_event_count": len(registry["dedup_events"]),
                    "layer_shared": registry["shared_across_layers"],
                    "gqa_shared": registry["shared_across_gqa_heads"],
                }
            )
            u_fp32, identity_fp32, projection_fp32 = direct_injections(
                model,
                reference,
                anchor_step,
                candidate.retained_positions,
                torch.float32,
            )
            _u_fp16, identity_fp16, projection_fp16 = direct_injections(
                model,
                reference,
                anchor_step,
                candidate.retained_positions,
                torch.float16,
            )
            for row in identity_fp32 + identity_fp16:
                tables["deletion_identity"].append({**common, **row})
            for row in projection_fp32 + projection_fp16:
                tables["projection_block"].append({**common, **row})

            for layer in (0, 7, 14, 21, 27):
                selection, cache_cfg = single_layer_selection(
                    reference, anchor_step, candidate, layer
                )
                physical = replay_physical(
                    model,
                    reference,
                    anchor_step,
                    selection,
                    cache_cfg,
                )
                manual_blocks = [
                    np.zeros_like(value) for value in u_fp32
                ]
                manual_blocks[layer] = u_fp32[layer]
                manual = pure_map.evaluate(manual_blocks)
                physical_delta = physical - base_first
                manual_delta = manual - base_first
                tables["single_layer"].append(
                    {
                        **common,
                        "layer": layer,
                        "physical_manual_relative_l2": relative_l2(
                            manual_delta, physical_delta
                        ),
                        "physical_manual_cosine": cosine(
                            manual_delta, physical_delta
                        ),
                        "physical_delta_norm": float(
                            np.linalg.norm(physical_delta)
                        ),
                        "manual_delta_norm": float(
                            np.linalg.norm(manual_delta)
                        ),
                    }
                )

            base_jvp, derivative, jvp_method = pure_map.jvp(u_fp32)
            radius = 1.0e-4
            finite_difference = pure_map.symmetric_fd(
                u_fp32, radius
            )["symmetric_delta"]
            scaled_jvp = radius * derivative
            tables["jvp_fd"].append(
                {
                    **common,
                    "radius": radius,
                    "jvp_method": jvp_method,
                    "base_jvp_vs_pure_cosine": cosine(
                        base_jvp, base_first
                    ),
                    "base_jvp_vs_pure_relative_l2": relative_l2(
                        base_jvp, base_first
                    ),
                    "jvp_fd_cosine": cosine(
                        scaled_jvp, finite_difference
                    ),
                    "jvp_fd_relative_l2": relative_l2(
                        scaled_jvp, finite_difference
                    ),
                    "jvp_norm": float(np.linalg.norm(derivative)),
                    "scaled_jvp_norm": float(
                        np.linalg.norm(scaled_jvp)
                    ),
                    "fd_norm": float(
                        np.linalg.norm(finite_difference)
                    ),
                    "finite": bool(
                        np.isfinite(derivative).all()
                        and np.isfinite(finite_difference).all()
                    ),
                }
            )

        cotangent = rng.standard_normal(base_first.shape).astype(np.float32)
        cotangent /= max(float(np.linalg.norm(cotangent)), 1.0e-30)
        vjp_blocks = pure_map.vjp(cotangent)
        fingerprint_after = pure_map.cache_fingerprint()
        tables["anchor_audit"].append(
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "split": "train",
                "anchor": anchor_step,
                "candidate_count": len(candidates),
                "candidate_distinct_count": len(
                    {candidate.mask_hash for candidate in candidates}
                ),
                "dedup_event_count": len(registry["dedup_events"]),
                "active_budget_min": min(
                    len(candidate.retained_positions)
                    for candidate in candidates
                ),
                "active_budget_max": max(
                    len(candidate.retained_positions)
                    for candidate in candidates
                ),
                "repeat_max_absolute_error": float(
                    np.max(np.abs(base_first - base_second))
                ),
                "pure_vs_reference_cosine": cosine(
                    base_first, reference_logits
                ),
                "pure_vs_reference_relative_l2": relative_l2(
                    base_first, reference_logits
                ),
                "full_replay_vs_reference_cosine": cosine(
                    full_physical, reference_logits
                ),
                "full_replay_vs_reference_relative_l2": relative_l2(
                    full_physical, reference_logits
                ),
                "cache_fingerprint_invariant": (
                    fingerprint_before == fingerprint_after
                ),
                "vjp_block_count": len(vjp_blocks),
                "vjp_all_finite": bool(
                    all(np.isfinite(value).all() for value in vjp_blocks)
                ),
                "vjp_total_norm": float(
                    math.sqrt(
                        sum(
                            float(np.dot(value, value))
                            for value in vjp_blocks
                        )
                    )
                ),
                "anchor_wall_seconds": time.perf_counter()
                - anchor_started,
            }
        )
        print(
            json.dumps(
                {
                    "event": "anchor_complete",
                    "sample_id": sample.sample_id,
                    "anchor": anchor_step,
                    "candidates": len(candidates),
                    "wall_seconds": tables["anchor_audit"][-1][
                        "anchor_wall_seconds"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del pure_map
        gc.collect()

    write_sequence_tables(sequence_dir, tables)
    status.update(
        {
            "state": "complete",
            "wall_seconds": time.perf_counter() - started,
            "prompt_length": reference.prompt_length,
            "prompt_truncated": reference.prompt_truncated,
            "generation_stopped_on_eos": (
                reference.generation_stopped_on_eos
            ),
            "anchors": anchors,
            "candidate_count_per_anchor": 8,
        }
    )
    atomic_json(sequence_dir / "status.json", status)
    return tables


def summarize(
    tables: Mapping[str, pd.DataFrame],
    model_info: Mapping[str, Any],
    dataset_events: List[Dict[str, Any]],
    started: float,
) -> Dict[str, Any]:
    identity = tables["deletion_identity"]
    projection = tables["projection_block"]
    manual = tables["single_layer"]
    jvp = tables["jvp_fd"]
    audit = tables["anchor_audit"]
    registry = tables["candidate_registry"]
    fp32 = identity[identity["dtype"].eq("float32")]
    fp32_fail = fp32[fp32["relative_error"].gt(1.0e-6)]
    checks = {
        "matrix_sequence_count": int(
            registry["sample_id"].nunique()
        )
        == 4,
        "matrix_task_counts": (
            registry.groupby("task")["sample_id"].nunique().to_dict()
            == {"gov_report": 2, "niah_single_1": 2}
        ),
        "matrix_anchor_count": int(registry["anchor"].nunique()) == 4,
        "matrix_candidates_per_group": bool(
            registry.groupby(["sample_id", "anchor"]).size().eq(8).all()
        ),
        "candidate_distinct_pass": bool(
            audit["candidate_distinct_count"].eq(8).all()
        ),
        "active_budget_pass": bool(
            audit["active_budget_min"].eq(128).all()
            and audit["active_budget_max"].eq(128).all()
        ),
        "identity_all_finite": bool(identity["finite"].all()),
        "fp32_identity_pass": bool(
            fp32["relative_error"].max() <= 1.0e-6
        ),
        "single_layer_pass": bool(
            manual["physical_manual_cosine"].median() >= 0.999
        ),
        "jvp_fd_pass": bool(jvp["jvp_fd_cosine"].median() >= 0.99),
        "jvp_all_finite": bool(jvp["finite"].all()),
        "projection_pass": bool(
            projection["sum_block_cosine"].min() >= 0.999
        ),
        "repeat_pass": bool(
            audit["repeat_max_absolute_error"].max() <= 1.0e-6
            and audit["cache_fingerprint_invariant"].all()
        ),
        "base_alignment_pass": bool(
            audit["pure_vs_reference_cosine"].min() >= 0.999999
            and audit["full_replay_vs_reference_cosine"].min()
            >= 0.999999
        ),
        "vjp_pass": bool(
            audit["vjp_block_count"].eq(28).all()
            and audit["vjp_all_finite"].all()
            and audit["cache_fingerprint_invariant"].all()
        ),
    }
    formal_pass = bool(all(checks.values()))
    return {
        "formal_p0": True,
        "split": "train",
        "heldout_touched": False,
        "formal_p0_passed": formal_pass,
        "stop_p1_plus": not formal_pass,
        "checks": checks,
        "metrics": {
            "fp32_max_relative_error": float(
                fp32["relative_error"].max()
            ),
            "fp32_median_relative_error": float(
                fp32["relative_error"].median()
            ),
            "fp32_failure_row_count": int(len(fp32_fail)),
            "fp32_failure_max_absolute_error": (
                float(fp32_fail["maximum_absolute_error"].max())
                if len(fp32_fail)
                else 0.0
            ),
            "fp32_failure_min_direct_norm": (
                float(fp32_fail["direct_norm"].min())
                if len(fp32_fail)
                else None
            ),
            "fp32_failure_max_denominator": (
                float(fp32_fail["denominator"].max())
                if len(fp32_fail)
                else None
            ),
            "single_layer_cosine_median": float(
                manual["physical_manual_cosine"].median()
            ),
            "single_layer_cosine_by_task": {
                str(key): float(value)
                for key, value in manual.groupby("task")[
                    "physical_manual_cosine"
                ].median().items()
            },
            "jvp_fd_cosine_median": float(
                jvp["jvp_fd_cosine"].median()
            ),
            "jvp_fd_cosine_by_task": {
                str(key): float(value)
                for key, value in jvp.groupby("task")[
                    "jvp_fd_cosine"
                ].median().items()
            },
            "jvp_fd_scaled_jvp_norm_median": float(
                jvp["scaled_jvp_norm"].median()
            ),
            "jvp_fd_fd_norm_median": float(jvp["fd_norm"].median()),
            "projection_min_cosine": float(
                projection["sum_block_cosine"].min()
            ),
            "base_repeat_max_absolute_error": float(
                audit["repeat_max_absolute_error"].max()
            ),
        },
        "row_counts": {
            name: int(len(frame)) for name, frame in tables.items()
        },
        "sequence_ids": sorted(registry["sample_id"].unique().tolist()),
        "dataset_events": dataset_events,
        "model_info": dict(model_info),
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
    }


def run(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    import mlx.core as mx

    started = time.perf_counter()
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = [16, 32, 48, 64]
    cfg.independent_fisher.segment_horizon = 1
    samples, dataset_events = load_discovery_tasks(cfg)
    actual_ids = [sample.sample_id for sample in samples]
    if actual_ids != expected_sample_ids():
        raise RuntimeError(
            f"formal P0 ID isolation failed: {actual_ids} "
            f"!= {expected_sample_ids()}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "formal_p0",
        "formal_p0": True,
        "split": "train",
        "heldout_touched": False,
        "state": "running",
        "completed_sequences": [],
        "errors": [],
    }
    atomic_json(output_dir / "status.json", status)
    model = MLXTemporalModel(cfg)
    all_tables: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in TABLE_NAMES
    }
    try:
        model_info = model.load()
        for sample in samples:
            sequence_dir = (
                output_dir / "checkpoints" / safe_id(sample.sample_id)
            )
            sequence_status_path = sequence_dir / "status.json"
            complete = False
            if cfg.runtime.resume and sequence_status_path.exists():
                sequence_status = json.loads(
                    sequence_status_path.read_text(encoding="utf-8")
                )
                complete = sequence_status.get("state") == "complete"
                complete = complete and all(
                    (sequence_dir / f"{name}.parquet").exists()
                    for name in TABLE_NAMES
                )
            if complete:
                tables = load_sequence_tables(sequence_dir)
                event = "sequence_resumed"
            else:
                tables = run_sequence(
                    model, sample, cfg, sequence_dir
                )
                event = "sequence_complete"
            for name in TABLE_NAMES:
                all_tables[name].extend(tables[name])
            status["completed_sequences"].append(sample.sample_id)
            atomic_json(output_dir / "status.json", status)
            print(
                json.dumps(
                    {
                        "event": event,
                        "sample_id": sample.sample_id,
                        "completed_sequences": len(
                            status["completed_sequences"]
                        ),
                        "total_sequences": len(samples),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        frames = {
            name: pd.DataFrame(rows)
            for name, rows in all_tables.items()
        }
        for name, frame in frames.items():
            atomic_frame(output_dir / f"{name}_rows.parquet", frame)
        summary = summarize(
            frames, model_info, dataset_events, started
        )
        summary["runtime"]["mlx_peak_memory_bytes"] = int(
            mx.get_peak_memory()
        )
        atomic_json(output_dir / "p0_formal_summary.json", summary)
        pd.DataFrame(
            [
                {
                    "formal_p0": True,
                    "formal_p0_passed": summary["formal_p0_passed"],
                    "stop_p1_plus": summary["stop_p1_plus"],
                    **summary["checks"],
                    **summary["metrics"],
                }
            ]
        ).to_csv(output_dir / "p0_formal_summary.csv", index=False)
        status.update(
            {
                "state": "complete",
                "formal_p0_passed": summary["formal_p0_passed"],
                "stop_p1_plus": summary["stop_p1_plus"],
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
        / "experiments/predictive_closure/raw/p0_alignment/formal_4bit",
    )
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
