#!/usr/bin/env python3
"""Run native-4bit P0 boundary/JVP smoke without touching held-out sequences."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config

from mlx_predictive_core import (
    PureMultiBoundaryMap,
    direct_injections,
    full_selection,
    joint_candidate_selection,
    make_smoke_candidates,
    replay_physical,
    single_layer_selection,
)
from predictive_core import (
    adaptive_path_fisher,
    exact_kl,
    fisher_score,
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


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return float(np.dot(a, b) / denominator)


def relative_l2(predicted: np.ndarray, truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(
        np.linalg.norm(predicted - truth)
        / max(float(np.linalg.norm(truth)), 1e-30)
    )


def prompt_text() -> str:
    paragraphs = []
    for index in range(80):
        paragraphs.append(
            "Archive entry %03d states that controlled cache experiments must "
            "preserve token positions, candidate budgets, and numerical audit "
            "trails before any held-out conclusion is read." % index
        )
    return "\n".join(paragraphs) + "\nSummarize the audit rule in one sentence."


def run(config_path: Path, output_dir: Path) -> Dict[str, Any]:
    import mlx.core as mx

    started = time.perf_counter()
    cfg = load_discovery_config(str(config_path))
    cfg.validate()
    # Reuse the existing reference runner's probe-logit capture without
    # enabling or analyzing the independent-Fisher protocol itself.
    cfg.independent_fisher.enabled = True
    cfg.independent_fisher.anchors = [16]
    cfg.independent_fisher.segment_horizon = 1
    model = MLXTemporalModel(cfg)
    identity_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    manual_rows: List[Dict[str, Any]] = []
    joint_rows: List[Dict[str, Any]] = []
    jvp_rows: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {
        "stage": "p0_smoke",
        "formal_p0": False,
        "heldout_touched": False,
        "state": "running",
        "errors": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "status.json", status)
    try:
        model_info = model.load()
        reference = model.generate_reference(
            sample_id="predictive_closure_p0_smoke_synthetic",
            task="p0_synthetic",
            prompt=prompt_text(),
        )
        anchor_step = 16
        if anchor_step not in reference.anchors:
            raise RuntimeError("P0 reference did not capture anchor 16")
        if anchor_step not in reference.probe_logits:
            raise RuntimeError("P0 reference did not capture probe logits")
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
        zero_blocks = [
            np.zeros(
                int(model.model_info["hidden_size"]), dtype=np.float32
            )
            for _ in range(int(model.model_info["num_layers"]))
        ]
        fingerprint_before = pure_map.cache_fingerprint()
        base_first = pure_map.evaluate(zero_blocks)
        base_second = pure_map.evaluate(zero_blocks)
        fingerprint_after_repeat = pure_map.cache_fingerprint()
        reference_logits = (
            reference.probe_logits[anchor_step]
            .double()
            .numpy()
        )
        full_core, full_cache_cfg = full_selection(reference, anchor_step)
        full_physical = replay_physical(
            model,
            reference,
            anchor_step,
            full_core,
            full_cache_cfg,
        )
        base_alignment = {
            "repeat_max_absolute_error": float(
                np.max(np.abs(base_first - base_second))
            ),
            "repeat_relative_l2": relative_l2(base_second, base_first),
            "pure_vs_reference_cosine": cosine(base_first, reference_logits),
            "pure_vs_reference_relative_l2": relative_l2(
                base_first, reference_logits
            ),
            "full_replay_vs_reference_cosine": cosine(
                full_physical, reference_logits
            ),
            "full_replay_vs_reference_relative_l2": relative_l2(
                full_physical, reference_logits
            ),
            "cache_fingerprint_before": fingerprint_before,
            "cache_fingerprint_after_repeat": fingerprint_after_repeat,
            "cache_fingerprint_invariant": (
                fingerprint_before == fingerprint_after_repeat
            ),
        }

        for candidate in candidates:
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
                identity_rows.append(
                    {
                        "sample_id": reference.sample_id,
                        "anchor": anchor_step,
                        "candidate_id": candidate.candidate_id,
                        "candidate_source": candidate.source,
                        **row,
                    }
                )
            for row in projection_fp32 + projection_fp16:
                projection_rows.append(
                    {
                        "sample_id": reference.sample_id,
                        "anchor": anchor_step,
                        "candidate_id": candidate.candidate_id,
                        **row,
                    }
                )

            for layer in (7, 14, 21, 27):
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
                manual_rows.append(
                    {
                        "sample_id": reference.sample_id,
                        "anchor": anchor_step,
                        "candidate_id": candidate.candidate_id,
                        "layer": layer,
                        "physical_manual_relative_l2": relative_l2(
                            manual_delta, physical_delta
                        ),
                        "physical_manual_cosine": cosine(
                            manual_delta, physical_delta
                        ),
                        "physical_kl": exact_kl(base_first, physical),
                        "manual_kl": exact_kl(base_first, manual),
                        "kl_absolute_difference": abs(
                            exact_kl(base_first, physical)
                            - exact_kl(base_first, manual)
                        ),
                    }
                )

            injected = pure_map.evaluate(u_fp32)
            physical = replay_physical(
                model,
                reference,
                anchor_step,
                joint_candidate_selection(
                    reference, anchor_step, candidate
                ),
                cfg.cache,
            )
            base_jvp, derivative, jvp_method = pure_map.jvp(u_fp32)
            radius = 1.0e-4
            fd = pure_map.symmetric_fd(u_fp32, radius)
            scaled_jvp = radius * derivative
            probability_direction = derivative
            pb0 = fisher_score(
                base_jvp, probability_direction, midpoint=False
            )
            pbmid = fisher_score(
                base_jvp, probability_direction, midpoint=True
            )
            delta_injected = injected - base_first
            delta_physical = physical - base_first
            joint_rows.append(
                {
                    "sample_id": reference.sample_id,
                    "anchor": anchor_step,
                    "candidate_id": candidate.candidate_id,
                    "candidate_source": candidate.source,
                    "mask_hash": candidate.mask_hash,
                    "active_budget": len(candidate.retained_positions),
                    "euclidean_direct": float(
                        sum(
                            np.dot(value, value)
                            for value in u_fp32
                        )
                    ),
                    "pb0": pb0,
                    "pbmid": pbmid,
                    "k_inj": exact_kl(base_first, injected),
                    "k_phys": exact_kl(base_first, physical),
                    "jvp_injected_cosine": cosine(
                        derivative, delta_injected
                    ),
                    "jvp_injected_relative_l2": relative_l2(
                        derivative, delta_injected
                    ),
                    "injected_physical_cosine": cosine(
                        delta_injected, delta_physical
                    ),
                    "injected_physical_relative_l2": relative_l2(
                        delta_injected, delta_physical
                    ),
                    "top_token_switched_injected": bool(
                        np.argmax(base_first) != np.argmax(injected)
                    ),
                    "top_token_switched_physical": bool(
                        np.argmax(base_first) != np.argmax(physical)
                    ),
                    **{
                        "adaptive_" + key: value
                        for key, value in adaptive_path_fisher(
                            base_first, delta_injected
                        ).items()
                    },
                }
            )
            jvp_rows.append(
                {
                    "sample_id": reference.sample_id,
                    "anchor": anchor_step,
                    "candidate_id": candidate.candidate_id,
                    "radius": radius,
                    "jvp_method": jvp_method,
                    "base_jvp_vs_pure_cosine": cosine(
                        base_jvp, base_first
                    ),
                    "base_jvp_vs_pure_relative_l2": relative_l2(
                        base_jvp, base_first
                    ),
                    "jvp_fd_cosine": cosine(
                        scaled_jvp, fd["symmetric_delta"]
                    ),
                    "jvp_fd_relative_l2": relative_l2(
                        scaled_jvp, fd["symmetric_delta"]
                    ),
                    "jvp_norm": float(np.linalg.norm(derivative)),
                    "fd_norm": float(
                        np.linalg.norm(fd["symmetric_delta"])
                    ),
                    "finite": bool(
                        np.isfinite(derivative).all()
                        and np.isfinite(fd["symmetric_delta"]).all()
                    ),
                }
            )

        rng = np.random.default_rng(cfg.runtime.seed)
        cotangent = rng.standard_normal(base_first.shape).astype(np.float32)
        cotangent /= max(float(np.linalg.norm(cotangent)), 1e-30)
        vjp_blocks = pure_map.vjp(cotangent)
        fingerprint_after_autograd = pure_map.cache_fingerprint()
        vjp_summary = {
            "block_count": len(vjp_blocks),
            "all_finite": bool(
                all(np.isfinite(value).all() for value in vjp_blocks)
            ),
            "total_norm": float(
                math.sqrt(
                    sum(float(np.dot(value, value)) for value in vjp_blocks)
                )
            ),
            "cache_fingerprint_after_autograd": fingerprint_after_autograd,
            "cache_fingerprint_invariant": (
                fingerprint_before == fingerprint_after_autograd
            ),
        }

        identity = pd.DataFrame(identity_rows)
        projection = pd.DataFrame(projection_rows)
        manual = pd.DataFrame(manual_rows)
        joint = pd.DataFrame(joint_rows)
        jvp = pd.DataFrame(jvp_rows)
        atomic_frame(output_dir / "deletion_identity_rows.parquet", identity)
        atomic_frame(output_dir / "projection_block_rows.parquet", projection)
        atomic_frame(output_dir / "single_layer_rows.parquet", manual)
        atomic_frame(output_dir / "joint_rows.parquet", joint)
        atomic_frame(output_dir / "jvp_fd_rows.parquet", jvp)

        fp32_identity = identity[identity["dtype"].eq("float32")]
        smoke_checks = {
            "identity_all_finite": bool(identity["finite"].all()),
            "fp32_max_relative_error": float(
                fp32_identity["relative_error"].max()
            ),
            "fp32_identity_pass": bool(
                fp32_identity["relative_error"].max() <= 1.0e-6
            ),
            "single_layer_cosine_median": float(
                manual["physical_manual_cosine"].median()
            ),
            "single_layer_pass": bool(
                manual["physical_manual_cosine"].median() >= 0.999
            ),
            "jvp_fd_cosine_median": float(
                jvp["jvp_fd_cosine"].median()
            ),
            "jvp_fd_pass": bool(
                jvp["jvp_fd_cosine"].median() >= 0.99
            ),
            "projection_min_cosine": float(
                projection["sum_block_cosine"].min()
            ),
            "projection_pass": bool(
                projection["sum_block_cosine"].min() >= 0.999
            ),
            "repeat_pass": bool(
                base_alignment["repeat_max_absolute_error"] <= 1.0e-6
                and base_alignment["cache_fingerprint_invariant"]
            ),
            "vjp_pass": bool(
                vjp_summary["block_count"]
                == int(model.model_info["num_layers"])
                and vjp_summary["all_finite"]
                and vjp_summary["cache_fingerprint_invariant"]
            ),
        }
        smoke_passed = bool(all(smoke_checks.values()))
        summary = {
            "formal_p0": False,
            "smoke_passed": smoke_passed,
            "checks": smoke_checks,
            "base_alignment": base_alignment,
            "vjp": vjp_summary,
            "row_counts": {
                "identity": len(identity),
                "projection": len(projection),
                "single_layer": len(manual),
                "joint": len(joint),
                "jvp_fd": len(jvp),
            },
            "model_info": model_info,
            "sequence": {
                "sample_id": reference.sample_id,
                "task": reference.task,
                "prompt_length": reference.prompt_length,
                "anchor": anchor_step,
                "candidate_count": len(candidates),
                "heldout": False,
            },
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "mlx_peak_memory_bytes": int(mx.get_peak_memory()),
            },
        }
        atomic_json(output_dir / "p0_smoke_summary.json", summary)
        pd.DataFrame(
            [
                {
                    "formal_p0": False,
                    "smoke_passed": smoke_passed,
                    **smoke_checks,
                }
            ]
        ).to_csv(output_dir / "p0_smoke_summary.csv", index=False)
        status.update(
            {
                "state": "complete",
                "smoke_passed": smoke_passed,
                "formal_p0": False,
                "heldout_touched": False,
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
        / "experiments/predictive_closure/configs/p0_smoke_4bit.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/predictive_closure/raw/p0_alignment/smoke_4bit",
    )
    args = parser.parse_args()
    summary = run(args.config, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
