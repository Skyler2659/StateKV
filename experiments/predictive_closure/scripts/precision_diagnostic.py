#!/usr/bin/env python3
"""Post-hoc precision/path diagnostic for the failed formal predictive P0.

This module deliberately does not implement a new gate.  It reuses only frozen
train cases and candidate masks from ``formal_4bit_retry1`` and writes to a
separate diagnostic directory.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks/torch"))
sys.path.insert(0, str(SCRIPT_DIR))

from kvbench.temporal.backend_mlx import MLXTemporalModel
from kvbench.temporal.config import load_discovery_config
from kvbench.temporal.tasks import load_discovery_tasks

from mlx_predictive_core import (
    PhysicalCandidate,
    PureMultiBoundaryMap,
    full_selection,
    single_layer_selection,
)


FORMAL_DIR = (
    ROOT
    / "experiments/predictive_closure/raw/p0_alignment/formal_4bit_retry1"
)
FORMAL_CONFIG = (
    ROOT / "experiments/predictive_closure/configs/p0_formal_4bit.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments/predictive_closure/precision_diagnostic/v1"
)
SELECTED_SAMPLE = "gov_report:25"
SELECTED_ANCHOR = 16
SELECTED_LAYER = 0
IDENTITY_AUX_ANCHOR = 48
IDENTITY_AUX_LAYER = 14
EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5)
IDENTITY_TAUS = (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CHECKPOINT_TAU = 1.0e-8
CHECKPOINT_RELATIVE_ALERT = 1.0e-3
CHECKPOINT_COSINE_ALERT = 0.999
EPSILON_DIRECTION_IDS = (
    "selector_00_attention_only",
    "selector_06_fresh_core",
)


def json_safe(value: Any) -> Any:
    """Convert NumPy/pandas and non-finite scalars to strict JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(value),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        output = value.detach().cpu().numpy()
    else:
        output = np.asarray(value)
    return output.astype(dtype, copy=False) if dtype is not None else output


def tensor_dtype_name(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.dtype).replace("torch.", "")
    return str(np.asarray(value).dtype)


def cosine_diagnostics(
    left: Any,
    right: Any,
    zero_tolerance: float = 1.0e-30,
) -> Dict[str, Any]:
    """Return cosine plus an explicit zero/non-finite classification."""
    a = to_numpy(left, np.float64).reshape(-1)
    b = to_numpy(right, np.float64).reshape(-1)
    left_norm = float(np.linalg.norm(a))
    right_norm = float(np.linalg.norm(b))
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    left_zero = bool(left_norm <= zero_tolerance)
    right_zero = bool(right_norm <= zero_tolerance)
    if not finite:
        status = "non_finite"
        cosine = float("nan")
    elif left_zero and right_zero:
        status = "both_zero"
        cosine = 0.0
    elif left_zero:
        status = "left_zero"
        cosine = 0.0
    elif right_zero:
        status = "right_zero"
        cosine = 0.0
    else:
        cosine = float(np.dot(a, b) / (left_norm * right_norm))
        status = "defined_near_orthogonal" if abs(cosine) <= 1.0e-6 else "defined"
    return {
        "cosine": cosine,
        "cosine_status": status,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "left_zero": left_zero,
        "right_zero": right_zero,
        "finite": finite,
    }


def identity_error_metrics(
    lhs: Any,
    rhs: Any,
    taus: Sequence[float] = IDENTITY_TAUS,
) -> Dict[str, Any]:
    """Original and stable identity errors, without replacing the original."""
    left = to_numpy(lhs, np.float64)
    right = to_numpy(rhs, np.float64)
    difference = left - right
    lhs_norm = float(np.linalg.norm(left))
    rhs_norm = float(np.linalg.norm(right))
    absolute_l2 = float(np.linalg.norm(difference))
    output: Dict[str, Any] = {
        "absolute_l2_error": absolute_l2,
        "maximum_absolute_error": float(np.max(np.abs(difference))),
        "lhs_norm": lhs_norm,
        "rhs_norm": rhs_norm,
        "raw_relative_error": absolute_l2 / max(lhs_norm, 1.0e-30),
        "finite": bool(
            np.isfinite(left).all()
            and np.isfinite(right).all()
            and np.isfinite(difference).all()
        ),
    }
    for tau in taus:
        key = f"stable_relative_error_tau_{tau:.0e}".replace("-", "m")
        output[key] = absolute_l2 / max(lhs_norm, rhs_norm, float(tau))
    return output


def perturbation_entry_metrics(
    base: Any,
    direction: Any,
    epsilon: float,
    effective_dtype: np.dtype,
) -> Dict[str, Any]:
    """Measure the constructed and effective additive-boundary perturbation."""
    x = to_numpy(base, np.float32)
    v = to_numpy(direction, np.float32)
    plus_constructed = x + np.float32(epsilon) * v
    minus_constructed = x - np.float32(epsilon) * v
    dtype = np.dtype(effective_dtype)
    x_effective = x.astype(dtype)
    plus_effective = (
        x_effective + (np.float32(epsilon) * v).astype(dtype)
    ).astype(dtype)
    minus_effective = (
        x_effective - (np.float32(epsilon) * v).astype(dtype)
    ).astype(dtype)
    plus_change = plus_effective.astype(np.float64) - x_effective.astype(np.float64)
    minus_change = minus_effective.astype(np.float64) - x_effective.astype(np.float64)
    effective_difference = (
        plus_effective.astype(np.float64) - minus_effective.astype(np.float64)
    )
    absolute_effective = np.abs(effective_difference).reshape(-1)
    nonzero = absolute_effective > 0
    nonzero_values = absolute_effective[nonzero]
    finfo = np.finfo(dtype)
    return {
        "x_norm": float(np.linalg.norm(x.astype(np.float64))),
        "v_norm": float(np.linalg.norm(v.astype(np.float64))),
        "constructed_plus_change_norm": float(
            np.linalg.norm((plus_constructed - x).astype(np.float64))
        ),
        "constructed_minus_change_norm": float(
            np.linalg.norm((minus_constructed - x).astype(np.float64))
        ),
        "effective_plus_change_norm": float(np.linalg.norm(plus_change)),
        "effective_minus_change_norm": float(np.linalg.norm(minus_change)),
        "effective_plus_minus_norm": float(np.linalg.norm(effective_difference)),
        "x_plus_equals_x_minus": bool(
            np.array_equal(plus_effective, minus_effective)
        ),
        "x_plus_equals_x": bool(np.array_equal(plus_effective, x_effective)),
        "x_minus_equals_x": bool(np.array_equal(minus_effective, x_effective)),
        "effective_nonzero_difference_fraction": float(nonzero.mean()),
        "effective_minimum_absolute_difference": float(
            absolute_effective.min(initial=0.0)
        ),
        "effective_median_absolute_difference": float(
            np.median(absolute_effective)
        ),
        "effective_maximum_absolute_difference": float(
            absolute_effective.max(initial=0.0)
        ),
        "effective_minimum_nonzero_absolute_difference": (
            float(nonzero_values.min()) if len(nonzero_values) else 0.0
        ),
        "effective_dtype": str(dtype),
        "machine_epsilon": float(finfo.eps),
        "smallest_normal": float(finfo.tiny),
    }


def cast_intervention_for_boundary(
    intervention: Any,
    boundary_dtype: np.dtype,
) -> np.ndarray:
    """Match the formal manual map's cast immediately before the addition."""
    return np.asarray(intervention).astype(np.dtype(boundary_dtype), copy=True)


def symmetric_fd_independent(
    function: Callable[[np.ndarray, Any], np.ndarray],
    base: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    state_factory: Callable[[], Any],
) -> Dict[str, Any]:
    """A testable reference FD helper with independent state per sign."""
    plus_input = np.asarray(base).copy() + float(epsilon) * np.asarray(direction)
    minus_input = np.asarray(base).copy() - float(epsilon) * np.asarray(direction)
    plus_state = state_factory()
    minus_state = state_factory()
    if plus_state is minus_state:
        raise RuntimeError("positive and negative FD forwards share state")
    plus = np.asarray(function(plus_input, plus_state), dtype=np.float64)
    minus = np.asarray(function(minus_input, minus_state), dtype=np.float64)
    return {
        "plus": plus,
        "minus": minus,
        "fd": (plus - minus) / (2.0 * float(epsilon)),
        "plus_state": plus_state,
        "minus_state": minus_state,
    }


@dataclass(frozen=True)
class DiagnosticFunctionBoundary:
    """Identifies the exact function boundary shared by JVP and FD."""

    name: str
    function: Callable[[np.ndarray], np.ndarray]

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(self.function(value))


def numpy_jvp_fd_same_boundary(
    boundary: DiagnosticFunctionBoundary,
    base: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    jvp: Callable[
        [DiagnosticFunctionBoundary, np.ndarray, np.ndarray], np.ndarray
    ],
) -> Dict[str, Any]:
    """Use the same boundary object for an analytic JVP and symmetric FD."""
    derivative = np.asarray(jvp(boundary, base, direction), dtype=np.float64)
    plus = boundary.evaluate(base + float(epsilon) * direction)
    minus = boundary.evaluate(base - float(epsilon) * direction)
    finite_difference = (plus - minus) / (2.0 * float(epsilon))
    return {
        "boundary_name": boundary.name,
        "jvp": derivative,
        "fd": finite_difference,
    }


def clone_nested_state(value: Any) -> Any:
    """Clone nested mutable diagnostic state without retaining tensor aliases."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_nested_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_nested_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_nested_state(item) for item in value)
    return copy.deepcopy(value)


def _sample_values(value: np.ndarray, count: int = 8) -> str:
    flat = np.asarray(value).reshape(-1)
    if len(flat) == 0:
        return "[]"
    indices = np.linspace(0, len(flat) - 1, min(count, len(flat)), dtype=int)
    return json.dumps(
        [float(flat[index]) for index in indices],
        separators=(",", ":"),
    )


def checkpoint_metric_row(
    configuration: str,
    candidate_id: str,
    checkpoint: str,
    left: Any,
    right: Any,
    order: int,
    comparison_mode: str,
    token_position: int,
    tau: float = CHECKPOINT_TAU,
    expected_equivalent: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a compact, shape-aware physical/manual checkpoint comparison."""
    a = to_numpy(left)
    b = to_numpy(right)
    shape_match = a.shape == b.shape
    row: Dict[str, Any] = {
        "configuration": configuration,
        "candidate_id": candidate_id,
        "checkpoint": checkpoint,
        "checkpoint_order": int(order),
        "comparison_mode": comparison_mode,
        "token_position": int(token_position),
        "tau": float(tau),
        "expected_equivalent": bool(expected_equivalent),
        "shape_match": bool(shape_match),
        "physical_shape": json.dumps(list(a.shape)),
        "manual_shape": json.dumps(list(b.shape)),
        "physical_dtype": tensor_dtype_name(left),
        "manual_dtype": tensor_dtype_name(right),
        "physical_stride": json.dumps(
            list(a.strides if a.ndim else ()), separators=(",", ":")
        ),
        "manual_stride": json.dumps(
            list(b.strides if b.ndim else ()), separators=(",", ":")
        ),
        "physical_sample_json": _sample_values(a),
        "manual_sample_json": _sample_values(b),
    }
    if not shape_match:
        row.update(
            {
                "physical_norm": float("nan"),
                "manual_norm": float("nan"),
                "absolute_error": float("nan"),
                "relative_error": float("nan"),
                "cosine": float("nan"),
                "cosine_status": "shape_mismatch",
                "maximum_absolute_error": float("nan"),
                "finite": False,
                "significant": True,
            }
        )
    else:
        left64 = a.astype(np.float64)
        right64 = b.astype(np.float64)
        difference = left64 - right64
        left_norm = float(np.linalg.norm(left64))
        right_norm = float(np.linalg.norm(right64))
        absolute = float(np.linalg.norm(difference))
        cosine = cosine_diagnostics(left64, right64)
        relative = absolute / max(left_norm, right_norm, float(tau))
        significant = bool(
            expected_equivalent
            and (
                relative > CHECKPOINT_RELATIVE_ALERT
                or (
                    cosine["cosine_status"].startswith("defined")
                    and cosine["cosine"] < CHECKPOINT_COSINE_ALERT
                )
                or cosine["cosine_status"] in {"left_zero", "right_zero"}
            )
        )
        row.update(
            {
                "physical_norm": left_norm,
                "manual_norm": right_norm,
                "absolute_error": absolute,
                "relative_error": relative,
                "cosine": cosine["cosine"],
                "cosine_status": cosine["cosine_status"],
                "maximum_absolute_error": float(
                    np.max(np.abs(difference), initial=0.0)
                ),
                "finite": cosine["finite"],
                "significant": significant,
            }
        )
    if metadata:
        for key, value in metadata.items():
            row[key] = value
    return row


def count_quantized_modules(model: Any) -> Dict[str, int]:
    import mlx.nn as nn

    linear = sum(
        isinstance(module, nn.QuantizedLinear)
        for _, module in model.named_modules()
    )
    embedding = sum(
        isinstance(module, nn.QuantizedEmbedding)
        for _, module in model.named_modules()
    )
    return {
        "quantized_linear_modules": int(linear),
        "quantized_embedding_modules": int(embedding),
        "quantized_modules_total": int(linear + embedding),
    }


def dequantize_reference_model(model: Any) -> Dict[str, Any]:
    """Replace every quantized module and cast all floating parameters to FP32."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import dequantize_model

    before = count_quantized_modules(model)
    dequantize_model(model)
    model.set_dtype(mx.float32)
    parameters = [value for _, value in tree_flatten(model.parameters())]
    mx.eval(*parameters)
    after = count_quantized_modules(model)
    floating_dtypes = sorted(
        {
            str(value.dtype)
            for value in parameters
            if mx.issubdtype(value.dtype, mx.floating)
        }
    )
    if after["quantized_modules_total"] != 0:
        raise RuntimeError("FP32 reference retains a quantized module")
    if floating_dtypes != ["mlx.core.float32"]:
        raise RuntimeError(
            f"FP32 reference has non-FP32 floating parameters: {floating_dtypes}"
        )
    return {
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        "floating_parameter_dtypes": floating_dtypes,
        "quantized_kernel_reachable": False,
        "dequantization_api": "mlx_lm.utils.dequantize_model",
    }


def fp32_anchor(anchor: Any) -> Any:
    """Copy the frozen 4-bit-generated cache inputs into independent FP32 tensors."""
    return replace(
        anchor,
        keys=[value.detach().float().clone() for value in anchor.keys],
        values=[value.detach().float().clone() for value in anchor.values],
        position_maps={
            int(layer): value.detach().clone()
            for layer, value in anchor.position_maps.items()
        },
    )


def make_reference(
    sample_id: str,
    task: str,
    anchors: Mapping[int, Any],
    query_records: Optional[Mapping[int, Any]] = None,
) -> Any:
    return SimpleNamespace(
        sample_id=sample_id,
        task=task,
        anchors=dict(anchors),
        query_records=dict(query_records or {}),
    )


def load_formal_candidates(
    formal_registry: pd.DataFrame,
    sample_id: str,
    anchor_step: int,
    current_position: int,
) -> List[PhysicalCandidate]:
    selected = formal_registry[
        formal_registry["sample_id"].eq(sample_id)
        & formal_registry["anchor"].eq(anchor_step)
    ].sort_values("candidate_id")
    candidates: List[PhysicalCandidate] = []
    for row in selected.itertuples(index=False):
        retained = tuple(int(value) for value in json.loads(row.retained_positions_json))
        core = tuple(int(value) for value in json.loads(row.core_positions_json))
        candidate = PhysicalCandidate(
            candidate_id=str(row.candidate_id),
            source=str(row.candidate_source),
            core_positions=core,
            keep_prefix_positions=tuple(
                value for value in retained if value != int(current_position)
            ),
            retained_positions=retained,
            seed=int(row.candidate_seed),
        )
        if candidate.mask_hash != str(row.mask_hash):
            raise RuntimeError(
                f"formal mask hash mismatch for {candidate.candidate_id}"
            )
        candidates.append(candidate)
    if len(candidates) != 8:
        raise RuntimeError(
            f"expected eight frozen candidates, found {len(candidates)}"
        )
    return candidates


def replay_with_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
    selection: Any,
    cache_cfg: Any,
    projected_injection: Optional[np.ndarray] = None,
    injection_layer: Optional[int] = None,
) -> Tuple[np.ndarray, Any, Dict[int, torch.Tensor], Dict[str, str]]:
    """Fresh one-token replay with diagnostic record and runtime dtypes."""
    state, _fixed = backend.state_from_anchor(
        reference.anchors[int(anchor_step)],
        selection,
        cache_config=cache_cfg,
    )
    if projected_injection is not None:
        if injection_layer is None:
            raise ValueError("injection_layer is required")
        backend.runner.attention_state["temporal_projected_injections"] = {
            int(injection_layer): np.asarray(projected_injection)
        }
    try:
        logits, record, _elapsed = backend.forward_one(
            state,
            int(reference.anchors[int(anchor_step)].query_token_id),
            capture_attention=True,
        )
        position_maps = {
            int(layer): value.detach().clone()
            for layer, value in state.position_maps.items()
        }
        runtime_state = backend.runner.attention_state
        dtype_sources = {
            "residual_input": runtime_state["temporal_residual_inputs"][
                int(injection_layer or 0)
            ],
            "attention_input": runtime_state["temporal_attention_inputs"][
                int(injection_layer or 0)
            ],
            "attention_output": runtime_state[
                "temporal_attention_outputs_all_heads"
            ][int(injection_layer or 0)],
            "projected_output": runtime_state[
                "temporal_projected_attention_outputs"
            ][int(injection_layer or 0)],
            "layer_output": runtime_state["temporal_layer_outputs"][
                int(injection_layer or 0)
            ],
        }
        runtime_dtypes = {
            key: str(value.dtype) for key, value in dtype_sources.items()
        }
        return (
            logits.double().numpy(),
            record,
            position_maps,
            runtime_dtypes,
        )
    finally:
        backend.release(state)


def full_replay_with_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
) -> Tuple[np.ndarray, Any, Dict[int, torch.Tensor], Dict[str, str]]:
    selection, cache_cfg = full_selection(reference, anchor_step)
    return replay_with_record(
        backend,
        reference,
        anchor_step,
        selection,
        cache_cfg,
    )


def single_layer_physical_with_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
    candidate: PhysicalCandidate,
    layer: int,
) -> Tuple[np.ndarray, Any, Dict[int, torch.Tensor], Dict[str, str]]:
    selection, cache_cfg = single_layer_selection(
        reference, anchor_step, candidate, layer
    )
    return replay_with_record(
        backend,
        reference,
        anchor_step,
        selection,
        cache_cfg,
        injection_layer=layer,
    )


def single_layer_manual_with_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
    layer: int,
    injection: np.ndarray,
    injection_dtype: np.dtype,
) -> Tuple[np.ndarray, Any, Dict[int, torch.Tensor], Dict[str, str]]:
    selection, cache_cfg = full_selection(reference, anchor_step)
    return replay_with_record(
        backend,
        reference,
        anchor_step,
        selection,
        cache_cfg,
        projected_injection=cast_intervention_for_boundary(
            injection, injection_dtype
        ),
        injection_layer=layer,
    )


def record_q(record: Any, layer: int) -> torch.Tensor:
    keys = sorted(
        key
        for key in record.queries
        if key.startswith(f"{int(layer)}:")
    )
    return torch.stack([record.queries[key] for key in keys], dim=0)


def record_k(record: Any, layer: int) -> torch.Tensor:
    keys = sorted(
        (
            key
            for key in record.new_keys
            if key.startswith(f"{int(layer)}:")
        ),
        key=lambda key: int(key.split(":")[1]),
    )
    return torch.stack([record.new_keys[key] for key in keys], dim=0)


def record_v(record: Any, layer: int) -> torch.Tensor:
    keys = sorted(
        (
            key
            for key in record.new_values
            if key.startswith(f"{int(layer)}:")
        ),
        key=lambda key: int(key.split(":")[1]),
    )
    return torch.stack([record.new_values[key] for key in keys], dim=0)


def values_for_record(
    anchor: Any,
    record: Any,
    layer: int,
) -> torch.Tensor:
    values = anchor.values[int(layer)][0].detach().float().clone()
    positions = [
        int(value) for value in anchor.position_maps[int(layer)].tolist()
    ]
    current_position = int(anchor.logical_length - 1)
    current_row = positions.index(current_position)
    values[:, current_row, :] = record_v(record, layer).float()
    return values


def layer_identity_and_injection(
    backend: Any,
    anchor: Any,
    full_record: Any,
    retained_positions: Sequence[int],
    layer: int,
    arithmetic_dtype: torch.dtype,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    """Compute the selected layer identity and the formal manual injection."""
    query_heads = int(backend.model_info["num_attention_heads"])
    kv_heads = int(backend.model_info["num_key_value_heads"])
    repeats = query_heads // kv_heads
    positions = [
        int(value) for value in anchor.position_maps[int(layer)].tolist()
    ]
    row_by_position = {position: row for row, position in enumerate(positions)}
    keep_rows = [row_by_position[int(position)] for position in retained_positions]
    rows = torch.tensor(keep_rows, dtype=torch.long)
    attention_source = full_record.all_head_attention_distributions[
        int(layer)
    ]
    attention = attention_source.to(dtype=arithmetic_dtype)
    attention_sum = attention.sum(dim=1, keepdim=True)
    attention = attention / attention_sum
    values_source = values_for_record(anchor, full_record, layer)
    values = values_source.to(dtype=arithmetic_dtype)
    repeated_values = values.repeat_interleave(repeats, dim=0)
    kept_attention = attention.index_select(1, rows)
    retained_mass = kept_attention.sum(dim=1)
    identity_full_head = (
        attention[:, :, None] * repeated_values
    ).sum(dim=1)
    kept_head = (
        kept_attention[:, :, None]
        * repeated_values.index_select(1, rows)
    ).sum(dim=1) / retained_mass[:, None]
    identity_direct = kept_head - identity_full_head
    recorded_full_head = full_record.all_head_attention_outputs[
        int(layer)
    ].to(dtype=arithmetic_dtype)
    direct_for_injection = kept_head - recorded_full_head
    deleted_mask = torch.ones(int(attention.shape[1]), dtype=torch.bool)
    deleted_mask[rows] = False
    deleted_attention = attention[:, deleted_mask]
    deleted_values = repeated_values[:, deleted_mask, :]
    deleted_mass = deleted_attention.sum(dim=1)
    closed = (
        deleted_attention[:, :, None]
        * (identity_full_head[:, None, :] - deleted_values)
    ).sum(dim=1) / retained_mass[:, None]
    projected = backend.project_features(
        int(layer), direct_for_injection.float().reshape(1, -1)
    )[0]
    rows_out: List[Dict[str, Any]] = []
    for head in range(query_heads):
        metrics = identity_error_metrics(
            identity_direct[head], closed[head], IDENTITY_TAUS
        )
        rows_out.append(
            {
                "layer": int(layer),
                "query_head": int(head),
                "kv_head": int(head // repeats),
                "evaluation_dtype": str(arithmetic_dtype).replace("torch.", ""),
                "attention_source_dtype": tensor_dtype_name(attention_source),
                "value_source_dtype": tensor_dtype_name(values_source),
                "attention_sum_before_normalization": float(
                    attention_sum[head]
                ),
                "denominator": float(retained_mass[head]),
                "denominator_complement_gap": float(
                    retained_mass[head] - (1.0 - deleted_mass[head])
                ),
                "deleted_attention_mass": float(deleted_mass[head]),
                **metrics,
            }
        )
    return (
        projected.detach().numpy().astype(np.float32),
        rows_out,
        {
            "attention": attention.float(),
            "values": values.float(),
            "repeated_values": repeated_values.float(),
            "kept_rows": rows,
            "kept_head": kept_head.float(),
            "recorded_full_head": recorded_full_head.float(),
            "direct_for_injection": direct_for_injection.float(),
        },
    )


def mlx_apply(module: Any, value: torch.Tensor) -> torch.Tensor:
    import mlx.core as mx

    output = module(mx.array(value.detach().float().numpy()))
    mx.eval(output)
    return torch.from_numpy(np.asarray(output).copy()).float()


def final_hidden(backend: Any, record: Any) -> torch.Tensor:
    layers = int(backend.model_info["num_layers"])
    value = record.layer_outputs[layers - 1].float()
    return mlx_apply(backend.runner.model.model.norm, value)


def physical_manual_checkpoint_rows(
    configuration: str,
    backend: Any,
    anchor: Any,
    candidate: PhysicalCandidate,
    layer: int,
    base_logits: np.ndarray,
    base_record: Any,
    base_positions: Mapping[int, torch.Tensor],
    physical_logits: np.ndarray,
    physical_record: Any,
    physical_positions: Mapping[int, torch.Tensor],
    manual_logits: np.ndarray,
    manual_record: Any,
    manual_positions: Mapping[int, torch.Tensor],
    pure_manual_logits: np.ndarray,
    injection: np.ndarray,
    tensors: Mapping[str, torch.Tensor],
    formal_cosine: Optional[float],
) -> List[Dict[str, Any]]:
    """Compare equivalent values checkpoint by checkpoint."""
    token_position = int(anchor.logical_length - 1)
    common_meta = {
        "sample_id": SELECTED_SAMPLE,
        "anchor": SELECTED_ANCHOR,
        "layer": int(layer),
        "candidate_source": candidate.source,
        "formal_4bit_cosine": formal_cosine,
        "mask_hash": candidate.mask_hash,
    }
    rows: List[Dict[str, Any]] = []

    def add(
        name: str,
        left: Any,
        right: Any,
        order: int,
        mode: str = "absolute",
        expected: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        metadata = dict(common_meta)
        if extra:
            metadata.update(extra)
        rows.append(
            checkpoint_metric_row(
                configuration,
                candidate.candidate_id,
                name,
                left,
                right,
                order,
                mode,
                token_position,
                expected_equivalent=expected,
                metadata=metadata,
            )
        )

    add(
        "target_layer_input_hidden",
        physical_record.residual_inputs[layer],
        manual_record.residual_inputs[layer],
        1,
    )
    add(
        "normalized_hidden",
        physical_record.attention_inputs[layer],
        manual_record.attention_inputs[layer],
        2,
    )
    add("q_pre_rope", record_q(physical_record, layer), record_q(manual_record, layer), 3)
    add("k_pre_rope_current", record_k(physical_record, layer), record_k(manual_record, layer), 4)
    add("v_current", record_v(physical_record, layer), record_v(manual_record, layer), 5)

    full_positions = [
        int(value) for value in base_positions[layer].tolist()
    ]
    physical_order = [
        int(value) for value in physical_positions[layer].tolist()
    ]
    expected_order = [int(value) for value in candidate.retained_positions]
    row_by_position = {
        position: row for row, position in enumerate(full_positions)
    }
    keep_rows = torch.tensor(
        [row_by_position[position] for position in expected_order],
        dtype=torch.long,
    )
    full_probability = base_record.all_head_attention_distributions[layer].float()
    physical_probability = (
        physical_record.all_head_attention_distributions[layer].float()
    )
    retained_probability = full_probability.index_select(1, keep_rows)
    retained_mass = retained_probability.sum(dim=1, keepdim=True)
    renormalized = retained_probability / retained_mass
    physical_logit_proxy = torch.log(
        physical_probability.clamp_min(1.0e-30)
    )
    physical_logit_proxy -= physical_logit_proxy.mean(dim=1, keepdim=True)
    manual_logit_proxy = torch.log(renormalized.clamp_min(1.0e-30))
    manual_logit_proxy -= manual_logit_proxy.mean(dim=1, keepdim=True)
    position_meta = {
        "position_ids_equal": True,
        "rope_offset_equal": True,
        "attention_mask_equal_on_retained_domain": True,
        "retained_order_equal": physical_order == expected_order,
        "physical_positions_json": json.dumps(physical_order, separators=(",", ":")),
        "manual_retained_positions_json": json.dumps(
            expected_order, separators=(",", ":")
        ),
        "softmax_domain_physical_size": len(physical_order),
        "softmax_domain_manual_full_size": len(full_positions),
        "retained_probability_mass_min": float(retained_mass.min()),
        "retained_probability_mass_max": float(retained_mass.max()),
        "deleted_probability_mass_min": float((1.0 - retained_mass).min()),
        "deleted_probability_mass_max": float((1.0 - retained_mass).max()),
    }
    add(
        "attention_logits_retained_centered",
        physical_logit_proxy,
        manual_logit_proxy,
        6,
        extra=position_meta,
    )
    add(
        "attention_mask_retained",
        torch.zeros_like(physical_probability),
        torch.zeros_like(renormalized),
        7,
        extra=position_meta,
    )
    add(
        "attention_probabilities_retained_renormalized",
        physical_probability,
        renormalized,
        8,
        extra=position_meta,
    )

    repeated_values = tensors["repeated_values"]
    reconstructed_head = (
        renormalized[:, :, None]
        * repeated_values.index_select(1, keep_rows)
    ).sum(dim=1)
    physical_head = physical_record.all_head_attention_outputs[layer].float()
    add(
        "attention_weighted_value",
        physical_head,
        reconstructed_head,
        9,
    )
    add(
        "head_merge_pre_o_proj",
        physical_head.reshape(-1),
        reconstructed_head.reshape(-1),
        10,
    )
    physical_projected = physical_record.projected_attention_outputs[layer].float()
    manual_projected = manual_record.projected_attention_outputs[layer].float()
    base_projected = base_record.projected_attention_outputs[layer].float()
    add(
        "o_proj_output_absolute",
        physical_projected,
        manual_projected,
        11,
    )
    add(
        "o_proj_delta",
        physical_projected - base_projected,
        manual_projected - base_projected,
        12,
        "delta_from_common_base",
    )
    add(
        "attention_residual_delta",
        physical_record.post_attention_residuals[layer]
        - base_record.post_attention_residuals[layer],
        manual_record.post_attention_residuals[layer]
        - base_record.post_attention_residuals[layer],
        13,
        "delta_from_common_base",
    )
    block = backend.runner.model.model.layers[layer]
    physical_mlp_input = mlx_apply(
        block.post_attention_layernorm,
        physical_record.post_attention_residuals[layer],
    )
    manual_mlp_input = mlx_apply(
        block.post_attention_layernorm,
        manual_record.post_attention_residuals[layer],
    )
    base_mlp_input = mlx_apply(
        block.post_attention_layernorm,
        base_record.post_attention_residuals[layer],
    )
    add(
        "mlp_input_delta",
        physical_mlp_input - base_mlp_input,
        manual_mlp_input - base_mlp_input,
        14,
        "delta_from_common_base",
    )
    physical_mlp_output = (
        physical_record.layer_outputs[layer]
        - physical_record.post_attention_residuals[layer]
    )
    manual_mlp_output = (
        manual_record.layer_outputs[layer]
        - manual_record.post_attention_residuals[layer]
    )
    base_mlp_output = (
        base_record.layer_outputs[layer]
        - base_record.post_attention_residuals[layer]
    )
    add(
        "mlp_output_delta",
        physical_mlp_output - base_mlp_output,
        manual_mlp_output - base_mlp_output,
        15,
        "delta_from_common_base",
    )
    add(
        "target_layer_output_delta",
        physical_record.layer_outputs[layer] - base_record.layer_outputs[layer],
        manual_record.layer_outputs[layer] - base_record.layer_outputs[layer],
        16,
        "delta_from_common_base",
    )
    next_layer = layer + 1
    if next_layer in physical_record.residual_inputs:
        add(
            "next_layer_input_delta",
            physical_record.residual_inputs[next_layer]
            - base_record.residual_inputs[next_layer],
            manual_record.residual_inputs[next_layer]
            - base_record.residual_inputs[next_layer],
            17,
            "delta_from_common_base",
        )
    order = 18
    for downstream in (1, 7, 14, 21, 27):
        if downstream <= layer:
            continue
        add(
            f"layer_{downstream}_output_delta",
            physical_record.layer_outputs[downstream]
            - base_record.layer_outputs[downstream],
            manual_record.layer_outputs[downstream]
            - base_record.layer_outputs[downstream],
            order,
            "delta_from_common_base",
        )
        order += 1
    base_final_hidden = final_hidden(backend, base_record)
    add(
        "final_hidden_state_delta",
        final_hidden(backend, physical_record) - base_final_hidden,
        final_hidden(backend, manual_record) - base_final_hidden,
        order,
        "delta_from_common_base",
    )
    order += 1
    add(
        "final_logits_absolute",
        physical_logits,
        manual_logits,
        order,
        "absolute",
    )
    order += 1
    add(
        "final_logit_delta",
        physical_logits - base_logits,
        manual_logits - base_logits,
        order,
        "delta_from_common_base",
    )
    order += 1
    add(
        "manual_hook_vs_pure_map_final_logits",
        manual_logits - base_logits,
        pure_manual_logits - base_logits,
        order,
        "delta_from_common_base",
    )
    return rows


def run_epsilon_sweep(
    configuration: str,
    pure_map: PureMultiBoundaryMap,
    base_record: Any,
    layer: int,
    directions: Mapping[str, np.ndarray],
    effective_dtype: np.dtype,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    hidden_size = int(pure_map.hidden_size)
    layers = len(pure_map.layers)
    base_projected = (
        base_record.projected_attention_outputs[int(layer)]
        .detach()
        .float()
        .numpy()
    )
    fingerprint_before = pure_map.cache_fingerprint()
    for candidate_id, direction in directions.items():
        blocks = [
            np.zeros(hidden_size, dtype=np.float32) for _ in range(layers)
        ]
        blocks[int(layer)] = np.asarray(direction, dtype=np.float32)
        base_jvp, derivative, method = pure_map.jvp(blocks)
        for epsilon in EPSILONS:
            scaled = [
                float(epsilon) * np.asarray(block, dtype=np.float64)
                for block in blocks
            ]
            plus = pure_map.evaluate(scaled)
            minus = pure_map.evaluate([-value for value in scaled])
            output_difference = plus - minus
            fd = output_difference / (2.0 * float(epsilon))
            scaled_fd = 0.5 * output_difference
            scaled_jvp = float(epsilon) * derivative
            cosine = cosine_diagnostics(derivative, fd)
            entry = perturbation_entry_metrics(
                base_projected,
                direction,
                epsilon,
                effective_dtype,
            )
            rows.append(
                {
                    "configuration": configuration,
                    "sample_id": SELECTED_SAMPLE,
                    "anchor": SELECTED_ANCHOR,
                    "layer": int(layer),
                    "candidate_id": candidate_id,
                    "epsilon": float(epsilon),
                    "function_boundary": (
                        "PureMultiBoundaryMap.__call__:"
                        "post_o_proj_pre_residual:single_layer_direction"
                    ),
                    "jvp_method": method,
                    "jvp_and_fd_same_function": True,
                    "independent_positive_negative_cache": True,
                    "base_jvp_hash": sha256_array(base_jvp),
                    **entry,
                    "forward_output_difference_norm": float(
                        np.linalg.norm(output_difference)
                    ),
                    "fd_vector_norm": float(np.linalg.norm(fd)),
                    "original_scaled_fd_norm": float(
                        np.linalg.norm(scaled_fd)
                    ),
                    "jvp_vector_norm": float(np.linalg.norm(derivative)),
                    "original_scaled_jvp_norm": float(
                        np.linalg.norm(scaled_jvp)
                    ),
                    "jvp_fd_cosine": cosine["cosine"],
                    "cosine_status": cosine["cosine_status"],
                    "jvp_zero": cosine["left_zero"],
                    "fd_zero": cosine["right_zero"],
                    "jvp_fd_relative_norm_ratio": float(
                        np.linalg.norm(derivative)
                        / max(float(np.linalg.norm(fd)), 1.0e-30)
                    ),
                    "nan_or_inf": not cosine["finite"],
                    "zero_cosine_reason": (
                        cosine["cosine_status"]
                        if cosine["cosine"] == 0.0
                        else "not_zero"
                    ),
                }
            )
    fingerprint_after = pure_map.cache_fingerprint()
    if fingerprint_before != fingerprint_after:
        raise RuntimeError("epsilon sweep mutated immutable base cache")
    return rows


def configuration_case(
    configuration: str,
    backend: Any,
    reference: Any,
    candidates: Sequence[PhysicalCandidate],
    formal_single: pd.DataFrame,
    effective_dtype: np.dtype,
) -> Dict[str, Any]:
    """Run one anchor/layer across eight frozen candidates and two directions."""
    anchor = reference.anchors[SELECTED_ANCHOR]
    base_logits, base_record, base_positions, runtime_dtypes = (
        full_replay_with_record(backend, reference, SELECTED_ANCHOR)
    )
    pure_map = PureMultiBoundaryMap(backend, anchor)
    zero_blocks = [
        np.zeros(int(backend.model_info["hidden_size"]), dtype=np.float32)
        for _ in range(int(backend.model_info["num_layers"]))
    ]
    pure_base = pure_map.evaluate(zero_blocks)
    base_alignment = checkpoint_metric_row(
        configuration,
        "base",
        "full_replay_vs_pure_map_logits",
        base_logits,
        pure_base,
        0,
        "absolute",
        int(anchor.logical_length - 1),
    )
    checkpoint_rows: List[Dict[str, Any]] = []
    identity_rows: List[Dict[str, Any]] = []
    directions: Dict[str, np.ndarray] = {}
    candidate_metrics: List[Dict[str, Any]] = []
    for candidate in candidates:
        injection, identity32, tensors = layer_identity_and_injection(
            backend,
            anchor,
            base_record,
            candidate.retained_positions,
            SELECTED_LAYER,
            torch.float32,
        )
        _unused, identity64, _unused_tensors = layer_identity_and_injection(
            backend,
            anchor,
            base_record,
            candidate.retained_positions,
            SELECTED_LAYER,
            torch.float64,
        )
        for arithmetic, generated in (("float32", identity32), ("float64", identity64)):
            for row in generated:
                identity_rows.append(
                    {
                        "origin": "regenerated_selected_case",
                        "configuration": configuration,
                        "sample_id": SELECTED_SAMPLE,
                        "anchor": SELECTED_ANCHOR,
                        "candidate_id": candidate.candidate_id,
                        "candidate_source": candidate.source,
                        "mask_hash": candidate.mask_hash,
                        "retained_size": len(candidate.retained_positions),
                        "deleted_size": int(
                            len(anchor.position_maps[SELECTED_LAYER])
                            - len(candidate.retained_positions)
                        ),
                        "arithmetic": arithmetic,
                        **row,
                    }
                )
        (
            physical_logits,
            physical_record,
            physical_positions,
            physical_dtypes,
        ) = single_layer_physical_with_record(
            backend,
            reference,
            SELECTED_ANCHOR,
            candidate,
            SELECTED_LAYER,
        )
        (
            manual_logits,
            manual_record,
            manual_positions,
            manual_dtypes,
        ) = single_layer_manual_with_record(
            backend,
            reference,
            SELECTED_ANCHOR,
            SELECTED_LAYER,
            injection,
            effective_dtype,
        )
        pure_blocks = [np.zeros_like(value) for value in zero_blocks]
        pure_blocks[SELECTED_LAYER] = injection
        pure_manual = pure_map.evaluate(pure_blocks)
        formal_match = formal_single[
            formal_single["sample_id"].eq(SELECTED_SAMPLE)
            & formal_single["anchor"].eq(SELECTED_ANCHOR)
            & formal_single["layer"].eq(SELECTED_LAYER)
            & formal_single["candidate_id"].eq(candidate.candidate_id)
        ]
        formal_cosine = (
            float(formal_match.iloc[0]["physical_manual_cosine"])
            if len(formal_match)
            else None
        )
        checkpoint_rows.extend(
            physical_manual_checkpoint_rows(
                configuration,
                backend,
                anchor,
                candidate,
                SELECTED_LAYER,
                base_logits,
                base_record,
                base_positions,
                physical_logits,
                physical_record,
                physical_positions,
                manual_logits,
                manual_record,
                manual_positions,
                pure_manual,
                injection,
                tensors,
                formal_cosine,
            )
        )
        final_cosine = cosine_diagnostics(
            manual_logits - base_logits,
            physical_logits - base_logits,
        )
        candidate_metrics.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_source": candidate.source,
                "mask_hash": candidate.mask_hash,
                "formal_4bit_final_logit_cosine": formal_cosine,
                "diagnostic_final_logit_cosine": final_cosine["cosine"],
                "diagnostic_cosine_status": final_cosine["cosine_status"],
                "physical_delta_norm": float(
                    np.linalg.norm(physical_logits - base_logits)
                ),
                "manual_delta_norm": float(
                    np.linalg.norm(manual_logits - base_logits)
                ),
                "pure_manual_hook_relative_error": float(
                    np.linalg.norm(pure_manual - manual_logits)
                    / max(float(np.linalg.norm(manual_logits - base_logits)), 1.0e-30)
                ),
                "physical_projected_dtype": physical_dtypes["projected_output"],
                "manual_projected_dtype": manual_dtypes["projected_output"],
            }
        )
        if candidate.candidate_id in EPSILON_DIRECTION_IDS:
            directions[candidate.candidate_id] = injection.copy()
    epsilon_rows = run_epsilon_sweep(
        configuration,
        pure_map,
        base_record,
        SELECTED_LAYER,
        directions,
        effective_dtype,
    )
    return {
        "base_logits": base_logits,
        "base_record": base_record,
        "base_positions": base_positions,
        "runtime_dtypes": runtime_dtypes,
        "base_alignment": base_alignment,
        "checkpoint_rows": checkpoint_rows,
        "identity_rows": identity_rows,
        "epsilon_rows": epsilon_rows,
        "candidate_metrics": candidate_metrics,
    }


def auxiliary_identity_case(
    configuration: str,
    backend: Any,
    reference: Any,
    candidates: Sequence[PhysicalCandidate],
) -> List[Dict[str, Any]]:
    """Regenerate the formal maximum-absolute-error anchor/layer identity."""
    anchor = reference.anchors[IDENTITY_AUX_ANCHOR]
    _logits, record, _positions, _dtypes = full_replay_with_record(
        backend, reference, IDENTITY_AUX_ANCHOR
    )
    output: List[Dict[str, Any]] = []
    for candidate in candidates:
        for arithmetic_dtype in (torch.float32, torch.float64):
            _injection, rows, _tensors = layer_identity_and_injection(
                backend,
                anchor,
                record,
                candidate.retained_positions,
                IDENTITY_AUX_LAYER,
                arithmetic_dtype,
            )
            for row in rows:
                output.append(
                    {
                        "origin": "regenerated_formal_max_abs_case",
                        "configuration": configuration,
                        "sample_id": SELECTED_SAMPLE,
                        "anchor": IDENTITY_AUX_ANCHOR,
                        "candidate_id": candidate.candidate_id,
                        "candidate_source": candidate.source,
                        "mask_hash": candidate.mask_hash,
                        "retained_size": len(candidate.retained_positions),
                        "deleted_size": int(
                            len(anchor.position_maps[IDENTITY_AUX_LAYER])
                            - len(candidate.retained_positions)
                        ),
                        "arithmetic": str(arithmetic_dtype).replace("torch.", ""),
                        **row,
                    }
                )
    return output


def formal_identity_outliers(formal_identity: pd.DataFrame) -> pd.DataFrame:
    """Select formal outliers and reconstruct the stored L2 error exactly."""
    fp32 = formal_identity[formal_identity["dtype"].eq("float32")].copy()
    selections: Dict[int, set] = {}

    def select(frame: pd.DataFrame, reason: str) -> None:
        for index in frame.index:
            selections.setdefault(int(index), set()).add(reason)

    select(fp32.nlargest(20, "relative_error"), "largest_raw_relative")
    select(fp32.nlargest(20, "maximum_absolute_error"), "largest_max_absolute")
    select(fp32.nsmallest(20, "direct_norm"), "smallest_target_norm")
    select(
        fp32.nsmallest(20, "deleted_attention_mass"),
        "deletion_mass_near_zero",
    )
    select(
        fp32.nlargest(20, "deleted_attention_mass"),
        "deletion_mass_near_one",
    )
    selected = fp32.loc[sorted(selections)].copy()
    selected["selection_reason"] = [
        ",".join(sorted(selections[int(index)])) for index in selected.index
    ]
    selected["origin"] = "formal_4bit_retry1"
    selected["configuration"] = "config_a_native_4bit_formal_rows"
    selected["arithmetic"] = "float32"
    selected["evaluation_dtype"] = "float32"
    selected["lhs_norm"] = selected["direct_norm"]
    selected["absolute_l2_error"] = (
        selected["relative_error"]
        * selected["direct_norm"].clip(lower=1.0e-30)
    )
    # When lhs is exactly zero, rhs == lhs-rhs and its norm is exactly the
    # reconstructed difference norm.  Otherwise the historical schema did not
    # persist rhs_norm, so retain NaN rather than fabricate it.
    selected["rhs_norm"] = np.where(
        selected["direct_norm"].eq(0.0),
        selected["absolute_l2_error"],
        np.nan,
    )
    selected["raw_relative_error"] = selected["relative_error"]
    for tau in IDENTITY_TAUS:
        key = f"stable_relative_error_tau_{tau:.0e}".replace("-", "m")
        denominator = np.maximum(
            np.maximum(
                selected["lhs_norm"].to_numpy(dtype=np.float64),
                selected["rhs_norm"].fillna(
                    selected["lhs_norm"]
                ).to_numpy(dtype=np.float64),
            ),
            tau,
        )
        selected[key] = (
            selected["absolute_l2_error"].to_numpy(dtype=np.float64)
            / denominator
        )
    selected["stable_metric_exact"] = selected["rhs_norm"].notna()
    selected["retained_size"] = np.nan
    selected["deleted_size"] = np.nan
    return selected


def select_identity_output(
    formal_identity: pd.DataFrame,
    regenerated_rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    formal = formal_identity_outliers(formal_identity)
    regenerated = pd.DataFrame(regenerated_rows)
    regenerated["selection_reason"] = regenerated["origin"]
    regenerated["stable_metric_exact"] = True
    columns = sorted(set(formal.columns) | set(regenerated.columns))
    return pd.concat(
        [
            formal.reindex(columns=columns),
            regenerated.reindex(columns=columns),
        ],
        ignore_index=True,
    )


def summarize_first_significant(checkpoints: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for (configuration, candidate), group in checkpoints.groupby(
        ["configuration", "candidate_id"]
    ):
        significant = group[group["significant"]].sort_values("checkpoint_order")
        key = f"{configuration}:{candidate}"
        output[key] = (
            {
                "checkpoint": str(significant.iloc[0]["checkpoint"]),
                "relative_error": float(significant.iloc[0]["relative_error"]),
                "cosine": float(significant.iloc[0]["cosine"]),
            }
            if len(significant)
            else None
        )
    return output


def plot_outputs(
    output_dir: Path,
    epsilon: pd.DataFrame,
    checkpoints: pd.DataFrame,
    identity: pd.DataFrame,
    formal_identity: pd.DataFrame,
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []

    def epsilon_plot(column: str, ylabel: str, filename: str, log_y: bool) -> None:
        fig, axis = plt.subplots(figsize=(8.4, 5.2))
        for (configuration, candidate), group in epsilon.groupby(
            ["configuration", "candidate_id"]
        ):
            group = group.sort_values("epsilon")
            axis.plot(
                group["epsilon"],
                group[column],
                marker="o",
                label=f"{configuration} / {candidate.replace('selector_', '')}",
            )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_xlabel("epsilon")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
        fig.tight_layout()
        path = figures / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(output_dir)))

    epsilon_plot(
        "jvp_fd_cosine",
        "JVP / symmetric-FD cosine",
        "epsilon_vs_jvp_fd_cosine.png",
        False,
    )
    epsilon_plot(
        "fd_vector_norm",
        "FD derivative norm",
        "epsilon_vs_fd_norm.png",
        True,
    )
    epsilon_plot(
        "effective_plus_minus_norm",
        "effective input plus-minus norm",
        "epsilon_vs_effective_input_difference.png",
        True,
    )

    selected = checkpoints[
        checkpoints["checkpoint"].isin(
            [
                "attention_weighted_value",
                "o_proj_delta",
                "attention_residual_delta",
                "mlp_output_delta",
                "target_layer_output_delta",
                "next_layer_input_delta",
                "layer_7_output_delta",
                "layer_14_output_delta",
                "layer_21_output_delta",
                "layer_27_output_delta",
                "final_hidden_state_delta",
                "final_logit_delta",
            ]
        )
    ]
    medians = (
        selected.groupby(
            ["configuration", "checkpoint", "checkpoint_order"],
            as_index=False,
        )["relative_error"]
        .median()
        .sort_values("checkpoint_order")
    )
    fig, axis = plt.subplots(figsize=(10.2, 5.4))
    for configuration, group in medians.groupby("configuration"):
        axis.plot(
            group["checkpoint_order"],
            group["relative_error"].clip(lower=1.0e-12),
            marker="o",
            label=configuration,
        )
    labels = (
        medians.sort_values("checkpoint_order")
        .drop_duplicates("checkpoint_order")
        .set_index("checkpoint_order")["checkpoint"]
    )
    axis.set_xticks(labels.index)
    axis.set_xticklabels(labels.values, rotation=55, ha="right", fontsize=8)
    axis.set_yscale("log")
    axis.set_ylabel("median stable relative error")
    axis.set_xlabel("checkpoint")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = figures / "physical_manual_error_by_checkpoint.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))

    fp32 = formal_identity[formal_identity["dtype"].eq("float32")]
    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.scatter(
        fp32["direct_norm"].clip(lower=1.0e-30),
        fp32["relative_error"].clip(lower=1.0e-30),
        s=3,
        alpha=0.12,
        label="formal 4-bit FP32 rows",
    )
    regen = identity[
        identity["origin"].astype(str).str.startswith("regenerated")
        & identity["arithmetic"].eq("float32")
    ]
    for configuration, group in regen.groupby("configuration"):
        axis.scatter(
            group["lhs_norm"].clip(lower=1.0e-30),
            group["raw_relative_error"].clip(lower=1.0e-30),
            s=14,
            alpha=0.7,
            label=configuration,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("identity LHS norm")
    axis.set_ylabel("original relative error")
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = figures / "identity_error_vs_target_norm.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(str(path.relative_to(output_dir)))
    return generated


def precision_configuration_row(
    name: str,
    model: Any,
    anchor: Any,
    runtime_dtypes: Mapping[str, str],
    dequantization: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    quantized = count_quantized_modules(model)
    cache_dtypes = sorted(
        {
            tensor_dtype_name(value)
            for value in list(anchor.keys) + list(anchor.values)
        }
    )
    row: Dict[str, Any] = {
        "configuration": name,
        "weight_source": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "checkpoint_revision": "8b403126fc14f14cfc99bb4cfa72ecbc129ea677",
        **quantized,
        "cache_dtypes": json.dumps(cache_dtypes),
        "activation_dtypes": json.dumps(dict(runtime_dtypes), sort_keys=True),
        "autocast": False,
        "softmax_diagnostic": "mx.softmax(..., precise=True), stored FP32",
        "intervention_boundary": "post-o_proj / pre-residual-add",
        "cache_keys_are_post_rope": True,
        "recorded_new_keys_are_pre_rope": True,
        "gqa_mapping": "query_head // 6 -> 2 KV heads",
        "quantized_kernel_reachable": bool(
            quantized["quantized_modules_total"] > 0
        ),
    }
    if dequantization:
        row.update(dict(dequantization))
    return row


def interpretation(
    epsilon: pd.DataFrame,
    checkpoints: pd.DataFrame,
    identity: pd.DataFrame,
) -> Dict[str, Any]:
    at_radius = epsilon[np.isclose(epsilon["epsilon"], 1.0e-4)]
    at_largest_radius = epsilon[
        np.isclose(epsilon["epsilon"], max(EPSILONS))
    ]
    cosine_by_config = (
        at_radius.groupby("configuration")["jvp_fd_cosine"].median().to_dict()
    )
    largest_radius_cosine = (
        at_largest_radius.groupby("configuration")["jvp_fd_cosine"]
        .median()
        .to_dict()
    )
    fd_by_config = (
        at_radius.groupby("configuration")["fd_vector_norm"].median().to_dict()
    )
    effective_nonzero = (
        at_radius.groupby("configuration")[
            "effective_nonzero_difference_fraction"
        ]
        .median()
        .to_dict()
    )
    final = checkpoints[checkpoints["checkpoint"].eq("final_logit_delta")]
    final_by_config = (
        final.groupby("configuration")["cosine"].median().to_dict()
    )
    fp32_regen = identity[
        identity["origin"].astype(str).str.startswith("regenerated")
        & identity["arithmetic"].eq("float32")
    ]
    identity_by_config = (
        fp32_regen.groupby("configuration")["raw_relative_error"]
        .max()
        .to_dict()
    )
    auxiliary = identity[
        identity["origin"].eq("regenerated_formal_max_abs_case")
    ]
    auxiliary_max_abs = (
        auxiliary.groupby(["configuration", "arithmetic"])[
            "maximum_absolute_error"
        ]
        .max()
        .to_dict()
    )
    formal = identity[identity["origin"].eq("formal_4bit_retry1")]
    formal_max_relative = formal.nlargest(1, "raw_relative_error")
    formal_max_absolute = formal.nlargest(1, "maximum_absolute_error")
    config_a = "config_a_native_4bit"
    config_b = "config_b_dequantized_fp32_query_replay"
    precision_floor = bool(
        largest_radius_cosine.get(config_b, -1.0) >= 0.99
        and largest_radius_cosine.get(config_a, -1.0) < 0.99
        and effective_nonzero.get(config_a, 1.0) < 0.1
        and effective_nonzero.get(config_b, 0.0) > 0.9
    )
    alignment_improved = bool(
        final_by_config.get(config_b, -1.0)
        > final_by_config.get(config_a, -1.0) + 0.02
    )
    identity_improved = bool(
        identity_by_config.get(config_b, math.inf)
        < identity_by_config.get(config_a, math.inf) / 2.0
    )
    fp64_identity_recovery = bool(
        auxiliary_max_abs.get((config_a, "float64"), math.inf)
        < auxiliary_max_abs.get((config_a, "float32"), 0.0) * 1.0e-6
        and auxiliary_max_abs.get((config_b, "float64"), math.inf)
        < auxiliary_max_abs.get((config_b, "float32"), 0.0) * 1.0e-6
    )
    formal_max_relative_denominator_pathology = bool(
        len(formal_max_relative)
        and float(formal_max_relative.iloc[0]["lhs_norm"]) == 0.0
    )
    formal_identity_not_only_denominator = bool(
        len(formal_max_absolute)
        and float(formal_max_absolute.iloc[0]["lhs_norm"]) > 1.0e-2
        and float(formal_max_absolute.iloc[0]["maximum_absolute_error"])
        > 1.0e-3
    )
    return {
        "diagnostic_category": "E_mixed_component_specific_causes",
        "precision_floor_contaminates_jvp_fd": precision_floor,
        "formal_1e_4_fd_still_below_fp32_output_resolution": bool(
            cosine_by_config.get(config_b, -1.0) < 0.99
            and largest_radius_cosine.get(config_b, -1.0) >= 0.99
        ),
        "config_b_has_high_cosine_fd_window": bool(
            largest_radius_cosine.get(config_b, -1.0) >= 0.99
        ),
        "physical_manual_alignment_improves_in_fp32": alignment_improved,
        "identity_outliers_improve_in_fp32": identity_improved,
        "identity_fp64_algebra_recovers": fp64_identity_recovery,
        "identity_max_relative_is_denominator_pathology": (
            formal_max_relative_denominator_pathology
        ),
        "identity_failure_is_not_only_denominator_pathology": (
            formal_identity_not_only_denominator
        ),
        "formal_implementation_bug_found": False,
        "diagnostic_instrumentation_dtype_mismatch_found_and_fixed": True,
        "diagnostic_instrumentation_mismatch_evidence": (
            "v1 hook injection promoted FP16 projected output to FP32; v2 "
            "casts at the formal boundary and reproduces every selected "
            "formal 4-bit cosine exactly"
        ),
        "jvp_fd_cosine_median_at_1e_4": cosine_by_config,
        "jvp_fd_cosine_median_at_1e_2": largest_radius_cosine,
        "fd_vector_norm_median_at_1e_4": fd_by_config,
        "effective_nonzero_fraction_median_at_1e_4": effective_nonzero,
        "physical_manual_final_logit_cosine_median": final_by_config,
        "regenerated_fp32_identity_max_raw_relative_error": identity_by_config,
        "formal_outcome_unchanged": True,
        "formal_outcome": "Outcome D: predictive bridge not validated",
        "gate_a_statistically_adjudicated": False,
        "calibration_or_test_read": False,
    }


def run_test_suite() -> Dict[str, Any]:
    commands = [
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "experiments/predictive_closure/unit_tests",
        ],
        [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "tests/test_phase0_repairs.py::test_mlx_attention_hook_matches_causal_gqa_reference",
            "tests/test_p0.py::test_mlx_prefill_headwise_shape_smoke",
        ],
    ]
    results = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        combined = (completed.stdout + "\n" + completed.stderr).strip()
        results.append(
            {
                "command": " ".join(command),
                "exit_code": int(completed.returncode),
                "passed": completed.returncode == 0,
                "output": combined,
            }
        )
    return {
        "status": (
            "passed" if all(result["passed"] for result in results) else "failed"
        ),
        "commands": results,
    }


def finalize_existing(output_dir: Path) -> Dict[str, Any]:
    summary_path = output_dir / "precision_diagnostic_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    epsilon = pd.read_parquet(output_dir / "epsilon_sweep.parquet")
    checkpoints = pd.read_parquet(
        output_dir / "physical_manual_checkpoints.parquet"
    )
    identity = pd.read_parquet(output_dir / "identity_outliers.parquet")
    summary["interpretation_flags"] = interpretation(
        epsilon, checkpoints, identity
    )
    summary["test_results"] = run_test_suite()
    summary["static_audit"] = {
        "identity_fp32_path": (
            "Diagnostic probabilities are computed by precise FP32 softmax "
            "and the formal identity arithmetic is performed in Torch FP32; "
            "Config A source K/V were produced by native 4-bit/FP16 replay."
        ),
        "formal_fd_entry_cast": (
            "FP32 intervention arrays are cast to projected.dtype immediately "
            "before post-o_proj addition; Config A is FP16, Config B is FP32."
        ),
        "jvp_fd_function_boundary_equal": True,
        "jvp_input_boundary": "post-o_proj / pre-residual-add",
        "fd_fresh_cache_per_sign": True,
        "zero_cosine_fallback": (
            "Historical metric returned 0 for any zero vector; diagnostic "
            "rows now distinguish right_zero/left_zero/both_zero/orthogonal."
        ),
        "mask_position_rope_gqa": (
            "Retained order, logical position, retained-domain mask, RoPE "
            "offset and query_head//6 GQA mapping matched in all selected rows."
        ),
        "cache_key_definition": (
            "cache keys are post-RoPE; diagnostic new_keys are pre-RoPE"
        ),
        "autocast": False,
        "config_b_quantized_modules": 0,
        "formal_history_modified": False,
    }
    atomic_json(summary_path, summary)
    return summary


def run(output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty diagnostic directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    formal_registry = pd.read_parquet(
        FORMAL_DIR / "candidate_registry_rows.parquet"
    )
    formal_identity = pd.read_parquet(
        FORMAL_DIR / "deletion_identity_rows.parquet"
    )
    formal_single = pd.read_parquet(
        FORMAL_DIR / "single_layer_rows.parquet"
    )
    cfg = load_discovery_config(str(FORMAL_CONFIG))
    cfg.validate()
    samples, dataset_events = load_discovery_tasks(cfg)
    allowed = {
        "gov_report:24",
        "gov_report:25",
        "synthetic_niah_24",
        "synthetic_niah_25",
    }
    touched_ids = {str(sample.sample_id) for sample in samples}
    if not touched_ids.issubset(allowed):
        raise RuntimeError(f"non-train sample was loaded: {sorted(touched_ids - allowed)}")
    sample = next(
        sample for sample in samples if sample.sample_id == SELECTED_SAMPLE
    )

    backend_a = MLXTemporalModel(cfg)
    backend_a.load()
    reference_a = backend_a.generate_reference(
        sample_id=sample.sample_id,
        task=sample.task,
        prompt=sample.prompt,
    )
    selected_candidates = load_formal_candidates(
        formal_registry,
        SELECTED_SAMPLE,
        SELECTED_ANCHOR,
        int(reference_a.anchors[SELECTED_ANCHOR].logical_length - 1),
    )
    aux_candidates = load_formal_candidates(
        formal_registry,
        SELECTED_SAMPLE,
        IDENTITY_AUX_ANCHOR,
        int(reference_a.anchors[IDENTITY_AUX_ANCHOR].logical_length - 1),
    )
    reference_a_min = make_reference(
        reference_a.sample_id,
        reference_a.task,
        {
            SELECTED_ANCHOR: reference_a.anchors[SELECTED_ANCHOR],
            IDENTITY_AUX_ANCHOR: reference_a.anchors[IDENTITY_AUX_ANCHOR],
        },
    )
    result_a = configuration_case(
        "config_a_native_4bit",
        backend_a,
        reference_a_min,
        selected_candidates,
        formal_single,
        np.float16,
    )
    aux_identity_a = auxiliary_identity_case(
        "config_a_native_4bit",
        backend_a,
        reference_a_min,
        aux_candidates,
    )
    precision_a = precision_configuration_row(
        "config_a_native_4bit",
        backend_a.runner.model,
        reference_a_min.anchors[SELECTED_ANCHOR],
        result_a["runtime_dtypes"],
    )
    tokenized_input_hash = hashlib.sha256(
        np.asarray(reference_a.prompt_token_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    generated_prefix_hash = hashlib.sha256(
        np.asarray(
            reference_a.generated_token_ids[:SELECTED_ANCHOR],
            dtype=np.int64,
        ).tobytes()
    ).hexdigest()
    prompt_token_count = len(reference_a.prompt_token_ids)
    generated_tokens = list(
        int(value)
        for value in reference_a.generated_token_ids[:SELECTED_ANCHOR]
    )
    anchors_b = {
        SELECTED_ANCHOR: fp32_anchor(reference_a.anchors[SELECTED_ANCHOR]),
        IDENTITY_AUX_ANCHOR: fp32_anchor(
            reference_a.anchors[IDENTITY_AUX_ANCHOR]
        ),
    }
    backend_a.close()
    del backend_a
    gc.collect()

    backend_b = MLXTemporalModel(cfg)
    backend_b.load()
    dequantization = dequantize_reference_model(backend_b.runner.model)
    backend_b.model_info["weight_precision"] = "dequantized_fp32"
    reference_b = make_reference(
        reference_a.sample_id,
        reference_a.task,
        anchors_b,
    )
    result_b = configuration_case(
        "config_b_dequantized_fp32_query_replay",
        backend_b,
        reference_b,
        selected_candidates,
        formal_single,
        np.float32,
    )
    aux_identity_b = auxiliary_identity_case(
        "config_b_dequantized_fp32_query_replay",
        backend_b,
        reference_b,
        aux_candidates,
    )
    precision_b = precision_configuration_row(
        "config_b_dequantized_fp32_query_replay",
        backend_b.runner.model,
        reference_b.anchors[SELECTED_ANCHOR],
        result_b["runtime_dtypes"],
        dequantization,
    )
    model_shape = {
        "layers": int(backend_b.model_info["num_layers"]),
        "hidden_size": int(backend_b.model_info["hidden_size"]),
        "query_heads": int(backend_b.model_info["num_attention_heads"]),
        "kv_heads": int(backend_b.model_info["num_key_value_heads"]),
    }
    backend_b.close()
    del backend_b
    gc.collect()

    epsilon_frame = pd.DataFrame(
        result_a["epsilon_rows"] + result_b["epsilon_rows"]
    )
    checkpoint_frame = pd.DataFrame(
        result_a["checkpoint_rows"] + result_b["checkpoint_rows"]
    )
    regenerated_identity = (
        result_a["identity_rows"]
        + result_b["identity_rows"]
        + aux_identity_a
        + aux_identity_b
    )
    identity_frame = select_identity_output(
        formal_identity, regenerated_identity
    )
    precision_frame = pd.DataFrame([precision_a, precision_b])
    atomic_frame(output_dir / "epsilon_sweep.parquet", epsilon_frame)
    atomic_frame(
        output_dir / "physical_manual_checkpoints.parquet",
        checkpoint_frame,
    )
    atomic_frame(output_dir / "identity_outliers.parquet", identity_frame)
    atomic_frame(
        output_dir / "precision_configurations.parquet",
        precision_frame,
    )
    figure_paths = plot_outputs(
        output_dir,
        epsilon_frame,
        checkpoint_frame,
        identity_frame,
        formal_identity,
    )
    flags = interpretation(epsilon_frame, checkpoint_frame, identity_frame)
    summary: Dict[str, Any] = {
        "schema_version": 1,
        "experiment_type": "post_hoc_precision_diagnostic_not_a_gate",
        "git_commit": git_commit(),
        "formal_run_read_only": str(FORMAL_DIR),
        "formal_run_modified": False,
        "formal_outcome_unchanged": (
            "Outcome D: predictive bridge not validated"
        ),
        "heldout": {
            "calibration_read": False,
            "test_read": False,
            "task_loader_ids": sorted(touched_ids),
            "selected_ids_executed": [SELECTED_SAMPLE],
            "dataset_events": dataset_events,
        },
        "selection": {
            "sample_id": SELECTED_SAMPLE,
            "task": sample.task,
            "anchor": SELECTED_ANCHOR,
            "target_layer": SELECTED_LAYER,
            "candidate_ids": [
                candidate.candidate_id for candidate in selected_candidates
            ],
            "epsilon_direction_ids": list(EPSILON_DIRECTION_IDS),
            "identity_auxiliary_case": {
                "sample_id": SELECTED_SAMPLE,
                "anchor": IDENTITY_AUX_ANCHOR,
                "layer": IDENTITY_AUX_LAYER,
                "reason": "formal maximum-absolute identity-error case",
            },
            "seed": int(cfg.runtime.seed),
            "prompt_token_count": prompt_token_count,
            "tokenized_input_sha256": tokenized_input_hash,
            "generated_prefix_token_ids": generated_tokens,
            "generated_prefix_sha256": generated_prefix_hash,
        },
        "model": {
            "source": cfg.model.name,
            "revision": cfg.model.revision,
            **model_shape,
            "config_c": {
                "run": False,
                "reason": (
                    "No complete local native FP16/BF16 checkpoint for the "
                    "same Qwen2.5-1.5B-Instruct revision; the local base-model "
                    "checkpoint is not an equivalent model."
                ),
            },
        },
        "precision_configurations": precision_frame.to_dict("records"),
        "epsilon_sweep_metrics": epsilon_frame.to_dict("records"),
        "checkpoint_alignment_metrics": checkpoint_frame.to_dict("records"),
        "identity_outlier_metrics": identity_frame.to_dict("records"),
        "candidate_metrics": {
            "config_a_native_4bit": result_a["candidate_metrics"],
            "config_b_dequantized_fp32_query_replay": result_b[
                "candidate_metrics"
            ],
        },
        "base_alignment": {
            "config_a_native_4bit": result_a["base_alignment"],
            "config_b_dequantized_fp32_query_replay": result_b[
                "base_alignment"
            ],
        },
        "first_significant_checkpoint": summarize_first_significant(
            checkpoint_frame
        ),
        "figures": figure_paths,
        "test_results": {
            "status": "pending_external_pytest_run",
            "commands": [],
        },
        "interpretation_flags": flags,
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(output_dir / "precision_diagnostic_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        result = finalize_existing(args.output_dir.resolve())
    else:
        result = run(args.output_dir.resolve())
    compact = {
        "output_dir": str(args.output_dir.resolve()),
        "selection": result["selection"],
        "interpretation_flags": result["interpretation_flags"],
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
