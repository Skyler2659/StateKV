"""Shared native-boundary and FP32 local-map primitives."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
PREDICTIVE_SCRIPTS = (
    ROOT / "experiments/predictive_closure/scripts"
)


@dataclass(frozen=True)
class LocalCandidate:
    candidate_id: str
    source: str
    core_positions: Tuple[int, ...]
    retained_positions: Tuple[int, ...]
    keep_prefix_positions: Tuple[int, ...]
    seed: int
    mask_hash: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


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


def cosine(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30)
    return float(np.dot(a, b) / denominator)


def relative_l2(predicted: Any, truth: Any) -> float:
    a = np.asarray(predicted, dtype=np.float64).reshape(-1)
    b = np.asarray(truth, dtype=np.float64).reshape(-1)
    return float(
        np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-30)
    )


def symmetric_norm_ratio(left: Any, right: Any) -> float:
    a = float(np.linalg.norm(np.asarray(left, dtype=np.float64)))
    b = float(np.linalg.norm(np.asarray(right, dtype=np.float64)))
    return float(2.0 * min(a, b) / max(a + b, 1e-30))


def load_candidates(
    registry_path: Path,
    sample_id: str,
    anchor: int,
    current_position: int,
    expected_sha256: str,
) -> List[LocalCandidate]:
    if sha256_file(registry_path) != expected_sha256:
        raise RuntimeError("candidate registry file checksum mismatch")
    frame = pd.read_parquet(registry_path)
    group = frame[
        frame["sample_id"].eq(sample_id)
        & frame["anchor"].eq(int(anchor))
    ].copy()
    source_order = {
        value: index
        for index, value in enumerate(
            (
                "attention_only",
                "aov",
                "aor",
                "v_ridge",
                "snapkv",
                "old_stale_core",
                "fresh_core",
                "random_reference",
            )
        )
    }
    group["_order"] = group["candidate_source"].map(source_order)
    group = group.sort_values("_order")
    if len(group) != 8 or group["mask_hash"].nunique() != 8:
        raise RuntimeError("candidate group is not eight-distinct")
    candidates = []
    for row in group.to_dict("records"):
        retained = tuple(
            int(value)
            for value in json.loads(row["retained_positions_json"])
        )
        core = tuple(
            int(value)
            for value in json.loads(row["core_positions_json"])
        )
        payload = ",".join(str(value) for value in retained)
        mask_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if mask_hash != row["mask_hash"]:
            raise RuntimeError("candidate mask checksum mismatch")
        if len(retained) != 128 or len(core) != 92:
            raise RuntimeError("candidate active/core budget mismatch")
        candidates.append(
            LocalCandidate(
                candidate_id=str(row["candidate_id"]),
                source=str(row["candidate_source"]),
                core_positions=core,
                retained_positions=retained,
                keep_prefix_positions=tuple(
                    value for value in retained
                    if value != int(current_position)
                ),
                seed=int(row["candidate_seed"]),
                mask_hash=mask_hash,
            )
        )
    return candidates


def to_physical_candidate(candidate: LocalCandidate) -> Any:
    import sys

    if str(PREDICTIVE_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PREDICTIVE_SCRIPTS))
    from mlx_predictive_core import PhysicalCandidate

    return PhysicalCandidate(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        core_positions=candidate.core_positions,
        keep_prefix_positions=candidate.keep_prefix_positions,
        retained_positions=candidate.retained_positions,
        seed=candidate.seed,
    )


def replay_record(
    backend: Any,
    reference: Any,
    anchor_step: int,
    selection: Any,
    cache_cfg: Any,
    injection_layer: Optional[int] = None,
    injection: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any, float]:
    state, _fixed = backend.state_from_anchor(
        reference.anchors[int(anchor_step)],
        selection,
        cache_config=cache_cfg,
    )
    try:
        if injection_layer is not None:
            if injection is None:
                raise ValueError("manual injection value is missing")
            backend.runner.attention_state[
                "temporal_projected_injections"
            ] = {
                int(injection_layer): np.asarray(
                    injection, dtype=np.float32
                )
            }
        logits, record, elapsed = backend.forward_one(
            state,
            int(reference.anchors[int(anchor_step)].query_token_id),
            capture_attention=True,
        )
        return logits.double().numpy(), record, float(elapsed)
    finally:
        backend.release(state)


def theoretical_injection(
    backend: Any,
    reference: Any,
    anchor_step: int,
    retained_positions: Sequence[int],
    layer: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    anchor = reference.anchors[int(anchor_step)]
    record = reference.query_records[int(anchor_step)]
    positions = [
        int(value) for value in anchor.position_maps[int(layer)].tolist()
    ]
    row_by_position = {
        position: row for row, position in enumerate(positions)
    }
    rows = torch.tensor(
        [row_by_position[int(position)] for position in retained_positions],
        dtype=torch.long,
    )
    attention = (
        record.all_head_attention_distributions[int(layer)].float()
    )
    attention = attention / attention.sum(dim=1, keepdim=True)
    values = anchor.values[int(layer)][0].float()
    query_heads = int(backend.model_info["num_attention_heads"])
    kv_heads = int(backend.model_info["num_key_value_heads"])
    repeated_values = values.repeat_interleave(
        query_heads // kv_heads, dim=0
    )
    kept_attention = attention.index_select(1, rows)
    retained_mass = kept_attention.sum(dim=1)
    if bool((retained_mass <= 0).any()):
        raise FloatingPointError("non-positive retained attention mass")
    masked_heads = (
        kept_attention[:, :, None]
        * repeated_values.index_select(1, rows)
    ).sum(dim=1) / retained_mass[:, None]
    full_heads = record.all_head_attention_outputs[int(layer)].float()
    projected = backend.project_features(
        int(layer), (masked_heads - full_heads).reshape(1, -1)
    )[0]
    return (
        projected.numpy().astype(np.float32),
        {
            "retained_mass_mean": float(retained_mass.mean()),
            "retained_mass_min": float(retained_mass.min()),
            "deleted_mass_mean": float((1.0 - retained_mass).mean()),
            "deleted_mass_max": float((1.0 - retained_mass).max()),
        },
    )


def _dequantize_linear(linear: Any) -> Tuple[Any, Dict[str, Any]]:
    import mlx.core as mx

    packed = linear["weight"]
    scales = linear["scales"]
    biases = linear.get("biases")
    weight = mx.dequantize(
        packed,
        scales=scales,
        biases=biases,
        group_size=int(linear.group_size),
        bits=int(linear.bits),
        mode=str(linear.mode),
    ).astype(mx.float32)
    mx.eval(packed, scales, weight)
    metadata = {
        "group_size": int(linear.group_size),
        "bits": int(linear.bits),
        "mode": str(linear.mode),
        "packed_weight_sha256": array_sha256(packed),
        "scales_sha256": array_sha256(scales),
        "biases_sha256": (
            array_sha256(biases) if biases is not None else None
        ),
        "dequantized_fp32_weight_sha256": array_sha256(weight),
        "shape": [int(value) for value in weight.shape],
    }
    return weight, metadata


class FP32LocalBlock:
    """Pure same-checkpoint FP32 map delta -> g_l(b + delta)."""

    def __init__(self, backend: Any, layer: int, native_b: Any):
        import mlx.core as mx

        self.layer = int(layer)
        block = backend.runner.model.model.layers[self.layer]
        self.base_b = mx.array(
            np.asarray(native_b, dtype=np.float32).reshape(-1)
        )
        self.norm_weight = block.post_attention_layernorm[
            "weight"
        ].astype(mx.float32)
        self.norm_eps = float(block.post_attention_layernorm.eps)
        self.gate_weight, gate_meta = _dequantize_linear(
            block.mlp.gate_proj
        )
        self.up_weight, up_meta = _dequantize_linear(
            block.mlp.up_proj
        )
        self.down_weight, down_meta = _dequantize_linear(
            block.mlp.down_proj
        )
        mx.eval(
            self.base_b,
            self.norm_weight,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
        )
        self.metadata = {
            "layer": self.layer,
            "path": "same_checkpoint_mlx_fp32_dequantized_local_block",
            "norm_weight_sha256": array_sha256(self.norm_weight),
            "norm_eps": self.norm_eps,
            "gate_proj": gate_meta,
            "up_proj": up_meta,
            "down_proj": down_meta,
        }

    def with_base(self, native_b: Any) -> "FP32LocalBlock":
        """Reuse immutable dequantized weights at a new operating point."""
        import mlx.core as mx

        value = object.__new__(FP32LocalBlock)
        value.layer = self.layer
        value.base_b = mx.array(
            np.asarray(native_b, dtype=np.float32).reshape(-1)
        )
        value.norm_weight = self.norm_weight
        value.norm_eps = self.norm_eps
        value.gate_weight = self.gate_weight
        value.up_weight = self.up_weight
        value.down_weight = self.down_weight
        value.metadata = self.metadata
        mx.eval(value.base_b)
        return value

    def __call__(self, delta: Any) -> Any:
        import mlx.core as mx

        x = self.base_b + delta.astype(mx.float32)
        mean_square = mx.mean(x * x, axis=-1, keepdims=True)
        normalized = (
            x
            * mx.rsqrt(mean_square + self.norm_eps)
            * self.norm_weight
        )
        gate = mx.matmul(normalized, self.gate_weight.T)
        up = mx.matmul(normalized, self.up_weight.T)
        activated = gate * mx.sigmoid(gate)
        down = mx.matmul(activated * up, self.down_weight.T)
        return x + down

    def zero(self) -> Any:
        import mlx.core as mx

        return mx.zeros_like(self.base_b).astype(mx.float32)

    def evaluate(self, delta: Any) -> np.ndarray:
        import mlx.core as mx

        value = self(mx.array(np.asarray(delta, dtype=np.float32)))
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    def baseline(self) -> np.ndarray:
        return self.evaluate(np.zeros(tuple(self.base_b.shape)))

    def jvp(self, direction: Any) -> Tuple[np.ndarray, np.ndarray, str]:
        import mlx.core as mx

        primal = self.zero()
        tangent = mx.array(
            np.asarray(direction, dtype=np.float32).reshape(-1)
        )
        try:
            output, derivative = mx.jvp(
                self, [primal], [tangent]
            )
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(derivative, (list, tuple)):
                derivative = derivative[0]
            method = "mx.jvp"
        except ValueError as error:
            message = str(error)
            if "Not implemented" not in message:
                raise
            output = self(primal)
            zero_cotangent = mx.zeros_like(output)

            def transpose_map(cotangent: Any) -> Any:
                _value, gradients = mx.vjp(
                    self, [primal], [cotangent]
                )
                return gradients

            _value, outer = mx.vjp(
                transpose_map, [zero_cotangent], [tangent]
            )
            derivative = outer[0]
            method = "mx.vjp_of_vjp"
        mx.eval(output, derivative)
        return (
            np.asarray(output).astype(np.float64),
            np.asarray(derivative).astype(np.float64),
            method,
        )

    def symmetric_fd(
        self, direction: Any, epsilon_relative: float
    ) -> Dict[str, Any]:
        direction_array = np.asarray(
            direction, dtype=np.float64
        ).reshape(-1)
        base_norm = float(
            np.linalg.norm(np.asarray(self.base_b).astype(np.float64))
        )
        direction_norm = float(np.linalg.norm(direction_array))
        epsilon_abs = (
            float(epsilon_relative)
            * base_norm
            / (direction_norm + 1.0e-12)
        )
        plus = self.evaluate(epsilon_abs * direction_array)
        minus = self.evaluate(-epsilon_abs * direction_array)
        derivative = (plus - minus) / (2.0 * epsilon_abs)
        return {
            "epsilon_relative": float(epsilon_relative),
            "epsilon_absolute": float(epsilon_abs),
            "plus": plus,
            "minus": minus,
            "derivative": derivative,
        }

    def nonlinear_delta(
        self, direction: Any, scale: float
    ) -> np.ndarray:
        direction_array = np.asarray(direction, dtype=np.float64)
        baseline = self.baseline()
        return self.evaluate(float(scale) * direction_array) - baseline


class FP32TransformerLayer:
    """Dequantized FP32 Qwen layer used inside a frozen-KV local stack."""

    def __init__(self, backend: Any, layer: int):
        import mlx.core as mx

        self.layer = int(layer)
        block = backend.runner.model.model.layers[self.layer]
        attention = block.self_attn
        self.input_norm_weight = block.input_layernorm[
            "weight"
        ].astype(mx.float32)
        self.input_norm_eps = float(block.input_layernorm.eps)
        self.post_norm_weight = block.post_attention_layernorm[
            "weight"
        ].astype(mx.float32)
        self.post_norm_eps = float(block.post_attention_layernorm.eps)
        self.q_weight, q_meta = _dequantize_linear(attention.q_proj)
        self.k_weight, k_meta = _dequantize_linear(attention.k_proj)
        self.v_weight, v_meta = _dequantize_linear(attention.v_proj)
        self.o_weight, o_meta = _dequantize_linear(attention.o_proj)
        self.q_bias = self._bias(attention.q_proj)
        self.k_bias = self._bias(attention.k_proj)
        self.v_bias = self._bias(attention.v_proj)
        self.o_bias = self._bias(attention.o_proj)
        self.gate_weight, gate_meta = _dequantize_linear(
            block.mlp.gate_proj
        )
        self.up_weight, up_meta = _dequantize_linear(
            block.mlp.up_proj
        )
        self.down_weight, down_meta = _dequantize_linear(
            block.mlp.down_proj
        )
        self.n_heads = int(attention.n_heads)
        self.n_kv_heads = int(attention.n_kv_heads)
        self.head_dim = int(self.q_weight.shape[0]) // self.n_heads
        self.scale = float(attention.scale)
        self.rope = attention.rope
        arrays = [
            self.input_norm_weight,
            self.post_norm_weight,
            self.q_weight,
            self.k_weight,
            self.v_weight,
            self.o_weight,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
        ]
        arrays.extend(
            value
            for value in (
                self.q_bias,
                self.k_bias,
                self.v_bias,
                self.o_bias,
            )
            if value is not None
        )
        mx.eval(*arrays)
        self.metadata = {
            "layer": self.layer,
            "path": (
                "same_checkpoint_dequantized_fp32_frozen_kv_full_layer"
            ),
            "input_norm_weight_sha256": array_sha256(
                self.input_norm_weight
            ),
            "input_norm_eps": self.input_norm_eps,
            "post_norm_weight_sha256": array_sha256(
                self.post_norm_weight
            ),
            "post_norm_eps": self.post_norm_eps,
            "q_proj": q_meta,
            "k_proj": k_meta,
            "v_proj": v_meta,
            "o_proj": o_meta,
            "gate_proj": gate_meta,
            "up_proj": up_meta,
            "down_proj": down_meta,
            "q_bias_sha256": (
                array_sha256(self.q_bias)
                if self.q_bias is not None
                else None
            ),
            "k_bias_sha256": (
                array_sha256(self.k_bias)
                if self.k_bias is not None
                else None
            ),
            "v_bias_sha256": (
                array_sha256(self.v_bias)
                if self.v_bias is not None
                else None
            ),
            "o_bias_sha256": (
                array_sha256(self.o_bias)
                if self.o_bias is not None
                else None
            ),
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attention_scale": self.scale,
        }

    @staticmethod
    def _bias(linear: Any) -> Optional[Any]:
        import mlx.core as mx

        value = linear.get("bias")
        return value.astype(mx.float32) if value is not None else None

    @staticmethod
    def _rms_norm(x: Any, weight: Any, epsilon: float) -> Any:
        import mlx.core as mx

        return (
            x
            * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + epsilon)
            * weight
        )

    @staticmethod
    def _linear(x: Any, weight: Any, bias: Optional[Any]) -> Any:
        import mlx.core as mx

        output = mx.matmul(x, weight.T)
        return output + bias if bias is not None else output

    def __call__(
        self,
        hidden: Any,
        fixed_keys: Any,
        fixed_values: Any,
        logical_position: int,
    ) -> Any:
        import mlx.core as mx

        normalized = self._rms_norm(
            hidden, self.input_norm_weight, self.input_norm_eps
        )
        query = self._linear(
            normalized, self.q_weight, self.q_bias
        ).reshape(self.n_heads, self.head_dim)
        key = self._linear(
            normalized, self.k_weight, self.k_bias
        ).reshape(self.n_kv_heads, self.head_dim)
        value = self._linear(
            normalized, self.v_weight, self.v_bias
        ).reshape(self.n_kv_heads, self.head_dim)
        query = self.rope(
            query[None, :, None, :], offset=int(logical_position)
        )
        key = self.rope(
            key[None, :, None, :], offset=int(logical_position)
        )
        keys = mx.concatenate([fixed_keys, key], axis=2)
        values = mx.concatenate(
            [fixed_values, value[None, :, None, :]], axis=2
        )
        repeats = self.n_heads // self.n_kv_heads
        repeated_keys = mx.repeat(keys, repeats, axis=1)
        repeated_values = mx.repeat(values, repeats, axis=1)
        logits = (
            mx.sum(
                query.astype(mx.float32)
                * repeated_keys.astype(mx.float32),
                axis=-1,
            )
            * self.scale
        )
        probabilities = mx.softmax(logits, axis=-1, precise=True)
        attended = mx.sum(
            probabilities[..., None] * repeated_values.astype(mx.float32),
            axis=2,
        ).reshape(-1)
        projected = self._linear(
            attended, self.o_weight, self.o_bias
        )
        post_attention = hidden + projected
        post_normalized = self._rms_norm(
            post_attention,
            self.post_norm_weight,
            self.post_norm_eps,
        )
        gate = mx.matmul(post_normalized, self.gate_weight.T)
        up = mx.matmul(post_normalized, self.up_weight.T)
        activated = gate * mx.sigmoid(gate)
        down = mx.matmul(activated * up, self.down_weight.T)
        return post_attention + down


class FP32TruncatedStack:
    """Depth-k derivative from post-attention b_l to h_(l+k)^in."""

    def __init__(
        self,
        backend: Any,
        start_layer: int,
        depth: int,
        native_b: Any,
        reference_record: Any,
        anchor: Any,
        local_templates: Dict[int, FP32LocalBlock],
        layer_templates: Dict[int, FP32TransformerLayer],
    ):
        import mlx.core as mx

        self.start_layer = int(start_layer)
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("truncated stack depth must be positive")
        layer_count = int(backend.model_info["num_layers"])
        if self.start_layer + self.depth > layer_count:
            raise ValueError("truncated stack extends beyond model depth")
        if self.start_layer not in local_templates:
            local_templates[self.start_layer] = FP32LocalBlock(
                backend, self.start_layer, native_b
            )
        self.local = local_templates[self.start_layer].with_base(native_b)
        self.logical_position = int(anchor.logical_length - 1)
        current_target = (
            reference_record.layer_outputs[self.start_layer]
            .float()
            .numpy()
        )
        current_raw = self.local.baseline()
        self.current_correction = mx.array(
            (current_target - current_raw).astype(np.float32)
        )
        self.downstream: List[
            Tuple[FP32TransformerLayer, Any, Any, Any]
        ] = []
        correction_metadata = [
            {
                "layer": self.start_layer,
                "correction_norm": float(
                    np.linalg.norm(current_target - current_raw)
                ),
            }
        ]
        for layer in range(
            self.start_layer + 1, self.start_layer + self.depth
        ):
            if layer not in layer_templates:
                layer_templates[layer] = FP32TransformerLayer(
                    backend, layer
                )
            weights = layer_templates[layer]
            positions = [
                int(value)
                for value in anchor.position_maps[layer].tolist()
            ]
            if self.logical_position not in positions:
                raise RuntimeError(
                    "current position missing from truncated-stack cache"
                )
            prefix_rows = [
                row
                for row, position in enumerate(positions)
                if position != self.logical_position
            ]
            fixed_keys = mx.array(
                anchor.keys[layer][:, :, prefix_rows, :].numpy()
            ).astype(mx.float32)
            fixed_values = mx.array(
                anchor.values[layer][:, :, prefix_rows, :].numpy()
            ).astype(mx.float32)
            native_input = (
                reference_record.residual_inputs[layer].float().numpy()
            )
            native_target = (
                reference_record.layer_outputs[layer].float().numpy()
            )
            raw = weights(
                mx.array(native_input).astype(mx.float32),
                fixed_keys,
                fixed_values,
                self.logical_position,
            )
            mx.eval(raw)
            raw_array = np.asarray(raw).astype(np.float64)
            correction_array = native_target - raw_array
            correction = mx.array(
                correction_array.astype(np.float32)
            )
            self.downstream.append(
                (weights, fixed_keys, fixed_values, correction)
            )
            correction_metadata.append(
                {
                    "layer": layer,
                    "correction_norm": float(
                        np.linalg.norm(correction_array)
                    ),
                }
            )
        mx.eval(
            self.current_correction,
            *[
                value
                for _weights, keys, values, correction in self.downstream
                for value in (keys, values, correction)
            ],
        )
        self.metadata = {
            "start_layer": self.start_layer,
            "depth": self.depth,
            "logical_position": self.logical_position,
            "baseline_anchor": (
                "constant_per_layer_native_reference_correction"
            ),
            "corrections": correction_metadata,
        }

    def __call__(self, delta: Any) -> Any:
        hidden = self.local(delta) + self.current_correction
        for weights, keys, values, correction in self.downstream:
            hidden = (
                weights(
                    hidden, keys, values, self.logical_position
                )
                + correction
            )
        return hidden

    def zero(self) -> Any:
        import mlx.core as mx

        return mx.zeros_like(self.local.base_b).astype(mx.float32)

    def evaluate(self, delta: Any) -> np.ndarray:
        import mlx.core as mx

        value = self(mx.array(np.asarray(delta, dtype=np.float32)))
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    def jvp(
        self, direction: Any
    ) -> Tuple[np.ndarray, np.ndarray, str, float, int]:
        import mlx.core as mx
        import time

        primal = self.zero()
        tangent = mx.array(
            np.asarray(direction, dtype=np.float32).reshape(-1)
        )
        mx.reset_peak_memory()
        started = time.perf_counter()
        try:
            output, derivative = mx.jvp(
                self, [primal], [tangent]
            )
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(derivative, (list, tuple)):
                derivative = derivative[0]
            method = "mx.jvp"
        except ValueError as error:
            if "Not implemented" not in str(error):
                raise
            output = self(primal)
            zero_cotangent = mx.zeros_like(output)

            def transpose_map(cotangent: Any) -> Any:
                _value, gradients = mx.vjp(
                    self, [primal], [cotangent]
                )
                return gradients

            _value, outer = mx.vjp(
                transpose_map, [zero_cotangent], [tangent]
            )
            derivative = outer[0]
            method = "mx.vjp_of_vjp"
        mx.eval(output, derivative)
        elapsed = time.perf_counter() - started
        peak_memory = int(mx.get_peak_memory())
        return (
            np.asarray(output).astype(np.float64),
            np.asarray(derivative).astype(np.float64),
            method,
            float(elapsed),
            peak_memory,
        )


def max_pairwise_noise(outputs: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for left in range(len(outputs)):
        for right in range(left + 1, len(outputs)):
            maximum = max(
                maximum,
                float(
                    np.linalg.norm(
                        np.asarray(outputs[left], dtype=np.float64)
                        - np.asarray(outputs[right], dtype=np.float64)
                    )
                ),
            )
    return maximum


def choose_radius(rows: pd.DataFrame) -> Dict[str, Any]:
    aggregates = []
    radii = sorted(float(value) for value in rows["radius"].unique())
    for radius in radii:
        group = rows[rows["radius"].eq(radius)]
        noise_threshold_pass = bool(
            (
                group["fd_norm"]
                >= 100.0 * np.maximum(group["noise_norm"], 1.0e-12)
            ).all()
        )
        finite = bool(group["finite"].all())
        median_cosine = float(group["jvp_fd_cosine"].median())
        median_relative = float(group["jvp_fd_relative_l2"].median())
        eligible = bool(
            finite
            and noise_threshold_pass
            and median_cosine >= 0.995
            and median_relative <= 0.05
        )
        aggregates.append(
            {
                "radius": radius,
                "row_count": int(len(group)),
                "finite": finite,
                "noise_threshold_pass": noise_threshold_pass,
                "median_cosine": median_cosine,
                "median_relative_l2": median_relative,
                "eligible": eligible,
            }
        )
    selected = None
    plateau = None
    for index in range(len(aggregates) - 1):
        left = aggregates[index]
        right = aggregates[index + 1]
        if not left["eligible"] or not right["eligible"]:
            continue
        if abs(left["median_cosine"] - right["median_cosine"]) > 0.002:
            continue
        if (
            abs(
                left["median_relative_l2"]
                - right["median_relative_l2"]
            )
            > 0.02
        ):
            continue
        next_row = (
            aggregates[index + 2]
            if index + 2 < len(aggregates)
            else right
        )
        if (
            next_row["median_cosine"]
            < right["median_cosine"] - 0.01
        ):
            continue
        if (
            next_row["median_relative_l2"]
            > right["median_relative_l2"] + 0.05
        ):
            continue
        selected = float(left["radius"])
        plateau = [float(left["radius"]), float(right["radius"])]
        break
    return {
        "aggregates": aggregates,
        "selected_radius": selected,
        "stable_plateau": plateau,
        "calibration_passed": selected is not None,
        "selection_rule": (
            "first_smaller_radius_in_frozen_stable_eligible_plateau"
        ),
    }
