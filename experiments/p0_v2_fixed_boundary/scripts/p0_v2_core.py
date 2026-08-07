"""Core primitives for the preregistered fixed-boundary P0-v2 experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
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
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
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


def stable_softmax(logits: Any) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = value - float(np.max(value))
    exponential = np.exp(shifted)
    return exponential / float(np.sum(exponential))


def exact_kl(full_logits: Any, changed_logits: Any) -> float:
    z0 = np.asarray(full_logits, dtype=np.float64).reshape(-1)
    z1 = np.asarray(changed_logits, dtype=np.float64).reshape(-1)
    p = stable_softmax(z0)
    log_p = z0 - (
        float(np.max(z0))
        + float(np.log(np.exp(z0 - float(np.max(z0))).sum()))
    )
    log_q = z1 - (
        float(np.max(z1))
        + float(np.log(np.exp(z1 - float(np.max(z1))).sum()))
    )
    # Roundoff can produce a tiny negative value although KL is non-negative.
    return max(0.0, float(np.dot(p, log_p - log_q)))


def fisher_variance(probability: Any, direction: Any) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    value = np.asarray(direction, dtype=np.float64).reshape(-1)
    centered = value - float(np.dot(p, value))
    return max(0.0, float(np.dot(p, centered * centered)))


def vector_metrics(
    predicted: Any,
    truth: Any,
    norm_floor: float = 1.0e-12,
    low_norm_threshold: float = 1.0e-8,
) -> Dict[str, Any]:
    left = np.asarray(predicted, dtype=np.float64).reshape(-1)
    right = np.asarray(truth, dtype=np.float64).reshape(-1)
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    difference = left - right
    error_norm = float(np.linalg.norm(difference))
    denominator = max(left_norm * right_norm, float(norm_floor) ** 2)
    cosine = (
        float(np.dot(left, right) / denominator)
        if finite and left_norm > norm_floor and right_norm > norm_floor
        else 0.0
    )
    return {
        "cosine": cosine,
        "relative_l2": error_norm / max(right_norm, float(norm_floor)),
        "symmetric_norm_ratio": (
            2.0 * min(left_norm, right_norm)
            / max(left_norm + right_norm, float(norm_floor))
        ),
        "maximum_absolute_error": (
            float(np.max(np.abs(difference), initial=0.0))
            if finite
            else float("nan")
        ),
        "predicted_norm": left_norm,
        "truth_norm": right_norm,
        "error_norm": error_norm,
        "finite": finite,
        "low_norm": bool(
            min(left_norm, right_norm) < float(low_norm_threshold)
        ),
    }


def prefixed_metrics(prefix: str, predicted: Any, truth: Any, **kwargs: Any) -> Dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in vector_metrics(predicted, truth, **kwargs).items()
    }


def _jvp_via_autodiff(
    function: Any,
    primal: Any,
    tangent: Any,
) -> Tuple[np.ndarray, np.ndarray, str]:
    import mlx.core as mx

    try:
        output, derivative = mx.jvp(function, [primal], [tangent])
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(derivative, (list, tuple)):
            derivative = derivative[0]
        method = "mx.jvp"
    except ValueError as error:
        if "Not implemented" not in str(error):
            raise
        output = function(primal)
        zero_cotangent = mx.zeros_like(output)

        def transpose_map(cotangent: Any) -> Any:
            _output, gradients = mx.vjp(
                function, [primal], [cotangent]
            )
            return gradients

        _transpose_output, outer = mx.vjp(
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


class P0V2FP32TemporalModel:
    """Factory wrapper returning an MLXTemporalModel with FP32 anchor storage."""

    @staticmethod
    def create(cfg: Any) -> Any:
        from kvbench.temporal.backend import AnchorState
        from kvbench.temporal.backend_mlx import MLXTemporalModel

        class _FP32Model(MLXTemporalModel):
            def _anchor_state(
                self,
                cache: List[Any],
                position_maps: Dict[int, torch.Tensor],
                logical_length: int,
                anchor_step: int,
                query_token_id: int,
            ) -> AnchorState:
                keys = []
                values = []
                for layer_cache in cache:
                    offset = int(layer_cache.offset)
                    keys.append(
                        self._torch(
                            layer_cache.keys[:, :, :offset, :],
                            torch.float32,
                        )
                    )
                    values.append(
                        self._torch(
                            layer_cache.values[:, :, :offset, :],
                            torch.float32,
                        )
                    )
                return AnchorState(
                    anchor_step=int(anchor_step),
                    logical_length=int(logical_length),
                    query_token_id=int(query_token_id),
                    keys=keys,
                    values=values,
                    position_maps={
                        int(layer): value.clone()
                        for layer, value in position_maps.items()
                    },
                    attention=self._attention_signals(),
                    query_head_observation=self._query_head_observation(),
                )

        return _FP32Model(cfg)


class AdjacentBoundaryMap:
    """Pure FP32 map from a post-O pulse to the next-layer input state."""

    def __init__(self, backend: Any, layer: int, base_record: Any):
        import mlx.core as mx

        self.layer = int(layer)
        self.block = backend.runner.model.model.layers[self.layer]
        self.base_post_attention = mx.array(
            base_record.post_attention_residuals[self.layer]
            .detach()
            .float()
            .numpy()
        )
        self.hidden_size = int(self.base_post_attention.shape[-1])
        mx.eval(self.base_post_attention)

    def __call__(self, pulse: Any) -> Any:
        hidden = self.base_post_attention + pulse.astype(
            self.base_post_attention.dtype
        )
        return hidden + self.block.mlp(
            self.block.post_attention_layernorm(hidden)
        )

    def zero(self) -> Any:
        import mlx.core as mx

        return mx.zeros((self.hidden_size,), dtype=mx.float32)

    def evaluate(self, pulse: Any) -> np.ndarray:
        import mlx.core as mx

        value = self(mx.array(np.asarray(pulse, dtype=np.float32)))
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    def baseline(self) -> np.ndarray:
        return self.evaluate(np.zeros(self.hidden_size, dtype=np.float32))

    def jvp(self, pulse: Any) -> Tuple[np.ndarray, np.ndarray, str]:
        import mlx.core as mx

        return _jvp_via_autodiff(
            self,
            self.zero(),
            mx.array(np.asarray(pulse, dtype=np.float32).reshape(-1)),
        )

    def vjp(self, cotangent: Any) -> np.ndarray:
        import mlx.core as mx

        zero = self.zero()
        vector = mx.array(np.asarray(cotangent, dtype=np.float32).reshape(-1))
        _output, gradients = mx.vjp(self, [zero], [vector])
        gradient = gradients[0]
        mx.eval(gradient)
        return np.asarray(gradient).astype(np.float64)


class FixedBoundaryReadoutMap:
    """Pure FP32 map from one fixed residual boundary to same-step logits."""

    def __init__(
        self,
        backend: Any,
        anchor: Any,
        base_record: Any,
        boundary_layer: int,
    ):
        import mlx.core as mx

        self.backend = backend
        self.model = backend.runner.model
        self.boundary_layer = int(boundary_layer)
        self.layers = list(
            self.model.model.layers[self.boundary_layer :]
        )
        self.layer_indices = list(
            range(
                self.boundary_layer,
                int(backend.model_info["num_layers"]),
            )
        )
        self.hidden_size = int(backend.model_info["hidden_size"])
        self.logical_position = int(anchor.logical_length - 1)
        self.base_input = mx.array(
            base_record.residual_inputs[self.boundary_layer]
            .detach()
            .float()
            .numpy()
        )
        self.keys: List[Any] = []
        self.values: List[Any] = []
        for layer in self.layer_indices:
            positions = [
                int(item)
                for item in anchor.position_maps[layer].tolist()
            ]
            current_row = positions.index(self.logical_position)
            prefix_rows = [
                row for row in range(len(positions))
                if row != current_row
            ]
            self.keys.append(
                mx.array(
                    anchor.keys[layer][
                        :, :, prefix_rows, :
                    ].detach().float().numpy()
                )
            )
            self.values.append(
                mx.array(
                    anchor.values[layer][
                        :, :, prefix_rows, :
                    ].detach().float().numpy()
                )
            )
        mx.eval(self.base_input, *self.keys, *self.values)

    def _fresh_caches(self) -> List[Any]:
        from mlx_lm.models.cache import KVCache

        caches = []
        for key, value in zip(self.keys, self.values):
            cache = KVCache()
            cache.state = (key, value)
            cache.logical_offset = int(self.logical_position)
            caches.append(cache)
        return caches

    def cache_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for key, value in zip(self.keys, self.values):
            digest.update(np.asarray(key).tobytes())
            digest.update(np.asarray(value).tobytes())
        digest.update(str(self.logical_position).encode("utf-8"))
        return digest.hexdigest()

    def __call__(self, delta: Any) -> Any:
        hidden = (
            self.base_input + delta.astype(self.base_input.dtype)
        ).reshape(1, 1, self.hidden_size)
        caches = self._fresh_caches()
        for layer, cache in zip(self.layers, caches):
            projected = layer.self_attn(
                layer.input_layernorm(hidden), None, cache
            )
            post_attention = hidden + projected
            hidden = post_attention + layer.mlp(
                layer.post_attention_layernorm(post_attention)
            )
        hidden = self.model.model.norm(hidden)
        if self.model.args.tie_word_embeddings:
            logits = self.model.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits.reshape(-1)

    def zero(self) -> Any:
        import mlx.core as mx

        return mx.zeros((self.hidden_size,), dtype=mx.float32)

    def evaluate(self, delta: Any) -> np.ndarray:
        import mlx.core as mx

        value = self(mx.array(np.asarray(delta, dtype=np.float32).reshape(-1)))
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    def baseline(self) -> np.ndarray:
        return self.evaluate(np.zeros(self.hidden_size, dtype=np.float32))

    def jvp(self, delta: Any) -> Tuple[np.ndarray, np.ndarray, str]:
        import mlx.core as mx

        fingerprint_before = self.cache_fingerprint()
        result = _jvp_via_autodiff(
            self,
            self.zero(),
            mx.array(np.asarray(delta, dtype=np.float32).reshape(-1)),
        )
        if fingerprint_before != self.cache_fingerprint():
            raise RuntimeError("downstream JVP mutated the frozen prefix cache")
        return result

    def vjp(self, cotangent: Any) -> np.ndarray:
        import mlx.core as mx

        zero = self.zero()
        vector = mx.array(np.asarray(cotangent, dtype=np.float32).reshape(-1))
        _output, gradients = mx.vjp(self, [zero], [vector])
        gradient = gradients[0]
        mx.eval(gradient)
        return np.asarray(gradient).astype(np.float64)

    def symmetric_fd(
        self,
        direction: Any,
        epsilon_relative: float,
        norm_floor: float = 1.0e-12,
    ) -> Dict[str, Any]:
        direction_array = np.asarray(
            direction, dtype=np.float64
        ).reshape(-1)
        base_norm = float(
            np.linalg.norm(np.asarray(self.base_input, dtype=np.float64))
        )
        direction_norm = float(np.linalg.norm(direction_array))
        epsilon_absolute = (
            float(epsilon_relative)
            * base_norm
            / max(direction_norm, float(norm_floor))
        )
        plus = self.evaluate(epsilon_absolute * direction_array)
        minus = self.evaluate(-epsilon_absolute * direction_array)
        derivative = (plus - minus) / (2.0 * epsilon_absolute)
        return {
            "epsilon_relative": float(epsilon_relative),
            "epsilon_absolute": float(epsilon_absolute),
            "base_norm": base_norm,
            "direction_norm": direction_norm,
            "plus": plus,
            "minus": minus,
            "derivative": derivative,
            "fd_norm": float(np.linalg.norm(derivative)),
        }


def full_replay(backend: Any, reference: Any, anchor_step: int) -> Tuple[Any, ...]:
    from mlx_predictive_core import full_selection
    from precision_diagnostic import replay_with_record

    selection, cache_cfg = full_selection(reference, anchor_step)
    return replay_with_record(
        backend,
        reference,
        anchor_step,
        selection,
        cache_cfg,
    )


def physical_single_layer_replay(
    backend: Any,
    reference: Any,
    anchor_step: int,
    candidate: Any,
    layer: int,
) -> Tuple[Any, ...]:
    from mlx_predictive_core import single_layer_selection
    from precision_diagnostic import replay_with_record

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


def manual_boundary_replay(
    backend: Any,
    reference: Any,
    anchor_step: int,
    boundary_layer: int,
    absolute_boundary_input: Any,
) -> Tuple[Any, ...]:
    from mlx_predictive_core import full_selection
    from precision_diagnostic import replay_with_record

    selection, cache_cfg = full_selection(reference, anchor_step)
    state, _fixed = backend.state_from_anchor(
        reference.anchors[int(anchor_step)],
        selection,
        cache_config=cache_cfg,
    )
    backend.runner.attention_state[
        "temporal_layer_input_overrides"
    ] = {
        int(boundary_layer): np.asarray(
            absolute_boundary_input, dtype=np.float32
        )
    }
    try:
        logits, record, _elapsed = backend.forward_one(
            state,
            int(reference.anchors[int(anchor_step)].query_token_id),
            capture_attention=True,
        )
        positions = {
            int(layer): value.detach().clone()
            for layer, value in state.position_maps.items()
        }
        runtime = backend.runner.attention_state
        runtime_dtypes = {
            "boundary_input": str(
                runtime["temporal_residual_inputs"][
                    int(boundary_layer)
                ].dtype
            ),
            "boundary_layer_output": str(
                runtime["temporal_layer_outputs"][
                    int(boundary_layer)
                ].dtype
            ),
        }
        return (
            logits.double().numpy(),
            record,
            positions,
            runtime_dtypes,
        )
    finally:
        backend.release(state)


def add_identity_conditioning(
    rows: Iterable[Mapping[str, Any]],
    identity_norm_floors: Sequence[float],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        retained_mass = float(row["denominator"])
        absolute = float(row.get("absolute_l2_error", 0.0))
        lhs = float(row.get("lhs_norm", row.get("direct_norm", 0.0)))
        rhs = float(row.get("rhs_norm", lhs))
        row["retained_mass"] = retained_mass
        row["kappa_mass"] = (
            1.0 / retained_mass if retained_mass > 0.0 else float("inf")
        )
        row["cancellation_sensitive"] = bool(
            lhs < 1.0e-8 or retained_mass < 1.0e-4
        )
        for floor in identity_norm_floors:
            label = f"stable_relative_error_floor_{floor:.0e}".replace(
                "-", "m"
            )
            row[label] = absolute / max(lhs, rhs, float(floor))
        output.append(row)
    return output


def spearman(predicted: Sequence[float], truth: Sequence[float]) -> float:
    from scipy.stats import rankdata

    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    if (
        len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.all(left == left[0])
        or np.all(right == right[0])
    ):
        return float("nan")
    return float(
        np.corrcoef(
            rankdata(left, method="average"),
            rankdata(right, method="average"),
        )[0, 1]
    )


def pairwise_sign_accuracy(
    predicted: Sequence[float], truth: Sequence[float]
) -> float:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(truth, dtype=np.float64)
    correct = 0
    total = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            correct += int(
                np.sign(left[first] - left[second])
                == np.sign(right[first] - right[second])
            )
            total += 1
    return float(correct / total) if total else float("nan")


def ranking_metrics(
    score: Sequence[float],
    truth: Sequence[float],
    top_k: int,
) -> Dict[str, Any]:
    predicted = np.asarray(score, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    predicted_order = np.argsort(predicted, kind="stable")
    truth_order = np.argsort(target, kind="stable")
    selected = int(predicted_order[0])
    score_norm = float(np.linalg.norm(predicted))
    truth_norm = float(np.linalg.norm(target))
    return {
        "spearman": spearman(predicted, target),
        "pairwise_sign_accuracy": pairwise_sign_accuracy(
            predicted, target
        ),
        "top1_accuracy": float(predicted_order[0] == truth_order[0]),
        "topk_overlap": float(
            len(
                set(predicted_order[: int(top_k)])
                & set(truth_order[: int(top_k)])
            )
            / float(top_k)
        ),
        "normalized_regret": float(
            (target[selected] - float(np.min(target)))
            / max(
                float(np.max(target) - np.min(target)),
                1.0e-30,
            )
        ),
        "symmetric_scale_ratio": (
            2.0
            * min(score_norm, truth_norm)
            / max(score_norm + truth_norm, 1.0e-30)
        ),
        "finite": bool(
            np.isfinite(predicted).all()
            and np.isfinite(target).all()
        ),
    }

