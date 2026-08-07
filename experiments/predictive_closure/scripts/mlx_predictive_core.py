"""Native MLX 4-bit multi-boundary intervention and physical replay helpers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from kvbench.temporal.config import CacheDiscoveryConfig
from kvbench.temporal.selectors import (
    CoreSelection,
    LayerSelection,
    ridge_leverage,
)


@dataclass(frozen=True)
class PhysicalCandidate:
    candidate_id: str
    source: str
    core_positions: Tuple[int, ...]
    keep_prefix_positions: Tuple[int, ...]
    retained_positions: Tuple[int, ...]
    seed: int

    @property
    def mask_hash(self) -> str:
        payload = ",".join(str(value) for value in self.retained_positions)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PureMultiBoundaryMap:
    """Pure U=(u_0,...,u_L-1)->logits map with immutable full-prefix K/V."""

    def __init__(self, backend: Any, anchor: Any):
        import mlx.core as mx

        self.backend = backend
        self.output_model = backend.runner.model
        self.layers = list(self.output_model.model.layers)
        self.hidden_size = int(backend.model_info["hidden_size"])
        self.query_token_id = int(anchor.query_token_id)
        self.logical_position = int(anchor.logical_length - 1)
        self.keys = []
        self.values = []
        for layer, (key, value) in enumerate(zip(anchor.keys, anchor.values)):
            positions = [
                int(item) for item in anchor.position_maps[layer].tolist()
            ]
            if self.logical_position not in positions:
                raise RuntimeError("anchor current position is absent from cache")
            current_row = positions.index(self.logical_position)
            prefix_rows = [
                row for row in range(len(positions)) if row != current_row
            ]
            self.keys.append(
                mx.array(key[:, :, prefix_rows, :].numpy())
            )
            self.values.append(
                mx.array(value[:, :, prefix_rows, :].numpy())
            )
        mx.eval(*self.keys, *self.values)
        self._clear_runtime_controls()

    def _clear_runtime_controls(self) -> None:
        state = self.backend.runner.attention_state
        for key in (
            "temporal_projected_injections",
            "temporal_query_overrides",
            "temporal_new_key_overrides",
            "temporal_new_value_overrides",
            "temporal_attention_input_overrides",
            "temporal_layer_input_overrides",
        ):
            state[key] = {}
        state["temporal_record_diagnostics"] = False

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
        import mlx.core as mx

        mx.eval(*self.keys, *self.values)
        digest = hashlib.sha256()
        for key, value in zip(self.keys, self.values):
            digest.update(np.asarray(key).tobytes())
            digest.update(np.asarray(value).tobytes())
        digest.update(str(self.logical_position).encode("utf-8"))
        return digest.hexdigest()

    def __call__(self, *blocks: Any) -> Any:
        import mlx.core as mx

        if len(blocks) != len(self.layers):
            raise ValueError(
                f"expected {len(self.layers)} intervention blocks, got {len(blocks)}"
            )
        self._clear_runtime_controls()
        cache = self._fresh_caches()
        token = mx.array([[self.query_token_id]])
        hidden = self.output_model.model.embed_tokens(token)
        for layer_index, (layer, layer_cache) in enumerate(
            zip(self.layers, cache)
        ):
            projected = layer.self_attn(
                layer.input_layernorm(hidden), None, layer_cache
            )
            injection = blocks[layer_index].reshape(
                1, 1, self.hidden_size
            ).astype(projected.dtype)
            # The registered boundary is after W_O and before the residual
            # add.  Preserve that parenthesization explicitly: FP16 addition
            # is not associative, and `(hidden + projected) + injection`
            # defines a different intervention from the hooked model path.
            intervened_projected = projected + injection
            post_attention = hidden + intervened_projected
            hidden = post_attention + layer.mlp(
                layer.post_attention_layernorm(post_attention)
            )
        hidden = self.output_model.model.norm(hidden)
        if self.output_model.args.tie_word_embeddings:
            logits = self.output_model.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.output_model.lm_head(hidden)
        return logits.reshape(-1)

    def zeros(self) -> List[Any]:
        import mlx.core as mx

        return [
            mx.zeros((self.hidden_size,), dtype=mx.float32)
            for _ in self.layers
        ]

    def arrays(self, blocks: Sequence[np.ndarray]) -> List[Any]:
        import mlx.core as mx

        return [
            mx.array(np.asarray(block, dtype=np.float32).reshape(-1))
            for block in blocks
        ]

    def evaluate(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        import mlx.core as mx

        output = self(*self.arrays(blocks))
        mx.eval(output)
        return np.asarray(output).astype(np.float64)

    def jvp(
        self, blocks: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        import mlx.core as mx

        primals = self.zeros()
        tangents = self.arrays(blocks)
        try:
            output, derivative = mx.jvp(self, primals, tangents)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if isinstance(derivative, (list, tuple)):
                derivative = derivative[0]
            method = "mx.jvp"
        except ValueError as error:
            if "Not implemented for Sum" not in str(error):
                raise
            # MLX 0.29.1 lacks a forward rule for the generic Sum primitive.
            # Compute the same Jv exactly as (J^T)^T v using two reverse-mode
            # products.  This is an autograd JVP, not finite differencing, and
            # still never materializes J.
            output = self(*primals)
            zero_cotangent = mx.zeros_like(output)

            def transpose_map(cotangent: Any) -> Any:
                _primal_output, gradients = mx.vjp(
                    self, primals, [cotangent]
                )
                return gradients

            _transpose_output, outer_gradient = mx.vjp(
                transpose_map, [zero_cotangent], tangents
            )
            derivative = outer_gradient[0]
            method = "mx.vjp_of_vjp"
        mx.eval(output, derivative)
        return (
            np.asarray(output).astype(np.float64),
            np.asarray(derivative).astype(np.float64),
            method,
        )

    def vjp(self, cotangent: np.ndarray) -> List[np.ndarray]:
        import mlx.core as mx

        zeros = self.zeros()
        vector = mx.array(np.asarray(cotangent, dtype=np.float32))
        _output, gradients = mx.vjp(self, zeros, [vector])
        if not isinstance(gradients, (list, tuple)):
            gradients = [gradients]
        mx.eval(*gradients)
        return [
            np.asarray(value).astype(np.float64) for value in gradients
        ]

    def symmetric_fd(
        self, blocks: Sequence[np.ndarray], radius: float
    ) -> Dict[str, np.ndarray]:
        scaled = [
            float(radius) * np.asarray(block, dtype=np.float64)
            for block in blocks
        ]
        plus = self.evaluate(scaled)
        minus = self.evaluate([-value for value in scaled])
        center = self.evaluate(
            [np.zeros_like(value) for value in scaled]
        )
        return {
            "center": center,
            "plus": plus,
            "minus": minus,
            "symmetric_delta": 0.5 * (plus - minus),
        }


def make_selection(
    reference: Any,
    anchor_step: int,
    selected_by_layer: Mapping[int, Sequence[int]],
    strategy: str,
) -> CoreSelection:
    anchor = reference.anchors[int(anchor_step)]
    by_layer: Dict[int, LayerSelection] = {}
    for layer in range(len(anchor.position_maps)):
        positions = [
            int(value) for value in anchor.position_maps[layer].tolist()
        ]
        selected = [
            int(value)
            for value in selected_by_layer.get(layer, [])
        ]
        by_layer[layer] = LayerSelection(
            layer=layer,
            selected_positions=selected,
            eligible_positions=positions,
            aggregate_scores=[0.0] * len(positions),
            metadata={"predictive_closure": True},
        )
    return CoreSelection(
        strategy=strategy,
        horizon_condition=None,
        by_layer=by_layer,
        metadata={"predictive_closure": True},
    )


def make_smoke_candidates(
    backend: Any,
    reference: Any,
    anchor_step: int,
    cache_cfg: CacheDiscoveryConfig,
    seed: int,
) -> List[PhysicalCandidate]:
    anchor = reference.anchors[int(anchor_step)]
    record = reference.query_records[int(anchor_step)]
    positions = [
        int(value) for value in anchor.position_maps[0].tolist()
    ]
    if any(
        [int(value) for value in anchor.position_maps[layer].tolist()]
        != positions
        for layer in range(len(anchor.position_maps))
    ):
        raise RuntimeError("smoke candidate requires a shared physical universe")
    current = int(anchor.logical_length - 1)
    sink = positions[: int(cache_cfg.sink_size)]
    recent = positions[-int(cache_cfg.recent_size) :]
    mandatory = set(sink + recent)
    eligible = [value for value in positions if value not in mandatory]
    core_size = int(cache_cfg.selected_core_budget)
    if len(eligible) < core_size:
        raise RuntimeError("anchor has insufficient eligible core positions")
    score = np.zeros(len(positions), dtype=np.float64)
    for layer in range(int(backend.model_info["num_layers"])):
        attention = (
            record.all_head_attention_distributions[layer]
            .double()
            .mean(dim=0)
            .numpy()
        )
        score[: len(attention)] += attention
    row_by_position = {position: row for row, position in enumerate(positions)}
    attention_core = sorted(
        eligible,
        key=lambda position: (
            -score[row_by_position[position]],
            position,
        ),
    )[:core_size]
    old_core = eligible[:core_size]
    candidates = []
    for candidate_id, source, core, candidate_seed in (
        ("smoke_attention", "attention_only", attention_core, seed),
        ("smoke_old", "old_stale_core", old_core, seed + 1),
    ):
        retained = sorted(mandatory | set(core))
        keep_prefix = [value for value in retained if value != current]
        if len(retained) != int(cache_cfg.total_budget):
            raise RuntimeError(
                f"active candidate budget {len(retained)} "
                f"!= {cache_cfg.total_budget}"
            )
        candidates.append(
            PhysicalCandidate(
                candidate_id=candidate_id,
                source=source,
                core_positions=tuple(sorted(core)),
                keep_prefix_positions=tuple(keep_prefix),
                retained_positions=tuple(retained),
                seed=int(candidate_seed),
            )
        )
    if len({candidate.mask_hash for candidate in candidates}) != len(candidates):
        raise RuntimeError("smoke candidates are not distinct")
    return candidates


def _derived_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(value) for value in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def make_selector_candidates(
    backend: Any,
    reference: Any,
    anchor_step: int,
    cache_cfg: CacheDiscoveryConfig,
    run_id: str,
    previous_attention_core: Optional[Sequence[int]] = None,
) -> Tuple[List[PhysicalCandidate], Dict[str, Any]]:
    """Build the preregistered eight selector-derived, layer-shared masks."""
    from src.runners.mlx_runner import snapkv_pool_scores_numpy

    anchor = reference.anchors[int(anchor_step)]
    record = reference.query_records[int(anchor_step)]
    positions = [
        int(value) for value in anchor.position_maps[0].tolist()
    ]
    if any(
        [int(value) for value in anchor.position_maps[layer].tolist()]
        != positions
        for layer in range(len(anchor.position_maps))
    ):
        raise RuntimeError(
            "selector candidates require a shared physical token universe"
        )
    current = int(anchor.logical_length - 1)
    sink = positions[: int(cache_cfg.sink_size)]
    recent = positions[-int(cache_cfg.recent_size) :]
    mandatory = set(sink + recent)
    eligible = [value for value in positions if value not in mandatory]
    core_size = int(cache_cfg.selected_core_budget)
    if len(eligible) < core_size:
        raise RuntimeError("anchor has insufficient eligible core positions")
    n_tokens = len(positions)
    query_heads = int(backend.model_info["num_attention_heads"])
    kv_heads = int(backend.model_info["num_key_value_heads"])
    repeats = query_heads // kv_heads
    scores = {
        name: np.zeros(n_tokens, dtype=np.float64)
        for name in ("attention_only", "aov", "aor", "v_ridge", "snapkv")
    }
    for layer in range(int(backend.model_info["num_layers"])):
        attention = (
            record.all_head_attention_distributions[layer].float()
        )
        attention = attention / attention.sum(dim=1, keepdim=True)
        values = anchor.values[layer][0].float()
        repeated_values = values.repeat_interleave(repeats, dim=0)
        full_heads = record.all_head_attention_outputs[layer].float()
        scores["attention_only"] += attention.mean(dim=0).numpy()
        pooled = snapkv_pool_scores_numpy(
            attention.mean(dim=0).numpy(),
            int(backend.cfg.selectors.snapkv_pooling_kernel),
            str(backend.cfg.selectors.snapkv_pooling),
        )
        scores["snapkv"] += np.asarray(pooled, dtype=np.float64)
        aov = (
            attention[:, :, None] * repeated_values
        ).permute(1, 0, 2).reshape(n_tokens, -1)
        safe = (1.0 - attention).clamp_min(1.0e-8)
        aor = (
            (attention / safe)[:, :, None]
            * (full_heads[:, None, :] - repeated_values)
        ).permute(1, 0, 2).reshape(n_tokens, -1)
        projected_aov = backend.project_features(
            layer, aov, chunk_size=128
        )
        projected_aor = backend.project_features(
            layer, aor, chunk_size=128
        )
        scores["aov"] += (
            projected_aov.square().sum(dim=1).double().numpy()
        )
        scores["aor"] += (
            projected_aor.square().sum(dim=1).double().numpy()
        )
        per_kv = []
        for kv_head in range(kv_heads):
            leverage, _diagnostics = ridge_leverage(
                values[kv_head],
                backend.cfg.selectors.ridge_lambda,
                backend.cfg.selectors.ridge_lambda_mode,
            )
            per_kv.append(leverage)
        scores["v_ridge"] += (
            torch.stack(per_kv).mean(dim=0).double().numpy()
        )

    row_by_position = {
        position: row for row, position in enumerate(positions)
    }

    def top_core(score: np.ndarray) -> List[int]:
        return sorted(
            sorted(
                eligible,
                key=lambda position: (
                    -float(score[row_by_position[position]]),
                    position,
                ),
            )[:core_size]
        )

    seed = _derived_seed(run_id, reference.sample_id, anchor_step)
    rng = np.random.default_rng(seed)
    random_core = sorted(
        int(value)
        for value in rng.choice(
            np.asarray(eligible, dtype=np.int64),
            size=core_size,
            replace=False,
        ).tolist()
    )
    cores: List[Tuple[str, List[int]]] = [
        ("attention_only", top_core(scores["attention_only"])),
        ("aov", top_core(scores["aov"])),
        ("aor", top_core(scores["aor"])),
        ("v_ridge", top_core(scores["v_ridge"])),
        ("snapkv", top_core(scores["snapkv"])),
        (
            "old_stale_core",
            sorted(
                int(value)
                for value in (
                    previous_attention_core
                    if previous_attention_core is not None
                    and len(previous_attention_core) == core_size
                    and set(previous_attention_core).issubset(set(eligible))
                    else eligible[:core_size]
                )
            ),
        ),
        ("fresh_core", sorted(eligible[-core_size:])),
        ("random_reference", random_core),
    ]
    candidates: List[PhysicalCandidate] = []
    seen = set()
    dedup_events = []
    for order, (source, proposed) in enumerate(cores):
        core = list(proposed)
        attempt = 0
        while tuple(core) in seen:
            attempt += 1
            replacement_rng = np.random.default_rng(
                _derived_seed(seed, source, "dedup", attempt)
            )
            core = sorted(
                int(value)
                for value in replacement_rng.choice(
                    np.asarray(eligible, dtype=np.int64),
                    size=core_size,
                    replace=False,
                ).tolist()
            )
        if attempt:
            dedup_events.append(
                {
                    "source": source,
                    "attempts": attempt,
                    "replacement_rule": "pre_action_seeded_uniform",
                }
            )
        seen.add(tuple(core))
        retained = sorted(mandatory | set(core))
        if len(retained) != int(cache_cfg.total_budget):
            raise RuntimeError(
                f"active candidate budget {len(retained)} "
                f"!= {cache_cfg.total_budget}"
            )
        candidates.append(
            PhysicalCandidate(
                candidate_id=f"selector_{order:02d}_{source}",
                source=source,
                core_positions=tuple(core),
                keep_prefix_positions=tuple(
                    value for value in retained if value != current
                ),
                retained_positions=tuple(retained),
                seed=int(_derived_seed(seed, source)),
            )
        )
    if len(candidates) != 8 or len({c.mask_hash for c in candidates}) != 8:
        raise RuntimeError("selector candidate registry is not eight-distinct")
    return candidates, {
        "attention_core": list(candidates[0].core_positions),
        "dedup_events": dedup_events,
        "shared_across_layers": True,
        "shared_across_gqa_heads": True,
        "candidate_count": len(candidates),
    }


def direct_injections(
    backend: Any,
    reference: Any,
    anchor_step: int,
    retained_positions: Sequence[int],
    arithmetic_dtype: torch.dtype,
) -> Tuple[List[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]]]:
    anchor = reference.anchors[int(anchor_step)]
    record = reference.query_records[int(anchor_step)]
    query_heads = int(backend.model_info["num_attention_heads"])
    kv_heads = int(backend.model_info["num_key_value_heads"])
    repeats = query_heads // kv_heads
    injections: List[np.ndarray] = []
    identity_rows: List[Dict[str, Any]] = []
    projection_rows: List[Dict[str, Any]] = []
    for layer in range(int(backend.model_info["num_layers"])):
        positions = [
            int(value) for value in anchor.position_maps[layer].tolist()
        ]
        row_by_position = {
            position: row for row, position in enumerate(positions)
        }
        keep_rows = [
            row_by_position[int(position)]
            for position in retained_positions
        ]
        attention = record.all_head_attention_distributions[
            layer
        ].to(dtype=arithmetic_dtype)
        # The diagnostic copy crosses MLX -> NumPy -> Torch.  Normalize after
        # the requested audit cast so the deletion identity is tested on a
        # bona-fide probability vector rather than on serialization roundoff.
        attention_sum = attention.sum(dim=1, keepdim=True)
        if bool((attention_sum <= 0).any()):
            raise FloatingPointError("non-positive attention normalizer")
        attention = attention / attention_sum
        values = anchor.values[layer][0].to(dtype=arithmetic_dtype)
        repeated_values = values.repeat_interleave(repeats, dim=0)
        rows = torch.tensor(keep_rows, dtype=torch.long)
        kept_attention = attention.index_select(1, rows)
        denominator = kept_attention.sum(dim=1)
        if bool((denominator <= 0).any()):
            raise FloatingPointError("zero retained attention mass")
        identity_full_head = (
            attention[:, :, None] * repeated_values
        ).sum(dim=1)
        kept_head = (
            kept_attention[:, :, None]
            * repeated_values.index_select(1, rows)
        ).sum(dim=1) / denominator[:, None]
        # For the algebra audit both sides use the same normalized diagnostic
        # probabilities.  For the actual residual injection, however, the
        # baseline must be the attention output that the model really used;
        # recomputing it from a serialized distribution adds avoidable error.
        identity_direct_head = kept_head - identity_full_head
        recorded_full_head = record.all_head_attention_outputs[layer].to(
            dtype=arithmetic_dtype
        )
        direct_head = kept_head - recorded_full_head
        deleted_mask = torch.ones(
            int(attention.shape[1]), dtype=torch.bool
        )
        deleted_mask[rows] = False
        deleted_attention = attention[:, deleted_mask]
        deleted_values = repeated_values[:, deleted_mask, :]
        deleted_mass = deleted_attention.sum(dim=1)
        closed = (
            deleted_attention[:, :, None]
            * (identity_full_head[:, None, :] - deleted_values)
        ).sum(dim=1) / denominator[:, None]

        projected = backend.project_features(
            layer, direct_head.float().reshape(1, -1)
        )[0]
        per_head = torch.zeros_like(projected)
        for head in range(query_heads):
            per_head += backend.project_features(
                layer,
                direct_head[head].float().reshape(1, -1),
                head=head,
            )[0]
        projection_rows.append(
            {
                "layer": layer,
                "dtype": str(arithmetic_dtype).replace("torch.", ""),
                "sum_block_relative_error": float(
                    (per_head - projected).norm()
                    / projected.norm().clamp_min(1.0e-30)
                ),
                "sum_block_cosine": float(
                    torch.dot(per_head.double(), projected.double())
                    / (
                        per_head.double().norm()
                        * projected.double().norm()
                    ).clamp_min(1.0e-30)
                ),
            }
        )
        injections.append(projected.numpy().astype(np.float32))
        for head in range(query_heads):
            difference = (
                identity_direct_head[head].float() - closed[head].float()
            )
            direct_norm = identity_direct_head[head].float().norm()
            identity_rows.append(
                {
                    "layer": layer,
                    "query_head": head,
                    "kv_head": head // repeats,
                    "dtype": str(arithmetic_dtype).replace("torch.", ""),
                    "maximum_absolute_error": float(
                        difference.abs().max()
                    ),
                    "relative_error": float(
                        difference.norm()
                        / direct_norm.clamp_min(1.0e-30)
                    ),
                    "direct_norm": float(direct_norm),
                    "attention_sum_before_normalization": float(
                        attention_sum[head]
                    ),
                    "denominator": float(denominator[head]),
                    "denominator_complement_gap": float(
                        denominator[head]
                        - (1.0 - deleted_mass[head])
                    ),
                    "deleted_attention_mass": float(deleted_mass[head]),
                    "finite": bool(
                        torch.isfinite(identity_direct_head[head]).all()
                        and torch.isfinite(closed[head]).all()
                    ),
                }
            )
    return injections, identity_rows, projection_rows


def replay_physical(
    backend: Any,
    reference: Any,
    anchor_step: int,
    selection: CoreSelection,
    cache_cfg: CacheDiscoveryConfig,
) -> np.ndarray:
    state, _fixed = backend.state_from_anchor(
        reference.anchors[int(anchor_step)],
        selection,
        cache_config=cache_cfg,
    )
    try:
        logits, _record, _elapsed = backend.forward_one(
            state,
            int(reference.anchors[int(anchor_step)].query_token_id),
            capture_attention=True,
        )
        backend.validate_active_budget(state, cache_config=cache_cfg)
        return logits.double().numpy()
    finally:
        backend.release(state)


def joint_candidate_selection(
    reference: Any,
    anchor_step: int,
    candidate: PhysicalCandidate,
) -> CoreSelection:
    layers = len(reference.anchors[int(anchor_step)].position_maps)
    return make_selection(
        reference,
        anchor_step,
        {
            layer: candidate.core_positions
            for layer in range(layers)
        },
        candidate.source,
    )


def single_layer_selection(
    reference: Any,
    anchor_step: int,
    candidate: PhysicalCandidate,
    masked_layer: int,
) -> Tuple[CoreSelection, CacheDiscoveryConfig]:
    anchor = reference.anchors[int(anchor_step)]
    current = int(anchor.logical_length - 1)
    selected = {}
    for layer in range(len(anchor.position_maps)):
        positions = [
            int(value) for value in anchor.position_maps[layer].tolist()
        ]
        prefix = [value for value in positions if value != current]
        selected[layer] = (
            list(candidate.keep_prefix_positions)
            if layer == int(masked_layer)
            else prefix
        )
    selection = make_selection(
        reference,
        anchor_step,
        selected,
        f"single_layer_{masked_layer}",
    )
    cache_cfg = CacheDiscoveryConfig(
        total_budget=int(anchor.logical_length),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor.logical_length - 1),
    )
    return selection, cache_cfg


def full_selection(
    reference: Any, anchor_step: int
) -> Tuple[CoreSelection, CacheDiscoveryConfig]:
    anchor = reference.anchors[int(anchor_step)]
    current = int(anchor.logical_length - 1)
    selected = {}
    for layer in range(len(anchor.position_maps)):
        selected[layer] = [
            int(value)
            for value in anchor.position_maps[layer].tolist()
            if int(value) != current
        ]
    cache_cfg = CacheDiscoveryConfig(
        total_budget=int(anchor.logical_length),
        sink_size=0,
        recent_size=1,
        selected_core_budget=int(anchor.logical_length - 1),
    )
    return (
        make_selection(reference, anchor_step, selected, "full"),
        cache_cfg,
    )
