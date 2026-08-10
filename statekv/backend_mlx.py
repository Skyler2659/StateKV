"""MLX 4-bit temporal backend using the repository's existing Qwen hooks."""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from statekv.backend import (
    AnchorState,
    QueryRecord,
    ReferenceTrajectory,
    ScoreState,
    peak_process_rss_bytes,
    set_discovery_determinism,
)
from statekv.config import CacheDiscoveryConfig, DiscoveryConfig
from statekv.selectors import CoreSelection
from kvbench.types import AttentionSignals


@dataclass
class MLXReplayState:
    cache: List[Any]
    position_maps: Dict[int, torch.Tensor]
    logical_next_position: int


class MLXTemporalModel:
    """Temporal protocol adapter for the cached MLX Instruct checkpoint."""

    def __init__(self, cfg: DiscoveryConfig):
        self.cfg = cfg
        self.device_name = "mps"
        self.model_info: Dict[str, Any] = {}
        self.selected_layers: List[int] = []
        self.selected_heads: Dict[int, List[int]] = {}
        self.runner: Any = None
        self.backend = None

    def _root_config(self) -> Any:
        from src.config import (
            BenchmarkConfig,
            EvictionConfig,
            ExperimentConfig,
            ModelConfig,
        )
        from src.model_adapters import infer_model_family

        return ExperimentConfig(
            experiment_name=self.cfg.experiment_name,
            model=ModelConfig(
                name=self.cfg.model.name,
                family=infer_model_family(self.cfg.model.name),
                backend="mlx",
                dtype=self.cfg.model.dtype,
                device="mps",
                quant_bits=int(self.cfg.model.quant_bits or 4),
                quant_group_size=64,
                mlx_weight_quantize=False,
                prefill_step_size=int(self.cfg.runtime.prefill_chunk_size),
                trust_remote_code=self.cfg.model.trust_remote_code,
                local_files_only=True,
                prompt_format={
                    "mode": self.cfg.model.prompt_format,
                    "system_prompt": self.cfg.model.system_prompt,
                    "template_kwargs": dict(
                        self.cfg.model.chat_template_kwargs
                    ),
                },
            ),
            eviction=EvictionConfig(
                method="attention_weighted_v_leverage",
                cache_size=self.cfg.cache.total_budget,
                sink_size=self.cfg.cache.sink_size,
                recent_size=self.cfg.cache.recent_size,
                observation_window=self.cfg.selectors.observation_window,
                attention_window=self.cfg.selectors.observation_window,
                pooling_kernel=self.cfg.selectors.snapkv_pooling_kernel,
                pooling_method=self.cfg.selectors.snapkv_pooling,
                attention_weighted_ridge_lambda=(
                    self.cfg.selectors.attention_weighted_ridge_lambda
                ),
                attention_weight_epsilon=(
                    self.cfg.selectors.attention_weight_epsilon
                ),
                update_policy="prefill_only",
                update_interval=0,
            ),
            benchmark=BenchmarkConfig(
                max_new_tokens=self.cfg.generation.max_new_tokens
            ),
            seed=self.cfg.runtime.seed,
        )

    def load(self) -> Dict[str, Any]:
        import mlx.core as mx
        from src.runners.mlx_runner import MLXRunner

        set_discovery_determinism(
            self.cfg.runtime.seed, self.cfg.runtime.deterministic
        )
        mx.random.seed(int(self.cfg.runtime.seed))
        self.runner = MLXRunner(self._root_config())
        started = time.perf_counter()
        self.runner.load_model()
        self.model_info = dict(self.runner.model_info)
        self.model_info["model_load_s"] = time.perf_counter() - started
        layers = int(self.model_info["num_layers"])
        if self.cfg.diagnostics.layer_selection == "explicit":
            self.selected_layers = [
                int(value)
                for value in self.cfg.diagnostics.explicit_layers
                if int(value) < layers
            ]
            if len(self.selected_layers) != len(
                self.cfg.diagnostics.explicit_layers
            ):
                raise ValueError("an explicit diagnostic layer is out of range")
        else:
            requested_layers = min(
                int(self.cfg.diagnostics.num_layers), layers
            )
            self.selected_layers = sorted(
                set(
                    int(round(value))
                    for value in np.linspace(
                        0, layers - 1, requested_layers
                    )
                )
            )
        query_heads = int(self.model_info["num_attention_heads"])
        if self.cfg.diagnostics.explicit_heads:
            heads = [
                int(value)
                for value in self.cfg.diagnostics.explicit_heads
                if int(value) < query_heads
            ]
            if len(heads) != len(self.cfg.diagnostics.explicit_heads):
                raise ValueError("an explicit diagnostic head is out of range")
        else:
            requested_heads = min(
                int(self.cfg.diagnostics.heads_per_layer), query_heads
            )
            heads = sorted(
                set(
                    int(round(value))
                    for value in np.linspace(
                        0, query_heads - 1, requested_heads
                    )
                )
            )
        self.selected_heads = {
            int(layer): list(heads) for layer in self.selected_layers
        }
        kv_heads = int(self.model_info["num_key_value_heads"])
        group = query_heads // kv_heads
        self.model_info.update(
            {
                "selected_diagnostic_layers": self.selected_layers,
                "selected_diagnostic_query_heads": self.selected_heads,
                "gqa_query_heads_per_kv_head": group,
                "gqa_mapping": {
                    str(head): int(head // group)
                    for head in range(query_heads)
                },
                "anchor_query_replay": "rewind_one_token",
                "device_resolved": "mps",
                "weight_precision": "4bit",
                "requested_bfloat16_overridden_by_user": True,
            }
        )
        self.backend = self
        return dict(self.model_info)

    def close(self) -> None:
        self.runner = None
        self.release()

    @property
    def tokenizer(self) -> Any:
        return self.runner.hf_tokenizer

    def encode_prompt(self, prompt: str) -> Tuple[List[int], bool]:
        from src.model_adapters import apply_prompt_format

        formatted = apply_prompt_format(
            self.runner.hf_tokenizer, prompt, self.runner.cfg.model
        )
        token_ids = [
            int(value) for value in self.runner.hf_tokenizer.encode(formatted)
        ]
        limit = int(self.cfg.runtime.max_prompt_tokens)
        if len(token_ids) <= limit:
            return token_ids, False
        half = limit // 2
        return token_ids[:half] + token_ids[-(limit - half) :], True

    @staticmethod
    def _numpy(value: Any, dtype: Any = np.float32) -> np.ndarray:
        # NumPy has no native MLX bfloat16 buffer representation. Convert only
        # the exported diagnostic copy through FP32; model and cache tensors
        # remain in their original MLX dtype.
        import mlx.core as mx

        exported = value.astype(mx.float32) if hasattr(value, "astype") else value
        return np.asarray(exported).astype(dtype, copy=False).copy()

    @staticmethod
    def _torch(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.from_numpy(MLXTemporalModel._numpy(value)).to(dtype=dtype)

    def _configure_attention(self, phase: str) -> None:
        state = self.runner.attention_state
        state["enabled"] = True
        state["phase"] = phase
        state["current_method"] = "attention_weighted_v_leverage"
        # Functional Stage 1 uses only the exact final observation window
        # (SnapKV/AOV/AOR), not full-prompt accumulated attention. Avoid the
        # otherwise quadratic all-query diagnostic during long prefill.
        state["record_all_queries"] = not bool(
            self.cfg.functional_probe.enabled
            or self.cfg.theory_closing.enabled
            or self.cfg.trajectory_model.enabled
            or self.cfg.robust_envelope.enabled
            or self.cfg.output_sensitivity.enabled
            or self.cfg.gauge_geometry.enabled
            or self.cfg.independent_fisher.enabled
            or getattr(self.cfg, "direct_policy_capture_only", False)
        )
        state["temporal_record_diagnostics"] = True
        state["temporal_record_direct_policy"] = False
        state["temporal_record_query_head_window"] = bool(
            self.cfg.functional_probe.enabled
            or self.cfg.theory_closing.enabled
            or self.cfg.trajectory_model.enabled
            or self.cfg.robust_envelope.enabled
            or self.cfg.output_sensitivity.enabled
            or self.cfg.gauge_geometry.enabled
            or self.cfg.independent_fisher.enabled
            or getattr(self.cfg, "direct_policy_capture_only", False)
        )
        state["temporal_selected_layers"] = list(self.selected_layers)
        state["temporal_selected_heads"] = {
            int(layer): list(heads)
            for layer, heads in self.selected_heads.items()
        }

    def _attention_signals(
        self, layers: Optional[Set[int]] = None
    ) -> AttentionSignals:
        state = self.runner.attention_state
        accumulated = {}
        observation = {}
        last = {}
        query_counts = {}
        for layer in range(int(self.model_info["num_layers"])):
            if layers is not None and layer not in layers:
                continue
            accumulated_value = state.get("accumulated_heads", {}).get(layer)
            last_value = state.get("last_heads", {}).get(layer)
            observe_rows = state.get("observe_heads", {}).get(layer, [])
            if accumulated_value is not None:
                accumulated[layer] = self._torch(accumulated_value)
            if last_value is not None:
                last[layer] = self._torch(last_value)
            if observe_rows:
                import mlx.core as mx

                target_length = max(int(row.shape[-1]) for row in observe_rows)
                padded_rows = [
                    (
                        row
                        if int(row.shape[-1]) == target_length
                        else mx.pad(
                            row,
                            [
                                (0, 0),
                                (0, target_length - int(row.shape[-1])),
                            ],
                        )
                    )
                    for row in observe_rows
                ]
                observation[layer] = self._torch(
                    mx.sum(mx.stack(padded_rows, axis=0), axis=0)
                )
            counts = state.get("query_counts", {}).get(layer, {})
            query_counts[layer] = int(counts.get("prefill", 0)) + int(
                counts.get("decode", 0)
            )
        return AttentionSignals(
            accumulated_by_layer=accumulated,
            observation_by_layer=observation,
            last_query_by_layer=last,
            query_counts=query_counts,
        )

    def _query_record(
        self,
        query_position: int,
        cache: List[Any],
    ) -> QueryRecord:
        state = self.runner.attention_state
        queries: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        distributions: Dict[str, torch.Tensor] = {}
        oracle: Dict[int, torch.Tensor] = {}
        new_values: Dict[str, torch.Tensor] = {}
        new_keys: Dict[str, torch.Tensor] = {}
        attention_inputs: Dict[int, torch.Tensor] = {}
        residual_inputs: Dict[int, torch.Tensor] = {}
        post_attention_residuals: Dict[int, torch.Tensor] = {}
        layer_outputs: Dict[int, torch.Tensor] = {}
        all_head_outputs: Dict[int, torch.Tensor] = {}
        all_head_distributions: Dict[int, torch.Tensor] = {}
        projected_outputs: Dict[int, torch.Tensor] = {}
        for layer in range(int(self.model_info["num_layers"])):
            pooled = state.get("last_heads", {}).get(layer)
            if pooled is None:
                raise RuntimeError(
                    "MLX attention hook produced no last-head signal at layer=%d"
                    % layer
                )
            oracle[layer] = self._torch(pooled, torch.float16)
            if layer not in self.selected_layers:
                continue
            q = state.get("temporal_queries", {}).get(layer)
            out = state.get("temporal_attention_outputs", {}).get(layer)
            attention = state.get(
                "temporal_attention_distributions", {}
            ).get(layer)
            values = state.get("temporal_new_values", {}).get(layer)
            keys = state.get("temporal_new_keys", {}).get(layer)
            attention_input = state.get(
                "temporal_attention_inputs", {}
            ).get(layer)
            residual_input = state.get(
                "temporal_residual_inputs", {}
            ).get(layer)
            post_attention_residual = state.get(
                "temporal_post_attention_residuals", {}
            ).get(layer)
            layer_output = state.get(
                "temporal_layer_outputs", {}
            ).get(layer)
            all_outputs = state.get(
                "temporal_attention_outputs_all_heads", {}
            ).get(layer)
            all_attention = state.get(
                "temporal_attention_distributions_all_heads", {}
            ).get(layer)
            projected = state.get(
                "temporal_projected_attention_outputs", {}
            ).get(layer)
            if any(
                value is None
                for value in (
                    q,
                    out,
                    attention,
                    values,
                    keys,
                    attention_input,
                    residual_input,
                    post_attention_residual,
                    layer_output,
                    all_outputs,
                    all_attention,
                    projected,
                )
            ):
                raise RuntimeError(
                    "MLX temporal diagnostic hook is incomplete at layer=%d"
                    % layer
                )
            q_t = self._torch(q)
            out_t = self._torch(out)
            attention_t = self._torch(attention, torch.float16)
            values_t = self._torch(values)
            keys_t = self._torch(keys)
            attention_inputs[layer] = self._torch(attention_input)
            residual_inputs[layer] = self._torch(residual_input)
            post_attention_residuals[layer] = self._torch(
                post_attention_residual
            )
            layer_outputs[layer] = self._torch(layer_output)
            all_head_outputs[layer] = self._torch(all_outputs)
            # Keep the one-query diagnostic distribution in FP32.  The model
            # and KV cache remain native MLX 4-bit/FP16; this conversion only
            # preserves enough mantissa for deletion-identity and finite-
            # difference audits.  Downstream code explicitly casts to FP16
            # when the preregistered FP16 arithmetic check is requested.
            all_head_distributions[layer] = self._torch(
                all_attention, torch.float32
            )
            projected_outputs[layer] = self._torch(projected)
            for local, head in enumerate(self.selected_heads[layer]):
                key = "%d:%d" % (layer, head)
                queries[key] = q_t[local]
                outputs[key] = out_t[local]
                distributions[key] = attention_t[local]
            for kv_head in range(int(values_t.shape[0])):
                new_values["%d:%d" % (layer, kv_head)] = values_t[kv_head]
                new_keys["%d:%d" % (layer, kv_head)] = keys_t[kv_head]
        return QueryRecord(
            query_position=int(query_position),
            queries=queries,
            attention_outputs=outputs,
            attention_distributions=distributions,
            oracle_attention_by_layer=oracle,
            new_values=new_values,
            all_head_attention_outputs=all_head_outputs,
            all_head_attention_distributions=all_head_distributions,
            projected_attention_outputs=projected_outputs,
            attention_inputs=attention_inputs,
            new_keys=new_keys,
            residual_inputs=residual_inputs,
            post_attention_residuals=post_attention_residuals,
            layer_outputs=layer_outputs,
        )

    def _query_head_observation(self) -> Dict[int, torch.Tensor]:
        """Mean attention over the exact retained pre-query window."""
        import mlx.core as mx

        output: Dict[int, torch.Tensor] = {}
        state = self.runner.attention_state
        for layer in range(int(self.model_info["num_layers"])):
            rows = state.get("observe_query_heads", {}).get(layer, [])
            if not rows:
                continue
            target_length = max(int(row.shape[-1]) for row in rows)
            padded = [
                (
                    row
                    if int(row.shape[-1]) == target_length
                    else mx.pad(
                        row,
                        [
                            (0, 0),
                            (0, target_length - int(row.shape[-1])),
                        ],
                    )
                )
                for row in rows
            ]
            output[layer] = self._torch(
                mx.mean(mx.stack(padded, axis=0), axis=0),
                torch.float32,
            )
        return output

    def _attention_observation_rows(self) -> Dict[int, List[torch.Tensor]]:
        """Copy the retained per-query attention window for closed-loop replay."""

        state = self.runner.attention_state
        return {
            int(layer): [
                self._torch(row, torch.float32)
                for row in state.get("observe_heads", {}).get(layer, [])
            ]
            for layer in range(int(self.model_info["num_layers"]))
        }

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
                    layer_cache.keys[:, :, :offset, :], torch.float16
                )
            )
            values.append(
                self._torch(
                    layer_cache.values[:, :, :offset, :], torch.float16
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
            attention_observation_rows=self._attention_observation_rows(),
        )

    def _score_state(
        self,
        cache: List[Any],
        position_maps: Dict[int, torch.Tensor],
        logical_length: int,
        step: int,
    ) -> ScoreState:
        return ScoreState(
            step=int(step),
            logical_length=int(logical_length),
            values={
                layer: self._torch(
                    cache[layer].values[
                        :, :, : int(cache[layer].offset), :
                    ],
                    torch.float16,
                )
                for layer in self.selected_layers
            },
            position_maps={
                layer: position_maps[layer].clone()
                for layer in self.selected_layers
            },
            attention=self._attention_signals(set(self.selected_layers)),
        )

    def _top_distribution(
        self, logits: Any
    ) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        values = self._numpy(logits, np.float64).reshape(-1)
        if not np.isfinite(values).all():
            raise FloatingPointError("reference logits contain NaN/Inf")
        maximum = float(values.max())
        shifted = values - maximum
        log_normalizer = maximum + float(np.log(np.exp(shifted).sum()))
        log_probabilities = values - log_normalizer
        token = int(np.argmax(values))
        top_k = min(int(self.cfg.metrics.logits_top_k), values.size)
        indices = np.argpartition(-log_probabilities, top_k - 1)[:top_k]
        indices = indices[
            np.argsort(-log_probabilities[indices], kind="stable")
        ]
        return (
            token,
            float(log_probabilities[token]),
            torch.from_numpy(indices.astype(np.int64)),
            torch.from_numpy(
                np.exp(log_probabilities[indices]).astype(np.float64)
            ),
        )

    def _cache_position_maps(
        self, cache: List[Any], logical_length: int
    ) -> Dict[int, torch.Tensor]:
        return {
            layer: torch.arange(logical_length, dtype=torch.long)
            for layer in range(len(cache))
        }

    def generate_reference(
        self,
        sample_id: str,
        task: str,
        prompt: str,
        extra_probe_target_indices: Optional[Sequence[int]] = None,
    ) -> ReferenceTrajectory:
        import mlx.core as mx

        mx.reset_peak_memory()
        self.runner.reset_attention_state()
        self._configure_attention("prefill")
        cache = self.runner.make_cache("full", self.cfg.cache.total_budget)
        prompt_ids, prompt_truncated = self.encode_prompt(prompt)
        if len(prompt_ids) < 2:
            raise ValueError("reference prompt requires at least two tokens")
        started = time.perf_counter()
        chunk_size = max(1, int(self.cfg.runtime.prefill_chunk_size))
        logits = None
        for offset in range(0, len(prompt_ids), chunk_size):
            chunk = prompt_ids[offset : offset + chunk_size]
            logits = self.runner.model(mx.array([chunk]), cache=cache)
            mx.eval(logits)
        logical_length = len(prompt_ids)
        position_maps = self._cache_position_maps(cache, logical_length)
        query_records = [
            self._query_record(logical_length - 1, cache)
        ]
        anchors: Dict[int, AnchorState] = {}
        score_states: Dict[int, ScoreState] = {}
        captured_anchor_steps = set(self.cfg.captured_anchor_steps())
        probe_target_indices = {
            int(anchor + lag)
            for anchor in self.cfg.functional_probe.base_anchor_steps
            for lag in self.cfg.functional_probe.probe_lags
            if int(anchor + lag) < int(self.cfg.generation.max_new_tokens)
        }
        probe_target_indices.update(
            int(value) for value in (extra_probe_target_indices or [])
        )
        if self.cfg.theory_closing.enabled:
            theory = self.cfg.theory_closing
            probe_target_indices.update(
                range(
                    int(theory.horizon_start_step),
                    int(theory.horizon_start_step)
                    + max(int(value) for value in theory.horizons),
                )
            )
            probe_target_indices.add(int(theory.subset_probe_step))
        if self.cfg.trajectory_model.enabled:
            trajectory = self.cfg.trajectory_model
            for anchor in trajectory.anchors:
                probe_target_indices.update(
                    range(
                        int(anchor),
                        int(anchor) + int(trajectory.horizon),
                    )
                )
        if self.cfg.robust_envelope.enabled:
            envelope = self.cfg.robust_envelope
            probe_target_indices.update(
                range(
                    int(envelope.anchor),
                    int(envelope.anchor) + int(envelope.horizon),
                    )
                )
        if self.cfg.output_sensitivity.enabled:
            output = self.cfg.output_sensitivity
            for anchor in output.anchors:
                probe_target_indices.update(
                    range(
                        int(anchor),
                        int(anchor) + int(output.segment_horizon),
                    )
                )
            probe_target_indices.update(
                range(
                    int(output.state_reference_anchor),
                    int(output.state_reference_anchor)
                    + int(output.state_reference_horizon),
                )
            )
        if self.cfg.gauge_geometry.enabled:
            gauge = self.cfg.gauge_geometry
            for anchor in gauge.anchors:
                probe_target_indices.update(
                    range(
                        int(anchor),
                        int(anchor) + int(gauge.segment_horizon),
                    )
                )
        if self.cfg.independent_fisher.enabled:
            independent = self.cfg.independent_fisher
            for anchor in independent.anchors:
                probe_target_indices.update(
                    range(
                        int(anchor),
                        int(anchor) + int(independent.segment_horizon),
                    )
                )
        probe_logits: Dict[int, torch.Tensor] = {}
        needed_score_steps = {
            int(anchor + lag)
            for anchor in self.cfg.anchor_steps
            for lag in [0] + list(self.cfg.signal_lags)
            if anchor + lag <= int(self.cfg.generation.max_new_tokens)
        }
        if 0 in captured_anchor_steps:
            anchors[0] = self._anchor_state(
                cache, position_maps, logical_length, 0, prompt_ids[-1]
            )
        if 0 in needed_score_steps:
            score_states[0] = self._score_state(
                cache, position_maps, logical_length, 0
            )
        generated: List[int] = []
        log_probabilities: List[float] = []
        top_ids: List[torch.Tensor] = []
        top_probabilities: List[torch.Tensor] = []
        stopped_on_eos = False
        eos_ids = set(
            int(value)
            for value in getattr(self.runner.tokenizer, "eos_token_ids", set())
        )
        self.runner.attention_state["phase"] = "decode"
        for target_index in range(int(self.cfg.generation.max_new_tokens)):
            if target_index in probe_target_indices:
                probe_logits[target_index] = self._torch(
                    logits[0, -1, :], torch.float32
                )
            token, log_probability, ids, probabilities = self._top_distribution(
                logits[0, -1, :]
            )
            generated.append(token)
            log_probabilities.append(log_probability)
            top_ids.append(ids)
            top_probabilities.append(probabilities)
            logits = self.runner.model(mx.array([[token]]), cache=cache)
            mx.eval(logits)
            query_records.append(
                self._query_record(logical_length, cache)
            )
            for layer in position_maps:
                position_maps[layer] = torch.cat(
                    [
                        position_maps[layer],
                        torch.tensor([logical_length], dtype=torch.long),
                    ]
                )
            logical_length += 1
            generated_count = len(generated)
            if generated_count in captured_anchor_steps:
                anchors[generated_count] = self._anchor_state(
                    cache,
                    position_maps,
                    logical_length,
                    generated_count,
                    token,
                )
            if generated_count in needed_score_steps:
                score_states[generated_count] = self._score_state(
                    cache, position_maps, logical_length, generated_count
                )
            if self.cfg.generation.stop_on_eos and token in eos_ids:
                stopped_on_eos = True
                break
        return ReferenceTrajectory(
            sample_id=sample_id,
            task=task,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated,
            prompt_length=len(prompt_ids),
            reference_log_probabilities=log_probabilities,
            top_ids=torch.stack(top_ids),
            top_probabilities=torch.stack(top_probabilities),
            query_records=query_records,
            anchors=anchors,
            score_states=score_states,
            selected_layers=list(self.selected_layers),
            selected_heads=dict(self.selected_heads),
            prompt_truncated=prompt_truncated,
            generation_stopped_on_eos=stopped_on_eos,
            generation_time_s=time.perf_counter() - started,
            peak_rss_bytes=peak_process_rss_bytes(),
            peak_accelerator_bytes=int(mx.get_peak_memory()),
            probe_logits=probe_logits,
        )

    def future_attention(
        self,
        reference: ReferenceTrajectory,
        anchor_step: int,
        horizon: int,
    ) -> Dict[int, torch.Tensor]:
        # query_records[t] is the query that predicts generated_token_ids[t].
        # Keep the oracle aligned to the teacher-forced loss window.
        start = int(anchor_step)
        stop = int(anchor_step) + int(horizon)
        records = reference.query_records[start:stop]
        if len(records) != int(horizon):
            raise ValueError("reference does not contain exact future oracle horizon")
        anchor_length = int(reference.anchors[anchor_step].logical_length)
        return {
            layer: torch.stack(
                [
                    record.oracle_attention_by_layer[layer][
                        :, :anchor_length
                    ].float()
                    for record in records
                ]
            ).sum(dim=0)
            for layer in range(int(self.model_info["num_layers"]))
        }

    def state_from_anchor(
        self,
        anchor: AnchorState,
        selection: CoreSelection,
        cache_config: Optional[CacheDiscoveryConfig] = None,
        *,
        cold_positions: Optional[Mapping[int, FrozenSet[int]]] = None,
        quant_bits: int = 4,
        quant_group: int = 64,
    ) -> Tuple[MLXReplayState, Dict[int, Set[int]]]:
        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        cache_cfg = cache_config or self.cfg.cache
        self.runner.reset_attention_state()
        self._configure_attention("decode")
        current_position = int(anchor.logical_length - 1)
        caches = []
        maps: Dict[int, torch.Tensor] = {}
        fixed: Dict[int, Set[int]] = {}
        for layer, (key, value) in enumerate(zip(anchor.keys, anchor.values)):
            positions = [
                int(item) for item in anchor.position_maps[layer].tolist()
            ]
            sink = positions[: min(len(positions), int(cache_cfg.sink_size))]
            recent_size = int(cache_cfg.recent_size)
            recent = (
                positions[max(0, len(positions) - recent_size) :]
                if recent_size > 0
                else []
            )
            core = selection.by_layer[layer].selected_positions
            fixed_positions = set(sink + core) - {current_position}
            keep_positions = sorted(
                fixed_positions | (set(recent) - {current_position})
            )
            row_by_position = {
                position: row for row, position in enumerate(positions)
            }
            rows = torch.tensor(
                [row_by_position[position] for position in keep_positions]
            )
            selected_keys = key.index_select(2, rows)
            selected_values = value.index_select(2, rows)
            if cold_positions is not None:
                cold = cold_positions.get(int(layer)) or frozenset()
                if cold:
                    cold_columns = [
                        row
                        for row, position in enumerate(keep_positions)
                        if int(position) in cold
                    ]
                    if cold_columns:
                        from statekv.value_tier import quantize_dequantize

                        selected_values[:, :, cold_columns, :] = (
                            quantize_dequantize(
                                selected_values[:, :, cold_columns, :],
                                bits=int(quant_bits),
                                group=int(quant_group),
                            )
                        )
            layer_cache = KVCache()
            layer_cache.state = (
                mx.array(selected_keys.numpy()),
                mx.array(selected_values.numpy()),
            )
            layer_cache.logical_offset = current_position
            caches.append(layer_cache)
            maps[layer] = torch.tensor(keep_positions, dtype=torch.long)
            fixed[layer] = fixed_positions
            if len(keep_positions) > int(cache_cfg.total_budget) - 1:
                raise RuntimeError("rewound MLX cache exceeds total_budget - 1")
        return (
            MLXReplayState(caches, maps, current_position),
            fixed,
        )

    @staticmethod
    def shallow_clone_state(state: MLXReplayState) -> MLXReplayState:
        """Duplicate replay bookkeeping while sharing immutable KV arrays.

        MLX arrays are lazy, immutable values, so the clone and the source
        state can reference the same keys/values.  The clone must be pruned
        (``apply_selection_in_place`` replaces the shared arrays with fresh
        ``mx.take`` outputs sized to the retained rows) before any forward,
        which guarantees its ``update_and_fetch`` allocates new buffers and
        never writes into the source state's storage.
        """
        from mlx_lm.models.cache import KVCache

        caches = []
        for layer_cache in state.cache:
            clone = KVCache()
            clone.keys = layer_cache.keys
            clone.values = layer_cache.values
            clone.offset = int(layer_cache.offset)
            clone.logical_offset = int(getattr(layer_cache, "logical_offset", 0))
            caches.append(clone)
        return MLXReplayState(
            caches,
            {
                int(layer): value.clone()
                for layer, value in state.position_maps.items()
            },
            int(state.logical_next_position),
        )

    def apply_selection_in_place(
        self,
        state: MLXReplayState,
        selection: CoreSelection,
        cache_config: Optional[CacheDiscoveryConfig] = None,
    ) -> Dict[int, Set[int]]:
        """Irreversibly prune an active MLX state without a CPU KV backing store."""
        import mlx.core as mx

        cache_cfg = cache_config or self.cfg.cache
        recent_before_query = max(0, int(cache_cfg.recent_size) - 1)
        fixed_by_layer: Dict[int, Set[int]] = {}
        for layer, layer_cache in enumerate(state.cache):
            positions = [
                int(item) for item in state.position_maps[int(layer)].tolist()
            ]
            available = set(positions)
            selected = set(
                int(value)
                for value in selection.by_layer[int(layer)].selected_positions
            )
            if not selected <= available:
                missing = sorted(selected - available)
                raise RuntimeError(
                    "pure eviction attempted to restore deleted positions: %s"
                    % missing[:8]
                )
            sink = positions[: min(len(positions), int(cache_cfg.sink_size))]
            recent = (
                positions[-recent_before_query:]
                if recent_before_query > 0
                else []
            )
            fixed = set(sink) | selected
            keep_positions = sorted(fixed | set(recent))
            if len(keep_positions) > int(cache_cfg.total_budget) - 1:
                raise RuntimeError(
                    "pure-eviction cache exceeds total_budget - 1 before query"
                )
            row_by_position = {
                position: row for row, position in enumerate(positions)
            }
            rows = mx.array(
                [row_by_position[position] for position in keep_positions]
            )
            offset = int(layer_cache.offset)
            layer_cache.keys = mx.take(
                layer_cache.keys[:, :, :offset, :], rows, axis=2
            )
            layer_cache.values = mx.take(
                layer_cache.values[:, :, :offset, :], rows, axis=2
            )
            layer_cache.offset = len(keep_positions)
            layer_cache.logical_offset = int(state.logical_next_position)
            state.position_maps[int(layer)] = torch.tensor(
                keep_positions, dtype=torch.long
            )
            fixed_by_layer[int(layer)] = fixed
        mx.eval(
            *[
                value
                for layer_cache in state.cache
                for value in (layer_cache.keys, layer_cache.values)
            ]
        )
        return fixed_by_layer

    def prune_recent_before_query(
        self,
        state: MLXReplayState,
        fixed_by_layer: Dict[int, Set[int]],
        cache_config: Optional[CacheDiscoveryConfig] = None,
    ) -> None:
        import mlx.core as mx

        cache_cfg = cache_config or self.cfg.cache
        recent_before_query = max(0, int(cache_cfg.recent_size) - 1)
        for layer, layer_cache in enumerate(state.cache):
            positions = [
                int(item) for item in state.position_maps[layer].tolist()
            ]
            fixed = fixed_by_layer[layer]
            dynamic = [position for position in positions if position not in fixed]
            dynamic_tail = (
                dynamic[-recent_before_query:]
                if recent_before_query > 0
                else []
            )
            keep_positions = sorted(fixed | set(dynamic_tail))
            row_by_position = {
                position: row for row, position in enumerate(positions)
            }
            rows = mx.array(
                [row_by_position[position] for position in keep_positions]
            )
            offset = int(layer_cache.offset)
            layer_cache.keys = mx.take(
                layer_cache.keys[:, :, :offset, :], rows, axis=2
            )
            layer_cache.values = mx.take(
                layer_cache.values[:, :, :offset, :], rows, axis=2
            )
            layer_cache.offset = len(keep_positions)
            layer_cache.logical_offset = int(state.logical_next_position)
            state.position_maps[layer] = torch.tensor(
                keep_positions, dtype=torch.long
            )

    def forward_one(
        self,
        state: MLXReplayState,
        token_id: int,
        capture_attention: bool = True,
    ) -> Tuple[torch.Tensor, QueryRecord, float]:
        import mlx.core as mx

        started = time.perf_counter()
        query_position = int(state.logical_next_position)
        logits = self.runner.model(mx.array([[int(token_id)]]), cache=state.cache)
        mx.eval(logits)
        record = self._query_record(query_position, state.cache)
        for layer in state.position_maps:
            state.position_maps[layer] = torch.cat(
                [
                    state.position_maps[layer],
                    torch.tensor([query_position], dtype=torch.long),
                ]
            )
        state.logical_next_position += 1
        return (
            self._torch(logits[0, -1, :]),
            record,
            time.perf_counter() - started,
        )

    def validate_active_budget(
        self,
        state: MLXReplayState,
        cache_config: Optional[CacheDiscoveryConfig] = None,
    ) -> None:
        cache_cfg = cache_config or self.cfg.cache
        lengths = [int(layer.offset) for layer in state.cache]
        if any(length > int(cache_cfg.total_budget) for length in lengths):
            raise RuntimeError("active MLX cache exceeded total budget: %s" % lengths)

    def project_features(
        self,
        layer: int,
        vectors: torch.Tensor,
        head: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply the linear part of the quantized output projection."""
        import mlx.core as mx

        if vectors.ndim != 2:
            raise ValueError("project_features expects a rank-2 matrix")
        query_heads = int(self.model_info["num_attention_heads"])
        head_dim = int(
            self.model_info.get("head_dim")
            or int(self.model_info["hidden_size"]) // query_heads
        )
        hidden_size = query_heads * head_dim
        if head is None and int(vectors.shape[1]) != hidden_size:
            raise ValueError(
                "layer features require hidden_size=%d, got %d"
                % (hidden_size, int(vectors.shape[1]))
            )
        if head is not None and int(vectors.shape[1]) != head_dim:
            raise ValueError(
                "head features require head_dim=%d, got %d"
                % (head_dim, int(vectors.shape[1]))
            )
        projection = self.runner.model.model.layers[
            int(layer)
        ].self_attn.o_proj
        rows: List[torch.Tensor] = []
        step = max(
            1,
            int(
                chunk_size
                or self.cfg.functional_probe.projection_chunk_size
            ),
        )
        zero = mx.zeros((1, hidden_size), dtype=mx.float32)
        zero_output = projection(zero)
        mx.eval(zero_output)
        for offset in range(0, int(vectors.shape[0]), step):
            block = vectors[offset : offset + step].float()
            if head is not None:
                expanded = torch.zeros(
                    (int(block.shape[0]), hidden_size), dtype=torch.float32
                )
                start = int(head) * head_dim
                expanded[:, start : start + head_dim] = block
                block = expanded
            projected = projection(mx.array(block.numpy()))
            projected = projected - zero_output
            mx.eval(projected)
            rows.append(self._torch(projected, torch.float32))
        return torch.cat(rows, dim=0)

    def active_cache_tokens(self, state: MLXReplayState) -> int:
        return max((int(layer.offset) for layer in state.cache), default=0)

    def release(self, *objects: Any) -> None:
        import mlx.core as mx

        del objects
        gc.collect()
        mx.clear_cache()

    def _peak_accelerator_memory(self) -> Optional[int]:
        import mlx.core as mx

        return int(mx.get_peak_memory())
