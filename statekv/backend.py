"""Reference generation and frozen-core replay on the existing HF backend."""
from __future__ import annotations

import gc
import math
import os
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from kvbench.backends.huggingface import (
    AttentionAccumulator,
    HFCacheState,
    HuggingFaceBackend,
    _legacy_cache,
)
from statekv.config import DiscoveryConfig
from statekv.selectors import CoreSelection
from kvbench.types import AttentionSignals, CacheSnapshot


def set_discovery_determinism(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> str:
    value = str(requested).lower()
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_process_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _clone_signals(
    signals: AttentionSignals,
    layers: Optional[Set[int]] = None,
) -> AttentionSignals:
    def clone_mapping(mapping: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        return {
            int(layer): value.detach().float().cpu().clone()
            for layer, value in mapping.items()
            if layers is None or int(layer) in layers
        }

    return AttentionSignals(
        accumulated_by_layer=clone_mapping(signals.accumulated_by_layer),
        observation_by_layer=clone_mapping(signals.observation_by_layer),
        last_query_by_layer=clone_mapping(signals.last_query_by_layer),
        query_counts={
            int(layer): int(value)
            for layer, value in signals.query_counts.items()
            if layers is None or int(layer) in layers
        },
    )


@dataclass
class QueryRecord:
    query_position: int
    queries: Dict[str, torch.Tensor]
    attention_outputs: Dict[str, torch.Tensor]
    attention_distributions: Dict[str, torch.Tensor]
    oracle_attention_by_layer: Dict[int, torch.Tensor]
    new_values: Dict[str, torch.Tensor]
    all_head_attention_outputs: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    all_head_attention_distributions: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    projected_attention_outputs: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    attention_inputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    new_keys: Dict[str, torch.Tensor] = field(default_factory=dict)
    residual_inputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    post_attention_residuals: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    layer_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass
class AnchorState:
    anchor_step: int
    logical_length: int
    query_token_id: int
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    position_maps: Dict[int, torch.Tensor]
    attention: AttentionSignals
    query_head_observation: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )
    attention_observation_rows: Dict[int, List[torch.Tensor]] = field(
        default_factory=dict
    )

    def snapshot(self, sample_id: str) -> CacheSnapshot:
        return CacheSnapshot(
            sample_id=sample_id,
            snapshot_id="%s:anchor:%d" % (sample_id, self.anchor_step),
            phase="reference_anchor",
            decode_step=self.anchor_step,
            logical_length=self.logical_length,
            keys=self.keys,
            values=self.values,
            position_maps=self.position_maps,
            attention=self.attention,
        )


@dataclass
class ScoreState:
    step: int
    logical_length: int
    values: Dict[int, torch.Tensor]
    position_maps: Dict[int, torch.Tensor]
    attention: AttentionSignals


@dataclass
class ReferenceTrajectory:
    sample_id: str
    task: str
    prompt_token_ids: List[int]
    generated_token_ids: List[int]
    prompt_length: int
    reference_log_probabilities: List[float]
    top_ids: torch.Tensor
    top_probabilities: torch.Tensor
    query_records: List[QueryRecord]
    anchors: Dict[int, AnchorState]
    score_states: Dict[int, ScoreState]
    selected_layers: List[int]
    selected_heads: Dict[int, List[int]]
    prompt_truncated: bool
    generation_stopped_on_eos: bool
    generation_time_s: float
    peak_rss_bytes: int
    peak_accelerator_bytes: Optional[int]
    probe_logits: Dict[int, torch.Tensor] = field(default_factory=dict)


class TemporalModel:
    """Thin discovery wrapper around the repository's HuggingFaceBackend."""

    def __init__(self, cfg: DiscoveryConfig):
        self.cfg = cfg
        self.device_name = resolve_device(cfg.runtime.device)
        backend_cfg = SimpleNamespace(
            runtime=SimpleNamespace(
                device=self.device_name,
                local_files_only=cfg.model.local_files_only,
                prefill_chunk_size=cfg.runtime.prefill_chunk_size,
                attention_prefill_chunk_size=cfg.runtime.prefill_chunk_size,
            ),
            model=SimpleNamespace(
                name=cfg.model.name,
                dtype=cfg.model.dtype,
                trust_remote_code=cfg.model.trust_remote_code,
                revision=cfg.model.revision,
                attn_implementation=cfg.model.attn_implementation,
                quantization="none",
                prompt_format=cfg.model.prompt_format,
                system_prompt=cfg.model.system_prompt,
            ),
            method=SimpleNamespace(
                observation_window=cfg.selectors.observation_window
            ),
        )
        self.backend = HuggingFaceBackend(backend_cfg)
        self.model_info: Dict[str, Any] = {}
        self.selected_layers: List[int] = []
        self.selected_heads: Dict[int, List[int]] = {}
        self._query_capture: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[Any] = []

    def load(self) -> Dict[str, Any]:
        set_discovery_determinism(
            self.cfg.runtime.seed, self.cfg.runtime.deterministic
        )
        started = time.perf_counter()
        self.model_info = self.backend.load()
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
        heads = int(self.model_info["num_attention_heads"])
        if self.cfg.diagnostics.explicit_heads:
            representative = [
                int(value)
                for value in self.cfg.diagnostics.explicit_heads
                if int(value) < heads
            ]
            if len(representative) != len(
                self.cfg.diagnostics.explicit_heads
            ):
                raise ValueError("an explicit diagnostic head is out of range")
        else:
            requested_heads = min(
                int(self.cfg.diagnostics.heads_per_layer), heads
            )
            representative = sorted(
                set(
                    int(round(value))
                    for value in np.linspace(
                        0, heads - 1, requested_heads
                    )
                )
            )
        self.selected_heads = {
            int(layer): list(representative) for layer in self.selected_layers
        }
        self._install_query_hooks()
        group = heads // int(self.model_info["num_key_value_heads"])
        self.model_info.update(
            {
                "selected_diagnostic_layers": self.selected_layers,
                "selected_diagnostic_query_heads": self.selected_heads,
                "gqa_query_heads_per_kv_head": group,
                "gqa_mapping": {
                    str(head): int(head // group) for head in range(heads)
                },
                "anchor_query_replay": "rewind_one_token",
                "device_resolved": self.device_name,
            }
        )
        return dict(self.model_info)

    def _install_query_hooks(self) -> None:
        model_layers = getattr(getattr(self.backend.model, "model", None), "layers", None)
        if model_layers is None:
            raise RuntimeError("Qwen model layers are unavailable for query diagnostics")

        for layer in self.selected_layers:
            projection = model_layers[layer].self_attn.q_proj

            def capture(_module, _inputs, output, layer_index=layer):
                tensor = output[0] if isinstance(output, tuple) else output
                self._query_capture[int(layer_index)] = tensor.detach()

            self._hook_handles.append(projection.register_forward_hook(capture))

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def encode_prompt(self, prompt: str) -> Tuple[List[int], bool]:
        token_ids = self.backend.encode_prompt(prompt)
        limit = int(self.cfg.runtime.max_prompt_tokens)
        if len(token_ids) <= limit:
            return token_ids, False
        half = limit // 2
        return token_ids[:half] + token_ids[-(limit - half) :], True

    def _diagnostics(
        self,
        attentions: Sequence[torch.Tensor],
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        query_position: int,
    ) -> QueryRecord:
        num_heads = int(self.model_info["num_attention_heads"])
        num_kv_heads = int(self.model_info["num_key_value_heads"])
        head_dim = int(self.model_info["hidden_size"]) // num_heads
        group = num_heads // num_kv_heads
        queries: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        distributions: Dict[str, torch.Tensor] = {}
        oracle: Dict[int, torch.Tensor] = {}
        new_values: Dict[str, torch.Tensor] = {}
        if attentions is None:
            raise RuntimeError("eager model call did not return attentions")
        for layer, raw in enumerate(attentions):
            if raw is None or raw.ndim != 4:
                raise RuntimeError("invalid attention output at layer=%d" % layer)
            query_attention = raw.detach()[0, :, -1, :].float()
            if int(query_attention.shape[0]) != num_heads:
                raise RuntimeError("attention query-head count mismatch")
            grouped = query_attention.reshape(
                num_kv_heads, group, int(query_attention.shape[-1])
            ).mean(dim=1)
            oracle[int(layer)] = grouped.to(dtype=torch.float16, device="cpu")
            if layer not in self.selected_layers:
                continue
            captured = self._query_capture.get(layer)
            if captured is None:
                raise RuntimeError("query hook produced no tensor at layer=%d" % layer)
            q = captured.detach()[0, -1].float().reshape(num_heads, head_dim)
            values = past[layer][1].detach()[0].float()
            for kv_head in range(num_kv_heads):
                new_values["%d:%d" % (layer, kv_head)] = (
                    values[kv_head, -1].cpu().clone()
                )
            for head in self.selected_heads[layer]:
                kv_head = int(head // group)
                key = "%d:%d" % (layer, head)
                weights = query_attention[head]
                queries[key] = q[head].cpu().clone()
                distributions[key] = weights.to(
                    dtype=torch.float16, device="cpu"
                )
                outputs[key] = torch.matmul(weights, values[kv_head]).cpu()
        return QueryRecord(
            query_position=int(query_position),
            queries=queries,
            attention_outputs=outputs,
            attention_distributions=distributions,
            oracle_attention_by_layer=oracle,
            new_values=new_values,
            all_head_attention_outputs={},
            all_head_attention_distributions={},
            projected_attention_outputs={},
        )

    @torch.no_grad()
    def _prefill(
        self, token_ids: List[int]
    ) -> Tuple[HFCacheState, torch.Tensor, QueryRecord]:
        accumulator = self.backend._new_accumulator()
        past: Optional[Any] = None
        start = 0
        logits: Optional[torch.Tensor] = None
        diagnostic: Optional[QueryRecord] = None
        chunk_size = max(1, int(self.cfg.runtime.prefill_chunk_size))
        for offset in range(0, len(token_ids), chunk_size):
            chunk = token_ids[offset : offset + chunk_size]
            inputs = torch.tensor(
                [chunk], device=self.backend.device, dtype=torch.long
            )
            outputs = self.backend._model_call(
                inputs, past, start, capture_attention=True
            )
            accumulator.update(outputs.attentions)
            past = _legacy_cache(outputs.past_key_values)
            logits = outputs.logits[0, -1, :]
            start += len(chunk)
            if offset + len(chunk) == len(token_ids):
                diagnostic = self._diagnostics(
                    outputs.attentions, past, start - 1
                )
            del outputs
        if logits is None or diagnostic is None:
            raise RuntimeError("prefill did not produce logits and diagnostics")
        legacy = _legacy_cache(past)
        maps = {
            layer: torch.arange(start, dtype=torch.long)
            for layer in range(len(legacy))
        }
        return HFCacheState(legacy, maps, start, accumulator), logits, diagnostic

    @torch.no_grad()
    def forward_one(
        self,
        state: HFCacheState,
        token_id: int,
        capture_attention: bool = True,
    ) -> Tuple[torch.Tensor, QueryRecord, float]:
        started = time.perf_counter()
        inputs = torch.tensor(
            [[int(token_id)]], device=self.backend.device, dtype=torch.long
        )
        query_position = int(state.logical_next_position)
        outputs = self.backend._model_call(
            inputs,
            state.past_key_values,
            query_position,
            capture_attention=capture_attention,
        )
        if capture_attention:
            state.attention.update(outputs.attentions)
        state.past_key_values = _legacy_cache(outputs.past_key_values)
        for layer in range(len(state.past_key_values)):
            state.position_maps[layer] = torch.cat(
                [
                    state.position_maps[layer],
                    torch.tensor([query_position], dtype=torch.long),
                ]
            )
        state.logical_next_position += 1
        logits = outputs.logits[0, -1, :]
        diagnostic = self._diagnostics(
            outputs.attentions, state.past_key_values, query_position
        )
        elapsed = time.perf_counter() - started
        del outputs
        return logits, diagnostic, elapsed

    def _anchor_state(
        self,
        state: HFCacheState,
        anchor_step: int,
        query_token_id: int,
    ) -> AnchorState:
        return AnchorState(
            anchor_step=int(anchor_step),
            logical_length=int(state.logical_next_position),
            query_token_id=int(query_token_id),
            keys=[
                pair[0].detach().cpu().clone() for pair in state.past_key_values
            ],
            values=[
                pair[1].detach().cpu().clone() for pair in state.past_key_values
            ],
            position_maps={
                int(layer): positions.detach().cpu().clone()
                for layer, positions in state.position_maps.items()
            },
            attention=_clone_signals(state.attention.signals()),
        )

    def _score_state(self, state: HFCacheState, step: int) -> ScoreState:
        layer_set = set(self.selected_layers)
        return ScoreState(
            step=int(step),
            logical_length=int(state.logical_next_position),
            values={
                int(layer): state.past_key_values[layer][1].detach().cpu().clone()
                for layer in self.selected_layers
            },
            position_maps={
                int(layer): state.position_maps[layer].detach().cpu().clone()
                for layer in self.selected_layers
            },
            attention=_clone_signals(state.attention.signals(), layer_set),
        )

    def _top_distribution(
        self, logits: torch.Tensor
    ) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        values = logits.detach().double()
        if not torch.isfinite(values).all():
            raise FloatingPointError("reference logits contain NaN/Inf")
        log_probs = torch.log_softmax(values, dim=-1)
        token = int(torch.argmax(values).item())
        top_k = min(int(self.cfg.metrics.logits_top_k), int(values.numel()))
        top_log_probs, top_ids = torch.topk(log_probs, top_k)
        return (
            token,
            float(log_probs[token].item()),
            top_ids.cpu(),
            torch.exp(top_log_probs).cpu(),
        )

    @torch.no_grad()
    def generate_reference(
        self,
        sample_id: str,
        task: str,
        prompt: str,
        extra_probe_target_indices: Optional[Sequence[int]] = None,
    ) -> ReferenceTrajectory:
        self._reset_peak_memory()
        started = time.perf_counter()
        prompt_ids, prompt_truncated = self.encode_prompt(prompt)
        if len(prompt_ids) < 2:
            raise ValueError("reference prompt requires at least two tokens")
        state, logits, query_record = self._prefill(prompt_ids)
        query_records = [query_record]
        anchors: Dict[int, AnchorState] = {}
        score_states: Dict[int, ScoreState] = {}
        captured_anchor_steps = set(self.cfg.captured_anchor_steps())
        needed_score_steps = {
            int(anchor + lag)
            for anchor in self.cfg.anchor_steps
            for lag in [0] + list(self.cfg.signal_lags)
            if anchor + lag <= int(self.cfg.generation.max_new_tokens)
        }
        if 0 in captured_anchor_steps:
            anchors[0] = self._anchor_state(state, 0, prompt_ids[-1])
        if 0 in needed_score_steps:
            score_states[0] = self._score_state(state, 0)
        generated: List[int] = []
        log_probabilities: List[float] = []
        top_ids: List[torch.Tensor] = []
        top_probabilities: List[torch.Tensor] = []
        stopped_on_eos = False
        probe_logits: Dict[int, torch.Tensor] = {}
        probe_targets = (
            {
                int(anchor) + int(lag)
                for anchor in self.cfg.functional_probe.base_anchor_steps
                for lag in self.cfg.functional_probe.probe_lags
            }
            if self.cfg.functional_probe.enabled
            else set()
        )
        probe_targets.update(
            int(value) for value in (extra_probe_target_indices or [])
        )
        if self.cfg.independent_fisher.enabled:
            for anchor in self.cfg.independent_fisher.anchors:
                probe_targets.update(
                    range(
                        int(anchor),
                        int(anchor)
                        + int(self.cfg.independent_fisher.segment_horizon),
                    )
                )
        eos_ids = self.backend.tokenizer.eos_token_id
        eos_set = (
            {int(value) for value in eos_ids}
            if isinstance(eos_ids, (list, tuple, set))
            else ({int(eos_ids)} if eos_ids is not None else set())
        )
        for _ in range(int(self.cfg.generation.max_new_tokens)):
            target_index = len(generated)
            if target_index in probe_targets:
                probe_logits[target_index] = (
                    logits.detach().float().cpu().clone()
                )
            token, log_probability, ids, probabilities = self._top_distribution(logits)
            generated.append(token)
            log_probabilities.append(log_probability)
            top_ids.append(ids)
            top_probabilities.append(probabilities)
            logits, record, _ = self.forward_one(state, token, capture_attention=True)
            query_records.append(record)
            generated_count = len(generated)
            if generated_count in captured_anchor_steps:
                anchors[generated_count] = self._anchor_state(
                    state, generated_count, token
                )
            if generated_count in needed_score_steps:
                score_states[generated_count] = self._score_state(
                    state, generated_count
                )
            if (
                self.cfg.generation.stop_on_eos
                and token in eos_set
            ):
                stopped_on_eos = True
                break
        missing_anchors = [
            value for value in captured_anchor_steps if value > len(generated)
        ]
        for value in missing_anchors:
            anchors.pop(int(value), None)
        elapsed = time.perf_counter() - started
        return ReferenceTrajectory(
            sample_id=sample_id,
            task=task,
            prompt_token_ids=prompt_ids,
            generated_token_ids=generated,
            prompt_length=len(prompt_ids),
            reference_log_probabilities=log_probabilities,
            top_ids=torch.stack(top_ids, dim=0),
            top_probabilities=torch.stack(top_probabilities, dim=0),
            query_records=query_records,
            anchors=anchors,
            score_states=score_states,
            selected_layers=list(self.selected_layers),
            selected_heads=dict(self.selected_heads),
            prompt_truncated=prompt_truncated,
            generation_stopped_on_eos=stopped_on_eos,
            generation_time_s=elapsed,
            peak_rss_bytes=peak_process_rss_bytes(),
            peak_accelerator_bytes=self._peak_accelerator_memory(),
            probe_logits=probe_logits,
        )

    def future_attention(
        self,
        reference: ReferenceTrajectory,
        anchor_step: int,
        horizon: int,
    ) -> Dict[int, torch.Tensor]:
        # query_records[t] is the query that predicts generated_token_ids[t].
        # Therefore the oracle window must align with the exact records used by
        # teacher-forced replay steps 1..H at this anchor.
        start = int(anchor_step)
        stop = int(anchor_step) + int(horizon)
        records = reference.query_records[start:stop]
        if len(records) != int(horizon):
            raise ValueError("reference does not contain exact future oracle horizon")
        anchor_length = int(reference.anchors[anchor_step].logical_length)
        result: Dict[int, torch.Tensor] = {}
        for layer in range(int(self.model_info["num_layers"])):
            values = [
                record.oracle_attention_by_layer[layer][:, :anchor_length].float()
                for record in records
            ]
            if any(int(value.shape[-1]) < anchor_length for value in values):
                raise ValueError("future attention is shorter than anchor history")
            result[layer] = torch.stack(values, dim=0).sum(dim=0)
        return result

    def state_from_anchor(
        self,
        anchor: AnchorState,
        selection: CoreSelection,
        cache_config: Optional[Any] = None,
    ) -> Tuple[HFCacheState, Dict[int, Set[int]]]:
        cache_cfg = cache_config or self.cfg.cache
        current_position = int(anchor.logical_length - 1)
        rebuilt = []
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
            row_by_position = {position: row for row, position in enumerate(positions)}
            keep_rows = torch.tensor(
                [row_by_position[position] for position in keep_positions],
                dtype=torch.long,
            )
            rebuilt.append(
                (
                    key.index_select(2, keep_rows).to(self.backend.device),
                    value.index_select(2, keep_rows).to(self.backend.device),
                )
            )
            maps[layer] = torch.tensor(keep_positions, dtype=torch.long)
            fixed[layer] = fixed_positions
            if len(keep_positions) > int(cache_cfg.total_budget) - 1:
                raise RuntimeError("rewound pre-query cache exceeds total_budget - 1")
        state = HFCacheState(
            past_key_values=tuple(rebuilt),
            position_maps=maps,
            logical_next_position=current_position,
            attention=self.backend._new_accumulator(),
        )
        return state, fixed

    def prune_recent_before_query(
        self,
        state: HFCacheState,
        fixed_by_layer: Dict[int, Set[int]],
        cache_config: Optional[Any] = None,
    ) -> None:
        cache_cfg = cache_config or self.cfg.cache
        recent_before_query = max(0, int(cache_cfg.recent_size) - 1)
        rebuilt = []
        for layer, (key, value) in enumerate(state.past_key_values):
            positions = [int(item) for item in state.position_maps[layer].tolist()]
            fixed = fixed_by_layer[layer]
            dynamic = [position for position in positions if position not in fixed]
            recent_dynamic = (
                dynamic[-recent_before_query:]
                if recent_before_query > 0
                else []
            )
            keep_positions = sorted(
                fixed | set(recent_dynamic)
            )
            row_by_position = {position: row for row, position in enumerate(positions)}
            keep_rows_cpu = torch.tensor(
                [row_by_position[position] for position in keep_positions],
                dtype=torch.long,
            )
            keep_rows = keep_rows_cpu.to(key.device)
            rebuilt.append(
                (
                    key.index_select(2, keep_rows),
                    value.index_select(2, keep_rows.to(value.device)),
                )
            )
            state.position_maps[layer] = torch.tensor(
                keep_positions, dtype=torch.long
            )
            if len(keep_positions) > int(cache_cfg.total_budget) - 1:
                raise RuntimeError("pre-query rolling cache exceeds total_budget - 1")
        state.past_key_values = tuple(rebuilt)

    def validate_active_budget(
        self, state: HFCacheState, cache_config: Optional[Any] = None
    ) -> None:
        cache_cfg = cache_config or self.cfg.cache
        lengths = [int(pair[0].shape[2]) for pair in state.past_key_values]
        if any(length > int(cache_cfg.total_budget) for length in lengths):
            raise RuntimeError("active cache exceeded total budget: %s" % lengths)

    def active_cache_tokens(self, state: HFCacheState) -> int:
        return max(
            (int(pair[0].shape[2]) for pair in state.past_key_values),
            default=0,
        )

    def release(self, *objects: Any) -> None:
        del objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _reset_peak_memory(self) -> None:
        if self.device_name.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.backend.device)

    def _peak_accelerator_memory(self) -> Optional[int]:
        if self.device_name.startswith("cuda"):
            return int(torch.cuda.max_memory_allocated(self.backend.device))
        if self.device_name == "mps" and hasattr(torch, "mps"):
            try:
                return int(torch.mps.current_allocated_memory())
            except Exception:
                return None
        return None
