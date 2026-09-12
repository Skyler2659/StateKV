"""CUDA generation, strict eviction and query-onset foresight on HF models."""
from __future__ import annotations

import importlib
import math
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from kvbench.backends.huggingface import HFCacheState, HuggingFaceBackend, _legacy_cache
from kvbench.types import SelectionDecision
from statekv.output_metrics import exact_distribution_metrics
from statekv.selectors import mandatory_and_eligible
from src.runners.mlx_runner import snapkv_pool_scores_numpy


POLICIES = ("FULL", "CURRENT_QK", "SNAPKV", "H2O", "LAQ", "LAQPP", "CHEAP_R2")


def retained_rows(positions: list[int], scores: np.ndarray, budget: int,
                  sink: int, recent: int) -> list[int]:
    """Reserve the pending query token before ranking the historical core."""
    sinks, recents, eligible = mandatory_and_eligible(positions, sink, recent - 1)
    take = budget - sink - recent
    if take < 0:
        raise ValueError("budget is smaller than sink + recent protection")
    indexes = {position: row for row, position in enumerate(positions)}
    core = sorted(eligible, key=lambda p: (-float(scores[indexes[p]]), p))[:take]
    keep = set(sinks + recents + core)
    return [row for row, position in enumerate(positions) if position in keep]


def cache_bytes(state: HFCacheState) -> int:
    return sum(t.numel() * t.element_size() for pair in state.past_key_values for t in pair)


class CudaRuntime:
    """Use eager attention for selection and SDPA for uninstrumented replay."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        model = dict(config["model"])
        model.setdefault("revision", None)
        model.setdefault("quantization", "none")
        model.setdefault("trust_remote_code", False)
        model.setdefault("system_prompt", None)
        model.setdefault("prompt_format", "chat_template")
        model.setdefault("chat_template_kwargs", {"enable_thinking": False})
        model["attn_implementation"] = "eager"
        self.backend = HuggingFaceBackend(SimpleNamespace(
            model=SimpleNamespace(**model),
            runtime=SimpleNamespace(device=config.get("device", "cuda:0"),
                local_files_only=config.get("local_files_only", False),
                prefill_chunk_size=config.get("prefill_chunk_size", 64),
                attention_prefill_chunk_size=config.get("prefill_chunk_size", 64)),
            method=SimpleNamespace(observation_window=config.get("snapkv_window", 32))))
        self.info = self.backend.load()
        count = self.info["num_layers"]
        self.layers = config.get("score_layers") or sorted(set(
            int(round(value)) for value in np.linspace(0, count - 1, min(6, count))))
        if any(layer < 0 or layer >= count for layer in self.layers):
            raise ValueError("score layer outside model")
        self.query_capture: dict[int, torch.Tensor] = {}
        self.capture_queries = False
        attention = self.backend.model.model.layers[0].self_attn
        self.attention_module = importlib.import_module(type(attention).__module__)
        self.original_eager = self.attention_module.eager_attention_forward

        def capture(module, query, key, value, *args, **kwargs):
            if self.capture_queries and module.layer_idx in self.layers:
                self.query_capture[module.layer_idx] = query.detach()[0, :, -1].float().cpu().clone()
            return self.original_eager(module, query, key, value, *args, **kwargs)

        self.attention_module.eager_attention_forward = capture
        self.info.update(score_layers=self.layers, cuda_runtime=torch.version.cuda,
                         torch_version=torch.__version__, head_dim=attention.head_dim)

    def close(self) -> None:
        self.attention_module.eager_attention_forward = self.original_eager

    def encode(self, sample, maximum: int) -> tuple[list[int], bool]:
        ids = self.backend.encode_prompt(sample.prompt,
            use_chat_template=not sample.metadata.get("disable_chat_template", False))
        truncated = len(ids) > maximum
        if truncated:
            half = maximum // 2
            ids = ids[:half] + ids[-(maximum - half):]
        return ids, truncated

    def prefill(self, ids: list[int], attention: bool = True):
        self.backend.model.config._attn_implementation = "eager" if attention else "sdpa"
        return self.backend.prefill(ids, capture_attention=attention)

    @torch.inference_mode()
    def forward(self, state: HFCacheState, token: int, attention: bool = True,
                queries: bool = False):
        self.backend.model.config._attn_implementation = "eager" if attention else "sdpa"
        self.capture_queries = queries
        self.query_capture.clear()
        self.backend.synchronize()
        started = time.perf_counter()
        inputs = torch.tensor([[token]], device=self.backend.device)
        try:
            outputs = self.backend._model_call(inputs, state.past_key_values,
                                               state.logical_next_position, attention)
            pooled = None
            if attention:
                pooled = torch.stack([outputs.attentions[layer][0, :, -1].float().mean(0)
                                      for layer in self.layers]).mean(0).cpu().numpy()
            state.past_key_values = _legacy_cache(outputs.past_key_values)
            for layer in state.position_maps:
                state.position_maps[layer] = torch.cat((state.position_maps[layer],
                    torch.tensor([state.logical_next_position])))
            state.logical_next_position += 1
            logits = outputs.logits[0, -1].detach().float().cpu()
            captured = dict(self.query_capture)
            del outputs
        finally:
            self.capture_queries = False
        self.backend.synchronize()
        return logits, pooled, captured, time.perf_counter() - started

    def evict(self, state: HFCacheState, keep: list[int], budget: int) -> float:
        positions = state.position_maps[0].tolist()
        decisions = [SelectionDecision(layer=layer, universe_positions=positions,
            selected_rows=keep, selected_positions=[positions[row] for row in keep],
            requested_budget=budget, effective_budget=len(keep), mandatory_positions=[],
            selectable_budget=budget, budget_scope="total_kv",
            budget_unit="shared_token_positions") for layer in range(len(state.past_key_values))]
        return self.backend.apply_decisions(state, decisions)

    def prompt_memory(self, state: HFCacheState):
        length = len(state.position_maps[0])
        cumulative = torch.stack([state.attention.accumulated[layer].float().mean(0)
                                  for layer in self.layers]).mean(0).cpu().numpy()
        rows = []
        available = min(len(state.attention.observation_rows[layer]) for layer in self.layers)
        for index in range(available):
            padded = [torch.nn.functional.pad(state.attention.observation_rows[layer][index].float().mean(0),
                       (0, length - state.attention.observation_rows[layer][index].shape[-1]))
                      for layer in self.layers]
            rows.append(dict(enumerate(torch.stack(padded).mean(0).cpu().numpy().tolist())))
        return dict(enumerate(cumulative.tolist())), rows

    def r2_scores(self, anchor: HFCacheState, current: int, eligible: list[int],
                  horizon: int) -> dict[int, float]:
        branch = self.backend.fork_state(anchor)
        logits, _, _, _ = self.forward(branch, current)
        token = int(logits.argmax())
        scores = np.zeros(len(eligible), dtype=np.float64)
        for _ in range(horizon):
            logits, pooled, _, _ = self.forward(branch, token)
            index = {p: i for i, p in enumerate(branch.position_maps[0].tolist())}
            scores += pooled[[index[p] for p in eligible]]
            token = int(logits.argmax())
        return dict(zip(eligible, scores.tolist()))

    def laq_scores(self, prompt: list[int], current: int, positions: list[int],
                   eligible: list[int], snap_scores: np.ndarray, budget: int,
                   window: int) -> dict[int, float]:
        # Match the existing LAQ/LAQ++ configuration: Q-cache=8, W=0/8.
        start = len(prompt) - max(0, window - 1)
        branch, _, _ = self.prefill(prompt[:start], attention=False)
        queries = {layer: [] for layer in self.layers}
        query_positions = []
        for offset, token in enumerate(prompt[start:] + [current], start):
            logits, _, captured, _ = self.forward(branch, token, queries=True)
            if window:
                query_positions.append(offset)
                for layer in self.layers:
                    queries[layer].append(captured[layer])
        original_keys = {layer: branch.past_key_values[layer][0][0] for layer in self.layers}
        original_indexes = {p: i for i, p in enumerate(branch.position_maps[0].tolist())}
        keep = retained_rows(positions, snap_scores, budget,
                             self.config.get("sink_size", 4), self.config.get("recent_size", 32))
        keep.append(len(positions))  # the already processed anchor query
        self.evict(branch, keep, budget)
        token = int(logits.argmax())
        for _ in range(self.config.get("laq_lookahead_size", 8)):
            query_positions.append(branch.logical_next_position)
            logits, _, captured, _ = self.forward(branch, token, queries=True)
            for layer in self.layers:
                queries[layer].append(captured[layer])
            token = int(logits.argmax())
        scores = torch.zeros(len(positions))
        mask = torch.tensor(positions)[None, :] > torch.tensor(query_positions)[:, None]
        for layer in self.layers:
            key = original_keys[layer][:, [original_indexes[p] for p in positions]].float().cpu()
            query = torch.stack(queries[layer])
            group = query.shape[1] // key.shape[0]
            layer_score = torch.zeros(len(positions))
            for head in range(query.shape[1]):
                raw = (query[:, head] @ key[head // group].T) / math.sqrt(key.shape[-1])
                layer_score += raw.masked_fill(mask, 0).sum(0)
            scores += layer_score / query.shape[1]
        scores /= len(self.layers)
        return {p: float(scores[i]) for i, p in enumerate(positions) if p in set(eligible)}

    def run_arm(self, anchor: HFCacheState, prompt: list[int], policy: str,
                budget: int, cycles: int, prefill_s: float, event=None) -> dict[str, Any]:
        if policy not in POLICIES:
            raise ValueError(policy)
        sink, recent = self.config.get("sink_size", 4), self.config.get("recent_size", 32)
        state = self.backend.fork_state(anchor)
        cumulative, window_rows = self.prompt_memory(anchor)
        current = prompt[-1]
        generated, logit_rows, rows = [], [], []
        cached = {}
        peak_cache = cache_bytes(state) if policy == "FULL" else 0
        teacher_s = 0.0
        onset_positions = []
        first_token_policy_s = 0.0
        if self.backend.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.backend.device)
        self.backend.synchronize()
        started = time.perf_counter()
        for cycle in range(cycles):
            scoring_s = compression_s = 0.0
            if policy != "FULL":
                positions = state.position_maps[0].tolist()
                scoring = self.backend.fork_state(state)
                _, pooled, _, scoring_s = self.forward(scoring, current)
                current_scores = pooled[:-1].astype(np.float64)
                del scoring
                for position, value in zip(positions, current_scores):
                    cumulative[position] = cumulative.get(position, 0.0) + float(value)
                window_rows.append(dict(zip(positions, current_scores.tolist())))
                window_rows = window_rows[-self.config.get("snapkv_window", 32):]
                _, _, eligible = mandatory_and_eligible(positions, sink, recent - 1)
                snap = snapkv_pool_scores_numpy(np.array([
                    sum(row.get(p, 0.0) for row in window_rows) for p in positions]),
                    self.config.get("snapkv_pooling_kernel", 63), "max")
                if policy == "CURRENT_QK":
                    scores = current_scores
                elif policy == "H2O":
                    scores = np.array([cumulative[p] for p in positions])
                elif policy == "SNAPKV":
                    scores = snap
                else:
                    if cycle == 0:
                        teacher_started = time.perf_counter()
                        if policy == "CHEAP_R2":
                            cached = self.r2_scores(anchor, current, eligible,
                                                    self.config.get("rollout_horizon", 32))
                        else:
                            cached = self.laq_scores(prompt[:-1], current, positions,
                                eligible, snap, budget, 8 if policy == "LAQPP" else 0)
                        self.backend.synchronize()
                        teacher_s = time.perf_counter() - teacher_started
                    scores = np.array([cached.get(p, current_scores[i]) for i, p in enumerate(positions)])
                keep = retained_rows(positions, scores, budget, sink, recent)
                compression_s = self.evict(state, keep, budget)
                if cycle == 0:
                    onset_positions = state.position_maps[0].tolist()
            logits, _, _, decode_s = self.forward(state, current, attention=False)
            token = int(logits.argmax())
            generated.append(token)
            logit_rows.append(logits)
            if cycle == 0:
                first_token_policy_s = time.perf_counter() - started
            active = len(state.position_maps[0])
            if policy != "FULL" and active > budget:
                raise RuntimeError("physical KV budget exceeded")
            peak_cache = max(peak_cache, cache_bytes(state))
            row = dict(cycle=cycle, token_id=token, active_cache_tokens=active,
                       scoring_s=scoring_s, compression_s=compression_s, decode_s=decode_s)
            rows.append(row)
            if event:
                event(row)
            current = token
            if self.config.get("stop_on_eos", True) and token == self.backend.tokenizer.eos_token_id:
                break
        self.backend.synchronize()
        policy_s = time.perf_counter() - started
        evaluation_peak = (torch.cuda.max_memory_allocated(self.backend.device)
                           if self.backend.device.type == "cuda" else None)
        del state
        # Same generated-prefix full-KV replay is outside policy timing.
        reference = self.backend.fork_state(anchor)
        metrics = []
        replay_started = time.perf_counter()
        for index, token in enumerate([prompt[-1]] + generated[:-1]):
            full, _, _, _ = self.forward(reference, token, attention=False)
            metrics.append(exact_distribution_metrics(full, logit_rows[index], generated[index]))
        del reference, logit_rows
        return dict(policy=policy, budget=budget, generated_token_ids=generated,
            generation_text=self.backend.decode(generated), generation_length_tokens=len(generated),
            mean_trajectory_exact_kl=float(np.mean([m["exact_kl"] for m in metrics])),
            p95_trajectory_exact_kl=float(np.quantile([m["exact_kl"] for m in metrics], .95)),
            max_trajectory_exact_kl=max(m["exact_kl"] for m in metrics),
            wall_time_s=prefill_s + policy_s, prefill_time_s=prefill_s,
            policy_time_s=policy_s, metric_replay_s=time.perf_counter() - replay_started,
            ttft_s=prefill_s + first_token_policy_s,
            decode_tokens_per_second=max(0, len(generated) - 1) / max(1e-9, policy_s - first_token_policy_s),
            query_onset_full_cache_bytes=cache_bytes(anchor),
            evaluation_peak_allocated_bytes=evaluation_peak,
            causal_teacher_time_s=teacher_s, causal_teacher_refreshes=int(bool(cached)),
            physical_cache_peak_bytes=peak_cache, onset_retained_positions=onset_positions,
            strict_pure_eviction=True, recoverable_cold_tokens=0, cycles=rows)
