"""Single-GPU HuggingFace backend with explicit legacy-cache compaction."""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvbench.backends.base import BackendAdapter
from kvbench.config import ExperimentConfig
from kvbench.errors import ConfigurationError, SignalUnavailableError
from kvbench.types import AttentionSignals, CacheSnapshot, SelectionDecision


def _legacy_cache(cache: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if cache is None:
        return tuple()
    if isinstance(cache, tuple):
        return cache
    if isinstance(cache, list):
        return tuple(cache)
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    try:
        return tuple((cache.key_cache[i], cache.value_cache[i]) for i in range(len(cache.key_cache)))
    except Exception as exc:
        raise RuntimeError("unsupported HuggingFace cache object: %s" % type(cache).__name__) from exc


class AttentionAccumulator:
    """Stream prefill/decode attention into aligned per-KV-head score vectors."""

    def __init__(self, num_kv_heads: int, observation_window: int):
        self.num_kv_heads = int(num_kv_heads)
        self.observation_window = max(1, int(observation_window))
        self.accumulated: Dict[int, torch.Tensor] = {}
        self.last: Dict[int, torch.Tensor] = {}
        self.observation_rows: Dict[int, List[torch.Tensor]] = {}
        self.query_counts: Dict[int, int] = {}

    def update(self, attentions: Any) -> None:
        if attentions is None:
            raise SignalUnavailableError(
                "model did not return attentions; use attn_implementation=eager"
            )
        for layer, raw in enumerate(attentions):
            if raw is None or raw.ndim != 4:
                raise SignalUnavailableError("invalid attention tensor at layer=%d" % layer)
            values = raw.detach()[0].float()  # [query_head, query, key]
            query_heads, query_len, key_len = values.shape
            if query_heads % self.num_kv_heads != 0:
                raise SignalUnavailableError(
                    "cannot map %d query heads to %d KV heads"
                    % (query_heads, self.num_kv_heads)
                )
            group = query_heads // self.num_kv_heads
            values = values.reshape(self.num_kv_heads, group, query_len, key_len).mean(dim=1)
            pooled = values.sum(dim=1)  # [kv_head, key]
            previous = self.accumulated.get(layer)
            if previous is None:
                previous = torch.zeros_like(pooled)
            elif int(previous.shape[-1]) < key_len:
                previous = torch.cat(
                    [
                        previous,
                        previous.new_zeros((self.num_kv_heads, key_len - int(previous.shape[-1]))),
                    ],
                    dim=-1,
                )
            previous[:, :key_len] += pooled
            self.accumulated[layer] = previous
            self.last[layer] = values[:, -1, :]
            buffer = self.observation_rows.setdefault(layer, [])
            for query_row in values.unbind(dim=1):
                buffer.append(query_row)
            if len(buffer) > self.observation_window:
                del buffer[:-self.observation_window]
            self.query_counts[layer] = self.query_counts.get(layer, 0) + int(query_len)

    def prune(self, keep_by_layer: Dict[int, torch.Tensor]) -> None:
        for layer, keep in keep_by_layer.items():
            current = self.accumulated.get(layer)
            current_length = int(current.shape[-1]) if current is not None else 0
            for mapping in (self.accumulated, self.last):
                value = mapping.get(layer)
                if value is not None:
                    mapping[layer] = value.index_select(-1, keep.to(value.device))
            rows = self.observation_rows.get(layer, [])
            rebuilt_rows = []
            for row in rows:
                # Causal observation rows recorded early in prefill have shorter
                # key axes.  Missing future positions have exactly zero mass.
                if int(row.shape[-1]) < current_length:
                    row = torch.cat(
                        [
                            row,
                            row.new_zeros(
                                (row.shape[0], current_length - int(row.shape[-1]))
                            ),
                        ],
                        dim=-1,
                    )
                rebuilt_rows.append(row.index_select(-1, keep.to(row.device)))
            self.observation_rows[layer] = rebuilt_rows

    def signals(self) -> AttentionSignals:
        observation: Dict[int, torch.Tensor] = {}
        for layer, rows in self.observation_rows.items():
            if rows:
                max_len = max(int(row.shape[-1]) for row in rows)
                padded = []
                for row in rows:
                    if int(row.shape[-1]) < max_len:
                        row = torch.cat(
                            [
                                row,
                                row.new_zeros((row.shape[0], max_len - int(row.shape[-1]))),
                            ],
                            dim=-1,
                        )
                    padded.append(row)
                observation[layer] = torch.stack(padded, dim=0).sum(dim=0)
        return AttentionSignals(
            accumulated_by_layer={key: value for key, value in self.accumulated.items()},
            observation_by_layer=observation,
            last_query_by_layer={key: value for key, value in self.last.items()},
            query_counts=dict(self.query_counts),
        )


@dataclass
class HFCacheState:
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    position_maps: Dict[int, torch.Tensor]
    logical_next_position: int
    attention: AttentionAccumulator


class HuggingFaceBackend(BackendAdapter):
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.runtime.device)
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_info: Dict[str, Any] = {}
        self._forward_params: set = set()

    def load(self) -> Dict[str, Any]:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.cfg.model.dtype]
        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.cfg.model.trust_remote_code,
            "local_files_only": self.cfg.runtime.local_files_only,
        }
        if self.cfg.model.revision:
            tokenizer_kwargs["revision"] = self.cfg.model.revision
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": self.cfg.model.trust_remote_code,
            "local_files_only": self.cfg.runtime.local_files_only,
            "low_cpu_mem_usage": True,
            "attn_implementation": self.cfg.model.attn_implementation,
        }
        if self.cfg.model.revision:
            load_kwargs["revision"] = self.cfg.model.revision
        quantization = self.cfg.model.quantization
        if quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ConfigurationError("bitsandbytes quantization requires BitsAndBytesConfig") from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "bnb_4bit",
                load_in_8bit=quantization == "bnb_8bit",
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs["device_map"] = {"": str(self.device)}
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model.name, **load_kwargs)
        if quantization == "none":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        self._forward_params = set(inspect.signature(self.model.forward).parameters)
        config = self.model.config
        num_heads = int(getattr(config, "num_attention_heads"))
        num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
        self.model_info = {
            "model_name": self.cfg.model.name,
            "revision": self.cfg.model.revision,
            "model_type": getattr(config, "model_type", None),
            "dtype": self.cfg.model.dtype,
            "quantization": quantization,
            "device": str(self.device),
            "num_layers": int(getattr(config, "num_hidden_layers")),
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": int(getattr(config, "hidden_size")),
            "attn_implementation": self.cfg.model.attn_implementation,
            "checkpoint_commit_hash": getattr(config, "_commit_hash", None),
            "model_name_or_path": getattr(config, "_name_or_path", None),
            "tokenizer_name_or_path": getattr(self.tokenizer, "name_or_path", None),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_vocab_size": int(len(self.tokenizer)),
        }
        return dict(self.model_info)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self, steps: int) -> float:
        """Run untimed synthetic forwards once, outside every benchmark sample."""
        count = int(steps)
        if count <= 0:
            return 0.0
        token = self.tokenizer.bos_token_id
        if token is None:
            token = self.tokenizer.eos_token_id
        if token is None:
            raise RuntimeError("warmup requires a tokenizer BOS or EOS token")
        self.synchronize()
        started = time.perf_counter()
        state, logits, _ = self.prefill([int(token)], capture_attention=False)
        current = int(torch.argmax(logits, dim=-1).item())
        for _ in range(max(0, count - 1)):
            logits, _ = self.step(state, current, capture_attention=False)
            current = int(torch.argmax(logits, dim=-1).item())
        self.synchronize()
        return time.perf_counter() - started

    def cache_bytes_per_token(self) -> int:
        """Theoretical physical K+V bytes for one shared token position."""
        if not self.model_info:
            raise RuntimeError("model must be loaded before cache size accounting")
        num_heads = int(self.model_info["num_attention_heads"])
        num_kv_heads = int(self.model_info["num_key_value_heads"])
        head_dim = int(self.model_info["hidden_size"]) // num_heads
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.cfg.model.dtype]
        element_size = torch.empty((), dtype=dtype).element_size()
        return (
            2
            * int(self.model_info["num_layers"])
            * num_kv_heads
            * head_dim
            * element_size
        )

    def encode_prompt(
        self, prompt: str, use_chat_template: Optional[bool] = None
    ) -> List[int]:
        text = prompt
        use_chat = (
            self.cfg.model.prompt_format == "chat_template"
            if use_chat_template is None
            else bool(use_chat_template)
        )
        if use_chat:
            messages = []
            if self.cfg.model.system_prompt:
                messages.append({"role": "system", "content": self.cfg.model.system_prompt})
            messages.append({"role": "user", "content": prompt})
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return [int(value) for value in self.tokenizer.encode(text, add_special_tokens=True)]

    def encode_text(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ]

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _new_accumulator(self) -> AttentionAccumulator:
        return AttentionAccumulator(
            int(self.model_info["num_key_value_heads"]),
            int(self.cfg.method.observation_window),
        )

    def _model_call(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any],
        position_start: int,
        capture_attention: bool,
    ) -> Any:
        positions = torch.arange(
            position_start,
            position_start + int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        )
        kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_attentions": capture_attention,
            "position_ids": positions.unsqueeze(0),
        }
        if "cache_position" in self._forward_params:
            kwargs["cache_position"] = positions
        return self.model(**kwargs)

    @torch.no_grad()
    def prefill(
        self, token_ids: List[int], capture_attention: bool
    ) -> Tuple[HFCacheState, torch.Tensor, float]:
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        accumulator = self._new_accumulator()
        past: Optional[Any] = None
        logits: Optional[torch.Tensor] = None
        start = 0
        chunk_size = (
            int(self.cfg.runtime.attention_prefill_chunk_size)
            if capture_attention
            else int(self.cfg.runtime.prefill_chunk_size)
        )
        self.synchronize()
        started = time.perf_counter()
        for offset in range(0, len(token_ids), max(1, chunk_size)):
            chunk = token_ids[offset : offset + max(1, chunk_size)]
            inputs = torch.tensor([chunk], device=self.device, dtype=torch.long)
            outputs = self._model_call(inputs, past, start, capture_attention)
            logits = outputs.logits[:, -1, :]
            if capture_attention:
                accumulator.update(outputs.attentions)
            past = _legacy_cache(outputs.past_key_values)
            start += len(chunk)
            del outputs
        if logits is None:
            raise RuntimeError("prefill produced no logits")
        self.synchronize()
        elapsed = time.perf_counter() - started
        legacy = _legacy_cache(past)
        maps = {
            layer: torch.arange(start, device="cpu", dtype=torch.long)
            for layer in range(len(legacy))
        }
        return HFCacheState(legacy, maps, start, accumulator), logits, elapsed

    @torch.no_grad()
    def step(
        self, state: HFCacheState, token_id: int, capture_attention: bool
    ) -> Tuple[torch.Tensor, float]:
        self.synchronize()
        started = time.perf_counter()
        inputs = torch.tensor([[int(token_id)]], device=self.device, dtype=torch.long)
        outputs = self._model_call(
            inputs,
            state.past_key_values,
            state.logical_next_position,
            capture_attention,
        )
        if capture_attention:
            state.attention.update(outputs.attentions)
        state.past_key_values = _legacy_cache(outputs.past_key_values)
        new_position = int(state.logical_next_position)
        for layer in range(len(state.past_key_values)):
            state.position_maps[layer] = torch.cat(
                [state.position_maps[layer], torch.tensor([new_position], dtype=torch.long)]
            )
        state.logical_next_position += 1
        logits = outputs.logits[:, -1, :]
        self.synchronize()
        elapsed = time.perf_counter() - started
        del outputs
        return logits, elapsed

    def snapshot(
        self,
        state: HFCacheState,
        sample_id: str,
        phase: str,
        decode_step: int,
    ) -> CacheSnapshot:
        keys = [pair[0] for pair in state.past_key_values]
        values = [pair[1] for pair in state.past_key_values]
        return CacheSnapshot(
            sample_id=sample_id,
            snapshot_id="%s:%s:%d:%d" % (
                sample_id,
                phase,
                int(decode_step),
                int(state.logical_next_position),
            ),
            phase=phase,
            decode_step=decode_step,
            logical_length=int(state.logical_next_position),
            keys=keys,
            values=values,
            position_maps={key: value.clone() for key, value in state.position_maps.items()},
            attention=state.attention.signals(),
        )

    def fork_state(self, state: HFCacheState) -> HFCacheState:
        """Fork a compressed shared-prefix cache for an independent future query.

        Legacy Hugging Face cache tensors are treated as immutable inputs: every
        subsequent forward returns a new cache tuple.  Sharing those tensors here
        avoids duplicating a long prefix on GPU, while position maps and attention
        state remain request-local.
        """
        return HFCacheState(
            past_key_values=tuple((key, value) for key, value in state.past_key_values),
            position_maps={
                layer: positions.clone()
                for layer, positions in state.position_maps.items()
            },
            logical_next_position=int(state.logical_next_position),
            attention=self._new_accumulator(),
        )

    @torch.no_grad()
    def apply_decisions(
        self, state: HFCacheState, decisions: List[SelectionDecision]
    ) -> float:
        if len(decisions) != len(state.past_key_values):
            raise RuntimeError("one cache decision is required for every model layer")
        if all(
            decision.selected_rows == list(range(int(pair[0].shape[2])))
            for pair, decision in zip(state.past_key_values, decisions)
        ):
            return 0.0
        self.synchronize()
        started = time.perf_counter()
        rebuilt = []
        keep_by_layer: Dict[int, torch.Tensor] = {}
        for layer, ((key, value), decision) in enumerate(
            zip(state.past_key_values, decisions)
        ):
            if decision.layer != layer:
                raise RuntimeError("cache decision layer order is invalid")
            keep = torch.tensor(decision.selected_rows, device=key.device, dtype=torch.long)
            rebuilt.append(
                (
                    key.index_select(2, keep),
                    value.index_select(2, keep.to(value.device)),
                )
            )
            cpu_keep = keep.cpu()
            state.position_maps[layer] = state.position_maps[layer].index_select(0, cpu_keep)
            keep_by_layer[layer] = keep
        state.past_key_values = tuple(rebuilt)
        state.attention.prune(keep_by_layer)
        self.synchronize()
        return time.perf_counter() - started

    def cache_length(self, state: HFCacheState) -> int:
        if not state.past_key_values:
            return 0
        lengths = {int(pair[0].shape[2]) for pair in state.past_key_values}
        if len(lengths) != 1:
            raise RuntimeError("model layers have inconsistent physical cache lengths")
        return lengths.pop()

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int:
        if self.device.type == "cuda":
            return int(torch.cuda.max_memory_allocated(self.device))
        return 0
