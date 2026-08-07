"""Strict, composable experiment configuration without a Hydra dependency."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

from kvbench.errors import ConfigurationError


@dataclass
class RuntimeConfig:
    backend: str = "torch"
    device: str = "cuda"
    seed: int = 42
    deterministic: bool = True
    prefill_chunk_size: int = 128
    attention_prefill_chunk_size: int = 16
    local_files_only: bool = True
    fail_on_error: bool = True
    resume: bool = True
    warmup_steps: int = 1


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: Optional[str] = None
    dtype: str = "float16"
    quantization: str = "none"
    trust_remote_code: bool = False
    attn_implementation: str = "eager"
    max_context: int = 32768
    prompt_format: str = "chat_template"
    system_prompt: Optional[str] = None


@dataclass
class BenchmarkConfig:
    name: str = "ruler"
    task: str = "niah_single_1"
    context_length: int = 4096
    num_samples: int = 2
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    split: Optional[str] = None
    dataset_revision: Optional[str] = None
    data_path: Optional[str] = None
    sample_strategy: str = "first"
    sample_indices: Optional[List[int]] = None
    require_official: bool = True
    use_official_prompt: bool = True
    use_official_generation_length: bool = True
    max_words: int = 0
    truncation: str = "error"


@dataclass
class ProtocolConfig:
    visibility: str = "query_visible"
    cache_mode: str = "prefill_only"
    update_policy: str = "prefill_once"
    update_interval: int = 32
    reuse_mode: str = "single_query"


@dataclass
class BudgetConfig:
    cache_budget: int = 128
    sink_size: int = 4
    recent_size: int = 32
    protect_current: bool = True
    scope: str = "total_kv"
    unit: str = "shared_token_positions"


@dataclass
class MethodConfig:
    name: str = "v_leverage"
    score_source: str = "v"
    leverage_estimator: str = "l2_exact"
    sketch_dim: int = 1024
    normalization: str = "rank"
    attention_ratio: float = 0.5
    alpha: float = 0.5
    residual_lambda: float = 1e-4
    residual_lambda_mode: str = "relative"
    observation_window: int = 32
    pooling_kernel: int = 5
    random_seed_offset: int = 0
    curdkv_projection_dim: int = 20


@dataclass
class GenerationConfig:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    stop_on_eos: bool = True
    compute_teacher_forced_ppl: bool = True


@dataclass
class DiagnosticsConfig:
    save_scores: bool = True
    save_selections: bool = True
    save_token_text: bool = False
    overlap: bool = True
    rank_correlation: bool = True
    evidence_recall: bool = True
    quadrants: bool = True
    reconstruction: bool = False
    attention_statistics: bool = True
    failure_cases: bool = True
    decode_event_interval: int = 1


@dataclass
class OutputConfig:
    root: str = "results"
    experiment_name: str = "torch_kvbench"
    run_id: Optional[str] = None
    overwrite: bool = False


@dataclass
class ExperimentConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.runtime.backend != "torch":
            raise ConfigurationError("The Torch benchmark supports runtime.backend=torch only")
        if self.runtime.device != "cpu" and not self.runtime.device.startswith("cuda"):
            raise ConfigurationError("runtime.device must be cpu or cuda[:index]")
        if self.runtime.warmup_steps < 0:
            raise ConfigurationError("runtime.warmup_steps must be non-negative")
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ConfigurationError("model.dtype must be float16, bfloat16, or float32")
        if self.model.quantization not in {"none", "bnb_4bit", "bnb_8bit"}:
            raise ConfigurationError("unsupported model.quantization")
        if self.protocol.visibility not in {"query_visible", "query_agnostic"}:
            raise ConfigurationError("unsupported protocol.visibility")
        if self.protocol.cache_mode not in {"prefill_only", "live_bounded"}:
            raise ConfigurationError("unsupported protocol.cache_mode")
        if self.protocol.update_policy not in {"prefill_once", "periodic", "every_step"}:
            raise ConfigurationError("unsupported protocol.update_policy")
        if self.protocol.update_policy == "periodic" and self.protocol.update_interval <= 0:
            raise ConfigurationError("periodic update requires update_interval > 0")
        if self.protocol.reuse_mode not in {"single_query", "multi_query"}:
            raise ConfigurationError(
                "protocol.reuse_mode must be single_query or multi_query"
            )
        if (
            self.protocol.reuse_mode == "multi_query"
            and self.protocol.visibility != "query_agnostic"
        ):
            raise ConfigurationError(
                "multi_query cache reuse is defined only for query_agnostic runs"
            )
        if self.budget.cache_budget <= 0:
            raise ConfigurationError("cache_budget must be positive")
        if min(self.budget.sink_size, self.budget.recent_size) < 0:
            raise ConfigurationError("sink_size and recent_size must be non-negative")
        if self.budget.scope not in {"total_kv", "prompt_prefill"}:
            raise ConfigurationError("unsupported budget.scope")
        if not 0.0 <= self.method.attention_ratio <= 1.0:
            raise ConfigurationError("method.attention_ratio must be in [0, 1]")
        if not 0.0 <= self.method.alpha <= 1.0:
            raise ConfigurationError("method.alpha must be in [0, 1]")
        if self.method.residual_lambda < 0:
            raise ConfigurationError("method.residual_lambda must be non-negative")
        if self.method.residual_lambda_mode not in {"absolute", "relative"}:
            raise ConfigurationError("unsupported residual_lambda_mode")
        if self.method.curdkv_projection_dim <= 0:
            raise ConfigurationError("method.curdkv_projection_dim must be positive")
        if self.generation.do_sample:
            raise ConfigurationError(
                "paper path currently requires deterministic greedy generation (do_sample=false)"
            )
        if self.generation.max_new_tokens <= 0:
            raise ConfigurationError("generation.max_new_tokens must be positive")
        if self.benchmark.truncation not in {"error", "head_tail"}:
            raise ConfigurationError("benchmark.truncation must be error or head_tail")


T = TypeVar("T")


def _dataclass_from_dict(cls: Type[T], data: Dict[str, Any], path: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError("unknown fields at %s: %s" % (path, unknown))
    return cls(**data)


def _deep_merge(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_mapping(path: Path, seen: Optional[set] = None) -> Dict[str, Any]:
    seen = set() if seen is None else seen
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigurationError("cyclic config include: %s" % resolved)
    seen.add(resolved)
    with open(resolved, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    includes = data.pop("include", data.pop("includes", []))
    if isinstance(includes, str):
        includes = [includes]
    merged: Dict[str, Any] = {}
    for include in includes:
        include_path = Path(os.path.expandvars(str(include)))
        if not include_path.is_absolute():
            include_path = resolved.parent / include_path
        merged = _deep_merge(merged, _load_mapping(include_path, seen))
    seen.remove(resolved)
    return _deep_merge(merged, data)


def _parse_override(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    result = dict(data)
    for override in overrides:
        if "=" not in override:
            raise ConfigurationError("override must use dotted.path=value: %s" % override)
        dotted, raw = override.split("=", 1)
        cursor = result
        parts = dotted.split(".")
        for part in parts[:-1]:
            child = cursor.get(part)
            if child is None:
                child = {}
                cursor[part] = child
            if not isinstance(child, dict):
                raise ConfigurationError("override path is not a mapping: %s" % dotted)
            cursor = child
        cursor[parts[-1]] = _parse_override(raw)
    return result


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def experiment_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    allowed = {
        "runtime", "model", "benchmark", "protocol", "budget", "method",
        "generation", "diagnostics", "output",
    }
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigurationError("unknown top-level config fields: %s" % unknown)
    cfg = ExperimentConfig(
        runtime=_dataclass_from_dict(RuntimeConfig, data.get("runtime", {}), "runtime"),
        model=_dataclass_from_dict(ModelConfig, data.get("model", {}), "model"),
        benchmark=_dataclass_from_dict(BenchmarkConfig, data.get("benchmark", {}), "benchmark"),
        protocol=_dataclass_from_dict(ProtocolConfig, data.get("protocol", {}), "protocol"),
        budget=_dataclass_from_dict(BudgetConfig, data.get("budget", {}), "budget"),
        method=_dataclass_from_dict(MethodConfig, data.get("method", {}), "method"),
        generation=_dataclass_from_dict(GenerationConfig, data.get("generation", {}), "generation"),
        diagnostics=_dataclass_from_dict(DiagnosticsConfig, data.get("diagnostics", {}), "diagnostics"),
        output=_dataclass_from_dict(OutputConfig, data.get("output", {}), "output"),
    )
    cfg.validate()
    return cfg


def load_experiment(path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    data = _load_mapping(Path(os.path.expandvars(path)))
    data = apply_overrides(data, overrides or [])
    data = _expand_environment(data)
    return experiment_from_dict(data)
