"""Backend-neutral contracts shared by protocols, methods, and artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class BenchmarkSample:
    sample_id: str
    prompt: str
    references: List[str]
    task: str
    answer_text: Optional[str] = None
    full_text: Optional[str] = None
    evidence_positions: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    shared_prefix: Optional[str] = None
    queries: List[str] = field(default_factory=list)


@dataclass
class AttentionSignals:
    accumulated_by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    observation_by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    last_query_by_layer: Dict[int, torch.Tensor] = field(default_factory=dict)
    query_counts: Dict[int, int] = field(default_factory=dict)

    def prune(self, keep_by_layer: Dict[int, torch.Tensor]) -> None:
        for mapping in (
            self.accumulated_by_layer,
            self.observation_by_layer,
            self.last_query_by_layer,
        ):
            for layer, keep in keep_by_layer.items():
                values = mapping.get(layer)
                if values is not None and values.shape[-1] >= int(keep.max().item()) + 1:
                    mapping[layer] = values.index_select(-1, keep.to(values.device))


@dataclass
class CacheSnapshot:
    sample_id: str
    snapshot_id: str
    phase: str
    decode_step: Optional[int]
    logical_length: int
    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    position_maps: Dict[int, torch.Tensor]
    attention: AttentionSignals

    @property
    def num_layers(self) -> int:
        return len(self.keys)


@dataclass
class ScoreBundle:
    aggregate: Dict[int, torch.Tensor] = field(default_factory=dict)
    by_head: Dict[int, torch.Tensor] = field(default_factory=dict)
    components: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    components_by_head: Dict[str, Dict[int, torch.Tensor]] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionDecision:
    layer: int
    universe_positions: List[int]
    selected_rows: List[int]
    selected_positions: List[int]
    requested_budget: int
    effective_budget: int
    mandatory_positions: List[int]
    selectable_budget: int
    budget_scope: str
    budget_unit: str
    selected_sources: Dict[str, List[str]] = field(default_factory=dict)
    scores: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    token_ids: List[int]
    text: str
    prefill_time_s: float
    score_time_s: float
    compression_time_s: float
    decode_time_s: float
    peak_gpu_memory_bytes: int
    cache_lengths: List[int]
    decisions: List[SelectionDecision]
    score_bundles: List[ScoreBundle] = field(default_factory=list)
    teacher_forced_ppl: Optional[float] = None
    query_texts: List[str] = field(default_factory=list)
    query_token_ids: List[List[int]] = field(default_factory=list)


@dataclass
class SampleResult:
    sample_id: str
    task: str
    prediction: str
    references: List[str]
    score: Optional[float]
    metric_name: Optional[str]
    correct: Optional[bool]
    status: str
    error: Optional[str]
    metadata: Dict[str, Any]
    timing: Dict[str, float]
    cache: Dict[str, Any]
    diagnostics: Dict[str, Any]
    predictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
