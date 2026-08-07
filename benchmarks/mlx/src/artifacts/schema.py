"""Schema-v2 score and selection artifacts.

The v1 files in existing result directories only contain ``layer -> values``.
They cannot establish snapshot, phase, head, token universe, or budget parity
and are intentionally not coerced into this schema.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


ARTIFACT_SCHEMA_VERSION = 2
_VALID_PHASES = {"prefill", "pre_answer", "decode", "post_generation"}
_VALID_BUDGET_SCOPES = {"total_kv", "prompt_prefill", "snapshot_offline"}
_VALID_BUDGET_UNITS = {
    "shared_token_positions",
    "token_slots_per_kv_head",
    "token_head_pairs",
}


def _int_tuple(values: Iterable[int], name: str) -> Tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} must contain non-negative positions")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicate positions")
    return normalized


def _unit_key(layer: int, head: Optional[int]) -> Tuple[int, Optional[int]]:
    return int(layer), None if head is None else int(head)


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    sample_id: str
    phase: str
    context_length: int
    decode_step: Optional[int] = None
    prompt_length: Optional[int] = None
    token_ids_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id is required")
        if self.phase not in _VALID_PHASES:
            raise ValueError(f"unsupported snapshot phase: {self.phase}")
        if int(self.context_length) < 0:
            raise ValueError("context_length must be non-negative")
        if self.decode_step is not None and int(self.decode_step) < 0:
            raise ValueError("decode_step must be non-negative")
        if self.prompt_length is not None and int(self.prompt_length) < 0:
            raise ValueError("prompt_length must be non-negative")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotRef":
        return cls(**data)


@dataclass(frozen=True)
class ScoreUnit:
    layer: int
    head: Optional[int]
    original_positions: Tuple[int, ...]
    universe_positions: Tuple[int, ...]
    scores: Tuple[float, ...]
    valid_mask: Optional[Tuple[bool, ...]] = None

    def __post_init__(self) -> None:
        layer, head = _unit_key(self.layer, self.head)
        if layer < 0 or (head is not None and head < 0):
            raise ValueError("layer and head must be non-negative")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "head", head)
        positions = _int_tuple(self.original_positions, "original_positions")
        universe = _int_tuple(self.universe_positions, "universe_positions")
        scores = tuple(float(value) for value in self.scores)
        if len(positions) != len(scores):
            raise ValueError("score count must match original position count")
        if not set(positions).issubset(set(universe)):
            raise ValueError("scored positions must be contained in the token universe")
        if any(not math.isfinite(value) for value in scores):
            raise ValueError("scores must be finite")
        if self.valid_mask is not None:
            valid_mask = tuple(bool(value) for value in self.valid_mask)
            if len(valid_mask) != len(positions):
                raise ValueError("valid_mask must match original position count")
            object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "original_positions", positions)
        object.__setattr__(self, "universe_positions", universe)
        object.__setattr__(self, "scores", scores)

    @property
    def key(self) -> Tuple[int, Optional[int]]:
        return _unit_key(self.layer, self.head)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreUnit":
        return cls(
            layer=data["layer"],
            head=data.get("head"),
            original_positions=tuple(data["original_positions"]),
            universe_positions=tuple(data["universe_positions"]),
            scores=tuple(data["scores"]),
            valid_mask=(tuple(data["valid_mask"]) if data.get("valid_mask") is not None else None),
        )


@dataclass(frozen=True)
class SelectionUnit:
    layer: int
    head: Optional[int]
    selected_positions: Tuple[int, ...]
    universe_positions: Tuple[int, ...]
    requested_budget: Optional[int] = None
    effective_budget: Optional[int] = None
    protected_positions: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        layer, head = _unit_key(self.layer, self.head)
        if layer < 0 or (head is not None and head < 0):
            raise ValueError("layer and head must be non-negative")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "head", head)
        selected = _int_tuple(self.selected_positions, "selected_positions")
        universe = _int_tuple(self.universe_positions, "universe_positions")
        protected = _int_tuple(self.protected_positions, "protected_positions")
        universe_set = set(universe)
        if not set(selected).issubset(universe_set):
            raise ValueError("selected positions must be contained in the token universe")
        if not set(protected).issubset(universe_set):
            raise ValueError("protected positions must be contained in the token universe")
        requested_budget = len(selected) if self.requested_budget is None else int(self.requested_budget)
        effective_budget = len(selected) if self.effective_budget is None else int(self.effective_budget)
        if requested_budget < 0 or effective_budget < 0:
            raise ValueError("unit budgets must be non-negative")
        if effective_budget != len(selected):
            raise ValueError("unit effective_budget must equal the physical selected slot count")
        object.__setattr__(self, "selected_positions", selected)
        object.__setattr__(self, "universe_positions", universe)
        object.__setattr__(self, "requested_budget", requested_budget)
        object.__setattr__(self, "effective_budget", effective_budget)
        object.__setattr__(self, "protected_positions", protected)

    @property
    def key(self) -> Tuple[int, Optional[int]]:
        return _unit_key(self.layer, self.head)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionUnit":
        return cls(
            layer=data["layer"],
            head=data.get("head"),
            selected_positions=tuple(data["selected_positions"]),
            universe_positions=tuple(data["universe_positions"]),
            requested_budget=data.get("requested_budget"),
            effective_budget=data.get("effective_budget"),
            protected_positions=tuple(data.get("protected_positions", ())),
        )


def _validate_unique_units(units: Iterable[Union[ScoreUnit, SelectionUnit]]) -> None:
    keys = [unit.key for unit in units]
    if len(keys) != len(set(keys)):
        raise ValueError("an artifact may contain only one unit per layer/head")


@dataclass(frozen=True)
class ScoreArtifact:
    artifact_id: str
    snapshot: SnapshotRef
    method: str
    score_type: str
    score_source: str
    units: Tuple[ScoreUnit, ...]
    definition: Dict[str, Any] = field(default_factory=dict)
    estimator: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    artifact_type: str = "score"

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION or self.artifact_type != "score":
            raise ValueError("invalid score artifact schema or type")
        if not self.artifact_id or not self.method or not self.score_type or not self.score_source:
            raise ValueError("score artifact identity fields are required")
        units = tuple(self.units)
        if not units:
            raise ValueError("score artifact must contain at least one layer/head unit")
        _validate_unique_units(units)
        object.__setattr__(self, "units", units)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoreArtifact":
        return cls(
            artifact_id=data["artifact_id"],
            snapshot=SnapshotRef.from_dict(data["snapshot"]),
            method=data["method"],
            score_type=data["score_type"],
            score_source=data["score_source"],
            units=tuple(ScoreUnit.from_dict(unit) for unit in data["units"]),
            definition=dict(data.get("definition", {})),
            estimator=dict(data.get("estimator", {})),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", ARTIFACT_SCHEMA_VERSION),
            artifact_type=data.get("artifact_type", "score"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionArtifact:
    artifact_id: str
    snapshot: SnapshotRef
    method: str
    requested_budget: int
    effective_budget: int
    budget_scope: str
    budget_unit: str
    units: Tuple[SelectionUnit, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = ARTIFACT_SCHEMA_VERSION
    artifact_type: str = "selection"

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION or self.artifact_type != "selection":
            raise ValueError("invalid selection artifact schema or type")
        if not self.artifact_id or not self.method:
            raise ValueError("selection artifact identity fields are required")
        if int(self.requested_budget) < 0 or int(self.effective_budget) < 0:
            raise ValueError("budgets must be non-negative")
        if self.budget_scope not in _VALID_BUDGET_SCOPES:
            raise ValueError(f"unsupported budget scope: {self.budget_scope}")
        if self.budget_unit not in _VALID_BUDGET_UNITS:
            raise ValueError(f"unsupported budget unit: {self.budget_unit}")
        units = tuple(self.units)
        if not units:
            raise ValueError("selection artifact must contain at least one layer/head unit")
        _validate_unique_units(units)
        object.__setattr__(self, "units", units)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelectionArtifact":
        return cls(
            artifact_id=data["artifact_id"],
            snapshot=SnapshotRef.from_dict(data["snapshot"]),
            method=data["method"],
            requested_budget=data["requested_budget"],
            effective_budget=data["effective_budget"],
            budget_scope=data["budget_scope"],
            budget_unit=data["budget_unit"],
            units=tuple(SelectionUnit.from_dict(unit) for unit in data["units"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=data.get("schema_version", ARTIFACT_SCHEMA_VERSION),
            artifact_type=data.get("artifact_type", "selection"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


Artifact = Union[ScoreArtifact, SelectionArtifact]


def artifact_content_hash(artifact: Artifact) -> str:
    payload = json.dumps(artifact.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_artifact(artifact: Artifact, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle, indent=2, ensure_ascii=False)


def load_artifact(path: Union[str, Path]) -> Artifact:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("legacy or unsupported artifact schema; v1 is not alignment-safe")
    artifact_type = data.get("artifact_type")
    if artifact_type == "score":
        return ScoreArtifact.from_dict(data)
    if artifact_type == "selection":
        return SelectionArtifact.from_dict(data)
    raise ValueError(f"unknown artifact type: {artifact_type}")
