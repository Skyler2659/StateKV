"""Strict alignment for score and selection artifacts.

Scientific comparisons must never truncate vectors to a common length. Units
are matched by snapshot, phase, layer, KV head, and original token universe,
then reordered by original token position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.artifacts.schema import (
    ScoreArtifact,
    ScoreUnit,
    SelectionArtifact,
    SelectionUnit,
)


UnitKey = Tuple[int, Optional[int]]


@dataclass(frozen=True)
class AlignedScores:
    layer: int
    head: Optional[int]
    positions: Tuple[int, ...]
    scores_a: np.ndarray
    scores_b: np.ndarray


@dataclass(frozen=True)
class AlignedSelections:
    layer: int
    head: Optional[int]
    universe_positions: Tuple[int, ...]
    selected_a: frozenset
    selected_b: frozenset


def _unit_map(units) -> Dict[UnitKey, object]:
    return {unit.key: unit for unit in units}


def _require_same_snapshot(left, right) -> None:
    if left.snapshot.snapshot_id != right.snapshot.snapshot_id:
        raise ValueError(
            "artifacts come from different snapshots: "
            f"{left.snapshot.snapshot_id} != {right.snapshot.snapshot_id}"
        )
    if left.snapshot.phase != right.snapshot.phase:
        raise ValueError(
            f"artifacts come from different phases: {left.snapshot.phase} != {right.snapshot.phase}"
        )
    if left.snapshot.sample_id != right.snapshot.sample_id:
        raise ValueError("artifacts come from different samples")


def _require_same_unit_keys(left: Dict[UnitKey, object], right: Dict[UnitKey, object]) -> None:
    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left), key=str)
        missing_right = sorted(set(left) - set(right), key=str)
        raise ValueError(
            "layer/head units do not match; "
            f"missing_from_left={missing_left[:10]} (n={len(missing_left)}), "
            f"missing_from_right={missing_right[:10]} (n={len(missing_right)})"
        )


def align_score_units(left: ScoreArtifact, right: ScoreArtifact) -> Dict[UnitKey, AlignedScores]:
    _require_same_snapshot(left, right)
    left_units = _unit_map(left.units)
    right_units = _unit_map(right.units)
    _require_same_unit_keys(left_units, right_units)

    aligned: Dict[UnitKey, AlignedScores] = {}
    for key in sorted(left_units, key=str):
        unit_a = left_units[key]
        unit_b = right_units[key]
        assert isinstance(unit_a, ScoreUnit) and isinstance(unit_b, ScoreUnit)
        if set(unit_a.universe_positions) != set(unit_b.universe_positions):
            raise ValueError(f"token universes do not match for layer/head {key}")
        if set(unit_a.original_positions) != set(unit_b.original_positions):
            raise ValueError(f"scored token positions do not match for layer/head {key}")

        order = tuple(sorted(unit_a.original_positions))
        score_a = dict(zip(unit_a.original_positions, unit_a.scores))
        score_b = dict(zip(unit_b.original_positions, unit_b.scores))
        aligned[key] = AlignedScores(
            layer=key[0],
            head=key[1],
            positions=order,
            scores_a=np.asarray([score_a[position] for position in order], dtype=np.float64),
            scores_b=np.asarray([score_b[position] for position in order], dtype=np.float64),
        )
    return aligned


def align_selection_units(
    left: SelectionArtifact,
    right: SelectionArtifact,
) -> Dict[UnitKey, AlignedSelections]:
    _require_same_snapshot(left, right)
    if left.budget_scope != right.budget_scope:
        raise ValueError(f"budget scopes do not match: {left.budget_scope} != {right.budget_scope}")
    if left.budget_unit != right.budget_unit:
        raise ValueError(f"budget units do not match: {left.budget_unit} != {right.budget_unit}")
    if left.requested_budget != right.requested_budget:
        raise ValueError(
            f"requested budgets do not match: {left.requested_budget} != {right.requested_budget}"
        )
    if left.effective_budget != right.effective_budget:
        raise ValueError(
            f"effective budgets do not match: {left.effective_budget} != {right.effective_budget}"
        )

    left_units = _unit_map(left.units)
    right_units = _unit_map(right.units)
    _require_same_unit_keys(left_units, right_units)

    aligned: Dict[UnitKey, AlignedSelections] = {}
    for key in sorted(left_units, key=str):
        unit_a = left_units[key]
        unit_b = right_units[key]
        assert isinstance(unit_a, SelectionUnit) and isinstance(unit_b, SelectionUnit)
        if set(unit_a.universe_positions) != set(unit_b.universe_positions):
            raise ValueError(f"token universes do not match for layer/head {key}")
        if unit_a.requested_budget != unit_b.requested_budget:
            raise ValueError(f"unit requested budgets differ for layer/head {key}")
        if unit_a.effective_budget != unit_b.effective_budget:
            raise ValueError(f"unit effective budgets differ for layer/head {key}")
        aligned[key] = AlignedSelections(
            layer=key[0],
            head=key[1],
            universe_positions=tuple(sorted(unit_a.universe_positions)),
            selected_a=frozenset(unit_a.selected_positions),
            selected_b=frozenset(unit_b.selected_positions),
        )
    return aligned
