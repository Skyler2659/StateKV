"""Decision utilities shared by selection and oracle refresh diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class SelectionDecision:
    """Lowest-risk candidate and the ordering induced by an evaluator."""

    candidate_id: str
    risk: float
    margin: float
    ordered_candidates: Tuple[str, ...]


def select_lowest_risk(risks: Mapping[str, float]) -> SelectionDecision:
    """Select the minimum finite risk with a deterministic identifier tie-break."""

    if not risks:
        raise ValueError("at least one candidate risk is required")
    normalized = []
    for candidate_id, value in risks.items():
        risk = float(value)
        if not math.isfinite(risk):
            raise ValueError("candidate risks must be finite")
        normalized.append((str(candidate_id), risk))
    ordered = sorted(normalized, key=lambda item: (item[1], item[0]))
    best_id, best_risk = ordered[0]
    margin = (
        float("inf") if len(ordered) == 1 else float(ordered[1][1] - best_risk)
    )
    return SelectionDecision(
        candidate_id=best_id,
        risk=best_risk,
        margin=margin,
        ordered_candidates=tuple(candidate_id for candidate_id, _ in ordered),
    )


def oracle_refresh_required(
    previous: SelectionDecision,
    current_risks: Mapping[str, float],
) -> bool:
    """Return whether exact re-evaluation changes the preferred candidate.

    This is an evaluator-side diagnostic, not a deployable state-drift detector.
    """

    return select_lowest_risk(current_risks).candidate_id != previous.candidate_id
