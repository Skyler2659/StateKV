"""Decision utilities shared by selection and oracle refresh diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple


@dataclass(frozen=True)
class SelectionDecision:
    """Lowest-risk candidate and the ordering induced by an evaluator."""

    candidate_id: str
    risk: float
    margin: float
    ordered_candidates: Tuple[str, ...]


@dataclass(frozen=True)
class ProxyRetentionDecision:
    """Budget-feasible action induced by one additive proxy risk."""

    retained_positions: Tuple[int, ...]
    risk: float
    retention_utility: float


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


def additive_retained_set_risk(
    deletion_costs: Mapping[int, float],
    retained_positions: Iterable[int],
) -> float:
    """Return the proxy risk of deleting every position outside ``C``.

    ``deletion_costs`` is one observable, training-free cost vector.  Defining
    the proxy on the retained-set action itself keeps selection and refresh in
    the same decision space: selection minimizes this value and refresh uses
    the regret of continuing with the previous retained set.
    """

    if not deletion_costs:
        raise ValueError("at least one deletion cost is required")
    normalized = {}
    for position, value in deletion_costs.items():
        index = int(position)
        cost = float(value)
        if index in normalized:
            raise ValueError("deletion-cost positions must be unique")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("deletion costs must be finite and non-negative")
        normalized[index] = cost
    retained = {int(position) for position in retained_positions}
    unknown = retained - set(normalized)
    if unknown:
        raise ValueError("retained position is outside the proxy universe")
    return float(
        sum(cost for position, cost in normalized.items() if position not in retained)
    )


def select_additive_retained_set(
    deletion_costs: Mapping[int, float], budget: int
) -> ProxyRetentionDecision:
    """Minimize additive deletion risk with a deterministic top-cost action."""

    keep = int(budget)
    if keep < 0 or keep > len(deletion_costs):
        raise ValueError("budget must lie within the proxy universe")
    normalized = []
    for position, value in deletion_costs.items():
        index = int(position)
        cost = float(value)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("deletion costs must be finite and non-negative")
        normalized.append((index, cost))
    if len({position for position, _ in normalized}) != len(normalized):
        raise ValueError("deletion-cost positions must be unique")
    retained = tuple(
        sorted(
            position
            for position, _ in sorted(
                normalized, key=lambda item: (-item[1], item[0])
            )[:keep]
        )
    )
    risk = additive_retained_set_risk(dict(normalized), retained)
    return ProxyRetentionDecision(
        retained_positions=retained,
        risk=risk,
        retention_utility=float(sum(dict(normalized)[position] for position in retained)),
    )


def additive_proxy_regret(
    deletion_costs: Mapping[int, float],
    retained_positions: Iterable[int],
    budget: int,
) -> float:
    """Return the refresh statistic from the same proxy used for selection."""

    current = tuple(int(position) for position in retained_positions)
    if len(set(current)) != int(budget):
        raise ValueError("current retained set must match the configured budget")
    current_risk = additive_retained_set_risk(deletion_costs, current)
    optimal = select_additive_retained_set(deletion_costs, int(budget))
    regret = float(current_risk - optimal.risk)
    if regret < -1.0e-12:
        raise RuntimeError("proxy regret is negative despite exact minimization")
    return max(regret, 0.0)


def proxy_refresh_required(
    deletion_costs: Mapping[int, float],
    retained_positions: Iterable[int],
    budget: int,
    threshold: float,
) -> bool:
    """Refresh when additive proxy regret exceeds a non-negative threshold."""

    if not math.isfinite(float(threshold)) or float(threshold) < 0.0:
        raise ValueError("refresh threshold must be finite and non-negative")
    return additive_proxy_regret(
        deletion_costs, retained_positions, int(budget)
    ) > float(threshold)
