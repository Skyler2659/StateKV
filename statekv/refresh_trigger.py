"""Frozen cheap-feature refresh trigger for the R2b selective-refresh gate.

A trigger rule is a conjunction of one or two clauses
``{"feature": <name>, "op": ">="|"<=", "threshold": <float>}`` loaded from a
JSON file.  The rule may ONLY reference online-computable cheap features; the
allowlist below is enforced at parse time so the trigger can never see
teacher-side quantities (exact_kl, stale_exact_kl, full-softmax telemetry,
refresh-benefit labels, ...).

Rule file schema::

    {
      "name": "r2b_trigger_v1",            # optional
      "clauses": [
        {"feature": "churn_jaccard_mean", "op": ">=", "threshold": 0.4},
        {"feature": "boundary_margin_mean", "op": "<=", "threshold": 0.05}
      ],
      ...                                   # any extra provenance keys kept
    }

NaN feature values never fire a clause.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# The only features a frozen trigger may consume.  All four are computed from
# the controller's own scores/selections and never touch the Full-KV teacher.
CHEAP_TRIGGER_FEATURES: Tuple[str, ...] = (
    "churn_jaccard_mean",
    "boundary_margin_mean",
    "score_tv_mean",
    "coverage_mass_mean",
)

_OPERATORS = (">=", "<=")


@dataclass(frozen=True)
class TriggerClause:
    feature: str
    op: str
    threshold: float

    def fires(self, value: float) -> bool:
        value = float(value)
        if value != value:  # NaN never alerts
            return False
        if self.op == ">=":
            return bool(value >= self.threshold)
        return bool(value <= self.threshold)


@dataclass(frozen=True)
class TriggerRule:
    clauses: Tuple[TriggerClause, ...]
    name: str = "unnamed"
    provenance: Mapping[str, Any] = None  # type: ignore[assignment]

    def evaluate(self, features: Mapping[str, float]) -> bool:
        """Conjunction of all clauses over the current step's features."""
        for clause in self.clauses:
            if clause.feature not in features:
                raise KeyError(f"trigger feature {clause.feature} was not computed")
            if not clause.fires(float(features[clause.feature])):
                return False
        return True


def _parse_clause(raw: Mapping[str, Any]) -> TriggerClause:
    if not isinstance(raw, Mapping):
        raise ValueError(f"trigger clause must be a mapping, got {type(raw).__name__}")
    feature = str(raw.get("feature", ""))
    if feature not in CHEAP_TRIGGER_FEATURES:
        raise ValueError(
            f"trigger clause feature {feature!r} is not an online-computable cheap "
            f"feature; allowlist={list(CHEAP_TRIGGER_FEATURES)} (teacher-side "
            "quantities are structurally forbidden)"
        )
    op = str(raw.get("op", ""))
    if op not in _OPERATORS:
        raise ValueError(f"trigger clause op must be one of {_OPERATORS}, got {op!r}")
    try:
        threshold = float(raw["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"trigger clause threshold must be numeric: {raw!r}") from exc
    return TriggerClause(feature=feature, op=op, threshold=threshold)


def decide_trigger_refresh(
    rule: TriggerRule, features: Mapping[str, float], cycle: int
) -> Tuple[bool, bool]:
    """Trigger-arm scheduling: cycle 0 always refreshes; later cycles refresh
    iff the frozen rule fires.  Returns (trigger_fired, refresh)."""
    fired = bool(int(cycle) > 0 and rule.evaluate(features))
    return fired, bool(int(cycle) == 0 or fired)


def load_trigger_rule(source: Any) -> TriggerRule:
    """Parse a frozen trigger rule from a JSON path or an already-loaded dict."""
    if isinstance(source, (str, Path)):
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    elif isinstance(source, Mapping):
        payload = dict(source)
    else:
        raise ValueError(f"cannot load a trigger rule from {type(source).__name__}")
    if not isinstance(payload, Mapping):
        raise ValueError("trigger rule JSON must be an object")
    raw_clauses = payload.get("clauses")
    if not isinstance(raw_clauses, Sequence) or isinstance(raw_clauses, (str, bytes)):
        raise ValueError("trigger rule JSON must contain a 'clauses' list")
    if not 1 <= len(raw_clauses) <= 2:
        raise ValueError(
            f"a trigger rule is a conjunction of one or two clauses, got {len(raw_clauses)}"
        )
    clauses = tuple(_parse_clause(raw) for raw in raw_clauses)
    provenance = {key: value for key, value in payload.items() if key not in ("clauses", "name")}
    return TriggerRule(
        clauses=clauses,
        name=str(payload.get("name", "unnamed")),
        provenance=provenance,
    )


__all__ = [
    "CHEAP_TRIGGER_FEATURES",
    "TriggerClause",
    "TriggerRule",
    "decide_trigger_refresh",
    "load_trigger_rule",
]
