"""One total-budget contract shared by every eviction method."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch

from kvbench.config import BudgetConfig
from kvbench.errors import BudgetError


def stable_topk(
    scores: torch.Tensor,
    k: int,
    eligible: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return highest-score rows with original row index as the tie breaker."""
    flat = scores.detach().float().flatten()
    if eligible is None:
        eligible = torch.arange(flat.numel(), device=flat.device, dtype=torch.long)
    else:
        eligible = eligible.to(flat.device, dtype=torch.long).flatten()
    if k <= 0 or eligible.numel() == 0:
        return torch.empty(0, device=flat.device, dtype=torch.long)
    if eligible.min().item() < 0 or eligible.max().item() >= flat.numel():
        raise BudgetError("eligible row is outside the score universe")
    values = flat.index_select(0, eligible)
    finite = torch.isfinite(values)
    eligible = eligible[finite]
    values = values[finite]
    if eligible.numel() == 0:
        return torch.empty(0, device=flat.device, dtype=torch.long)
    # Stable descending sort preserves ascending eligible row order for ties.
    order_by_row = torch.argsort(eligible, stable=True)
    eligible = eligible[order_by_row]
    values = values[order_by_row]
    order = torch.argsort(values, descending=True, stable=True)
    return eligible[order[: min(int(k), int(order.numel()))]].sort().values


@dataclass
class BudgetSelection:
    rows: List[int]
    mandatory_rows: List[int]
    selectable_budget: int
    sources: Dict[int, List[str]] = field(default_factory=dict)


class BudgetAllocator:
    """Allocate one strict total token budget for all methods.

    Sink, recent, and current rows all count toward ``cache_budget``. If those
    mandatory policies cannot fit, the run fails instead of silently granting a
    method extra cache or truncating a protected class.
    """

    def __init__(self, cfg: BudgetConfig):
        self.cfg = cfg

    def mandatory_rows(self, seq_len: int) -> List[int]:
        if seq_len < 0:
            raise BudgetError("seq_len must be non-negative")
        sink = list(range(min(int(self.cfg.sink_size), seq_len)))
        recent_count = min(int(self.cfg.recent_size), seq_len)
        recent = list(range(max(0, seq_len - recent_count), seq_len))
        current = [seq_len - 1] if self.cfg.protect_current and seq_len else []
        mandatory = sorted(set(sink + recent + current))
        if len(mandatory) > int(self.cfg.cache_budget):
            raise BudgetError(
                "mandatory tokens (%d) exceed total cache budget (%d); "
                "reduce sink/recent or increase cache_budget"
                % (len(mandatory), int(self.cfg.cache_budget))
            )
        return mandatory

    def select(
        self,
        scores: torch.Tensor,
        source: str,
        budget: Optional[int] = None,
        exclude: Optional[Iterable[int]] = None,
    ) -> BudgetSelection:
        seq_len = int(scores.numel())
        total_budget = min(seq_len, int(self.cfg.cache_budget if budget is None else budget))
        mandatory = self.mandatory_rows(seq_len)
        if len(mandatory) > total_budget:
            raise BudgetError("mandatory tokens exceed the requested effective budget")
        excluded: Set[int] = set(int(row) for row in (exclude or []))
        excluded.update(mandatory)
        candidates = torch.tensor(
            [row for row in range(seq_len) if row not in excluded],
            device=scores.device,
            dtype=torch.long,
        )
        selectable = total_budget - len(mandatory)
        chosen = stable_topk(scores, selectable, candidates).tolist()
        rows = sorted(set(mandatory + [int(row) for row in chosen]))
        if len(rows) != total_budget:
            raise BudgetError(
                "selection underfilled: selected=%d expected=%d" % (len(rows), total_budget)
            )
        sources: Dict[int, List[str]] = {}
        for row in mandatory:
            sources.setdefault(row, []).append("mandatory")
        for row in chosen:
            sources.setdefault(int(row), []).append(source)
        self.validate(rows, seq_len, total_budget)
        return BudgetSelection(rows, mandatory, selectable, sources)

    def select_partitioned(
        self,
        component_scores: Dict[str, torch.Tensor],
        component_counts: Dict[str, int],
        backfill_scores: torch.Tensor,
    ) -> BudgetSelection:
        if not component_scores:
            raise BudgetError("partitioned selection requires component scores")
        seq_lengths = {int(score.numel()) for score in component_scores.values()}
        seq_lengths.add(int(backfill_scores.numel()))
        if len(seq_lengths) != 1:
            raise BudgetError("component score universes do not align")
        seq_len = seq_lengths.pop()
        total_budget = min(seq_len, int(self.cfg.cache_budget))
        mandatory = self.mandatory_rows(seq_len)
        selectable = total_budget - len(mandatory)
        requested_total = sum(max(0, int(value)) for value in component_counts.values())
        if requested_total > selectable:
            raise BudgetError(
                "partition counts (%d) exceed selectable budget (%d)"
                % (requested_total, selectable)
            )
        selected: Set[int] = set(mandatory)
        sources: Dict[int, List[str]] = {row: ["mandatory"] for row in mandatory}

        # Select every component against the same candidate universe.  This is
        # a true independent split: overlaps remain visible in ``sources`` and
        # are filled deterministically from the shared backfill score below.
        for name, requested in component_counts.items():
            score = component_scores[name]
            eligible = torch.tensor(
                [row for row in range(seq_len) if row not in set(mandatory)],
                device=score.device,
                dtype=torch.long,
            )
            for row in stable_topk(score, max(0, int(requested)), eligible).tolist():
                selected.add(int(row))
                sources.setdefault(int(row), []).append(name)

        remaining = total_budget - len(selected)
        if remaining > 0:
            eligible = torch.tensor(
                [row for row in range(seq_len) if row not in selected],
                device=backfill_scores.device,
                dtype=torch.long,
            )
            for row in stable_topk(backfill_scores, remaining, eligible).tolist():
                selected.add(int(row))
                sources.setdefault(int(row), []).append("backfill")

        rows = sorted(selected)
        self.validate(rows, seq_len, total_budget)
        return BudgetSelection(rows, mandatory, selectable, sources)

    @staticmethod
    def validate(rows: Sequence[int], seq_len: int, expected: int) -> None:
        normalized = [int(row) for row in rows]
        if normalized != sorted(set(normalized)):
            raise BudgetError("selected rows must be unique and sorted")
        if normalized and (normalized[0] < 0 or normalized[-1] >= seq_len):
            raise BudgetError("selected row is outside the cache")
        if len(normalized) != int(expected):
            raise BudgetError(
                "actual retained count %d does not equal effective budget %d"
                % (len(normalized), int(expected))
            )
