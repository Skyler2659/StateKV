import pytest
import torch

from kvbench.cache.budget import BudgetAllocator, stable_topk
from kvbench.config import BudgetConfig
from kvbench.errors import BudgetError


def test_stable_topk_breaks_ties_by_lower_row():
    scores = torch.tensor([1.0, 3.0, 3.0, 3.0, 0.0])
    assert stable_topk(scores, 2).tolist() == [1, 2]


def test_total_budget_includes_sink_recent_and_current():
    allocator = BudgetAllocator(
        BudgetConfig(cache_budget=6, sink_size=2, recent_size=2, protect_current=True)
    )
    selection = allocator.select(torch.arange(10, dtype=torch.float32), "score")
    assert selection.rows == [0, 1, 6, 7, 8, 9]
    assert len(selection.rows) == 6
    assert set(selection.mandatory_rows) == {0, 1, 8, 9}


def test_budget_smaller_than_mandatory_fails_explicitly():
    allocator = BudgetAllocator(
        BudgetConfig(cache_budget=3, sink_size=2, recent_size=2, protect_current=True)
    )
    with pytest.raises(BudgetError, match="mandatory tokens"):
        allocator.select(torch.ones(8), "score")


def test_independent_overlap_is_recorded_and_backfilled():
    allocator = BudgetAllocator(
        BudgetConfig(cache_budget=5, sink_size=1, recent_size=0, protect_current=False)
    )
    attention = torch.tensor([0.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    leverage = torch.tensor([0.0, 9.0, 8.0, 1.0, 2.0, 3.0])
    selection = allocator.select_partitioned(
        {"attention": attention, "leverage": leverage},
        {"attention": 2, "leverage": 2},
        attention + leverage,
    )
    assert len(selection.rows) == 5
    assert selection.sources[1] == ["attention", "leverage"]
    assert any("backfill" in sources for sources in selection.sources.values())

