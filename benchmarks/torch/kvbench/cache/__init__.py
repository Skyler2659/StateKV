"""Cache accounting and backend-neutral selection decisions."""

from kvbench.cache.budget import BudgetAllocator, BudgetSelection, stable_topk

__all__ = ["BudgetAllocator", "BudgetSelection", "stable_topk"]

