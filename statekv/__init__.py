"""State-conditioned KV-cache selection and refresh research package.

The package contains the reusable StateKV mechanism, evaluator, analysis, and
experiment primitives.  Protocol-aware eviction benchmarking remains in the
independent :mod:`kvbench` package.
"""

from statekv.config import DiscoveryConfig, load_discovery_config
from statekv.core import (
    SelectionDecision,
    functional_history_state,
    oracle_refresh_required,
    select_lowest_risk,
    set_level_attention_delta,
    state_conditioned_quadratic_risk,
)

__all__ = [
    "DiscoveryConfig",
    "SelectionDecision",
    "functional_history_state",
    "load_discovery_config",
    "oracle_refresh_required",
    "select_lowest_risk",
    "set_level_attention_delta",
    "state_conditioned_quadratic_risk",
]
