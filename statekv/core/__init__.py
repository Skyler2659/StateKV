"""Stable, paper-facing StateKV primitives.

The modules in :mod:`statekv.core` encode the state, action, risk, and
decision objects described by the StateKV formulation.  Model execution,
candidate generation, and frozen experiment protocols remain outside this
package.
"""

from statekv.core.actions import (
    functional_history_state,
    set_level_attention_delta,
)
from statekv.core.decision import (
    ProxyRetentionDecision,
    SelectionDecision,
    additive_proxy_regret,
    additive_retained_set_risk,
    oracle_refresh_required,
    proxy_refresh_required,
    select_additive_retained_set,
    select_lowest_risk,
)
from statekv.core.risk import (
    fisher_vector_product,
    midpoint_path_response,
    reference_kl,
    reference_kl_increment,
    state_conditioned_quadratic_risk,
)

__all__ = [
    "ProxyRetentionDecision",
    "SelectionDecision",
    "additive_proxy_regret",
    "additive_retained_set_risk",
    "fisher_vector_product",
    "functional_history_state",
    "midpoint_path_response",
    "oracle_refresh_required",
    "proxy_refresh_required",
    "reference_kl",
    "reference_kl_increment",
    "select_additive_retained_set",
    "select_lowest_risk",
    "set_level_attention_delta",
    "state_conditioned_quadratic_risk",
]
