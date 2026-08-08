from __future__ import annotations

import pytest

from statekv.core.decision import (
    additive_proxy_regret,
    additive_retained_set_risk,
    proxy_refresh_required,
    select_additive_retained_set,
)


def test_additive_selection_minimizes_the_same_set_risk() -> None:
    costs = {0: 0.1, 1: 0.7, 2: 0.4, 3: 0.2}
    decision = select_additive_retained_set(costs, budget=2)
    assert decision.retained_positions == (1, 2)
    assert decision.risk == pytest.approx(0.3)
    assert decision.retention_utility == pytest.approx(1.1)


def test_proxy_regret_is_zero_for_current_optimum() -> None:
    costs = {0: 0.1, 1: 0.7, 2: 0.4, 3: 0.2}
    assert additive_proxy_regret(costs, (1, 2), budget=2) == pytest.approx(0.0)
    assert not proxy_refresh_required(costs, (1, 2), budget=2, threshold=0.0)


def test_proxy_regret_drives_refresh_without_a_second_signal() -> None:
    costs = {0: 0.1, 1: 0.7, 2: 0.4, 3: 0.2}
    assert additive_retained_set_risk(costs, (0, 3)) == pytest.approx(1.1)
    assert additive_proxy_regret(costs, (0, 3), budget=2) == pytest.approx(0.8)
    assert proxy_refresh_required(costs, (0, 3), budget=2, threshold=0.5)


def test_proxy_contract_rejects_invalid_costs_and_actions() -> None:
    with pytest.raises(ValueError):
        select_additive_retained_set({0: -1.0}, budget=1)
    with pytest.raises(ValueError):
        additive_proxy_regret({0: 1.0, 1: 0.5}, (0,), budget=2)
    with pytest.raises(ValueError):
        proxy_refresh_required({0: 1.0}, (0,), budget=1, threshold=-1.0)
