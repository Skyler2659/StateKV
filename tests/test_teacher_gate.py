"""Unit tests for the strict-pure-eviction teacher panel (Gate 0/1).

Covers the panel evaluator selection logic, the stale_prev candidate,
deterministic tie-breaking, the teacher-mode validation guards, and the
"uniform" control policy.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pytest
import torch

from statekv.budget_dynamics import (
    DirectBudgetController,
    LayerCacheView,
)
from statekv.config import CacheDiscoveryConfig
from statekv.core.risk import reference_kl
from statekv.oracle_policy_comparison import AttentionPolicyMemory
from statekv.selectors import CoreSelection, LayerSelection
from statekv.statekv_gate_runner import (
    _teacher_panel_decision,
    run_pure_eviction_policy,
)


class _FakeDecision:
    """Minimal BudgetDecision stand-in: a CoreSelection + requested budgets."""

    def __init__(self, strategy: str, requested: int = 3) -> None:
        self.selection = CoreSelection(
            strategy=strategy,
            horizon_condition=None,
            by_layer={
                0: LayerSelection(
                    layer=0,
                    selected_positions=[1, 2, 3],
                    eligible_positions=[0, 1, 2, 3, 4],
                    aggregate_scores=[],
                )
            },
            metadata={},
        )
        self.requested_budgets = {0: int(requested)}


class _FakeRunner:
    """Wraps a fake model behind the runner.model interface."""

    def __init__(self, model: "_FakeModel") -> None:
        self.model = model


class _FakeModel:
    """Deterministic fake: logits per candidate encode a controllable KL."""

    def __init__(self, logits_by_name: Mapping[str, torch.Tensor]) -> None:
        self.logits_by_name = dict(logits_by_name)
        self.order: List[str] = []
        self.forwarded: List[str] = []

    def shallow_clone_state(self, state: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            source=state,
            position_maps={
                0: torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
            },
        )

    def apply_selection_in_place(
        self, clone: Any, selection: Any, cache_config: Any = None
    ) -> None:
        pass

    def forward_one(
        self, clone: Any, token: int, capture_attention: bool = False
    ) -> Tuple[torch.Tensor, Any, float]:
        # _teacher_panel_decision evaluates candidates in insertion order.
        name = self.order[len(self.forwarded)]
        self.forwarded.append(name)
        return self.logits_by_name[name], object(), 0.001


def _full_logits() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 8)


def _perturbed(coefficient: float) -> torch.Tensor:
    """Rank-one perturbation whose KL over the full distribution grows with c."""
    full = _full_logits()
    direction = torch.linspace(-1.0, 1.0, 8)
    return full + coefficient * direction


def _panel_logits() -> Dict[str, torch.Tensor]:
    return {
        "attention": _perturbed(0.05),
        "b2_uniform": _perturbed(0.05),
        "a2_temporal_volatility": _perturbed(0.30),
        "uniform": _perturbed(0.10),
        "snapkv": _perturbed(0.60),
    }


def _candidates() -> Dict[str, _FakeDecision]:
    return {
        name: _FakeDecision(name)
        for name in (
            "attention",
            "b2_uniform",
            "a2_temporal_volatility",
            "uniform",
            "snapkv",
        )
    }


def _rolling() -> CacheDiscoveryConfig:
    return CacheDiscoveryConfig(
        total_budget=8, sink_size=1, recent_size=2, selected_core_budget=3
    )


def test_teacher_picks_minimum_kl_candidate() -> None:
    full = _full_logits()
    logits = _panel_logits()
    model = _FakeModel(logits)
    model.order = list(logits)
    candidates = _candidates()
    decision, rows = _teacher_panel_decision(
        _FakeRunner(model), object(), 0, full, candidates, _rolling()
    )
    # attention and b2_uniform tie at the lowest risk; alphabetical wins.
    assert decision is candidates["attention"]
    assert len(rows) == 5
    assert rows[0]["candidate"] == "attention"
    assert rows[0]["selected"] is True
    assert rows[0]["risk_rank"] == 0
    assert rows[1]["candidate"] == "b2_uniform"
    assert rows[1]["risk_rank"] == 1
    expected_kl = {
        name: float(reference_kl(full, logits[name]).item())
        for name in logits
    }
    for row in rows:
        assert abs(row["exact_kl"] - expected_kl[row["candidate"]]) < 1.0e-9
    assert model.forwarded == model.order
    # ordering by risk rank matches the KL ordering
    ranked = sorted(rows, key=lambda row: row["risk_rank"])
    assert [row["candidate"] for row in ranked] == [
        "attention",
        "b2_uniform",
        "uniform",
        "a2_temporal_volatility",
        "snapkv",
    ]


def test_teacher_selection_follows_actual_kl_not_policy_order() -> None:
    full = _full_logits()
    logits = _panel_logits()
    # snapkv becomes the best action under this panel
    logits["snapkv"] = _perturbed(0.01)
    model = _FakeModel(logits)
    model.order = list(logits)
    decision, rows = _teacher_panel_decision(
        _FakeRunner(model), object(), 0, full, _candidates(), _rolling()
    )
    assert decision.selection.strategy == "snapkv"
    assert next(row for row in rows if row["candidate"] == "snapkv")["selected"]


def test_stale_prev_candidate_included_when_previous_decision_given() -> None:
    full = _full_logits()
    logits = _panel_logits()
    logits["stale_prev"] = _perturbed(0.02)  # stale_prev is the best action
    model = _FakeModel(logits)
    model.order = list(logits)
    previous = _FakeDecision("attention")
    decision, rows = _teacher_panel_decision(
        _FakeRunner(model), object(), 0, full, _candidates(), _rolling(),
        previous_decision=previous,
    )
    assert decision is previous
    assert rows[-1]["candidate"] == "stale_prev"
    assert rows[-1]["selected"] is True
    assert rows[-1]["risk_rank"] == 0


def test_stale_prev_absent_without_previous_decision() -> None:
    full = _full_logits()
    model = _FakeModel(_panel_logits())
    model.order = list(_panel_logits())
    _, rows = _teacher_panel_decision(
        _FakeRunner(model), object(), 0, full, _candidates(), _rolling()
    )
    assert all(row["candidate"] != "stale_prev" for row in rows)


def test_teacher_mode_requires_panel_and_full_evaluator() -> None:
    args = dict(
        runner=None,
        reference=None,
        sample=None,
        policy="teacher_panel",
        controller=None,
        cycles=1,
        monitor_labels=[],
        evidence_positions=[],
    )
    with pytest.raises(ValueError, match="panel_policies"):
        run_pure_eviction_policy(
            **args, evaluate_exact_kl=True, refresh_mode="teacher"
        )
    with pytest.raises(ValueError, match="Full-KV evaluator"):
        run_pure_eviction_policy(
            **args,
            evaluate_exact_kl=False,
            refresh_mode="teacher",
            panel_policies=["attention"],
        )
    with pytest.raises(ValueError, match="label_mode"):
        run_pure_eviction_policy(
            **args,
            evaluate_exact_kl=True,
            refresh_mode="teacher",
            panel_policies=["attention"],
            label_mode=True,
        )
    with pytest.raises(ValueError, match="telemetry"):
        run_pure_eviction_policy(
            **args,
            evaluate_exact_kl=True,
            refresh_mode="teacher",
            panel_policies=["attention"],
            collect_diagnostic_telemetry=True,
        )


def _memory() -> AttentionPolicyMemory:
    memory = AttentionPolicyMemory((0, 1), 4)
    for layer in (0, 1):
        values = [0.01, 0.85, 0.04, 0.04, 0.04, 0.02]
        memory.latest[layer] = dict(enumerate(values))
        memory.latest_by_head[layer] = {
            position: np.asarray([value, value], dtype=np.float64)
            for position, value in enumerate(values)
        }
        memory.cumulative[layer] = dict(memory.latest[layer])
        memory.window[layer] = [
            {position: value * 0.5 for position, value in enumerate(values)},
            dict(memory.latest[layer]),
        ]
    return memory


def _view() -> LayerCacheView:
    positions = tuple(range(6))
    return LayerCacheView(
        positions_by_layer={layer: positions for layer in (0, 1)},
        values_by_layer={
            layer: torch.arange(6, dtype=torch.float32).reshape(1, 1, 6, 1)
            for layer in (0, 1)
        },
    )


def test_uniform_control_policy_is_selectable() -> None:
    controller = DirectBudgetController(
        core_budget=2,
        sink_size=1,
        recent_size=1,
        pooling_kernel=1,
        pooling_method="max",
        maximum_delta=1,
    )
    decision = controller.select("uniform", _memory(), _view(), 0, "s")
    assert decision.requested_budgets == {0: 2, 1: 2}
    for layer in (0, 1):
        positions = decision.selection.by_layer[layer].selected_positions
        assert len(positions) == 2
        assert set(positions) <= set(_view().positions_by_layer[layer])


class _LadderFakeModel(_FakeModel):
    """Fake with prune support for the multi-step rollout."""

    def prune_recent_before_query(self, state, fixed, cache_config=None):
        pass


class _LadderReference:
    def __init__(self, tokens, probe_logits):
        self.generated_token_ids = list(tokens)
        self.probe_logits = {i: probe_logits[i] for i in range(len(tokens))}


def _ladder_candidate(strategy="attention"):
    return _FakeDecision(strategy)


def test_ladder_rollout_records_all_horizons() -> None:
    from statekv.statekv_gate_runner import _ladder_rollout

    # reference: logits that the candidates never match at deep steps
    n = 12
    tokens = list(range(n))
    direction = torch.linspace(-1.0, 1.0, 8)
    probes = {i: torch.linspace(-1.0, 1.0, 8) + 0.01 * i * direction for i in range(n)}
    ref = _LadderReference(tokens, probes)
    model = _LadderFakeModel({})
    model.order = ["attention"]  # not used; logits fixed below
    # candidate logits: closer to probe at step 1, diverging later
    model.logits_by_name = {"attention": _perturbed(0.05)}

    class _StatefulModel(_LadderFakeModel):
        def forward_one(self, clone, token, capture_attention=False):
            # step 1: near-identical logits; later steps: diverged
            name = self.order[min(len(self.forwarded), len(self.order) - 1)]
            self.forwarded.append(name)
            if len(self.forwarded) == 1:
                return probes[0].clone(), object(), 0.001
            return probes[1] + 3.0 * direction, object(), 0.001

    model = _StatefulModel({})
    model.order = ["attention"]
    rolling = CacheDiscoveryConfig(
        total_budget=8, sink_size=1, recent_size=2, selected_core_budget=3
    )
    rows = _ladder_rollout(
        _FakeRunner(model), object(), 7, ref, _ladder_candidate(), rolling,
        sink_size=1, horizons=[1, 2], probe_offset=1,
    )
    assert [row["horizon"] for row in rows] == [1, 2]
    # step 1 nearly matches the probe -> KL tiny
    assert rows[0]["exact_kl"] < 1.0e-4
    assert rows[0]["cumulative_kl"] < 1.0e-4
    # step 2 diverges -> larger KL, cumulative rises
    assert rows[1]["exact_kl"] > rows[0]["exact_kl"]
    assert rows[1]["cumulative_kl"] > rows[0]["cumulative_kl"]


def test_swap_selection_replaces_exactly_one_layer() -> None:
    from statekv.statekv_gate_runner import _swap_selection

    base = _FakeDecision("attention")
    # base has only layer 0 with [1,2,3]
    swapped = _swap_selection(base, 0, [1], [9])
    assert sorted(swapped.by_layer[0].selected_positions) == [2, 3, 9]
    assert swapped.by_layer[0] is not base.selection.by_layer[0]
    # other layers untouched (none here) and budgets preserved
    assert len(swapped.by_layer[0].selected_positions) == len(
        base.selection.by_layer[0].selected_positions
    )
