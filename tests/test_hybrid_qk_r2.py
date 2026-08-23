import math

import numpy as np
import pytest

from statekv.causal_closed_loop import hybrid_trigger


def test_margin_trigger_fires_on_flat_scores() -> None:
    scores = np.full(64, 0.01, dtype=np.float64)
    fired, stat = hybrid_trigger(
        "margin", scores, k=16, cycle=3, cfg={"margin_threshold": 0.1}
    )
    assert fired
    assert stat == pytest.approx(0.0, abs=1e-6)


def test_margin_trigger_skips_peaked_scores() -> None:
    scores = np.concatenate(
        [np.full(16, 1.0), np.full(48, 0.001)]
    ).astype(np.float64)
    fired, stat = hybrid_trigger(
        "margin", scores, k=16, cycle=3, cfg={"margin_threshold": 0.1}
    )
    assert not fired
    assert stat > 0.1


def test_margin_threshold_is_strict() -> None:
    # Two scores [3, 1]: margin = 2 / (std + 1e-12) = 2 / (1 + 1e-12) < 2.
    fired, stat = hybrid_trigger(
        "margin", [3.0, 1.0], k=1, cycle=1, cfg={"margin_threshold": 2.0}
    )
    assert fired
    assert stat == pytest.approx(2.0, rel=1e-9)
    fired, _ = hybrid_trigger(
        "margin", [3.0, 1.0], k=1, cycle=1, cfg={"margin_threshold": 1.999}
    )
    assert not fired


def test_margin_without_boundary_position_never_triggers() -> None:
    # k == len(scores): no position beyond the budget, margin is infinite.
    fired, stat = hybrid_trigger(
        "margin", [0.5, 0.5], k=2, cycle=1, cfg={"margin_threshold": 0.1}
    )
    assert not fired
    assert math.isinf(stat)


def test_entropy_trigger_fires_on_flat_scores() -> None:
    scores = np.full(64, 0.01, dtype=np.float64)
    fired, stat = hybrid_trigger(
        "entropy", scores, k=16, cycle=3, cfg={"entropy_threshold": 0.95}
    )
    assert fired
    assert stat == pytest.approx(1.0, rel=1e-6)


def test_entropy_trigger_skips_peaked_scores() -> None:
    scores = np.full(64, 0.001, dtype=np.float64)
    scores[0] = 50.0
    fired, stat = hybrid_trigger(
        "entropy", scores, k=16, cycle=3, cfg={"entropy_threshold": 0.95}
    )
    assert not fired
    assert stat < 0.95


def test_periodic_margin_fires_on_schedule_regardless_of_scores() -> None:
    peaked = np.concatenate(
        [np.full(16, 1.0), np.full(48, 0.001)]
    ).astype(np.float64)
    cfg = {"margin_threshold": 0.1, "base_refresh": 8}
    fired, _ = hybrid_trigger("periodic_margin", peaked, k=16, cycle=16, cfg=cfg)
    assert fired
    fired, _ = hybrid_trigger("periodic_margin", peaked, k=16, cycle=9, cfg=cfg)
    assert not fired
    # Cycle 0 is on schedule (0 % base_refresh == 0).
    fired, _ = hybrid_trigger("periodic_margin", peaked, k=16, cycle=0, cfg=cfg)
    assert fired


def test_periodic_margin_also_fires_on_uncertain_margin_off_schedule() -> None:
    flat = np.full(64, 0.01, dtype=np.float64)
    cfg = {"margin_threshold": 0.1, "base_refresh": 8}
    fired, stat = hybrid_trigger("periodic_margin", flat, k=16, cycle=9, cfg=cfg)
    assert fired
    assert stat == pytest.approx(0.0, abs=1e-6)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown hybrid trigger mode"):
        hybrid_trigger("bogus", [1.0, 0.5], k=1, cycle=0, cfg={})


def test_defaults_match_config_contract() -> None:
    # Empty cfg must behave like the documented defaults.
    flat = np.full(64, 0.01, dtype=np.float64)
    fired, _ = hybrid_trigger("margin", flat, k=16, cycle=3, cfg={})
    assert fired
    peaked = np.concatenate(
        [np.full(16, 1.0), np.full(48, 0.001)]
    ).astype(np.float64)
    fired, _ = hybrid_trigger("margin", peaked, k=16, cycle=3, cfg={})
    assert not fired
