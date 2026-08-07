import pytest

from statekv.core.decision import (
    oracle_refresh_required,
    select_lowest_risk,
)


def test_selection_is_deterministic_and_reports_margin() -> None:
    decision = select_lowest_risk({"candidate-b": 0.2, "candidate-a": 0.2, "c": 0.5})
    assert decision.candidate_id == "candidate-a"
    assert decision.margin == pytest.approx(0.0)
    assert decision.ordered_candidates == ("candidate-a", "candidate-b", "c")


def test_oracle_refresh_only_tracks_a_changed_argmin() -> None:
    previous = select_lowest_risk({"a": 0.1, "b": 0.2})
    assert not oracle_refresh_required(previous, {"a": 0.3, "b": 0.4})
    assert oracle_refresh_required(previous, {"a": 0.5, "b": 0.2})


def test_selection_rejects_non_finite_risk() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_lowest_risk({"a": float("nan")})
