import pandas as pd

from statekv.direct_policy_trigger import _activation, _conditional_rows


def test_activation_directions_are_complementary_away_from_threshold() -> None:
    features = pd.DataFrame({"signal": [0.1, 0.4, 0.9]})
    high = _activation(
        features,
        {"feature": "signal", "direction": "high", "threshold": 0.5},
    )
    low = _activation(
        features,
        {"feature": "signal", "direction": "low", "threshold": 0.5},
    )
    assert high.tolist() == [False, False, True]
    assert low.tolist() == [True, True, False]


def test_conditional_rows_select_exactly_one_source_action_per_unit() -> None:
    rows = []
    for policy, shift in (("base", 0.0), ("alternative", -0.2)):
        for sample, anchor in (("a", 16), ("b", 32)):
            for horizon in (1, 2):
                rows.append(
                    {
                        "sample_id": sample,
                        "task": "task",
                        "anchor": anchor,
                        "horizon_offset": horizon,
                        "policy": policy,
                        "exact_kl": float(horizon + shift),
                    }
                )
    replay = pd.DataFrame(rows)
    features = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "task": ["task", "task"],
            "anchor": [16, 32],
            "signal": [0.9, 0.1],
        }
    )
    selected, activation = _conditional_rows(
        replay,
        features,
        "base",
        "alternative",
        {
            "name": "selective",
            "feature": "signal",
            "direction": "high",
            "threshold": 0.5,
        },
    )
    assert len(selected) == 4
    assert selected[selected["sample_id"] == "a"]["exact_kl"].tolist() == [0.8, 1.8]
    assert selected[selected["sample_id"] == "b"]["exact_kl"].tolist() == [1.0, 2.0]
    assert activation["activated"].tolist() == [True, False]
