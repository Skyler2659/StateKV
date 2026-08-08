import numpy as np

import pandas as pd

from statekv.direct_policy_analysis import (
    paired_bootstrap_mean,
    paired_tail_migration,
    select_direct_policy_candidate,
    select_protected_rescue_candidate,
)


def test_paired_bootstrap_mean_is_deterministic_and_order_invariant() -> None:
    values = np.asarray([0.1, 0.2, -0.01, 0.05])
    first = paired_bootstrap_mean(values, 2000, 7)
    second = paired_bootstrap_mean(values[::-1], 2000, 7)
    repeated = paired_bootstrap_mean(values, 2000, 7)
    assert first == repeated
    assert first[0] > -0.02
    assert second[0] > -0.02


def test_paired_tail_migration_counts_created_and_escaped_steps() -> None:
    rows = []
    for policy, values in (("base", [1.0, 2.0, 3.0, 4.0]), ("primary", [1.0, 2.0, 5.0, 3.0])):
        for offset, value in enumerate(values, start=1):
            rows.append(
                {
                    "sample_id": "sample",
                    "task": "task",
                    "anchor": 16,
                    "horizon_offset": offset,
                    "policy": policy,
                    "exact_kl": value,
                }
            )
    summary, paired = paired_tail_migration(
        pd.DataFrame(rows), "base", "primary", quantile=0.75
    )
    assert summary["escaped_tail_steps"] == 1
    assert summary["new_primary_tail_steps"] == 1
    assert set(paired["tail_category"]) >= {"escaped_tail", "new_primary_tail"}


def test_protected_rescue_selection_requires_every_constraint() -> None:
    metrics = pd.DataFrame(
        [
            {
                "policy": "base",
                "mean_exact_kl": 1.0,
                "p95_exact_kl": 2.0,
                "cvar95_exact_kl": 3.0,
                "maximum_exact_kl": 4.0,
                "large_loss_rate": 0.1,
            },
            {
                "policy": "m4",
                "mean_exact_kl": 0.9,
                "p95_exact_kl": 1.9,
                "cvar95_exact_kl": 3.1,
                "maximum_exact_kl": 3.9,
                "large_loss_rate": 0.09,
            },
        ]
    )
    stratified = pd.DataFrame(
        [
            {"stratum": "task", "value": "a", "policy": "base", "mean_exact_kl": 1.0},
            {"stratum": "task", "value": "a", "policy": "m4", "mean_exact_kl": 0.8},
        ]
    )
    inventory = pd.DataFrame(
        [{"policy": "m4", "core_changes_vs_attention": 4}]
    )
    result, audit = select_protected_rescue_candidate(
        metrics, stratified, inventory, "base", ["m4"], {"m4": 4}
    )
    assert not audit.iloc[0]["cvar95_exact_kl_nonworse"]
    assert not audit.iloc[0]["eligible"]
    assert result["selected_policy"] is None
    assert not result["independent_run_authorized"]


def test_direct_policy_selection_accepts_a_distinct_signal_family() -> None:
    metrics = pd.DataFrame(
        [
            {
                "policy": "base",
                "mean_exact_kl": 1.0,
                "p95_exact_kl": 2.0,
                "cvar95_exact_kl": 3.0,
                "maximum_exact_kl": 4.0,
                "large_loss_rate": 0.1,
            },
            {
                "policy": "geometry",
                "mean_exact_kl": 0.8,
                "p95_exact_kl": 1.9,
                "cvar95_exact_kl": 2.9,
                "maximum_exact_kl": 3.9,
                "large_loss_rate": 0.09,
            },
        ]
    )
    stratified = pd.DataFrame(
        [
            {"stratum": "task", "value": "a", "policy": "base", "mean_exact_kl": 1.0},
            {"stratum": "task", "value": "a", "policy": "geometry", "mean_exact_kl": 0.8},
            {"stratum": "task", "value": "b", "policy": "base", "mean_exact_kl": 0.5},
            {"stratum": "task", "value": "b", "policy": "geometry", "mean_exact_kl": 0.4},
        ]
    )
    result, audit = select_direct_policy_candidate(
        metrics, stratified, "base", ["geometry"]
    )
    assert bool(audit.iloc[0]["eligible"])
    assert result["selected_policy"] == "geometry"
    assert result["independent_run_authorized"]
