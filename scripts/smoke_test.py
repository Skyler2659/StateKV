#!/usr/bin/env python3
"""Dependency-light smoke test for the StateKV research path."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statekv.config import load_discovery_config  # noqa: E402
from statekv.core import (  # noqa: E402
    functional_history_state,
    select_lowest_risk,
    set_level_attention_delta,
    state_conditioned_quadratic_risk,
)
from statekv.functional_features import (  # noqa: E402
    LayerFeatureMatrices,
    functional_measurement,
)
from statekv.metrics import approximate_kl, loss_shape  # noqa: E402
from statekv.selectors import mandatory_and_eligible, ridge_leverage  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    config = load_discovery_config(
        str(ROOT / "configs/discovery/smoke.yaml")
    )

    positions = list(range(20))
    sink, recent, eligible = mandatory_and_eligible(
        positions, sink_size=2, recent_size=3
    )
    assert sink == [0, 1]
    assert recent == [17, 18, 19]
    assert eligible == list(range(2, 17))

    values = torch.randn(12, 4)
    scores, diagnostics = ridge_leverage(values, 1.0e-3, "relative")
    assert scores.shape == (12,)
    assert torch.isfinite(scores).all()
    assert diagnostics["ridge"] > 0

    base_matrix = torch.randn(8, 5)
    current_matrix = torch.cat([base_matrix, torch.randn(2, 5)], dim=0)
    key = ("raw_v", "layer", None)
    base = LayerFeatureMatrices(
        layer=0,
        positions=list(range(8)),
        matrices={key: base_matrix},
        observation_window_queries=4,
        observation_weight_source="smoke",
    )
    current = LayerFeatureMatrices(
        layer=0,
        positions=list(range(10)),
        matrices={key: current_matrix},
        observation_window_queries=4,
        observation_weight_source="smoke",
    )
    measurement = functional_measurement(
        base_features=base,
        current_features=current,
        key=key,
        old_history_positions={0, 1, 3, 6},
        fresh_history_positions={0, 1, 3, 6},
        base_old_history_positions={0, 1, 3, 6},
        epsilon=1.0e-12,
        ridge_coefficient=1.0e-3,
        ridge_mode="relative",
    )
    assert measurement["delta_e"]["raw_sum"] == 0.0
    assert measurement["arrival_residual"]["token_count"] == 2

    logits = torch.tensor([1.0, 0.5, -0.25, -1.0])
    probabilities = torch.softmax(logits.double(), dim=0)
    top_probabilities, top_ids = torch.topk(probabilities, 2)
    risk = approximate_kl(top_ids, top_probabilities, logits)
    assert abs(risk["approx_kl"]) < 1.0e-10

    curve = loss_shape([0.01, 0.02, 0.30, 0.05], 0.25)
    assert curve["first_large_loss_spike"] == 3
    refresh_offsets = [int(round((index + 1) * 16 / 4)) for index in range(3)]
    assert refresh_offsets == [4, 8, 12]

    reference_boundary = torch.tensor([0.0, 1.0], dtype=torch.float64)
    history_boundary = torch.tensor([0.25, 0.75], dtype=torch.float64)
    state = functional_history_state(history_boundary, reference_boundary)
    torch.testing.assert_close(
        state, torch.tensor([0.25, -0.25], dtype=torch.float64)
    )

    attention = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    values = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=torch.float64
    )
    action_delta = set_level_attention_delta(attention, values, [0, 2])
    retained_output = (0.2 * values[0] + 0.5 * values[2]) / 0.7
    full_output = torch.sum(attention.unsqueeze(-1) * values, dim=0)
    torch.testing.assert_close(action_delta, retained_output - full_output)

    local_risk = state_conditioned_quadratic_risk(
        torch.tensor([0.5, -0.5], dtype=torch.float64),
        torch.tensor([0.4, -0.4], dtype=torch.float64),
        torch.tensor([0.02, -0.02], dtype=torch.float64),
    )
    assert torch.isfinite(local_risk)
    decision = select_lowest_risk(
        {"attention": 0.2, "statekv-teacher": float(local_risk)}
    )
    assert decision.candidate_id == "statekv-teacher"

    print(
        json.dumps(
            {
                "status": "ok",
                "config": config.experiment_name,
                "eligible_tokens": len(eligible),
                "ridge_effective_dimension": diagnostics[
                    "effective_dimension"
                ],
                "arrival_tokens": measurement["arrival_residual"][
                    "token_count"
                ],
                "core_api": "ok",
                "refresh_offsets": refresh_offsets,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
