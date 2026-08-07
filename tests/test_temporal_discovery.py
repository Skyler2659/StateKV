import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from kvbench.backends.huggingface import AttentionAccumulator, HFCacheState
from statekv.backend import AnchorState, QueryRecord, TemporalModel
from statekv.metrics import approximate_kl, loss_shape
from statekv.selectors import (
    CoreSelection,
    LayerSelection,
    fit_online_ridge_factor,
    mandatory_and_eligible,
    ridge_leverage,
)
from statekv.config import DiscoveryConfig, MechanismDiscoveryConfig
from statekv.signals import candidate_layer_records
from kvbench.types import AttentionSignals
from src.runners.mlx_runner import snapkv_pool_scores_numpy


def _tiny_temporal_model():
    model = TemporalModel.__new__(TemporalModel)
    model.cfg = SimpleNamespace(
        cache=SimpleNamespace(
            total_budget=8,
            sink_size=2,
            recent_size=2,
            selected_core_budget=4,
        )
    )
    model.backend = SimpleNamespace(
        device=torch.device("cpu"),
        _new_accumulator=lambda: AttentionAccumulator(1, 2),
    )
    return model


def _tiny_anchor():
    key = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20, 1)
    value = key + 100
    return AnchorState(
        anchor_step=0,
        logical_length=20,
        query_token_id=19,
        keys=[key],
        values=[value],
        position_maps={0: torch.arange(20)},
        attention=AttentionSignals(),
    )


def _tiny_selection():
    layer = LayerSelection(
        layer=0,
        selected_positions=[2, 3, 4, 5],
        eligible_positions=list(range(2, 18)),
        aggregate_scores=[float(index) for index in range(20)],
    )
    return CoreSelection("synthetic", None, {0: layer})


def test_tiny_budget_core_freeze_and_recent_fifo():
    model = _tiny_temporal_model()
    state, fixed = model.state_from_anchor(_tiny_anchor(), _tiny_selection())
    assert state.position_maps[0].tolist() == [0, 1, 2, 3, 4, 5, 18]
    assert fixed[0] == {0, 1, 2, 3, 4, 5}
    assert state.past_key_values[0][0].shape[2] == 7

    # Simulate replaying anchor query position 19. The next pre-query pruning
    # keeps the frozen sink/core and the newest one-token FIFO prefix.
    state.past_key_values = (
        (
            torch.cat(
                [state.past_key_values[0][0], torch.tensor([[[[19.0]]]])],
                dim=2,
            ),
            torch.cat(
                [state.past_key_values[0][1], torch.tensor([[[[119.0]]]])],
                dim=2,
            ),
        ),
    )
    state.position_maps[0] = torch.cat(
        [state.position_maps[0], torch.tensor([19])]
    )
    model.prune_recent_before_query(state, fixed)
    assert state.position_maps[0].tolist() == [0, 1, 2, 3, 4, 5, 19]
    assert fixed[0] == {0, 1, 2, 3, 4, 5}
    assert state.past_key_values[0][0].shape[2] == 7


def test_mandatory_tokens_do_not_compete_for_core():
    sink, recent, eligible = mandatory_and_eligible(
        list(range(20)), sink_size=2, recent_size=3
    )
    assert sink == [0, 1]
    assert recent == [17, 18, 19]
    assert eligible == list(range(2, 17))


def test_snapkv_shared_pooling_reuses_repository_implementation():
    values = snapkv_pool_scores_numpy(
        [0.0, 1.0, 0.0, 2.0, 0.0], kernel=3, method="max"
    )
    assert values.tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_candidate_records_label_sink_recent_and_core():
    records = candidate_layer_records(
        _tiny_selection(),
        _tiny_anchor(),
        list(range(20)),
        sink_size=2,
        recent_size=2,
    )
    layer = records[0]
    assert layer["sink_positions"] == [0, 1]
    assert layer["recent_positions"] == [18, 19]
    assert {
        row["cache_role"] for row in layer["mandatory_token_records"]
    } == {"sink", "recent"}
    assert {
        row["position"]
        for row in layer["eligible_token_records"]
        if row["cache_role"] == "core"
    } == {2, 3, 4, 5}


def test_future_oracle_window_aligns_with_teacher_forced_targets():
    model = _tiny_temporal_model()
    model.model_info = {"num_layers": 1}
    query_records = [
        QueryRecord(
            query_position=index,
            queries={},
            attention_outputs={},
            attention_distributions={},
            oracle_attention_by_layer={
                0: torch.tensor([[float(index + 1)] * 20])
            },
            new_values={},
        )
        for index in range(4)
    ]
    reference = SimpleNamespace(
        query_records=query_records,
        anchors={1: _tiny_anchor()},
    )
    future = model.future_attention(reference, anchor_step=1, horizon=2)
    assert future[0].shape == (1, 20)
    assert torch.all(future[0] == 5.0)


def test_ridge_leverage_is_finite_and_uses_no_inverse_result():
    rows = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    )
    scores, diagnostics = ridge_leverage(rows, 1e-3, "relative")
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()
    assert diagnostics["calculation"] == "ridge_eigh_no_inverse"
    assert diagnostics["ridge"] > 0


def test_online_ridge_factor_matches_linear_solve_without_inverse():
    history = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, -1.0]]
    )
    vector = torch.tensor([0.5, -0.25])
    factor = fit_online_ridge_factor(history, 1e-3, "relative")
    gram = history.double().T @ history.double()
    expected = vector.double() @ torch.linalg.solve(
        gram + factor.ridge * torch.eye(2, dtype=torch.float64),
        vector.double(),
    )
    assert torch.allclose(factor.score(vector), expected, rtol=1e-10, atol=1e-12)
    assert factor.diagnostics["calculation"] == "cholesky_solve_no_inverse"


def test_mechanism_capture_steps_cover_dense_refresh_and_exit_probe():
    cfg = DiscoveryConfig(
        anchor_steps=[0, 16, 48],
        mechanism=MechanismDiscoveryConfig(
            enabled=True,
            base_anchor_steps=[0, 16, 48],
            refresh_lags=[1, 8, 64],
            recent_exit_enabled=True,
            recent_exit_base_anchor=0,
            recent_exit_search_max_offset=2,
            recent_exit_relative_lags=[-1, 0, 1],
        ),
    )
    captured = cfg.captured_anchor_steps()
    assert {0, 1, 16, 17, 48, 49, 64, 80, 112}.issubset(captured)
    # Offsets 1 and 2 first leave a 32-token recent window at lags 33/34.
    assert {32, 33, 34, 35}.issubset(captured)


def test_recent_window_exit_occurs_at_token_offset_plus_window():
    positions_at_before = list(range(32))
    _, recent_before, eligible_before = mandatory_and_eligible(
        positions_at_before, sink_size=0, recent_size=32
    )
    assert 0 not in eligible_before
    assert 0 in recent_before
    positions_at_exit = list(range(33))
    _, recent_exit, eligible_exit = mandatory_and_eligible(
        positions_at_exit, sink_size=0, recent_size=32
    )
    assert 0 in eligible_exit
    assert 0 not in recent_exit


def test_approximate_kl_topk_plus_other_is_normalized():
    logits = torch.tensor([1.0, 0.5, -0.25, -1.0])
    probabilities = torch.softmax(logits.double(), dim=0)
    top_probabilities, top_ids = torch.topk(probabilities, 2)
    result = approximate_kl(top_ids, top_probabilities, logits, floor=1e-12)
    assert abs(result["approx_kl"]) < 1e-10
    assert 0.0 <= result["full_other_mass"] <= 1.0
    assert 0.0 <= result["compressed_other_mass"] <= 1.0


def test_approximate_kl_records_float32_mass_rounding_correction():
    logits = torch.tensor([12.0, 3.0, -4.0, -8.0])
    probabilities = torch.softmax(logits.double(), dim=0)
    top_probabilities, top_ids = torch.topk(probabilities, 4)
    rounded = top_probabilities.float().double() * (1.0 + 2e-6)
    result = approximate_kl(top_ids, rounded, logits, floor=1e-12)
    assert result["full_top_mass"] == 1.0
    assert 0 < result["full_top_mass_rounding_correction"] < 1e-5
    assert torch.isfinite(torch.tensor(result["approx_kl"]))


def test_loss_shape_reports_derivatives_and_spike():
    result = loss_shape([0.0, 0.1, 0.4, 0.2], large_spike_threshold=0.25)
    assert result["cumulative"] == [0.0, 0.1, 0.5, 0.7]
    assert result["first_large_loss_spike"] == 3
    assert len(result["slope"]) == 4
    assert len(result["curvature"]) == 4
