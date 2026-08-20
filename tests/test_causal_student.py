"""CPU-only tests for the deployable R2 student pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from statekv.causal_predictors import FixedProjector, MultiHorizonMLP, artifact_boundary
from statekv.causal_student import (
    FEATURE_SEGMENTS,
    FORBIDDEN_RUNTIME_KEYS,
    RuntimeFeatureHistory,
    RuntimeStudentScorer,
    StudentScorer,
    load_student_checkpoint,
    save_student_checkpoint,
)
from statekv.oracle_policy_comparison import _selection_from_scores
from statekv.qkv_decomposition import rank_and_margin
from statekv.selectors import mandatory_and_eligible


SCORE_LAYERS = [0, 7]
KV_HEADS = 2
QUERY_HEADS = 4
SINK = 2
RECENT = 3
CYCLES = 6
POSITIONS = 40
HIDDEN = 4096
HEAD_DIM = 128


def _synthetic_artifact(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    positions = np.arange(POSITIONS, dtype=np.int32)
    return {
        "attention": rng.random(
            (CYCLES, len(SCORE_LAYERS), KV_HEADS, POSITIONS), dtype=np.float32
        )
        + 0.01,
        "position_ids": np.tile(positions, (CYCLES, 1)),
        "position_lengths": np.full(CYCLES, POSITIONS, dtype=np.int32),
        "query_post": rng.normal(
            size=(CYCLES, len(SCORE_LAYERS), QUERY_HEADS, HEAD_DIM)
        ).astype(np.float32),
        "residual": rng.normal(
            size=(CYCLES, len(SCORE_LAYERS), HIDDEN)
        ).astype(np.float32),
        "attention_input": rng.normal(
            size=(CYCLES, len(SCORE_LAYERS), HIDDEN)
        ).astype(np.float32),
        "global_features": rng.normal(size=(CYCLES, 4)).astype(np.float32),
        "keys": rng.normal(
            size=(len(SCORE_LAYERS), KV_HEADS, POSITIONS, HEAD_DIM)
        ).astype(np.float32),
        "values": rng.normal(
            size=(len(SCORE_LAYERS), KV_HEADS, POSITIONS, HEAD_DIM)
        ).astype(np.float32),
        "kv_position_ids": positions,
        "layers": np.asarray(SCORE_LAYERS, dtype=np.int16),
        "sample_id": np.asarray("synthetic_0"),
        "task": np.asarray("synthetic"),
        "split": np.asarray("train"),
    }


def _observation(artifact: Dict[str, np.ndarray], cycle: int) -> Dict[str, object]:
    return {
        "per_head_attention": {
            layer: artifact["attention"][cycle, index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
        "post_rope_queries": {
            layer: artifact["query_post"][cycle, index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
        "residual": {
            layer: artifact["residual"][cycle, index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
        "attention_input": {
            layer: artifact["attention_input"][cycle, index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
        "global_features": artifact["global_features"][cycle],
    }


def _kv(artifact: Dict[str, np.ndarray]) -> Dict[str, Dict[int, np.ndarray]]:
    return {
        "keys": {
            layer: artifact["keys"][index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
        "values": {
            layer: artifact["values"][index]
            for index, layer in enumerate(SCORE_LAYERS)
        },
    }


def test_feature_segments_cover_exactly_the_contract_width() -> None:
    segments = sorted(FEATURE_SEGMENTS.values())
    assert segments[0][0] == 0
    assert segments[-1][1] == 120
    for (_, stop, _), (next_start, _, _) in zip(segments, segments[1:]):
        assert stop == next_start


def test_artifact_boundary_feature_shape() -> None:
    artifact = _synthetic_artifact()
    boundary = artifact_boundary(
        artifact, 2, 0, 0, [1], SINK, RECENT, 10, FixedProjector(7),
        feature_only=True,
    )
    _, _, eligible = mandatory_and_eligible(
        list(range(POSITIONS)), SINK, RECENT
    )
    assert boundary.features.shape == (len(eligible), 120)
    assert np.isfinite(boundary.features).all()


def test_runtime_history_matches_artifact_boundary() -> None:
    """The runtime feature path must reproduce training features exactly."""

    artifact = _synthetic_artifact()
    projector = FixedProjector(7)
    cycle = 4
    history = RuntimeFeatureHistory(
        score_layers=SCORE_LAYERS,
        kv_heads=KV_HEADS,
        query_heads=QUERY_HEADS,
        sink_size=SINK,
        recent_size=RECENT,
        total_cycles=CYCLES,
    )
    positions = list(range(POSITIONS))
    for step in range(cycle + 1):
        history.observe(cycle=step, positions=positions, **_observation(artifact, step))
    view = history.artifact_view(cycle=cycle, positions=positions, **_kv(artifact))
    for layer_index in range(len(SCORE_LAYERS)):
        for head in range(KV_HEADS):
            runtime = artifact_boundary(
                view, cycle, layer_index, head, [1], SINK, RECENT, 1,
                projector, feature_only=True,
            ).features
            reference = artifact_boundary(
                artifact, cycle, layer_index, head, [1], SINK, RECENT, 1,
                projector, feature_only=True,
            ).features
            np.testing.assert_allclose(runtime, reference, rtol=1.0e-5, atol=1.0e-6)


def test_runtime_history_repacks_shrinking_pools() -> None:
    artifact = _synthetic_artifact()
    rng = np.random.default_rng(11)
    history = RuntimeFeatureHistory(
        score_layers=SCORE_LAYERS,
        kv_heads=KV_HEADS,
        query_heads=QUERY_HEADS,
        sink_size=SINK,
        recent_size=RECENT,
        total_cycles=CYCLES,
    )
    pools = [list(range(40)), list(range(20)), list(range(20)) + [40]]
    new_token_attention = rng.random((len(SCORE_LAYERS), KV_HEADS, 1)).astype(
        np.float32
    )
    for step, pool in enumerate(pools):
        observation = _observation(artifact, step)
        per_head = {}
        for layer, values in observation["per_head_attention"].items():
            kept = values[:, [position for position in pool if position < 40]]
            if 40 in pool:
                kept = np.concatenate(
                    [kept, new_token_attention[SCORE_LAYERS.index(layer)]], axis=1
                )
            per_head[layer] = kept
        observation["per_head_attention"] = per_head
        history.observe(cycle=step, positions=pool, **observation)
    new_token_kv = rng.normal(size=(2, 1, HEAD_DIM)).astype(np.float32)
    keys, values = _kv(artifact)["keys"], _kv(artifact)["values"]
    current = pools[-1]
    columns = np.asarray([position for position in current if position < 40])
    view = history.artifact_view(
        cycle=2,
        positions=current,
        keys={
            layer: np.concatenate([array[:, columns], new_token_kv], axis=1)
            for layer, array in keys.items()
        },
        values={
            layer: np.concatenate([array[:, columns], new_token_kv], axis=1)
            for layer, array in values.items()
        },
    )
    # Surviving position 5 keeps its cycle-0 attention value in its new column.
    column = current.index(5)
    np.testing.assert_allclose(
        view["attention"][0, :, :, column],
        artifact["attention"][0, :, :, 5],
        rtol=1.0e-6,
    )
    # The freshly generated position 40 was absent at cycles 0 and 1.
    generated_column = current.index(40)
    assert np.isnan(view["attention"][0, :, :, generated_column]).all()
    assert np.isnan(view["attention"][1, :, :, generated_column]).all()
    assert np.isfinite(view["attention"][2, :, :, generated_column]).all()
    # Evicted position 25 contributes no columns at cycle 2.
    assert 25 not in view["position_ids"][2]


def test_runtime_view_contains_no_future_information() -> None:
    artifact = _synthetic_artifact()
    history = RuntimeFeatureHistory(
        score_layers=SCORE_LAYERS,
        kv_heads=KV_HEADS,
        query_heads=QUERY_HEADS,
        sink_size=SINK,
        recent_size=RECENT,
        total_cycles=CYCLES,
    )
    positions = list(range(POSITIONS))
    cycle = 3
    for step in range(cycle + 1):
        history.observe(cycle=step, positions=positions, **_observation(artifact, step))
    view = history.artifact_view(cycle=cycle, positions=positions, **_kv(artifact))
    # Rows beyond the current cycle stay NaN; any accidental future read would
    # propagate NaN into the features.
    assert np.isnan(view["attention"][cycle + 1 :]).all()
    assert int(view["position_lengths"][cycle + 1 :].sum()) == 0
    for key in FORBIDDEN_RUNTIME_KEYS:
        assert key not in view
    boundary = artifact_boundary(
        view, cycle, 0, 0, [1], SINK, RECENT, 1, FixedProjector(7),
        feature_only=True,
    )
    assert np.isfinite(boundary.features).all()


def test_runtime_history_rejects_nonconsecutive_cycles() -> None:
    artifact = _synthetic_artifact()
    history = RuntimeFeatureHistory(
        score_layers=SCORE_LAYERS,
        kv_heads=KV_HEADS,
        query_heads=QUERY_HEADS,
        sink_size=SINK,
        recent_size=RECENT,
        total_cycles=CYCLES,
    )
    positions = list(range(POSITIONS))
    history.observe(cycle=0, positions=positions, **_observation(artifact, 0))
    with pytest.raises(RuntimeError):
        history.observe(cycle=2, positions=positions, **_observation(artifact, 2))


def _gbdt_checkpoint(seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4000, 120)).astype(np.float32)
    signal = 3.0 * features[:, 0] - 1.5 * features[:, 7] + features[:, 30]
    scaler = StandardScaler().fit(features)
    models = {}
    for horizon, noise_scale in ((1, 0.05), (4, 0.20)):
        target = np.log1p(
            signal - signal.min() + noise_scale * rng.normal(size=len(signal)) + 1.0
        )
        models[horizon] = HistGradientBoostingRegressor(
            max_iter=40, random_state=seed
        ).fit(scaler.transform(features), target)
    return {
        "kind": "hist_gbdt",
        "models": models,
        "scaler": scaler,
        "horizons": [1, 4],
        "projector_seed": 7,
        "_signal_features": features,
    }


def test_gbdt_checkpoint_round_trip_and_ranking(tmp_path: Path) -> None:
    built = _gbdt_checkpoint()
    signal = 3.0 * built["_signal_features"][:, 0] - 1.5 * built["_signal_features"][:, 7] + built["_signal_features"][:, 30]
    path = save_student_checkpoint(
        tmp_path / "student.joblib",
        kind="hist_gbdt",
        models=built["models"],
        scaler=built["scaler"],
        horizons=built["horizons"],
        projector_seed=7,
    )
    scorer = StudentScorer(load_student_checkpoint(path))
    prediction = scorer.predict(built["_signal_features"])
    assert prediction.shape == (len(signal), 2)
    for column in range(2):
        rho = spearmanr(signal, prediction[:, column]).statistic
        assert rho > 0.9


def test_mlp_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = MultiHorizonMLP(120, 2)
    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(16, 120)))
    path = save_student_checkpoint(
        tmp_path / "student.pt",
        kind="mlp",
        models=model.state_dict(),
        scaler=scaler,
        horizons=[1, 4],
        projector_seed=7,
    )
    scorer = StudentScorer(load_student_checkpoint(path))
    features = np.random.default_rng(1).normal(size=(32, 120)).astype(np.float32)
    first = scorer.predict(features)
    second = StudentScorer(load_student_checkpoint(path)).predict(features)
    assert first.shape == (32, 2)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_checkpoint_validation_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        load_student_checkpoint(tmp_path / "absent.joblib")
    built = _gbdt_checkpoint()
    path = save_student_checkpoint(
        tmp_path / "student.joblib",
        kind="hist_gbdt",
        models=built["models"],
        scaler=built["scaler"],
        horizons=built["horizons"],
        projector_seed=7,
    )
    scorer = StudentScorer(load_student_checkpoint(path))
    with pytest.raises(RuntimeError, match="n_tokens"):
        scorer.predict(np.zeros((4, 119), dtype=np.float32))
    with pytest.raises(RuntimeError, match="horizon"):
        scorer.horizon_column(16)


def test_runtime_student_scorer_scores_exactly_the_eligible_set() -> None:
    artifact = _synthetic_artifact()
    built = _gbdt_checkpoint()
    scorer = RuntimeStudentScorer(
        {
            "kind": "hist_gbdt",
            "models": built["models"],
            "scaler": built["scaler"],
            "horizons": [1, 4],
            "projector_seed": 7,
        },
        score_layers=SCORE_LAYERS,
        kv_heads=KV_HEADS,
        query_heads=QUERY_HEADS,
        sink_size=SINK,
        recent_size=RECENT,
        horizon=1,
    )
    scorer.reset(total_cycles=CYCLES)
    positions = list(range(POSITIONS))
    scores = None
    for step in range(3):
        scores = scorer.observe_and_score(
            cycle=step,
            positions=positions,
            keys=_kv(artifact)["keys"],
            values=_kv(artifact)["values"],
            **_observation(artifact, step),
        )
    _, _, eligible = mandatory_and_eligible(positions, SINK, RECENT)
    assert scores is not None
    assert sorted(scores) == sorted(eligible)
    assert all(np.isfinite(value) for value in scores.values())
    # Sink and recent positions are never scored: they can never be evicted.
    assert not (set(positions[:SINK]) | set(positions[-RECENT:])) & set(scores)


def test_student_selection_satisfies_strict_invariants() -> None:
    """The student branch feeds the same strict enforcement path as every
    other strict policy: sink/recent protection, exact budget, shared core."""

    rng = np.random.default_rng(3)
    positions = list(range(100))
    sink, recent, core_budget = 4, 8, 20
    _, _, eligible = mandatory_and_eligible(positions, sink, recent)
    fake_scores = {int(position): float(rng.random()) for position in eligible}
    # Mirror the policy branch: -inf outside the eligible set.
    shared_scores = np.full(len(positions), -np.inf, dtype=np.float64)
    for index, position in enumerate(positions):
        if position in fake_scores:
            shared_scores[index] = fake_scores[position]
    _, _, core = rank_and_margin(shared_scores, positions, eligible, core_budget)
    assert len(core) == core_budget
    assert set(core) <= set(eligible)
    assert not (set(positions[:sink]) | set(positions[-recent:])) & set(core)
    layers = 3
    selection = _selection_from_scores(
        "STRICT_STATEKV_STUDENT",
        positions,
        eligible,
        {layer: core for layer in range(layers)},
        {layer: shared_scores for layer in range(layers)},
    )
    cores = {
        layer: tuple(layer_selection.selected_positions)
        for layer, layer_selection in selection.by_layer.items()
    }
    assert len(set(cores.values())) == 1  # shared-token core across layers
    # Deterministic under ties: equal scores keep the lowest positions.
    tied = np.zeros(len(positions), dtype=np.float64)
    _, _, tied_core = rank_and_margin(tied, positions, eligible, core_budget)
    assert list(tied_core) == eligible[:core_budget]
