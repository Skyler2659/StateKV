"""CPU-only tests for the structured R2 student pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from statekv.causal_predictors import FixedProjector
from statekv.causal_student import (
    RuntimeFeatureHistory,
    load_student_checkpoint,
    save_student_checkpoint,
)
from statekv.selectors import mandatory_and_eligible
from statekv.structured_student import (
    GLOBAL_FEATURE_WIDTH,
    HEAD_FEATURE_WIDTH,
    STATE_FEATURE_WIDTH,
    StructuredStudent,
    StructuredStudentScorer,
    RuntimeStructuredScorer,
    structured_boundary,
)


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
    attention = rng.random(
        (CYCLES, len(SCORE_LAYERS), KV_HEADS, POSITIONS), dtype=np.float32
    ) + 0.01
    # Later positions do not exist at early cycles, like the real artifacts.
    for cycle in range(CYCLES):
        present = POSITIONS // 2 + cycle * POSITIONS // (2 * CYCLES)
        attention[cycle, :, :, present:] = np.nan
    return {
        "attention": attention,
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


def test_structured_boundary_shapes() -> None:
    artifact = _synthetic_artifact()
    boundary = structured_boundary(artifact, 3, SINK, RECENT, FixedProjector(7))
    _, _, eligible = mandatory_and_eligible(list(range(POSITIONS)), SINK, RECENT)
    n = len(eligible)
    assert boundary["positions"].shape == (n,)
    assert list(boundary["positions"]) == [int(v) for v in eligible]
    assert boundary["head_features"].shape == (
        n, len(SCORE_LAYERS), KV_HEADS, HEAD_FEATURE_WIDTH
    )
    assert boundary["state_features"].shape == (
        len(SCORE_LAYERS), STATE_FEATURE_WIDTH
    )
    assert boundary["token_features"].shape == (n, 2)
    assert boundary["global_features"].shape == (GLOBAL_FEATURE_WIDTH,)
    for key in ("head_features", "state_features", "token_features", "global_features"):
        assert np.isfinite(boundary[key]).all(), key


def test_structured_boundary_is_causal() -> None:
    """Perturbing cycles after c must not change features at cycle c."""

    artifact = _synthetic_artifact()
    perturbed = {key: np.array(value, copy=True) for key, value in artifact.items()}
    rng = np.random.default_rng(5)
    cycle = 2
    for key in ("attention", "query_post", "residual", "attention_input", "global_features"):
        shape = perturbed[key].shape
        noise = rng.normal(size=(CYCLES - cycle - 1, *shape[1:])).astype(
            perturbed[key].dtype
        )
        perturbed[key][cycle + 1 :] = perturbed[key][cycle + 1 :] + noise
    projector = FixedProjector(7)
    reference = structured_boundary(artifact, cycle, SINK, RECENT, projector)
    changed = structured_boundary(perturbed, cycle, SINK, RECENT, projector)
    for key in ("positions", "head_features", "state_features", "token_features", "global_features"):
        np.testing.assert_array_equal(reference[key], changed[key], err_msg=key)


def test_runtime_history_matches_structured_boundary() -> None:
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
    runtime = structured_boundary(view, cycle, SINK, RECENT, projector)
    reference = structured_boundary(artifact, cycle, SINK, RECENT, projector)
    for key in ("positions", "head_features", "state_features", "token_features", "global_features"):
        np.testing.assert_allclose(
            runtime[key], reference[key], rtol=1.0e-5, atol=1.0e-6, err_msg=key
        )


def _structured_checkpoint(seed: int = 0) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = StructuredStudent(layers=len(SCORE_LAYERS), kv_heads=KV_HEADS)
    rng = np.random.default_rng(seed)
    scalers = {
        "head": StandardScaler().fit(
            rng.normal(size=(64, HEAD_FEATURE_WIDTH)).astype(np.float32)
        ),
        "state": StandardScaler().fit(
            rng.normal(size=(16, STATE_FEATURE_WIDTH)).astype(np.float32)
        ),
        "token": StandardScaler().fit(rng.normal(size=(64, 2)).astype(np.float32)),
        "global": StandardScaler().fit(
            rng.normal(size=(16, GLOBAL_FEATURE_WIDTH)).astype(np.float32)
        ),
    }
    return {
        "model": model,
        "scalers": scalers,
    }


def test_structured_checkpoint_round_trip(tmp_path: Path) -> None:
    built = _structured_checkpoint()
    path = save_student_checkpoint(
        tmp_path / "structured.pt",
        kind="structured_mlp",
        models=built["model"].state_dict(),
        scaler=built["scalers"],
        horizons=[1],
        projector_seed=7,
        score_channel=0,
        metadata={
            "architecture": {
                "layers": len(SCORE_LAYERS),
                "kv_heads": KV_HEADS,
                "head_feature_width": HEAD_FEATURE_WIDTH,
            }
        },
    )
    scorer = StructuredStudentScorer(load_student_checkpoint(path))
    artifact = _synthetic_artifact()
    boundary = structured_boundary(artifact, 3, SINK, RECENT, FixedProjector(7))
    first = scorer.predict(boundary)
    second = StructuredStudentScorer(load_student_checkpoint(path)).predict(boundary)
    assert first.shape == (len(boundary["positions"]),)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_model_forward_variants() -> None:
    torch.manual_seed(0)
    n = 12
    for kwargs in (
        {},
        {"use_identity": False},
        {"use_context": False},
        {"head_structure": False},
    ):
        model = StructuredStudent(layers=2, kv_heads=2, **kwargs)
        score, per_head = model(
            torch.randn(n, 2, 2, HEAD_FEATURE_WIDTH),
            torch.randn(2, STATE_FEATURE_WIDTH),
            torch.randn(n, GLOBAL_FEATURE_WIDTH + 2),
        )
        assert score.shape == (n,)
        assert torch.isfinite(score).all()
        if kwargs.get("head_structure") is False:
            assert per_head is None
        else:
            assert per_head.shape == (n, 2, 2)


def test_runtime_structured_scorer_scores_exactly_the_eligible_set() -> None:
    artifact = _synthetic_artifact()
    built = _structured_checkpoint()
    checkpoint = {
        "kind": "structured_mlp",
        "models": built["model"].state_dict(),
        "scaler": built["scalers"],
        "horizons": [1],
        "projector_seed": 7,
        "metadata": {
            "architecture": {
                "layers": len(SCORE_LAYERS),
                "kv_heads": KV_HEADS,
                "head_feature_width": HEAD_FEATURE_WIDTH,
            }
        },
    }
    scorer = RuntimeStructuredScorer(
        checkpoint,
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
