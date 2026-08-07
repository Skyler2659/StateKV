import json
import math

import numpy as np
import pytest
import torch
from scipy.stats import kendalltau

from scripts.run_analysis import analyze_overlap, analyze_rank_correlation
from src.analysis.alignment import align_score_units, align_selection_units
from src.analysis.overlap import OverlapAnalyzer, selection_frequency
from src.analysis.rank_correlation import (
    RankCorrelationAnalyzer,
    _kendall_tau,
    _ranks,
)
from src.artifacts.schema import (
    ScoreArtifact,
    ScoreUnit,
    SelectionArtifact,
    SelectionUnit,
    SnapshotRef,
    load_artifact,
    save_artifact,
)
from src.cache.position_map import PositionMap


def snapshot(snapshot_id="snapshot-1", phase="pre_answer"):
    return SnapshotRef(
        snapshot_id=snapshot_id,
        sample_id="sample-7",
        phase=phase,
        context_length=8,
        prompt_length=8,
    )


def score_artifact(method, units, snapshot_ref=None):
    return ScoreArtifact(
        artifact_id=f"score-{method}",
        snapshot=snapshot_ref or snapshot(),
        method=method,
        score_type="test_score",
        score_source="value",
        units=tuple(units),
    )


def selection_artifact(
    method,
    units,
    snapshot_ref=None,
    budget=2,
    scope="snapshot_offline",
    unit="token_slots_per_kv_head",
):
    return SelectionArtifact(
        artifact_id=f"selection-{method}",
        snapshot=snapshot_ref or snapshot(),
        method=method,
        requested_budget=budget,
        effective_budget=budget,
        budget_scope=scope,
        budget_unit=unit,
        units=tuple(units),
    )


def test_position_map_append_prune_padding_and_reverse_lookup():
    positions = PositionMap.identity(4).append([7, 9]).prune([0, 3, 4, 5]).with_padding(6)
    assert positions.positions == (0, 3, 7, 9, None, None)
    assert positions.valid_mask == (True, True, True, True, False, False)
    assert positions.rows_for_original([9, 0]) == (3, 0)
    with pytest.raises(KeyError):
        positions.rows_for_original([8])
    with pytest.raises(ValueError):
        PositionMap.from_iterable([1, 1])


def test_artifact_round_trip_and_legacy_rejection(tmp_path):
    artifact = score_artifact(
        "value_l2",
        [ScoreUnit(0, 0, (0, 2), (0, 1, 2), (0.1, 0.9))],
    )
    path = tmp_path / "score.json"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded == artifact

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"0": [0.1, 0.2]}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        load_artifact(legacy)


def test_score_alignment_uses_original_positions_not_vector_order():
    left = score_artifact(
        "a",
        [ScoreUnit(2, 1, (3, 1, 2), (1, 2, 3), (30.0, 10.0, 20.0))],
    )
    right = score_artifact(
        "b",
        [ScoreUnit(2, 1, (2, 3, 1), (3, 2, 1), (200.0, 300.0, 100.0))],
    )
    aligned = align_score_units(left, right)[(2, 1)]
    assert aligned.positions == (1, 2, 3)
    assert aligned.scores_a.tolist() == [10.0, 20.0, 30.0]
    assert aligned.scores_b.tolist() == [100.0, 200.0, 300.0]


@pytest.mark.parametrize(
    "change,match",
    [
        ("snapshot", "different snapshots"),
        ("universe", "token universes"),
        ("head", "layer/head units"),
    ],
)
def test_score_alignment_rejects_mismatched_scientific_units(change, match):
    left = score_artifact("a", [ScoreUnit(0, 0, (0, 1), (0, 1), (1.0, 2.0))])
    if change == "snapshot":
        right = score_artifact(
            "b",
            [ScoreUnit(0, 0, (0, 1), (0, 1), (1.0, 2.0))],
            snapshot("snapshot-2"),
        )
    elif change == "universe":
        right = score_artifact("b", [ScoreUnit(0, 0, (0, 1), (0, 1, 2), (1.0, 2.0))])
    else:
        right = score_artifact("b", [ScoreUnit(0, 1, (0, 1), (0, 1), (1.0, 2.0))])
    with pytest.raises(ValueError, match=match):
        align_score_units(left, right)


def test_tie_aware_ranks_and_true_kendall_tau_b():
    values = torch.tensor([3.0, 3.0, 1.0, 0.0])
    assert torch.allclose(
        _ranks(values),
        torch.tensor([1.5, 1.5, 3.0, 4.0], dtype=torch.float64),
    )

    left = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0])
    right = torch.tensor([1.0, 2.0, 2.0, 3.0, 1.0])
    expected = kendalltau(left.numpy(), right.numpy(), variant="b").statistic
    assert _kendall_tau(left, right) == pytest.approx(expected)


def test_rank_correlation_rejects_truncation_and_marks_constants():
    analyzer = RankCorrelationAnalyzer()
    with pytest.raises(ValueError, match="equal length"):
        analyzer.pairwise(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))

    result = analyzer.pairwise(torch.ones(4), torch.arange(4.0))
    assert math.isnan(result.spearman)
    assert next(iter(result.unit_results.values())).status == "constant_or_undefined"


def test_artifact_rank_correlation_is_macro_over_heads():
    left = score_artifact(
        "a",
        [
            ScoreUnit(0, 0, (0, 1, 2), (0, 1, 2), (1.0, 2.0, 3.0)),
            ScoreUnit(0, 1, (0, 1, 2), (0, 1, 2), (1.0, 2.0, 3.0)),
        ],
    )
    right = score_artifact(
        "b",
        [
            ScoreUnit(0, 0, (0, 1, 2), (0, 1, 2), (1.0, 2.0, 3.0)),
            ScoreUnit(0, 1, (0, 1, 2), (0, 1, 2), (3.0, 2.0, 1.0)),
        ],
    )
    result = RankCorrelationAnalyzer().artifact_pairwise(left, right)
    assert result.spearman == pytest.approx(0.0)
    assert result.n_units == 2
    assert set(result.head_wise_spearman) == {(0, 0), (0, 1)}


def test_overlap_macro_does_not_union_positions_across_layers():
    selected_a = {
        0: torch.tensor([0, 1, 2, 3]),
        1: torch.tensor([0]),
    }
    selected_b = {
        0: torch.tensor([0, 1, 2, 3]),
        1: torch.tensor([4]),
    }
    result = OverlapAnalyzer().pairwise_overlap(selected_a, selected_b, "a", "b")
    assert result.layer_wise_jaccard == {0: 1.0, 1: 0.0}
    assert result.jaccard == pytest.approx(0.5)
    # The prohibited cross-layer union would have been 4/5 = 0.8.
    assert result.jaccard != pytest.approx(0.8)
    assert result.micro_jaccard == pytest.approx(4 / 6)


def test_selection_alignment_rejects_budget_scope_and_universe_mismatch():
    unit = SelectionUnit(0, 0, (0, 1), (0, 1, 2, 3))
    left = selection_artifact("a", [unit])
    wrong_scope = selection_artifact("b", [unit], scope="total_kv")
    with pytest.raises(ValueError, match="budget scopes"):
        align_selection_units(left, wrong_scope)

    wrong_universe = selection_artifact(
        "b",
        [SelectionUnit(0, 0, (0, 1), (0, 1, 2, 4))],
    )
    with pytest.raises(ValueError, match="token universes"):
        align_selection_units(left, wrong_universe)

    with pytest.raises(ValueError, match="physical selected slot count"):
        SelectionUnit(
            0,
            0,
            (0, 1),
            (0, 1, 2, 3),
            requested_budget=2,
            effective_budget=3,
        )


def test_selection_frequency_and_headwise_overlap():
    left = selection_artifact(
        "a",
        [
            SelectionUnit(0, 0, (0, 1), (0, 1, 2)),
            SelectionUnit(0, 1, (0, 2), (0, 1, 2)),
        ],
    )
    right = selection_artifact(
        "b",
        [
            SelectionUnit(0, 0, (0, 1), (0, 1, 2)),
            SelectionUnit(0, 1, (1, 2), (0, 1, 2)),
        ],
    )
    assert selection_frequency(left) == {0: 1.0, 1: 0.5, 2: 0.5}
    analyzer = OverlapAnalyzer()
    overlap = analyzer.artifact_pairwise(left, right)
    assert overlap.head_wise_jaccard[(0, 0)] == 1.0
    assert overlap.head_wise_jaccard[(0, 1)] == pytest.approx(1 / 3)
    frequency = analyzer.frequency_correlation(left, right)
    assert frequency.n_positions == 3
    assert np.isfinite(frequency.spearman)


def test_analysis_runner_skips_legacy_artifacts_instead_of_guessing(tmp_path):
    legacy_selection = tmp_path / "legacy_selection.json"
    legacy_score = tmp_path / "legacy_score.json"
    legacy_selection.write_text(json.dumps({"0": [0, 1]}), encoding="utf-8")
    legacy_score.write_text(json.dumps({"0": [0.1, 0.2]}), encoding="utf-8")
    rows = [
        {
            "method": "legacy",
            "sample_id": 0,
            "budget": 2,
            "selected_tokens_path": str(legacy_selection),
            "scores_path": str(legacy_score),
        }
    ]
    overlap = analyze_overlap(rows, tmp_path)
    correlation = analyze_rank_correlation(rows, tmp_path)
    assert not overlap["pairs"]
    assert not correlation["pairs"]
    assert overlap["skipped_artifacts"]["legacy_or_invalid_selection_artifact"] == 1
    assert correlation["skipped_artifacts"]["legacy_or_invalid_score_artifact"] == 1


def test_analysis_runner_uses_schema_v2_artifacts(tmp_path):
    units_a = [SelectionUnit(0, 0, (0, 1), (0, 1, 2))]
    units_b = [SelectionUnit(0, 0, (1, 2), (0, 1, 2))]
    scores_a = [ScoreUnit(0, 0, (0, 1, 2), (0, 1, 2), (1.0, 2.0, 3.0))]
    scores_b = [ScoreUnit(0, 0, (0, 1, 2), (0, 1, 2), (3.0, 2.0, 1.0))]

    rows = []
    for method, selections, scores in (
        ("a", units_a, scores_a),
        ("b", units_b, scores_b),
    ):
        selection_path = tmp_path / f"{method}_selection.json"
        score_path = tmp_path / f"{method}_score.json"
        save_artifact(selection_artifact(method, selections), selection_path)
        save_artifact(score_artifact(method, scores), score_path)
        rows.append(
            {
                "method": method,
                "sample_id": 7,
                "budget": 2,
                "selection_artifact_path": str(selection_path),
                "score_artifact_path": str(score_path),
            }
        )

    overlap = analyze_overlap(rows, tmp_path)
    correlation = analyze_rank_correlation(rows, tmp_path)
    assert overlap["pairs"]["a_vs_b"]["jaccard"] == pytest.approx(1 / 3)
    assert correlation["pairs"]["a_vs_b"]["spearman"] == pytest.approx(-1.0)
