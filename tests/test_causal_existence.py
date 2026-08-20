from pathlib import Path

import yaml

from statekv.backend import QueryRecord
from statekv.causal_existence import _atomic_npz, expand_split_ids, sample_id_for, task_overrides
from statekv.causal_existence_analysis import (
    aggregate_sequence_metrics,
    boundary_metrics,
    future_attention_labels,
)
from statekv.causal_rollout import _near_cutoff_groups
from statekv.causal_predictors import _history_features, ema_score, feature_groups


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "statekv_existence" / "causal_existence_qwen3_8b.yaml"


def test_fresh_split_is_disjoint_and_preregistered() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    splits = expand_split_ids(config)
    assert {key: len(value) for key, value in splits.items()} == {
        "debug": 12,
        "train": 60,
        "validation": 21,
        "fresh_test": 24,
    }
    assert not (set(splits["train"]) & set(splits["fresh_test"]))
    assert sample_id_for("govreport_or_qmsum", 192) == "gov_report:192"


def test_task_overrides_cover_the_frozen_index_universe() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tasks = task_overrides(config)
    assert tasks["ruler_niah"]["sample_offset"] == 161
    assert tasks["ruler_niah"]["num_samples"] == 39
    assert tasks["govreport_or_qmsum"]["sample_indices"] == list(range(161, 200))


def test_query_record_post_rope_field_is_backward_compatible() -> None:
    record = QueryRecord(0, {}, {}, {}, {}, {})
    assert record.post_rope_queries == {}


def test_atomic_npz_temporary_file_never_matches_publication_glob(tmp_path) -> None:
    import numpy as np

    path = tmp_path / "artifact.npz"
    _atomic_npz(path, values=np.asarray([1, 2, 3]))
    assert [value.name for value in tmp_path.glob("*.npz")] == ["artifact.npz"]
    with np.load(path, allow_pickle=False) as payload:
        assert payload["values"].tolist() == [1, 2, 3]


def test_future_labels_follow_position_identity_and_exclude_current_step() -> None:
    import numpy as np

    attention = np.full((3, 1, 1, 3), np.nan, dtype=np.float32)
    attention[0, 0, 0, :2] = [0.7, 0.3]
    attention[1, 0, 0, :3] = [0.1, 0.2, 0.7]
    attention[2, 0, 0, :3] = [0.4, 0.5, 0.1]
    positions = np.asarray([[10, 20, -1], [10, 20, 30], [10, 20, 30]])
    labels = future_attention_labels(attention, positions, np.asarray([2, 3, 3]), [1, 2])
    assert np.allclose(labels[1][0, 0, 0, :2], [0.1, 0.2])
    assert np.allclose(labels[2][0, 0, 0, :2], [0.5, 0.7])
    assert np.isnan(labels[2][1:]).all()


def test_boundary_metrics_reports_exact_oracle_gap_recovery() -> None:
    import numpy as np

    truth = np.asarray([4.0, 3.0, 2.0, 1.0])
    baseline = np.asarray([1.0, 2.0, 3.0, 4.0])
    metrics = boundary_metrics(truth, truth, baseline, k=2)
    assert metrics["future_topk_recall"] == 1.0
    assert metrics["oracle_gap_recovery"] == 1.0


def test_counterfactual_panel_is_fixed_width_and_near_cutoff() -> None:
    import numpy as np

    positions = list(range(40))
    eligible = list(range(4, 36))
    scores = np.arange(40, dtype=np.float64)[::-1]
    groups = _near_cutoff_groups(
        positions, eligible, scores, core_budget=12, group_size=4, group_count=4
    )
    assert len(groups) == 4
    assert all(len(group) == 4 for group in groups)
    assert len({position for group in groups for position in group}) == 16


def test_feature_ladder_is_monotone_and_covers_full_state() -> None:
    groups = feature_groups()
    sizes = [len(columns) for columns in groups.values()]
    assert sizes == sorted(sizes)
    assert sizes[0] == 16
    assert sizes[-1] == 120


def test_causal_history_initializes_tokens_at_first_observation() -> None:
    import numpy as np

    rows = np.asarray([[1.0, np.nan], [0.5, 0.8]], dtype=np.float32)
    scalar, temporal = _history_features(rows)
    assert np.isfinite(scalar).all()
    assert np.allclose(ema_score(rows, 0.9), [0.95, 0.8])
    assert temporal[1, -2, 1] == 0.0
    assert temporal[1, -1, 1] == 1.0


def test_sequence_recovery_uses_aggregate_values_not_mean_of_ratios() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "sample_id": ["s", "s"],
            "task": ["t", "t"],
            "split": ["v", "v"],
            "method": ["m", "m"],
            "future_horizon": [4, 4],
            "future_topk_recall": [0.5, 0.5],
            "spearman": [0.2, 0.2],
            "pairwise_accuracy": [0.6, 0.6],
            "ndcg": [0.7, 0.7],
            "oracle_value": [2.0, 101.0],
            "selected_value": [1.5, 51.0],
            "baseline_value": [1.0, 1.0],
        }
    )
    row = aggregate_sequence_metrics(frame).iloc[0]
    assert row["oracle_gap_recovery"] == 50.5 / 101.0


def test_causal_runtime_paths_never_generate_a_saved_real_future() -> None:
    for name in (
        "causal_existence.py",
        "causal_rollout.py",
        "causal_closed_loop.py",
    ):
        source = (ROOT / "statekv" / name).read_text(encoding="utf-8")
        assert ".generate_reference(" not in source
