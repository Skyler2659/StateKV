from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.p3pr_generalization.scripts.analyze_generalization import (
    calibration_boundary_scan,
    exact_paired_sign_flip,
)
from experiments.p3pr_generalization.scripts.run_generalization import (
    build_backend_config,
    relative_boundaries,
)
from statekv.backend_mlx import MLXTemporalModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "experiments/p3pr_generalization/p3pr_generalization_config.yaml"
)


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_role_ids_are_pairwise_disjoint():
    roles = _config()["data"]["role_isolation"]
    sets = [
        set(roles["calibration_ids"]),
        set(roles["formal_ids"]),
        set(roles["replication_ids"]),
    ]
    assert all(
        not (left & right)
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    )


def test_relative_boundary_rule_changes_with_depth():
    config = _config()
    assert relative_boundaries(24, config) == [6, 12, 18, 23]
    assert relative_boundaries(16, config) == [4, 8, 12, 15]
    assert 24 not in relative_boundaries(24, config)
    assert 16 not in relative_boundaries(16, config)


def test_backend_family_is_inferred_for_each_checkpoint():
    config = _config()
    for model_key, expected in (
        ("qwen25_05b", "qwen"),
        ("llama32_1b", "llama"),
    ):
        discovery = build_backend_config(config, model_key)
        root = MLXTemporalModel(discovery)._root_config()
        assert root.model.family == expected


def test_exact_paired_sign_flip_detects_uniform_gain():
    result = exact_paired_sign_flip(
        np.ones(8, dtype=np.float64),
        np.zeros(8, dtype=np.float64),
    )
    assert result["mean_spearman_gain"] == 1.0
    assert result["exact_one_sided_sign_flip_p"] == 1.0 / 256.0
    assert result["all_sequence_gains_positive"]


def test_calibration_scan_excludes_boundaries_absent_for_a_model():
    rows = []
    for model_key, layers, boundary in (
        ("model_a", 24, 23),
        ("model_b", 16, 15),
    ):
        for sample_index in range(2):
            for candidate_index in range(8):
                rows.append(
                    {
                        "stage": "calibration",
                        "model_key": model_key,
                        "model_family": model_key,
                        "sample_id": f"{model_key}_{sample_index}",
                        "task": "task",
                        "num_layers": layers,
                        "exact_physical_kl": float(candidate_index),
                        "b23_path_k1_risk": (
                            float(candidate_index)
                            if boundary == 23
                            else np.nan
                        ),
                        "b15_path_k1_risk": (
                            float(candidate_index)
                            if boundary == 15
                            else np.nan
                        ),
                    }
                )
    scan = calibration_boundary_scan(pd.DataFrame(rows))
    observed = {
        (row["model_key"], row["boundary"]) for row in scan
    }
    assert observed == {("model_a", 23), ("model_b", 15)}
