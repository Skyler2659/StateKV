from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = (
    ROOT
    / "experiments/p0_v2_fixed_boundary/scripts/p0_v2_core.py"
)
SPEC = importlib.util.spec_from_file_location("p0_v2_core", CORE_PATH)
assert SPEC is not None and SPEC.loader is not None
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def test_vector_metrics_identity() -> None:
    value = np.array([1.0, -2.0, 3.0])
    metrics = CORE.vector_metrics(value, value)
    assert metrics["finite"]
    assert np.isclose(metrics["cosine"], 1.0)
    assert np.isclose(metrics["relative_l2"], 0.0)
    assert np.isclose(metrics["symmetric_norm_ratio"], 1.0)


def test_fisher_is_common_shift_invariant() -> None:
    probability = np.array([0.2, 0.3, 0.5])
    direction = np.array([-0.7, 0.1, 1.2])
    shifted = direction + 100.0
    assert np.isclose(
        CORE.fisher_variance(probability, direction),
        CORE.fisher_variance(probability, shifted),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_exact_kl_nonnegative_and_zero_at_identity() -> None:
    logits = np.array([2.0, 0.0, -1.0])
    changed = np.array([1.5, 0.2, -0.5])
    assert CORE.exact_kl(logits, changed) >= 0.0
    assert np.isclose(CORE.exact_kl(logits, logits), 0.0)


def test_ranking_uses_lowest_risk_candidate() -> None:
    truth = [0.1, 0.2, 0.3, 0.4]
    result = CORE.ranking_metrics(truth, truth, top_k=2)
    assert np.isclose(result["spearman"], 1.0)
    assert np.isclose(result["pairwise_sign_accuracy"], 1.0)
    assert np.isclose(result["top1_accuracy"], 1.0)
    assert np.isclose(result["topk_overlap"], 1.0)
    assert np.isclose(result["normalized_regret"], 0.0)


def test_frozen_config_has_disjoint_splits_and_no_future_oracle_candidate() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/frozen/p0_v2_config.yaml").read_text(encoding="utf-8")
    )
    calibration = set(
        config["data"]["calibration"]["gov_report_indices"]
    )
    evaluation = set(
        config["data"]["evaluation"]["gov_report_indices"]
    )
    assert calibration.isdisjoint(evaluation)
    assert (
        "future_attention_oracle"
        not in config["candidates"]["sources"]
    )
    assert config["scope"]["action_only"]
    assert config["scope"]["isolated_single_layer_mask"]
    assert config["data"]["evaluation"]["layers"] == [0, 14, 26]

