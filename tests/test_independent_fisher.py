from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from statekv.candidate_pullback import (
    covariance_energy,
    principal_angles,
    psd_interaction_parameters,
    state_action_energy_decomposition,
)
from statekv.config import load_discovery_config
from statekv.independent_fisher import adaptive_fisher_geometry
from statekv.independent_fisher_analysis import (
    CROSS_SCHEMA,
    LOW_RANK_SCHEMA,
    PULLBACK_SCHEMA,
    Q_STATE_SCHEMA,
    REFRESH_SCHEMA,
    _pairwise_sign_accuracy,
    analyze_trust_region,
    build_action_rows,
    write_later_stage_skips,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_config_is_frozen_and_local_4bit():
    cfg = load_discovery_config(str(ROOT / "configs" / "stages" / "independent_fisher_config.yaml"))
    assert cfg.model.name == "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    assert cfg.model.local_files_only
    assert cfg.model.quant_bits == 4
    assert cfg.independent_fisher.enabled
    assert cfg.independent_fisher.anchors == [16, 32, 48]
    assert cfg.independent_fisher.evaluation_horizons == [1, 4, 8, 16]


def test_independent_config_uses_disjoint_registered_indices():
    cfg = load_discovery_config(str(ROOT / "configs" / "stages" / "independent_fisher_config.yaml"))
    assert cfg.tasks["ruler_niah"]["sample_offset"] == 12
    assert cfg.tasks["govreport_or_qmsum"]["sample_indices"] == list(
        range(12, 24)
    )


def test_stage_b_candidate_pool_is_fixed_and_unique():
    cfg = load_discovery_config(str(ROOT / "configs" / "stages" / "independent_fisher_config.yaml"))
    sources = cfg.independent_fisher.stage_b_candidate_sources
    assert len(sources) == len(set(sources)) == 8
    assert "aor" in sources
    assert "aov" not in sources


def test_adaptive_integral_matches_exact_kl():
    z = torch.linspace(-2.0, 2.0, 101)
    delta = 0.7 * torch.sin(torch.arange(101))
    geometry, curvature = adaptive_fisher_geometry(
        z, z + delta, 16, 1.0e-8, 1.0e-10, 50
    )
    assert curvature["adaptive_weighted_integral"] == pytest.approx(
        geometry["exact_kl"], rel=1.0e-8, abs=1.0e-10
    )


def test_curvature_peak_extraction_is_finite():
    z = torch.tensor([8.0, 7.0, 0.0, -1.0])
    delta = torch.tensor([-5.0, 5.0, 0.0, 0.0])
    _, curvature = adaptive_fisher_geometry(
        z, z + delta, 4, 1.0e-8, 1.0e-10, 50
    )
    assert 0.0 <= curvature["curvature_peak_location"] <= 1.0
    assert curvature["curvature_max"] >= 0.0
    assert curvature["effective_curvature_width"] >= 0.0


def test_top_token_switch_detection():
    z = torch.tensor([3.0, 2.0, -4.0])
    delta = torch.tensor([-4.0, 4.0, 0.0])
    _, curvature = adaptive_fisher_geometry(
        z, z + delta, 3, 1.0e-8, 1.0e-10, 50
    )
    assert curvature["top1_changed_along_path"]
    assert curvature["top1_change_count"] >= 1
    assert curvature["final_top1_margin"] < 0.0


def test_gl5_counterexample_does_not_block_config():
    cfg = load_discovery_config(str(ROOT / "configs" / "stages" / "independent_fisher_config.yaml"))
    assert not hasattr(
        cfg.independent_fisher, "quadrature_relative_error_gate"
    )


def test_pairwise_sign_accuracy_direction():
    truth = np.array([1.0, 2.0, 3.0])
    assert _pairwise_sign_accuracy(truth, truth) == pytest.approx(1.0)
    assert _pairwise_sign_accuracy(-truth, truth) == pytest.approx(0.0)


def test_action_rows_use_equal_candidate_groups():
    rows = []
    for candidate, value in enumerate([1.0, 2.0, 3.0]):
        for offset in (1, 2):
            rows.append(
                {
                    "task": "toy",
                    "sample_id": "s0",
                    "anchor": 16,
                    "candidate_id": "c%d" % candidate,
                    "horizon_offset": offset,
                    "exact_kl": value,
                    "g0_raw_l2_sq": 4.0 - value,
                    "g2_base_fisher": value,
                    "g3_midpoint_fisher": value,
                }
            )
    action = build_action_rows(pd.DataFrame(rows), [1, 2])
    assert len(action) == 6
    assert set(action["candidate_count"]) == {3}
    assert action[
        action["family"] == "G3_MIDPOINT_FISHER"
    ]["top1_regret"].eq(0.0).all()


def test_trust_region_outer_split_has_no_task_feature():
    rows = []
    for sample in range(4):
        task = "a" if sample < 2 else "b"
        for index in range(20):
            value = index / 20.0
            rows.append(
                {
                    "sample_id": "s%d" % sample,
                    "task": task,
                    "anchor": 16,
                    "horizon_offset": 1,
                    "candidate_id": "c%d" % index,
                    "g3_relative_error": 0.05 if value <= 0.6 else 0.5,
                    "g3_midpoint_fisher": value,
                    "exact_kl": value,
                    "trust_t0_base_fisher_distance": value,
                    "trust_t1_fisher_margin_ratio": value,
                    "trust_t2_top_switch_margin_ratio": value,
                    "trust_t3_g2_g3_disagreement": value,
                }
            )
    cfg = load_discovery_config(
        str(ROOT / "configs" / "stages" / "independent_fisher_config.yaml")
    ).independent_fisher
    summary = analyze_trust_region(pd.DataFrame(rows), cfg)
    assert not summary["thresholds_use_task_id"]
    assert not summary["heldout_sequence_leakage"]


@pytest.mark.parametrize(
    "filename,schema",
    [
        ("pullback_operating_point_rows.parquet", PULLBACK_SCHEMA),
        ("state_action_cross_term_rows.parquet", CROSS_SCHEMA),
        ("pullback_low_rank_rows.parquet", LOW_RANK_SCHEMA),
        ("q_state_envelope_rows.parquet", Q_STATE_SCHEMA),
        ("q_refresh_policy_rows.parquet", REFRESH_SCHEMA),
    ],
)
def test_skipped_parquet_schema_is_stable(tmp_path, filename, schema):
    write_later_stage_skips(tmp_path, "Stage A-prime", "test")
    frame = pd.read_parquet(tmp_path / filename)
    assert list(frame.columns) == list(schema)
    assert len(frame) == 0


def test_skipped_json_is_explicit(tmp_path):
    write_later_stage_skips(tmp_path, "Stage A-prime", "test")
    value = json.loads(
        (tmp_path / "pullback_jvp_validation_summary.json").read_text()
    )
    assert value["status"] == "not_run_by_preregistered_gate"
    assert value["blocking_stage"] == "Stage A-prime"
    assert value["rows"] == 0


def test_no_nan_inf_in_adaptive_geometry():
    z = torch.randn(257)
    compressed = z + 0.2 * torch.randn(257)
    geometry, curvature = adaptive_fisher_geometry(
        z, compressed, 16, 1.0e-8, 1.0e-10, 50
    )
    numeric = [
        value
        for value in list(geometry.values()) + list(curvature.values())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    assert np.isfinite(np.asarray(numeric, dtype=np.float64)).all()


def test_state_action_cross_term_decomposition_identity():
    rng = np.random.default_rng(91)
    probability = rng.dirichlet(np.ones(17))
    state = rng.normal(size=17)
    direct = rng.normal(size=17)
    result = state_action_energy_decomposition(
        probability, state, direct
    )
    assert result["decomposition_abs_error"] < 1.0e-12


def test_cross_covariance_is_symmetric():
    rng = np.random.default_rng(92)
    probability = rng.dirichlet(np.ones(11))
    left = rng.normal(size=11)
    right = rng.normal(size=11)
    assert covariance_energy(
        probability, left, right
    ) == pytest.approx(covariance_energy(probability, right, left))


def test_cross_term_cauchy_schwarz_and_scalar_bound():
    rng = np.random.default_rng(93)
    for _ in range(100):
        probability = rng.dirichlet(np.ones(9))
        state = rng.normal(size=9)
        direct = rng.normal(size=9)
        result = state_action_energy_decomposition(
            probability, state, direct
        )
        assert result["cauchy_schwarz_holds"]
        assert result["scalar_bound_holds"]


def test_psd_interaction_parameterization():
    for alpha, beta, eta in (
        (1.0, 2.0, -100.0),
        (-3.0, 4.0, 0.0),
        (5.0, -2.0, 100.0),
    ):
        a, b, gamma = psd_interaction_parameters(alpha, beta, eta)
        assert a >= 0.0 and b >= 0.0
        assert abs(gamma) <= np.sqrt(a * b) + 1.0e-12


def test_principal_angles_identical_subspace_are_zero():
    left = np.eye(8)[:, :3]
    angles = principal_angles(left, left)
    assert np.allclose(angles, 0.0)


def test_principal_angles_orthogonal_subspaces_are_pi_over_two():
    left = np.eye(8)[:, :2]
    right = np.eye(8)[:, 2:4]
    angles = principal_angles(left, right)
    assert np.allclose(angles, np.pi / 2.0)
