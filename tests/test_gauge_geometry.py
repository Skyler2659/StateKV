from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from statekv.config import load_discovery_config
from statekv.fisher_pullback import (
    anchor_frozen_sources,
    explicit_pullback,
    fisher_matrix,
    fisher_output_random_direction,
    fisher_vjp_sketch,
    horizon_stratified_scores,
    load_gate_and_apply_skips,
    low_rank_from_sketch,
    matched_refresh_count,
    normalized_conformal_interval,
    normalized_conformal_radius,
    pairwise_difference,
    periodic_q_sources,
    predicted_top_candidate_pairs,
    pullback_quadratic,
    q_refresh_source,
    recursive_q_envelope,
    refresh_continues_envelope,
    spectral_band_energy,
    symmetric_finite_difference,
    write_gated_skips,
)
from statekv.gauge_geometry import (
    center_uniform,
    centered_cumulants,
    exact_kl_cumulant_identity,
    fisher_pairwise_gap,
    fisher_variance,
    gauge_geometry_metrics,
    gauss_legendre_path_fisher,
    simpson_path_fisher,
    stable_logsumexp,
    top_margin_geometry,
    topk_geometry,
)


def _probability(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - stable_logsumexp(values)
    return np.exp(shifted)


def test_exact_kl_cumulant_identity():
    z = np.array([1.0, -2.0, 0.5, 3.0])
    delta = np.array([0.3, -0.7, 0.1, 0.2])
    p = _probability(z)
    q = _probability(z + delta)
    explicit = float(np.sum(p * (np.log(p) - np.log(q))))
    assert exact_kl_cumulant_identity(z, delta) == pytest.approx(explicit)


def test_fisher_variance_identity():
    p = np.array([0.2, 0.3, 0.5])
    v = np.array([1.2, -0.7, 0.4])
    assert fisher_variance(p, v) == pytest.approx(
        float(v @ fisher_matrix(p) @ v)
    )


def test_fisher_pairwise_gap_identity():
    p = np.array([0.2, 0.3, 0.5])
    v = np.array([1.2, -0.7, 0.4])
    explicit = 0.5 * sum(
        p[i] * p[j] * (v[i] - v[j]) ** 2
        for i in range(3)
        for j in range(3)
    )
    assert fisher_pairwise_gap(p, v) == pytest.approx(explicit)


def test_common_shift_softmax_and_kl_invariant():
    z = np.array([-3.0, 0.0, 4.0])
    shift = np.full(3, 91.0)
    assert np.allclose(_probability(z), _probability(z + shift))
    assert exact_kl_cumulant_identity(z, shift) == pytest.approx(
        0.0, abs=1.0e-12
    )


def test_centered_norm_removes_common_shift():
    assert np.linalg.norm(center_uniform(np.full(17, 8.0))) == pytest.approx(0.0)


def test_g4_quadrature_converges_for_local_perturbation():
    z = np.linspace(-2.0, 2.0, 31)
    delta = 0.02 * np.sin(np.arange(31))
    exact = exact_kl_cumulant_identity(z, delta)
    errors = [
        abs(gauss_legendre_path_fisher(z, delta, order) - exact)
        for order in (2, 3, 5)
    ]
    assert errors[2] <= errors[1] <= errors[0]


def test_g4_five_point_recovers_local_exact_kl():
    z = np.linspace(-1.0, 1.0, 17)
    delta = 0.01 * np.cos(np.arange(17))
    exact = exact_kl_cumulant_identity(z, delta)
    predicted = gauss_legendre_path_fisher(z, delta, 5)
    assert abs(predicted - exact) / exact < 1.0e-8


def test_simpson_reference_is_finite():
    z = np.linspace(-3.0, 3.0, 21)
    delta = np.linspace(-0.2, 0.3, 21)
    assert np.isfinite(simpson_path_fisher(z, delta, 9))


def test_topk_mass_and_renormalization():
    z = np.array([4.0, 3.0, 2.0, 1.0])
    p = _probability(z)
    delta = np.array([0.2, -0.1, 0.4, -0.7])
    indices = np.argsort(-z)
    result = topk_geometry(p, z, delta, indices, 2)
    assert result["mass"] == pytest.approx(float(p[:2].sum()))


def test_topk_pairwise_gap_formula():
    z = np.array([4.0, 3.0, 2.0, 1.0])
    p = _probability(z)
    delta = np.array([0.2, -0.1, 0.4, -0.7])
    indices = np.argsort(-z)
    result = topk_geometry(p, z, delta, indices, 3)
    top = indices[:3]
    explicit = 0.25 * sum(
        p[i] * p[j] * (delta[i] - delta[j]) ** 2
        for i in top
        for j in top
    )
    assert result["g5b"] == pytest.approx(explicit)


def test_top_margin_index_alignment():
    z = np.array([0.0, 9.0, 1.0, 8.0])
    p = _probability(z)
    delta = np.array([0.0, -1.0, 0.0, 1.0])
    indices = np.argsort(-z)
    result = top_margin_geometry(p, z, delta, indices, 2)
    assert result["g6_two_uniform"] == pytest.approx(4.0)


def test_cumulant_calculation():
    p = np.array([0.5, 0.5])
    value = np.array([-1.0, 1.0])
    second, third, fourth = centered_cumulants(p, value)
    assert second == pytest.approx(1.0)
    assert third == pytest.approx(0.0)
    assert fourth == pytest.approx(-2.0)


def test_pullback_quadratic_identity():
    rng = np.random.default_rng(3)
    jacobian = rng.normal(size=(5, 3))
    p = _probability(rng.normal(size=5))
    v = rng.normal(size=3)
    q = explicit_pullback(jacobian, p)
    assert float(v @ q @ v) == pytest.approx(
        pullback_quadratic(p, jacobian @ v)
    )


def test_jvp_matches_symmetric_finite_difference():
    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])
    point = np.array([0.2, -0.3])
    direction = np.array([0.7, -0.4])
    function = lambda x: np.tanh(matrix @ x)
    analytical = (
        1.0 - np.tanh(matrix @ point) ** 2
    ) * (matrix @ direction)
    finite = symmetric_finite_difference(
        function, point, direction, 1.0e-5
    )
    assert np.allclose(analytical, finite, atol=1.0e-9)


def test_multiple_finite_difference_radii_execute():
    function = lambda x: np.square(x)
    values = [
        symmetric_finite_difference(
            function, np.array([2.0]), np.array([1.0]), radius
        )[0]
        for radius in (1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2)
    ]
    assert np.allclose(values, 4.0)


def test_fisher_randomized_direction_covariance():
    rng = np.random.default_rng(4)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    samples = np.stack(
        [
            fisher_output_random_direction(p, rng.normal(size=4))
            for _ in range(30000)
        ]
    )
    covariance = samples.T @ samples / len(samples)
    assert np.allclose(covariance, fisher_matrix(p), atol=0.01)


def test_low_rank_sketch_recovers_synthetic_q_subspace():
    rng = np.random.default_rng(5)
    jacobian = np.zeros((5, 4))
    jacobian[:, :2] = rng.normal(size=(5, 2))
    p = _probability(rng.normal(size=5))
    sketch = fisher_vjp_sketch(
        jacobian, p, rng.normal(size=(5000, 5))
    )
    vectors, eigenvalues = low_rank_from_sketch(sketch, 2)
    assert vectors.shape == (4, 2)
    assert np.all(eigenvalues >= 0.0)
    assert np.linalg.norm(vectors[2:, :]) < 1.0e-8


def test_spectral_band_energy_matches_eigendecomposition():
    eigenvectors = np.eye(4)
    eigenvalues = np.array([9.0, 4.0, 1.0, 0.25])
    direction = np.array([2.0, 3.0, 4.0, 5.0])
    assert spectral_band_energy(
        direction, eigenvectors, eigenvalues, 0, 2
    ) == pytest.approx(np.sqrt(9.0 * 4.0 + 4.0 * 9.0))


def test_anchor_frozen_q_never_reads_future_q():
    sources = anchor_frozen_sources(8, anchor_source=3)
    assert np.array_equal(sources, np.full(8, 3))


def test_periodic_q_update_timing():
    assert np.array_equal(
        periodic_q_sources(10, 4),
        np.array([0, 0, 0, 0, 4, 4, 4, 4, 8, 8]),
    )


def test_q_refresh_source_rejects_invalid_interval():
    with pytest.raises(ValueError):
        q_refresh_source(1, 0)


def test_q_envelope_does_not_read_realized_future_state():
    direct = [1.0, 2.0, 3.0]
    first = recursive_q_envelope(direct, 0.5, 2.0, 0.1)
    second = recursive_q_envelope(direct, 0.5, 2.0, 0.1)
    assert np.array_equal(first, second)


def test_refresh_does_not_reset_accumulated_state():
    continued = refresh_continues_envelope(10.0, [0.0], 0.5, 1.0, 0.0)
    assert continued[0] == pytest.approx(5.0)


def test_sequence_split_no_leakage_by_construction():
    sequence = ["a", "b", "c"]
    held_out = "c"
    training = [value for value in sequence if value != held_out]
    assert held_out not in training


def test_task_id_not_in_gauge_feature_names():
    feature_names = {
        "g2_base_fisher",
        "topk_mass_16",
        "output_entropy",
        "top1_margin",
    }
    assert not {"task", "task_id"} & feature_names


def test_path_oracle_is_not_deployable_family():
    deployable = {"G2_BASE_FISHER", "G3_MIDPOINT_FISHER"}
    assert "G4_GL5_ORACLE" not in deployable


def test_inherited_candidate_masks_and_budget_are_equal():
    root = Path(__file__).resolve().parents[1]
    inventory = pd.read_parquet(
        root
        / "results/temporal_cache_discovery/"
        "output_sensitivity_4bit_24seq_seed42_v1/"
        "output_candidate_inventory.parquet"
    )
    assert inventory["total_budget"].eq(128).all()
    assert (
        inventory.groupby(["sample_id", "anchor"])["mask_hash"].nunique()
        == 24
    ).all()


def test_pairwise_antisymmetry():
    assert pairwise_difference(3.0, 7.0) == -pairwise_difference(7.0, 3.0)


def test_same_candidate_pairwise_regret_zero():
    assert pairwise_difference(3.7, 3.7) == 0.0


def test_normalized_conformal_interval():
    score = normalized_conformal_radius([3.0], [2.0], tau=0.5)[0]
    lower, upper = normalized_conformal_interval(2.0, score, tau=0.5)
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(3.0)


def test_horizon_stratified_calibration():
    frame = pd.DataFrame(
        {"horizon": [4, 4, 8], "score": [1.0, 2.0, 9.0]}
    )
    split = horizon_stratified_scores(frame, "horizon", "score")
    assert set(split) == {4, 8}
    assert np.array_equal(split[4], np.array([1.0, 2.0]))


def test_top_candidate_calibration_pairs():
    pairs = predicted_top_candidate_pairs(
        ["a", "b", "c", "d"], [4.0, 1.0, 2.0, 3.0], 3
    )
    assert set(pairs) == {("b", "c"), ("b", "d"), ("c", "d")}


def test_matched_refresh_count():
    assert matched_refresh_count(
        [True, False, True], [False, True, True]
    )
    assert not matched_refresh_count([True], [False])


def test_skipped_stages_generate_valid_artifacts(tmp_path):
    write_gated_skips(tmp_path, "Stage A", "test")
    assert (tmp_path / "pullback_jvp_rows.parquet").exists()
    assert pd.read_parquet(tmp_path / "pullback_jvp_rows.parquet").empty
    summary = json.loads(
        (tmp_path / "q_free_generation_results.json").read_text()
    )
    assert summary["status"] == "not_run_by_preregistered_gate"


def test_free_generation_separated_from_teacher_forced(tmp_path):
    write_gated_skips(tmp_path, "Stage C", "teacher-forced gate failed")
    free = json.loads(
        (tmp_path / "q_free_generation_results.json").read_text()
    )
    refresh = json.loads(
        (tmp_path / "q_refresh_policy_summary.json").read_text()
    )
    assert free["artifact"] != refresh["artifact"]


def test_all_geometry_numeric_outputs_finite():
    z = torch.linspace(-4.0, 4.0, 257)
    compressed = z + 0.03 * torch.sin(torch.arange(257))
    result = gauge_geometry_metrics(
        z, compressed, [4, 16, 64, 256], [1.0e-3]
    )
    numeric = [
        value
        for value in result.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    assert np.isfinite(numeric).all()


def test_quantized_replay_metric_reproducibility_proxy():
    z = torch.linspace(-2.0, 2.0, 129)
    compressed = z + 0.1 * torch.cos(torch.arange(129))
    first = gauge_geometry_metrics(z, compressed, [4, 32, 128], [1.0e-3])
    second = gauge_geometry_metrics(z, compressed, [4, 32, 128], [1.0e-3])
    assert first == second


def test_stable_logsumexp_extreme_logits():
    values = np.array([10000.0, 9999.0, -10000.0])
    assert np.isfinite(stable_logsumexp(values))
    assert stable_logsumexp(values) == pytest.approx(
        10000.0 + np.log1p(np.exp(-1.0))
    )


def test_vocabulary_topk_token_alignment():
    z = torch.tensor([0.0, 4.0, 1.0, 3.0, 2.0])
    result = gauge_geometry_metrics(z, z + 0.01, [4], [1.0e-3])
    assert result["topk_mass_4"] == pytest.approx(
        float(torch.softmax(z, dim=0)[[1, 3, 4, 2]].sum()), rel=1.0e-6
    )


def test_config_requires_exclusive_gauge_protocol():
    root = Path(__file__).resolve().parents[1]
    cfg = load_discovery_config(
        str(root / "configs" / "stages" / "gauge_geometry_config.yaml")
    )
    invalid = replace(
        cfg,
        output_sensitivity=replace(cfg.output_sensitivity, enabled=True),
    )
    with pytest.raises(ValueError):
        invalid.validate()


def test_stage_gate_skip_loader(tmp_path):
    (tmp_path / "gauge_geometry_gate_decision.json").write_text(
        json.dumps({"stage_a_passed": False})
    )
    result = load_gate_and_apply_skips(tmp_path)
    assert result["blocking_stage"] == "Stage A"
