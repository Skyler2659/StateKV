import numpy as np
import pytest
import torch

from kvbench.config import MethodConfig
from kvbench.scoring.core import ScoreEngine, normalize_scores


def _kv(rows: torch.Tensor, heads: int = 2) -> torch.Tensor:
    return rows.unsqueeze(0).repeat(1, heads, 1, 1)


def test_rank_normalization_uses_average_tie_ranks():
    normalized = normalize_scores(torch.tensor([4.0, 4.0, 1.0]), "rank")
    assert normalized[0].item() == pytest.approx(normalized[1].item())
    assert normalized[0].item() == pytest.approx(0.75)
    assert normalized[2].item() == pytest.approx(0.0)


@pytest.mark.parametrize("mode", ["none", "rank", "minmax", "zscore", "log"])
def test_all_normalizations_are_finite(mode):
    result = normalize_scores(torch.tensor([1.0, 1.0, 3.0]), mode)
    assert result.shape == (3,)
    assert torch.isfinite(result).all()


def test_l2_leverage_shape_nonnegative_and_rank_sum():
    rows = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
    )
    engine = ScoreEngine(MethodConfig(leverage_estimator="l2_exact"), seed=7)
    score, by_head, diagnostics = engine.geometry(_kv(rows), _kv(rows), 0, source="v")
    assert score.shape == (4,)
    assert by_head.shape == (2, 4)
    assert torch.all(score >= 0)
    assert float(score.sum()) == pytest.approx(2.0, abs=1e-5)
    assert diagnostics["head_diagnostics"][0]["effective_rank"] == 2


def test_l2_leverage_is_invariant_to_invertible_feature_transform():
    generator = torch.Generator().manual_seed(3)
    rows = torch.randn(9, 4, generator=generator)
    transform = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0], [0.1, 1.0, 0.0, 0.0],
         [0.0, 0.2, 0.5, 0.0], [0.0, 0.0, 0.3, 1.5]]
    )
    engine = ScoreEngine(MethodConfig(), seed=0)
    left, _, _ = engine.geometry(_kv(rows, 1), _kv(rows, 1), 0, source="v")
    right_rows = rows @ transform
    right, _, _ = engine.geometry(
        _kv(right_rows, 1), _kv(right_rows, 1), 0, source="v"
    )
    assert torch.allclose(left, right, atol=2e-5, rtol=2e-5)


def test_rank_deficient_and_zero_matrices_are_supported():
    rows = torch.tensor([[1.0, 2.0], [2.0, 4.0], [0.0, 0.0]])
    engine = ScoreEngine(MethodConfig(), seed=0)
    score, _, diagnostics = engine.geometry(_kv(rows, 1), _kv(rows, 1), 0)
    assert diagnostics["head_diagnostics"][0]["effective_rank"] == 1
    assert torch.isfinite(score).all()
    zero, _, _ = engine.geometry(_kv(torch.zeros_like(rows), 1), _kv(torch.zeros_like(rows), 1), 0)
    assert torch.equal(zero, torch.zeros(3))


def test_residual_projection_and_ridge_leverage_match_direct_formula():
    rows = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=torch.float32,
    )
    cfg = MethodConfig(residual_lambda=1e-4, residual_lambda_mode="absolute")
    score, by_head, diagnostics = ScoreEngine(cfg, seed=0).residual_v(
        _kv(rows, 1), [0], 0
    )
    gram = rows[[0]].T @ rows[[0]]
    projector = torch.linalg.solve(gram + 1e-4 * torch.eye(2), gram)
    residual = rows - rows @ projector
    residual_gram = residual.T @ residual
    direct = torch.sum(
        residual
        * torch.linalg.solve(residual_gram + 1e-4 * torch.eye(2), residual.T).T,
        dim=-1,
    )
    assert torch.allclose(score, direct, atol=1e-5, rtol=1e-5)
    assert by_head.shape == (1, 4)
    selected_residual = diagnostics["head_diagnostics"][0]["selected_residual_max"]
    assert selected_residual < 1e-3


def test_l1_sketch_is_seeded_and_finite():
    rows = torch.arange(1, 49, dtype=torch.float32).reshape(12, 4)
    cfg = MethodConfig(leverage_estimator="l1_approx", sketch_dim=8)
    first, _, _ = ScoreEngine(cfg, seed=11).geometry(
        _kv(rows, 1), _kv(rows, 1), 0
    )
    second, _, _ = ScoreEngine(cfg, seed=11).geometry(
        _kv(rows, 1), _kv(rows, 1), 0
    )
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert np.all(first.numpy() >= 0)


def test_l2_sketch_matches_exact_when_embedding_is_not_compressed():
    rows = torch.randn(20, 5, generator=torch.Generator().manual_seed(17))
    exact, _, _ = ScoreEngine(
        MethodConfig(leverage_estimator="l2_exact"), seed=3
    ).geometry(_kv(rows, 1), _kv(rows, 1), 0)
    sketched, _, diagnostics = ScoreEngine(
        MethodConfig(leverage_estimator="l2_sketch", sketch_dim=32), seed=3
    ).geometry(_kv(rows, 1), _kv(rows, 1), 0)
    assert torch.allclose(exact, sketched, atol=2e-5, rtol=2e-5)
    assert diagnostics["head_diagnostics"][0]["used_count_sketch"] is False


def test_l2_countsketch_is_seeded_nonnegative_and_finite():
    rows = torch.randn(200, 8, generator=torch.Generator().manual_seed(23))
    cfg = MethodConfig(leverage_estimator="l2_sketch", sketch_dim=32)
    first, _, diagnostics = ScoreEngine(cfg, seed=11).geometry(
        _kv(rows, 1), _kv(rows, 1), 0
    )
    second, _, _ = ScoreEngine(cfg, seed=11).geometry(
        _kv(rows, 1), _kv(rows, 1), 0
    )
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert torch.all(first >= 0)
    assert diagnostics["head_diagnostics"][0]["used_count_sketch"] is True
