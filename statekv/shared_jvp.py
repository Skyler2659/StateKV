"""Randomized shared-JVP pullback primitives for low-cost action scoring."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def randomized_action_basis(
    actions: np.ndarray,
    rank: int,
    *,
    seed: int,
    oversampling: int = 4,
) -> np.ndarray:
    """Build an ordered randomized range basis for row-wise action vectors."""

    matrix = np.asarray(actions, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("actions must be a non-empty matrix")
    target = min(int(rank), min(matrix.shape))
    if target < 1:
        raise ValueError("rank must be positive")
    width = min(matrix.shape[0], target + max(int(oversampling), 0))
    generator = np.random.default_rng(int(seed))
    sample = generator.normal(size=(matrix.shape[0], width))
    range_basis, _ = np.linalg.qr(matrix.T @ sample, mode="reduced")
    compressed = matrix @ range_basis
    _, _, right = np.linalg.svd(compressed, full_matrices=False)
    basis = range_basis @ right[:target].T
    # Fix the arbitrary SVD sign for byte-stable artifacts.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


def categorical_fisher_gram(
    probability: np.ndarray,
    output_directions: np.ndarray,
) -> np.ndarray:
    """Return ``B.T @ (diag(p) - p p.T) @ B`` without forming Fisher."""

    probability = np.asarray(probability, dtype=np.float64)
    directions = np.asarray(output_directions, dtype=np.float64)
    if probability.ndim != 1 or directions.ndim != 2:
        raise ValueError("probability and directions must be vector/matrix")
    if len(probability) != directions.shape[0]:
        raise ValueError("vocabulary dimensions do not align")
    if np.any(probability < 0.0) or not np.isclose(probability.sum(), 1.0):
        raise ValueError("probability must be non-negative and sum to one")
    mean = probability @ directions
    centered = directions - mean[None, :]
    gram = centered.T @ (probability[:, None] * centered)
    return 0.5 * (gram + gram.T)


def randomized_fisher_gram(
    probability: np.ndarray,
    output_directions: np.ndarray,
    sketch_width: int,
    *,
    seed: int,
) -> np.ndarray:
    """Approximate the Fisher Gram with one shared Rademacher output sketch."""

    probability = np.asarray(probability, dtype=np.float64)
    directions = np.asarray(output_directions, dtype=np.float64)
    if int(sketch_width) < 1:
        raise ValueError("sketch_width must be positive")
    if len(probability) != directions.shape[0]:
        raise ValueError("vocabulary dimensions do not align")
    mean = probability @ directions
    weighted = np.sqrt(probability)[:, None] * (
        directions - mean[None, :]
    )
    generator = np.random.default_rng(int(seed))
    signs = generator.integers(
        0,
        2,
        size=(len(probability), int(sketch_width)),
        dtype=np.int8,
    )
    projection = (2.0 * signs.astype(np.float64) - 1.0) / math.sqrt(
        float(sketch_width)
    )
    sketch = projection.T @ weighted
    gram = sketch.T @ sketch
    return 0.5 * (gram + gram.T)


def state_action_scores(
    actions: np.ndarray,
    states: np.ndarray,
    basis: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    """Score many candidate actions under one shared low-rank pullback."""

    actions = np.asarray(actions, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    if actions.shape != states.shape or actions.ndim != 2:
        raise ValueError("actions and states must be aligned matrices")
    if actions.shape[1] != basis.shape[0]:
        raise ValueError("hidden dimensions do not align")
    if gram.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("gram must match basis rank")
    action_coordinates = actions @ basis
    state_coordinates = states @ basis
    linear = np.einsum(
        "ni,ij,nj->n", state_coordinates, gram, action_coordinates
    )
    quadratic = 0.5 * np.einsum(
        "ni,ij,nj->n", action_coordinates, gram, action_coordinates
    )
    return linear + quadratic


def gram_variants(
    probability: np.ndarray,
    output_directions: np.ndarray,
    sketch_widths: list,
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Return full, diagonal, and randomized Fisher Gram variants."""

    full = categorical_fisher_gram(probability, output_directions)
    result = {"full": full, "diagonal": np.diag(np.diag(full))}
    for width in sketch_widths:
        result["randomized_%d" % int(width)] = randomized_fisher_gram(
            probability,
            output_directions,
            int(width),
            seed=int(seed) + int(width),
        )
    return result


__all__ = [
    "categorical_fisher_gram",
    "gram_variants",
    "randomized_action_basis",
    "randomized_fisher_gram",
    "state_action_scores",
]
