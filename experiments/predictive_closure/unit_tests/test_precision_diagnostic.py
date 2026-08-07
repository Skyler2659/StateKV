from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "experiments/predictive_closure/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from precision_diagnostic import (
    DiagnosticFunctionBoundary,
    checkpoint_metric_row,
    cast_intervention_for_boundary,
    clone_nested_state,
    cosine_diagnostics,
    count_quantized_modules,
    dequantize_reference_model,
    identity_error_metrics,
    numpy_jvp_fd_same_boundary,
    perturbation_entry_metrics,
    symmetric_fd_independent,
)


def test_fd_positive_negative_inputs_are_distinct() -> None:
    base = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    direction = np.array([0.5, 1.0, -0.25], dtype=np.float32)
    metrics = perturbation_entry_metrics(
        base, direction, 1.0e-3, np.float32
    )
    assert not metrics["x_plus_equals_x_minus"]
    assert metrics["effective_plus_minus_norm"] > 0.0
    assert metrics["effective_nonzero_difference_fraction"] == 1.0


def test_fd_uses_independent_base_state() -> None:
    initial = {"cache": np.array([3.0], dtype=np.float64), "calls": []}

    def factory():
        return clone_nested_state(initial)

    def mutating_function(value, state):
        state["cache"] += 1.0
        state["calls"].append(float(value[0]))
        return value + state["cache"]

    result = symmetric_fd_independent(
        mutating_function,
        np.array([2.0]),
        np.array([1.0]),
        1.0e-3,
        factory,
    )
    assert result["plus_state"] is not result["minus_state"]
    assert not np.shares_memory(
        result["plus_state"]["cache"], result["minus_state"]["cache"]
    )
    np.testing.assert_allclose(initial["cache"], [3.0])
    np.testing.assert_allclose(result["fd"], [1.0], rtol=1e-10)


def test_jvp_and_fd_wrap_same_function() -> None:
    calls = []

    def function(value):
        calls.append(np.asarray(value).copy())
        return np.asarray(value) ** 3

    boundary = DiagnosticFunctionBoundary("cube:input->output", function)

    def analytic_jvp(received_boundary, base, direction):
        assert received_boundary is boundary
        return 3.0 * base**2 * direction

    result = numpy_jvp_fd_same_boundary(
        boundary,
        np.array([2.0, -1.5]),
        np.array([0.25, 0.5]),
        1.0e-5,
        analytic_jvp,
    )
    assert result["boundary_name"] == "cube:input->output"
    assert len(calls) == 2
    np.testing.assert_allclose(result["jvp"], result["fd"], rtol=1e-8)


def test_cosine_zero_vector_diagnostics() -> None:
    fd_zero = cosine_diagnostics(np.ones(3), np.zeros(3))
    orthogonal = cosine_diagnostics(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    )
    assert fd_zero["cosine"] == 0.0
    assert fd_zero["cosine_status"] == "right_zero"
    assert orthogonal["cosine"] == 0.0
    assert orthogonal["cosine_status"] == "defined_near_orthogonal"


def test_manual_physical_checkpoint_alignment() -> None:
    probabilities = np.array([0.2, 0.3, 0.5], dtype=np.float64)
    values = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    retained = np.array([0, 2])
    physical_probability = probabilities[retained]
    physical_probability /= physical_probability.sum()
    physical = physical_probability @ values[retained]
    manual_reconstruction = (
        probabilities[retained] @ values[retained]
    ) / probabilities[retained].sum()
    row = checkpoint_metric_row(
        "tiny",
        "candidate",
        "attention_weighted_value",
        physical,
        manual_reconstruction,
        1,
        "absolute",
        token_position=7,
    )
    assert row["shape_match"]
    assert row["token_position"] == 7
    assert row["relative_error"] < 1.0e-12
    assert row["cosine"] > 1.0 - 1.0e-12
    intervention = np.array([1.0e-4, -2.0e-4], dtype=np.float32)
    cast = cast_intervention_for_boundary(intervention, np.float16)
    assert cast.dtype == np.float16
    assert (np.zeros(2, dtype=np.float16) + cast).dtype == np.float16


def test_fp32_reference_avoids_quantized_kernel() -> None:
    import mlx.core as mx
    import mlx.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.QuantizedLinear.from_linear(
                nn.Linear(32, 4, bias=False),
                group_size=32,
                bits=4,
            )

        def __call__(self, value):
            return self.linear(value)

    model = Tiny()
    assert count_quantized_modules(model)["quantized_modules_total"] == 1
    evidence = dequantize_reference_model(model)
    assert evidence["after_quantized_modules_total"] == 0
    assert evidence["quantized_kernel_reachable"] is False
    output = model(mx.ones((1, 32), dtype=mx.float32))
    mx.eval(output)
    assert output.dtype == mx.float32


def test_identity_reports_absolute_and_stable_relative_error() -> None:
    lhs = np.zeros(4, dtype=np.float64)
    rhs = np.full(4, 1.0e-10, dtype=np.float64)
    metrics = identity_error_metrics(lhs, rhs)
    assert metrics["absolute_l2_error"] == 2.0e-10
    assert metrics["raw_relative_error"] > 1.0e20
    assert metrics["stable_relative_error_tau_1em08"] == 0.02
    assert "maximum_absolute_error" in metrics
    assert metrics["lhs_norm"] == 0.0
    assert metrics["rhs_norm"] == 2.0e-10
