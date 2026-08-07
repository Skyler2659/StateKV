import numpy as np

from experiments.local_truncated_jacobian.scripts.run_l2_formal import (
    available_depths,
    stable_exact_kl,
)


def test_available_depths_preserve_terminal_layer() -> None:
    assert available_depths(0, 28) == [0, 1, 2, 4]
    assert available_depths(26, 28) == [0, 1, 2]


def test_exact_kl_identity() -> None:
    logits = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    assert abs(stable_exact_kl(logits, logits)) <= 1.0e-15


def test_exact_kl_is_positive_for_changed_distribution() -> None:
    full = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    changed = np.array([-1.0, 2.0, 0.5], dtype=np.float64)
    assert stable_exact_kl(full, changed) > 0.0
