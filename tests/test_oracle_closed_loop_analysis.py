from __future__ import annotations

import pandas as pd

from statekv.oracle_closed_loop_analysis import _finite_spearman


def test_finite_spearman_marks_constant_panels_undefined() -> None:
    assert pd.isna(
        _finite_spearman(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))
    )


def test_finite_spearman_recovers_exact_order() -> None:
    assert _finite_spearman(
        pd.Series([3.0, 1.0, 2.0]), pd.Series([30.0, 10.0, 20.0])
    ) == 1.0
