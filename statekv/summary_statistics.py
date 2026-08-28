"""Small statistical summaries shared by active experiment reports."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "sample_id",
    samples: int = 2000,
    seed: int = 42,
    statistic: str = "median",
) -> Dict[str, float]:
    """Bootstrap a mean or median after one aggregation per cluster."""

    values = frame.groupby(cluster, sort=True)[value].agg(statistic).to_numpy(
        dtype=np.float64
    )
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "clusters": 0,
        }
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    reducer = np.median if statistic == "median" else np.mean
    for index in range(int(samples)):
        draws[index] = float(reducer(rng.choice(values, size=len(values), replace=True)))
    return {
        "estimate": float(reducer(values)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(values)),
    }
