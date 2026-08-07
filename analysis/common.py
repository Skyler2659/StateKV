"""Shared I/O and statistical helpers for offline theory-discovery analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_TABLES = {
    "reference": "reference_trajectories.parquet",
    "candidate": "candidate_sets.parquet",
    "step": "step_losses.parquet",
    "horizon": "horizon_losses.parquet",
    "temporal": "temporal_signals.parquet",
}


def ensure_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_required_tables(input_dir: Path) -> Dict[str, pd.DataFrame]:
    input_dir = Path(input_dir).resolve()
    missing = [
        filename
        for filename in REQUIRED_TABLES.values()
        if not (input_dir / filename).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing required experiment outputs: %s" % ", ".join(missing)
        )
    return {
        name: pd.read_parquet(input_dir / filename)
        for name, filename in REQUIRED_TABLES.items()
    }


def parse_json(value: Any, label: str) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        raise TypeError("%s must be JSON text, got %s" % (label, type(value).__name__))
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON in %s: %s" % (label, exc)) from exc


def write_dual(frame: pd.DataFrame, stem: Path) -> Tuple[Path, Path]:
    stem = Path(stem)
    ensure_directory(stem.parent)
    parquet = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    return parquet, csv


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_raise(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError("%s missing columns: %s" % (label, missing))
    values = frame[list(columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("%s contains NaN/Inf in required columns" % label)


def robust_zscore(values: Sequence[float], epsilon: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(array)
    mad = np.nanmedian(np.abs(array - median))
    scale = max(1.4826 * float(mad), float(epsilon))
    return (array - median) / scale


def cluster_bootstrap_statistic(
    frame: pd.DataFrame,
    cluster: str,
    value: str,
    statistic: str,
    rng: np.random.Generator,
    draws: int,
) -> Tuple[float, float, float, int]:
    clean = frame[[cluster, value]].replace([np.inf, -np.inf], np.nan).dropna()
    clusters = list(clean[cluster].drop_duplicates())
    if not clusters:
        return float("nan"), float("nan"), float("nan"), 0
    grouped = {key: group[value].to_numpy(np.float64) for key, group in clean.groupby(cluster)}

    def evaluate(rows: Iterable[np.ndarray]) -> float:
        array = np.concatenate(list(rows))
        if statistic == "median":
            return float(np.median(array))
        if statistic == "mean":
            return float(np.mean(array))
        raise ValueError("unsupported statistic=%s" % statistic)

    point = evaluate(grouped[key] for key in clusters)
    boot = []
    for _ in range(int(draws)):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        boot.append(evaluate(grouped[key] for key in sampled))
    return (
        point,
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
        len(clusters),
    )


def cluster_bootstrap_correlation(
    frame: pd.DataFrame,
    cluster: str,
    x: str,
    y: str,
    rng: np.random.Generator,
    draws: int,
) -> Dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr

    clean = (
        frame[[cluster, x, y]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    clusters = list(clean[cluster].drop_duplicates())

    def correlation(sample: pd.DataFrame, method: str) -> float:
        xv = sample[x].to_numpy(np.float64)
        yv = sample[y].to_numpy(np.float64)
        if len(sample) < 3 or np.std(xv) <= 0 or np.std(yv) <= 0:
            return float("nan")
        value = (
            spearmanr(xv, yv).statistic
            if method == "spearman"
            else pearsonr(xv, yv).statistic
        )
        return float(value)

    point_p = correlation(clean, "pearson")
    point_s = correlation(clean, "spearman")
    boot_p, boot_s = [], []
    grouped = {key: group for key, group in clean.groupby(cluster)}
    for _ in range(int(draws)):
        if not clusters:
            break
        sampled = rng.choice(clusters, len(clusters), replace=True)
        sample = pd.concat(
            [grouped[key].assign(_bootstrap_cluster=index) for index, key in enumerate(sampled)],
            ignore_index=True,
        )
        p = correlation(sample, "pearson")
        s = correlation(sample, "spearman")
        if np.isfinite(p):
            boot_p.append(p)
        if np.isfinite(s):
            boot_s.append(s)

    def interval(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    p_low, p_high = interval(boot_p)
    s_low, s_high = interval(boot_s)
    return {
        "n_rows": int(len(clean)),
        "n_clusters": int(len(clusters)),
        "pearson": point_p,
        "pearson_ci_low": p_low,
        "pearson_ci_high": p_high,
        "spearman": point_s,
        "spearman_ci_low": s_low,
        "spearman_ci_high": s_high,
    }


def json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_directory(Path(path).parent)

    def default(value: Any) -> Any:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError("not JSON serializable: %s" % type(value).__name__)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=default)
