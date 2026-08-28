"""Pure metrics for the dynamic-horizon oracle analysis."""
from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Tuple

import numpy as np
from scipy.stats import rankdata


def _top(values: np.ndarray, valid: np.ndarray, count: int) -> np.ndarray:
    rows = np.flatnonzero(valid & np.isfinite(values))
    take = min(int(count), int(rows.size))
    if take <= 0:
        return np.asarray([], dtype=np.int64)
    return rows[np.lexsort((rows, -values[rows]))[:take]]


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    x = rankdata(left[valid], method="average")
    y = rankdata(right[valid], method="average")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fixed_metrics(
    score: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    topk: int,
) -> Dict[str, float]:
    recalls = []
    correlations = []
    for cycle in range(score.shape[0]):
        valid = eligible[cycle] & np.isfinite(score[cycle]) & np.isfinite(target[cycle])
        if int(valid.sum()) < 3:
            continue
        predicted = set(_top(score[cycle], valid, topk).tolist())
        oracle = set(_top(target[cycle], valid, topk).tolist())
        recalls.append(len(predicted & oracle) / max(1, len(oracle)))
        correlations.append(_spearman(score[cycle, valid], target[cycle, valid]))
    return {
        "future_topk_recall": float(np.nanmean(recalls)),
        "mean_step_spearman": float(np.nanmean(correlations)),
        "decisions": int(len(recalls)),
    }


def _percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    count = int(valid.sum())
    if count:
        output[valid] = rankdata(values[valid], method="average") / float(count)
    return output


def _token_time_oracle_metrics(
    scores: Mapping[str, np.ndarray],
    target: np.ndarray,
    eligible: np.ndarray,
    topk: int,
) -> Tuple[Dict[str, float], Counter]:
    recalls = []
    correlations = []
    choices: Counter = Counter()
    methods = list(scores)
    for cycle in range(target.shape[0]):
        valid = eligible[cycle] & np.isfinite(target[cycle])
        for values in scores.values():
            valid &= np.isfinite(values[cycle])
        if int(valid.sum()) < 3:
            continue
        target_rank = _percentile(target[cycle], valid)
        candidate_ranks = np.stack(
            [_percentile(scores[name][cycle], valid) for name in methods], axis=0
        )
        selected = np.argmin(
            np.abs(candidate_ranks - target_rank[None, :])[:, valid], axis=0
        )
        oracle_score = np.full(target.shape[1], np.nan, dtype=np.float64)
        valid_rows = np.flatnonzero(valid)
        oracle_score[valid_rows] = candidate_ranks[selected, valid_rows]
        for index in selected.tolist():
            choices[methods[int(index)]] += 1
        predicted = set(_top(oracle_score, valid, topk).tolist())
        oracle = set(_top(target[cycle], valid, topk).tolist())
        recalls.append(len(predicted & oracle) / max(1, len(oracle)))
        correlations.append(_spearman(oracle_score[valid], target[cycle, valid]))
    return (
        {
            "future_topk_recall": float(np.nanmean(recalls)),
            "mean_step_spearman": float(np.nanmean(correlations)),
            "decisions": int(len(recalls)),
        },
        choices,
    )
