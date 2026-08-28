"""Dependency-light text diagnostics for free-generation reports."""
from __future__ import annotations

from collections import Counter


def ngram_f1(prediction: str, reference: str, n: int) -> float:
    """Compute whitespace-token n-gram F1 used by the legacy reports."""

    left = prediction.split()
    right = reference.split()
    left_counts = Counter(
        tuple(left[index : index + n]) for index in range(max(0, len(left) - n + 1))
    )
    right_counts = Counter(
        tuple(right[index : index + n])
        for index in range(max(0, len(right) - n + 1))
    )
    overlap = sum((left_counts & right_counts).values())
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if not overlap or not left_total or not right_total:
        return 0.0
    precision = overlap / left_total
    recall = overlap / right_total
    return float(2.0 * precision * recall / (precision + recall))


def repetition_4gram_rate(text: str) -> float:
    """Return the fraction of repeated whitespace-token four-grams."""

    tokens = text.split()
    if len(tokens) < 4:
        return 0.0
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    return float(1.0 - len(set(grams)) / len(grams))
