import torch

from kvbench.analysis.diagnostics import find_subsequence_positions


def test_evidence_alignment_finds_all_duplicate_occurrences():
    assert find_subsequence_positions([1, 2, 3, 1, 2], [1, 2]) == [0, 1, 3, 4]


def test_evidence_alignment_handles_empty_and_missing_patterns():
    assert find_subsequence_positions([1, 2], []) == []
    assert find_subsequence_positions([1, 2], [3]) == []

