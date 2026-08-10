"""External-validity substrate invariant tests.

Covers the machinery guards added for the long-context external-validity
gate (analysis/statekv_external_validity_log.md): stage-config runtime
overrides, the silent prompt-truncation hard fail, and the reasoning-task
metric bucketing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from statekv.config import RuntimeDiscoveryConfig, apply_named_overrides
from statekv.oracle_policy_freegen import _check_prompt_truncation, _metric_row


def test_apply_named_overrides_sets_known_runtime_fields() -> None:
    runtime = RuntimeDiscoveryConfig()
    apply_named_overrides(runtime, {"max_prompt_tokens": 8192}, "runtime")
    assert runtime.max_prompt_tokens == 8192


def test_apply_named_overrides_rejects_unknown_fields() -> None:
    runtime = RuntimeDiscoveryConfig()
    with pytest.raises(ValueError, match="unknown runtime override"):
        apply_named_overrides(runtime, {"not_a_field": 1}, "runtime")


def test_apply_named_overrides_accepts_none() -> None:
    runtime = RuntimeDiscoveryConfig()
    before = runtime.max_prompt_tokens
    apply_named_overrides(runtime, None, "runtime")
    assert runtime.max_prompt_tokens == before


def test_prompt_truncation_guard_raises_by_default() -> None:
    reference = SimpleNamespace(prompt_truncated=True)
    with pytest.raises(RuntimeError, match="middle-truncated"):
        _check_prompt_truncation(reference, "synthetic_niah_86", False)


def test_prompt_truncation_guard_allows_explicit_opt_in() -> None:
    reference = SimpleNamespace(prompt_truncated=True)
    _check_prompt_truncation(reference, "synthetic_niah_86", True)


def test_prompt_truncation_guard_passes_untruncated() -> None:
    reference = SimpleNamespace(prompt_truncated=False)
    _check_prompt_truncation(reference, "synthetic_niah_86", False)


def _fake_runner(text: str) -> SimpleNamespace:
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: text
    )
    return SimpleNamespace(model=SimpleNamespace(tokenizer=tokenizer))


def test_metric_row_buckets_reasoning_separately() -> None:
    sample = SimpleNamespace(
        sample_id="reasoning_long_generation_0",
        task="reasoning_long_generation",
        references=["1234"],
    )
    row = _metric_row(
        _fake_runner("a long derivation ending in 1234"),
        sample,
        "qk_pool",
        [1, 2, 3],
        0.01,
    )
    assert row["task_bucket"] == "Reasoning"
    assert row["needle_retrieval_accuracy"] == 1.0


def test_metric_row_buckets_niah_and_govreport() -> None:
    niah = SimpleNamespace(
        sample_id="synthetic_niah_86",
        task="ruler_niah",
        references=["needle-value"],
    )
    row = _metric_row(
        _fake_runner("the magic value is needle-value"), niah, "qk_pool", [1], 0.0
    )
    assert row["task_bucket"] == "NIAH"
    gov = SimpleNamespace(
        sample_id="gov_report:86",
        task="gov_report",
        references=["a report about budgets"],
    )
    row = _metric_row(
        _fake_runner("a report about budgets"), gov, "qk_pool", [1], 0.0
    )
    assert row["task_bucket"] == "GovReport"


def test_multikey_niah_generator_structure() -> None:
    from statekv.tasks import _synthetic_niah_multikey

    samples = _synthetic_niah_multikey(20260808, 2, 3072, 4)
    assert len(samples) == 2
    sample = samples[0]
    assert len(sample.references) == 4
    assert len(set(sample.references)) >= 2
    for value in sample.references:
        assert value in sample.prompt
    assert len(sample.metadata["needle_depths"]) == 4
    assert sample.task == "niah_multikey_1"


def test_metric_row_multikey_scores_fraction_found() -> None:
    sample = SimpleNamespace(
        sample_id="synthetic_niah_multikey_0",
        task="niah_multikey_1",
        references=["1111111", "2222222", "3333333", "4444444"],
    )
    row = _metric_row(
        _fake_runner("the numbers are 1111111 and 3333333"),
        sample,
        "qk_pool",
        [1, 2],
        0.01,
    )
    assert row["task_bucket"] == "NIAH"
    assert row["needle_retrieval_accuracy"] == 0.5


def test_reasoning_answer_first_prompt_variant() -> None:
    from statekv.tasks import _reasoning_samples

    plain = _reasoning_samples(20260808, 1, 280, answer_first=False)[0]
    first = _reasoning_samples(20260808, 1, 280, answer_first=True)[0]
    assert plain.sample_id == first.sample_id
    assert plain.references == first.references
    assert "final numeric answer on the first line" in first.prompt
    assert "final numeric answer on the first line" not in plain.prompt
    assert first.metadata["answer_first"] is True
