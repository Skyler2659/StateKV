"""Counterfactual-utility phase task loading (no model inference).

Covers the new load_discovery_tasks families:
- ruler_variable_tracking (mlx-side RULER generator wrapper)
- ruler_niah_multiquery (shared-value multi-query synthetic generator)
- passage_retrieval_en / hotpotqa (official LongBench, no fallback)

LongBench tests skip when the local HF snapshot cannot serve the task.
"""
from __future__ import annotations

import pytest

from statekv.causal_existence import sample_id_for, task_overrides
from statekv.config import DiscoveryConfig
from statekv.tasks import load_discovery_tasks


PHASE_SEED = 20260820


def _load(task_name: str, settings: dict):
    cfg = DiscoveryConfig(tasks={task_name: dict(settings)})
    cfg.runtime.seed = PHASE_SEED
    return load_discovery_tasks(cfg)


def _check_synthetic_family(task_name: str, settings: dict, task_string: str) -> None:
    settings = {"num_samples": 2, "sample_offset": 200, **settings}
    samples_a, events_a = _load(task_name, settings)
    samples_b, _ = _load(task_name, settings)

    assert len(samples_a) == 2
    # Deterministic: the same call twice yields identical prompts.
    assert [sample.prompt for sample in samples_a] == [
        sample.prompt for sample in samples_b
    ]
    # sample_offset slicing: indices 200/201 match sample_id_for, and the
    # offset-200 sample differs from the offset-201 sample.
    assert [sample.sample_id for sample in samples_a] == [
        sample_id_for(task_name, 200),
        sample_id_for(task_name, 201),
    ]
    (other,), _ = _load(task_name, {**settings, "num_samples": 1, "sample_offset": 201})
    assert other.prompt == samples_a[1].prompt
    assert other.prompt != samples_a[0].prompt

    for sample in samples_a:
        assert sample.prompt.strip()
        assert sample.references and str(sample.references[0]).strip()
        assert sample.answer_text and str(sample.answer_text).strip()
        assert sample.task == task_string
        assert sample.metadata["dataset_official"] is False
        assert sample.metadata["requested_discovery_task"] == task_name

    (event,) = events_a
    assert event["task"] == task_name
    assert event["dataset_official"] is False
    assert event["sample_offset"] == 200
    assert event["count"] == 2


def test_ruler_variable_tracking_offset_200() -> None:
    _check_synthetic_family(
        "ruler_variable_tracking",
        {"context_length": 768, "n_variables": 8},
        task_string="vt",
    )


def test_ruler_niah_multiquery_offset_200() -> None:
    settings = {"context_length": 768, "n_queries": 4}
    _check_synthetic_family("ruler_niah_multiquery", settings, task_string="niah_multiquery")
    samples, events = _load(
        "ruler_niah_multiquery", {"num_samples": 2, "sample_offset": 200, **settings}
    )
    (event,) = events
    assert event["source"] == "repository_synthetic_ruler_niah_multiquery"
    assert event["n_queries"] == 4
    for sample in samples:
        # Multi-query semantics: n_queries distinct keys share one value and
        # all keys are queried; the reference is the shared value.
        assert sample.metadata["n_queries"] == 4
        assert len(sample.metadata["evidence_texts"]) == 4
        assert len({sample.references[0]} | {sample.metadata["shared_value"]}) == 1
        assert sample.prompt.count("What is the special magic number for") == 4


def _load_longbench_or_skip(task_name: str, sample_indices):
    try:
        return _load(
            task_name,
            {"num_samples": len(sample_indices), "sample_indices": sample_indices},
        )
    except Exception as exc:  # dataset not in the local HF cache
        pytest.skip(
            "LongBench %s not loadable offline: %s: %s"
            % (task_name, type(exc).__name__, exc)
        )


def _check_longbench_family(task_name: str) -> None:
    # The local THUDM/LongBench snapshot has exactly 200 rows per task, so
    # the phase's 200-279 index range cannot be served; use the last valid
    # rows for determinism/slicing checks and assert out-of-range raises.
    samples_a, events_a = _load_longbench_or_skip(task_name, [198, 199])
    samples_b, _ = _load_longbench_or_skip(task_name, [198, 199])

    assert len(samples_a) == 2
    assert [sample.prompt for sample in samples_a] == [
        sample.prompt for sample in samples_b
    ]
    assert [sample.sample_id for sample in samples_a] == [
        sample_id_for(task_name, 198),
        sample_id_for(task_name, 199),
    ]
    assert samples_a[0].prompt != samples_a[1].prompt

    for sample in samples_a:
        assert sample.prompt.strip()
        assert sample.references and str(sample.references[0]).strip()
        assert sample.answer_text and str(sample.answer_text).strip()
        assert sample.task == task_name
        assert sample.metadata["dataset_official"] is True
        assert sample.metadata["requested_discovery_task"] == task_name

    (event,) = events_a
    assert event["task"] == task_name
    assert event["source"] == "official_longbench_%s" % task_name
    assert event["dataset_official"] is True
    assert event["count"] == 2

    # Fail loudly (no synthetic fallback) when indices exceed the dataset.
    with pytest.raises(Exception):
        _load(
            task_name,
            {"num_samples": 2, "sample_indices": [200, 201]},
        )


def test_passage_retrieval_en_longbench() -> None:
    _check_longbench_family("passage_retrieval_en")


def test_hotpotqa_longbench() -> None:
    _check_longbench_family("hotpotqa")


def test_sample_id_for_covers_new_families() -> None:
    assert sample_id_for("ruler_variable_tracking", 200) == "synthetic_vt_200"
    assert (
        sample_id_for("ruler_niah_multiquery", 200)
        == "synthetic_niah_multiquery_200"
    )
    assert sample_id_for("passage_retrieval_en", 200) == "passage_retrieval_en:200"
    assert sample_id_for("hotpotqa", 200) == "hotpotqa:200"


def test_task_overrides_cover_new_families() -> None:
    config = {
        "task_families": [
            "ruler_variable_tracking",
            "ruler_niah_multiquery",
            "passage_retrieval_en",
            "hotpotqa",
        ],
        "task_settings": {
            "ruler_variable_tracking": {"context_length": 768},
            "ruler_niah_multiquery": {"context_length": 768, "n_queries": 4},
            "passage_retrieval_en": {"max_words": 700},
            "hotpotqa": {"max_words": 700},
        },
        "split_indices": {
            "train": [200, 201],
            "validation": [202],
            "fresh_test": [203],
        },
    }
    tasks = task_overrides(config)
    assert tasks["ruler_variable_tracking"]["sample_offset"] == 200
    assert tasks["ruler_variable_tracking"]["num_samples"] == 4
    assert tasks["ruler_niah_multiquery"]["sample_offset"] == 200
    assert tasks["ruler_niah_multiquery"]["n_queries"] == 4
    assert tasks["passage_retrieval_en"]["sample_indices"] == [200, 201, 202, 203]
    assert tasks["hotpotqa"]["sample_indices"] == [200, 201, 202, 203]
