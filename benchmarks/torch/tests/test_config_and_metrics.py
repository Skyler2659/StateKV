from pathlib import Path

import pytest

from kvbench.config import ConfigurationError, load_experiment
from kvbench.evaluation.metrics import evaluate_prediction


def test_config_composition_environment_and_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "/models/qwen")
    base = tmp_path / "base.yaml"
    base.write_text(
        "runtime:\n  device: cpu\nmodel:\n  name: ${MODEL_DIR}\n  dtype: float32\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        "include: base.yaml\nbudget:\n  cache_budget: 16\n",
        encoding="utf-8",
    )
    cfg = load_experiment(str(experiment), ["budget.cache_budget=8"])
    assert cfg.model.name == "/models/qwen"
    assert cfg.budget.cache_budget == 8


def test_unknown_config_field_fails(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("runtime:\n  magic: true\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown fields"):
        load_experiment(str(path))


def test_ruler_qa_uses_partial_match_but_niah_requires_all():
    qa = evaluate_prediction("ruler", "qa_1", "answer beta", ["alpha", "beta"])
    niah = evaluate_prediction(
        "ruler", "niah_multikey_1", "answer beta", ["alpha", "beta"]
    )
    assert qa["score"] == 100.0
    assert niah["score"] == 50.0


def test_longbench_count_and_retrieval_match_official_semantics():
    count = evaluate_prediction("longbench", "passage_count", "There are 4.", ["4"])
    retrieval = evaluate_prediction(
        "longbench", "passage_retrieval_en", "Paragraph 7", ["Paragraph 7"]
    )
    assert count["score"] == 100.0
    assert retrieval["score"] == 100.0

