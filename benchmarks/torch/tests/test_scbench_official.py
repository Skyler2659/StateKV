import json

import pytest

from kvbench.benchmarks.scbench import SCBenchBenchmark
from kvbench.config import BenchmarkConfig
from kvbench.evaluation.metrics import evaluate_prediction


def _official_metadata(label, task):
    return {
        "scbench_schema": "official_context_multi_turns",
        "query_label": label,
        "query_task": task,
    }


def test_official_scbench_adapter_ports_scdq_prefix_and_labels(tmp_path):
    path = tmp_path / "scbench_kv.jsonl"
    row = {
        "id": 7,
        "context": '{"alpha": "red", "beta": "blue"}',
        "multi_turns": [
            {"input": "What is alpha?", "answer": "red"},
            {"input": "What is beta?", "answer": "blue"},
        ],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = BenchmarkConfig(
        name="scbench",
        task="scbench_kv",
        data_path=str(path),
        num_samples=1,
        require_official=True,
        use_official_prompt=True,
    )
    sample = SCBenchBenchmark(cfg, seed=42).load()[0]
    assert sample.sample_id == "scbench_kv:7"
    assert sample.shared_prefix.startswith("Extract the value corresponding")
    assert sample.shared_prefix.endswith(row["context"])
    assert sample.queries == ["What is alpha?", "What is beta?"]
    assert sample.metadata["query_labels"] == ["red", "blue"]
    assert sample.metadata["official_max_new_tokens"] == 150
    assert sample.metadata["scbench_schema"] == "official_context_multi_turns"


def test_normalized_scbench_is_never_accepted_as_official(tmp_path):
    path = tmp_path / "normalized.jsonl"
    path.write_text(
        json.dumps(
            {"shared_prefix": "prefix", "queries": ["q"], "answers": ["a"]}
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = BenchmarkConfig(
        name="scbench",
        task="scbench_kv",
        data_path=str(path),
        num_samples=1,
        require_official=True,
    )
    with pytest.raises(RuntimeError, match="raw context/multi_turns"):
        SCBenchBenchmark(cfg, seed=42).load()


@pytest.mark.parametrize(
    "task,prediction,label,expected",
    [
        ("scbench_kv", "The value is red.", "red", 100.0),
        ("scbench_passkey", "The passkey is 12345.", "12345", 100.0),
        ("scbench_mf", "17 is the result", 17, 100.0),
        ("scbench_qa_eng", "Snowfield", ["Snowfield"], 100.0),
        ("scbench_choice_eng", "B", ["Snowfield", "B"], 100.0),
        ("scbench_qa_chn", "答案是北京。", ["北京"], 57.1429),
        ("scbench_vt", "alpha was found", ["alpha", "beta"], 50.0),
        ("scbench_many_shot", "LABEL_X", ["label_x"], 100.0),
    ],
)
def test_scbench_task_specific_official_metrics(task, prediction, label, expected):
    result = evaluate_prediction(
        "scbench",
        task,
        prediction,
        [str(label)],
        _official_metadata(label, task),
    )
    assert result["score"] == expected
    assert result["metric_implementation"] == "scbench_official_python_port"


def test_scbench_special_scorer_task_fails_instead_of_emitting_fake_score():
    with pytest.raises(RuntimeError, match="special/multi-task scorer"):
        evaluate_prediction(
            "scbench",
            "scbench_repoqa",
            "prediction",
            ["label"],
            _official_metadata("label", "scbench_repoqa"),
        )
