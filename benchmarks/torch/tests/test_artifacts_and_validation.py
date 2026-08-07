from pathlib import Path

from kvbench.analysis.validate import validate_results
from kvbench.analysis.aggregate import aggregate
from kvbench.artifacts.writer import RunWriter
from kvbench.config import experiment_from_dict
from kvbench.types import SampleResult, ScoreBundle, SelectionDecision


def _config(output_root: Path):
    return experiment_from_dict(
        {
            "runtime": {"device": "cpu"},
            "model": {"name": "tiny", "dtype": "float32"},
            "benchmark": {
                "name": "ruler",
                "task": "niah_single_1",
                "require_official": False,
                "num_samples": 1,
            },
            "budget": {
                "cache_budget": 3,
                "sink_size": 1,
                "recent_size": 1,
            },
            "method": {"name": "recency"},
            "output": {"root": str(output_root), "experiment_name": "unit"},
        }
    )


def test_atomic_artifacts_resume_and_validation(tmp_path):
    cfg = _config(tmp_path / "results")
    writer = RunWriter(cfg, command=["unit-test"])
    writer.save_environment(
        {
            "model_name": "tiny",
            "revision": "r1",
            "checkpoint_commit_hash": "abc",
            "tokenizer_name_or_path": "tiny",
            "tokenizer_class": "FakeTokenizer",
            "tokenizer_vocab_size": 10,
        }
    )
    writer.write_sample_manifest(
        [
            {
                "sample_id": "s0",
                "task": "niah_single_1",
                "prompt_sha256": "a",
                "reference_sha256": "b",
            }
        ]
    )
    decision = SelectionDecision(
        layer=0,
        universe_positions=[0, 1, 2, 3, 4],
        selected_rows=[0, 3, 4],
        selected_positions=[0, 3, 4],
        requested_budget=3,
        effective_budget=3,
        mandatory_positions=[0, 4],
        selectable_budget=1,
        budget_scope="total_kv",
        budget_unit="shared_token_positions",
    )
    result = SampleResult(
        sample_id="s0",
        task="niah_single_1",
        prediction="42",
        references=["42"],
        score=100.0,
        metric_name="string_match_all",
        correct=True,
        status="complete",
        error=None,
        metadata={
            "benchmark": "ruler",
            "method": "recency",
            "method_variant": "recency",
            "dataset_official": False,
            "target_used_for_generation": False,
            "truncation": {"truncated": False},
            "metric_implementation": "ruler_public_string_match",
        },
        timing={"prefill_s": 0.1, "scoring_s": 0.0, "compression_s": 0.0, "decode_s": 0.1},
        cache={"occupancy_trace": [3], "peak_gpu_memory_bytes": 0},
        diagnostics={},
        predictions=["42"],
    )
    writer.mark_running("s0")
    writer.save_sample(result, [decision], [ScoreBundle()])
    writer.finalize(expected_samples=1)
    assert writer.is_complete("s0")
    report = validate_results(cfg.output.root and Path(cfg.output.root))
    assert report["valid"], report["errors"]
    outputs = aggregate(Path(cfg.output.root), tmp_path / "aggregate")
    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["diagnostics_parquet"]).exists()


def test_storage_root_does_not_change_scientific_config_hash(tmp_path):
    left = RunWriter(_config(tmp_path / "left"), command=["unit-test"])
    right = RunWriter(_config(tmp_path / "right"), command=["unit-test"])
    assert left.config_hash == right.config_hash
