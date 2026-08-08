from pathlib import Path

import pandas as pd
import yaml

from src.config import ModelConfig
from src.model_adapters import apply_prompt_format
from statekv.config import load_discovery_config
from statekv.backend_mlx import MLXTemporalModel
from statekv.oracle_policy_freegen import _aggregate_free_results


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "configs"
    / "stages"
    / "oracle_policy_freegen_qwen3_8b_n10_protocol.yaml"
)


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "formatted"


def test_qwen3_protocol_disables_thinking_and_has_ten_samples() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["model_overrides"]["name"] == "mlx-community/Qwen3-8B-4bit"
    assert protocol["model_overrides"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert protocol["expected_sample_count"] == 10
    assert len(protocol["sample_ids"]) == 10
    assert protocol["total_budget"] == 256
    assert protocol["primary_decision"]["strict_pareto_status"] == "diagnostic_only"


def test_qwen3_checkpoint_passes_discovery_validation() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    cfg = load_discovery_config(str(ROOT / protocol["base_config"]))
    for key, value in protocol["model_overrides"].items():
        setattr(cfg.model, key, value)
    cfg.cache.total_budget = int(protocol["total_budget"])
    cfg.cache.sink_size = int(protocol["sink_size"])
    cfg.cache.recent_size = int(protocol["recent_size"])
    cfg.cache.selected_core_budget = int(protocol["core_budget"])
    cfg.validate()


def test_chat_template_forwards_qwen3_non_thinking_switch() -> None:
    tokenizer = _RecordingTokenizer()
    cfg = ModelConfig(
        prompt_format={
            "mode": "chat_template",
            "system_prompt": None,
            "template_kwargs": {"enable_thinking": False},
        }
    )
    assert apply_prompt_format(tokenizer, "hello", cfg) == "formatted"
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_mlx_bfloat16_diagnostic_export_uses_fp32_bridge() -> None:
    import mlx.core as mx
    import torch

    value = mx.array([1.25, -0.5], dtype=mx.bfloat16)
    exported = MLXTemporalModel._torch(value)
    exported_numpy = MLXTemporalModel._numpy(value, dtype=float)
    assert exported.dtype == torch.float32
    assert exported.tolist() == [1.25, -0.5]
    assert exported_numpy.tolist() == [1.25, -0.5]


def test_overall_quality_and_fidelity_are_reported_separately() -> None:
    rows = []
    values = {
        "statekv_exact_mean": [(8.0, 0.10), (100.0, 0.20)],
        "attention": [(9.0, 0.40), (0.0, 0.50)],
        "snapkv": [(10.0, 0.20), (100.0, 0.25)],
        "h2o": [(7.0, 0.30), (0.0, 0.35)],
        "full_cache": [(11.0, 0.0), (100.0, 0.0)],
    }
    for policy, policy_values in values.items():
        rows.extend(
            [
                {
                    "sample_id": "gov:1",
                    "policy": policy,
                    "task_bucket": "GovReport",
                    "official_score": policy_values[0][0],
                    "mean_trajectory_exact_kl": policy_values[0][1],
                    "rouge_l": policy_values[0][0] / 100.0,
                    "needle_retrieval_accuracy": None,
                },
                {
                    "sample_id": "niah:1",
                    "policy": policy,
                    "task_bucket": "NIAH",
                    "official_score": policy_values[1][0],
                    "mean_trajectory_exact_kl": policy_values[1][1],
                    "rouge_l": None,
                    "needle_retrieval_accuracy": policy_values[1][0] / 100.0,
                },
            ]
        )
    result = _aggregate_free_results(
        pd.DataFrame(rows), bootstrap_seed=7, bootstrap_samples=100
    )
    overall = result["overall_comparisons"]
    assert overall["best_fixed_quality_policy"] == "snapkv"
    assert overall["mean_official_score_statekv_minus_best_fixed"] == -1.0
    assert overall["best_fixed_kl_policy"] == "snapkv"
    assert overall["mean_trajectory_kl_best_fixed_minus_statekv"] > 0.0
