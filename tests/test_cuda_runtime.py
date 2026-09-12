"""CPU checks for CUDA policy semantics using the same HF model forward path."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvbench.backends.huggingface import HuggingFaceBackend
from statekv.cuda_runtime import CudaRuntime, retained_rows
from src.evaluation.official_metrics import longbench_score


def test_retention_reserves_pending_query_and_breaks_ties_by_position():
    positions = list(range(32))
    rows = retained_rows(positions, np.ones(32), 16, 2, 4)
    assert rows == list(range(12)) + [29, 30, 31]
    assert len(rows) + 1 == 16


def test_longbench_task_specific_metrics():
    assert longbench_score("2wikimqa", "Paris", ["Paris"], official=True) == 100
    assert longbench_score("passage_count", "3 4", ["3"], official=True) == 50
    assert longbench_score("passage_retrieval_en", "Paragraph 2", ["Paragraph 2"], official=True) == 100
    assert longbench_score("trec", "ABBR", ["ABBR"], all_classes=["ABBR", "NUM"], official=True) == 100


def test_r2_scores_use_future_attention_without_current_query():
    runtime = object.__new__(CudaRuntime)
    state = SimpleNamespace(position_maps={0: torch.arange(4)})
    runtime.backend = SimpleNamespace(fork_state=lambda anchor: state)
    calls = []

    def forward(branch, token):
        value = 100 if not calls else len(calls)
        calls.append(token)
        branch.position_maps[0] = torch.arange(4 + len(calls))
        return torch.tensor([0., 1.]), np.full(4 + len(calls), value), {}, 0.

    runtime.forward = forward
    assert runtime.r2_scores(None, 7, [0, 2], 3) == {0: 6., 2: 6.}


@pytest.fixture
def runtime(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen3.modeling_qwen3")
    from transformers import Qwen3Config

    def load(backend):
        torch.manual_seed(7)
        cfg = Qwen3Config(vocab_size=128, hidden_size=96, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=32, max_position_embeddings=512)
        cfg._attn_implementation = "eager"
        backend.model = modeling.Qwen3ForCausalLM(cfg).eval()
        import inspect

        backend._forward_params = set(inspect.signature(backend.model.forward).parameters)
        backend.tokenizer = SimpleNamespace(eos_token_id=127,
                                            decode=lambda ids, **kwargs: " ".join(map(str, ids)))
        backend.model_info = {"num_layers": 2, "num_key_value_heads": 2}
        return dict(backend.model_info)

    monkeypatch.setattr(HuggingFaceBackend, "load", load)
    value = CudaRuntime({"model": {"name": "random_qwen3", "dtype": "float32"},
                         "device": "cpu", "sink_size": 2, "recent_size": 4,
                         "rollout_horizon": 3, "stop_on_eos": False, "prefill_chunk_size": 16})
    yield value
    value.close()


def test_full_budget_has_full_kv_outputs_and_branch_does_not_mutate_anchor(runtime):
    prompt = list(range(1, 33))
    anchor, _, seconds = runtime.prefill(prompt[:-1])
    original = [(key.clone(), value.clone()) for key, value in anchor.past_key_values]
    full = runtime.run_arm(anchor, prompt, "FULL", 0, 4, seconds)
    current = runtime.run_arm(anchor, prompt, "CURRENT_QK", 128, 4, seconds)
    assert full["generated_token_ids"] == current["generated_token_ids"]
    assert current["mean_trajectory_exact_kl"] < 1e-6
    assert anchor.logical_next_position == 31
    for pair, copy in zip(anchor.past_key_values, original):
        for tensor, expected in zip(pair, copy):
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


def test_r2_excludes_current_attention_and_only_runs_once(runtime, monkeypatch):
    prompt = list(range(1, 33))
    anchor, _, seconds = runtime.prefill(prompt[:-1])
    calls = []
    original = runtime.r2_scores

    def score(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "r2_scores", score)
    result = runtime.run_arm(anchor, prompt, "CHEAP_R2", 16, 4, seconds)
    assert len(calls) == result["causal_teacher_refreshes"] == 1
    assert all(row["active_cache_tokens"] == 16 for row in result["cycles"])
    assert result["recoverable_cold_tokens"] == 0


@pytest.mark.parametrize("policy", ["SNAPKV", "H2O", "LAQ", "LAQPP"])
def test_remaining_policies_execute_with_physical_budget(runtime, policy):
    prompt = list(range(1, 33))
    anchor, _, seconds = runtime.prefill(prompt[:-1])
    result = runtime.run_arm(anchor, prompt, policy, 16, 4, seconds)
    assert all(row["active_cache_tokens"] == 16 for row in result["cycles"])
    assert np.isfinite(result["mean_trajectory_exact_kl"])
    assert len(result["generated_token_ids"]) == 4
