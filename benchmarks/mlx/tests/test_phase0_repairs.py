import math

import numpy as np
import pytest
import torch

from src.benchmarks.vt_parser import parse_vt_prompt
from src.eviction.attention import WindowedAttentionEviction
from src.eviction.l2_leverage import l2_row_leverage_scores
from src.runners.mlx_runner import _record_attention_from_hook
from src.runners.mlx_runner import MLXCacheEvictor
from src.config import ExperimentConfig
from src.sketching.woodruff_l1 import WoodruffL1Estimator


class Encoding(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharacterTokenizer:
    def __call__(self, text, return_offsets_mapping=False, return_tensors=None, add_special_tokens=True):
        result = Encoding(input_ids=torch.tensor([[ord(ch) for ch in text]]))
        if return_tensors is None:
            result["input_ids"] = result["input_ids"].tolist()[0]
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


def test_vt_parser_recovers_target_chain_and_distractor_spans():
    prompt = (
        "VAR AAA = 111 VAR BBB = VAR AAA "
        "VAR CCC = 222 VAR DDD = VAR BBB\n"
        "Question: Find all variables that are assigned the value 111. Answer:"
    )
    parsed = parse_vt_prompt(prompt, CharacterTokenizer())
    assert parsed["parser_complete"] is True
    assert parsed["target_variables"] == ["AAA", "BBB", "DDD"]
    assert parsed["target_edge_ids"] == [0, 1, 3]
    assert parsed["evidence_positions"]
    assert parsed["distractor_positions"]
    evidence_text = "".join(prompt[index] for index in parsed["evidence_positions"])
    assert "VAR AAA = 111" in evidence_text
    assert "VAR CCC = 222" not in evidence_text


def test_rank_deficient_l2_is_exact_and_never_norm_fallback():
    rows = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [0.0, 0.0], [3.0, 0.0]]
    )
    scores, diagnostics = l2_row_leverage_scores(rows, return_diagnostics=True)
    expected = torch.tensor([1.0, 4.0, 0.0, 9.0]) / 14.0
    assert torch.allclose(scores, expected, atol=1e-6)
    assert float(scores.sum()) == pytest.approx(1.0)
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["fallback"] is False


def test_l1_estimator_is_seed_deterministic_and_refit_telemetry_is_real():
    rows = torch.Generator().manual_seed(9)
    matrix = torch.randn(24, 4, generator=rows)
    first = WoodruffL1Estimator(sketch_dim=12, seed=17)
    second = WoodruffL1Estimator(sketch_dim=12, seed=17)
    scores_first = first.scores(matrix, force_refit=True)
    scores_second = second.scores(matrix, force_refit=True)
    assert torch.allclose(scores_first, scores_second)
    assert first.fit_count == 1
    assert first.last_diagnostics["fallback"] is False
    first.scores(matrix, force_refit=False)
    assert first.fit_count == 1
    assert first.last_diagnostics["refit"] is False


def test_torch_attention_state_is_pruned_with_cache_positions():
    eviction = WindowedAttentionEviction(
        cache_size=2, k_seq_dim=2, v_seq_dim=2, attention_window=2
    )
    eviction._acc_scores[0] = torch.tensor([10.0, 20.0, 30.0, 40.0])
    eviction._windows[0] = [torch.tensor([1.0, 2.0, 3.0, 4.0])]
    eviction.on_cache_pruned(0, torch.tensor([1, 3]), 4)
    assert eviction._acc_scores[0].tolist() == [20.0, 40.0]
    assert eviction._windows[0][0].tolist() == [2.0, 4.0]


def test_mlx_attention_hook_matches_causal_gqa_reference():
    mx = pytest.importorskip("mlx.core")
    rng = np.random.default_rng(3)
    queries = rng.normal(size=(1, 4, 3, 2)).astype(np.float32)
    keys = rng.normal(size=(1, 2, 5, 2)).astype(np.float32)
    state = {
        "enabled": True,
        "phase": "prefill",
        "current_method": "attention",
        "record_all_queries": True,
        "max_observe": 2,
        "attention_chunk_size": 2,
        "decay_gamma": 0.9,
    }
    module = type("Attention", (), {})()
    module._l1kv_attention_state = state
    module._l1kv_layer_idx = 0
    module.scale = 1.0 / math.sqrt(2.0)
    _record_attention_from_hook(module, mx.array(queries), mx.array(keys), query_len=3)
    assert state.get("hook_errors", 0) == 0

    reference = np.zeros((2, 5), dtype=np.float64)
    for kv_head in range(2):
        for query_index in range(3):
            distributions = []
            for group in range(2):
                q_head = kv_head * 2 + group
                logits = queries[0, q_head, query_index] @ keys[0, kv_head].T * module.scale
                allowed = 5 - 3 + query_index + 1
                logits[allowed:] = -np.inf
                finite = np.exp(logits - np.max(logits))
                distributions.append(finite / finite.sum())
            reference[kv_head] += np.mean(distributions, axis=0)
    actual = np.asarray(state["accumulated_heads"][0].tolist())
    assert np.allclose(actual, reference, atol=1e-5)
    assert state["query_counts"][0]["prefill"] == 3


def test_mlx_position_map_stays_aligned_across_multiple_evictions():
    mx = pytest.importorskip("mlx.core")
    cache = type("Cache", (), {})()
    cache.keys = mx.arange(1 * 2 * 10 * 2).reshape(1, 2, 10, 2).astype(mx.float32)
    cache.values = cache.keys
    cache.offset = 10
    cache.logical_offset = 10
    cfg = ExperimentConfig()
    cfg.seed = 5
    evictor = MLXCacheEvictor("random", 6, cfg, 1)
    evictor.evict([cache], 6)
    first = set(evictor.last_selected[0])
    assert len(first) == 6
    cache.keys = mx.concatenate(
        [cache.keys, mx.zeros((1, 2, 2, 2), dtype=cache.keys.dtype)], axis=2
    )
    cache.values = mx.concatenate(
        [cache.values, mx.zeros((1, 2, 2, 2), dtype=cache.values.dtype)], axis=2
    )
    cache.offset = 8
    cache.logical_offset = 12
    evictor.evict([cache], 6)
    second = evictor.last_selected[0]
    assert len(second) == len(set(second)) == 6
    assert set(second).issubset(first | {10, 11})
    assert int(cache.keys.shape[2]) == int(cache.values.shape[2]) == 6


def test_mlx_update_interval_counts_real_eviction_decisions_not_noop_calls():
    mx = pytest.importorskip("mlx.core")

    class Cache:
        pass

    def make_cache(seq_len):
        cache = Cache()
        cache.keys = mx.arange(1 * 2 * seq_len * 2).reshape(1, 2, seq_len, 2).astype(mx.float32)
        cache.values = cache.keys
        cache.offset = seq_len
        cache.logical_offset = seq_len
        return cache

    cfg = ExperimentConfig()
    cfg.eviction.update_policy = "every_n_steps"
    cfg.eviction.update_interval = 2
    evictor = MLXCacheEvictor("random", 6, cfg, 1)

    cache = make_cache(8)
    evictor.evict([cache], 6)
    assert evictor.eviction_step == 1

    # The post-token call in the decode loop is a no-op at the exact budget.
    # It must not consume an update-interval step.
    evictor.evict([cache], 6)
    assert evictor.eviction_step == 1

    cache.keys = mx.concatenate(
        [cache.keys, mx.zeros((1, 2, 1, 2), dtype=cache.keys.dtype)], axis=2
    )
    cache.values = mx.concatenate(
        [cache.values, mx.zeros((1, 2, 1, 2), dtype=cache.values.dtype)], axis=2
    )
    cache.offset = 7
    cache.logical_offset = 9
    evictor.evict([cache], 6)
    assert evictor.eviction_step == 2

    estimator = type("Estimator", (), {"fit_count": 1})()
    evictor.method = "l2_leverage"
    assert evictor._should_refit_estimator(estimator) is True


def test_mlx_l2_update_interval_is_a_real_score_refresh_interval():
    mx = pytest.importorskip("mlx.core")

    class Cache:
        pass

    cache = Cache()
    cache.keys = mx.random.normal((1, 2, 8, 4))
    cache.values = mx.random.normal((1, 2, 8, 4))
    cache.offset = 8
    cache.logical_offset = 8

    cfg = ExperimentConfig()
    cfg.eviction.sink_size = 1
    cfg.eviction.recent_size = 2
    cfg.eviction.score_source = "v"
    cfg.eviction.update_policy = "every_n_steps"
    cfg.eviction.update_interval = 2
    evictor = MLXCacheEvictor("l2_leverage", 6, cfg, 1)
    evictor.set_phase("prefill")
    evictor.evict([cache], 6)
    assert evictor.score_refit_count == 2  # one fit per KV head
    assert evictor.eviction_step == 1

    def append_token(logical_offset):
        cache.keys = mx.concatenate(
            [cache.keys, mx.random.normal((1, 2, 1, 4))], axis=2
        )
        cache.values = mx.concatenate(
            [cache.values, mx.random.normal((1, 2, 1, 4))], axis=2
        )
        cache.offset = int(cache.keys.shape[2])
        cache.logical_offset = logical_offset

    evictor.set_phase("decode")
    append_token(9)
    evictor.evict([cache], 6)
    assert evictor.score_refit_count == 2
    assert evictor.eviction_step == 2

    append_token(10)
    evictor.evict([cache], 6)
    assert evictor.score_refit_count == 4
    assert evictor.eviction_step == 3
