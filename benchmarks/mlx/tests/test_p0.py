import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.run_benchmark import instantiate_benchmark
from src.benchmarks.niah import NIAHBenchmark
from src.config import ExperimentConfig
from src.eviction.base import validate_selected_indices
from src.eviction.l2_leverage import l2_row_leverage_scores
from src.eviction.registry import create_eviction


class Enc(dict):
    @property
    def input_ids(self):
        return self["input_ids"]


class CharTokenizer:
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        out = Enc()
        tensor = torch.tensor([ids], dtype=torch.long)
        out["input_ids"] = tensor if return_tensors == "pt" else ids
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()
        return "".join(chr(int(i)) for i in ids)


def test_config_load_utf8(tmp_path):
    path = tmp_path / "quick_utf8.yaml"
    text = Path("configs/experiments/dev/tiny_niah_cpu.yaml").read_text(encoding="utf-8")
    path.write_text(text.replace("tiny_niah_cpu", "tiny_niah_cpu_中文"), encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.experiment_name == "tiny_niah_cpu_中文"


def test_mlx_config_fields_load():
    cfg = ExperimentConfig.from_yaml("configs/experiments/dev/qwen25_05b_mlx_method_sanity.yaml")
    from src.runners.mlx_runner import MLXRunner, SUPPORTED_MLX_METHODS, canonical_method

    assert cfg.model.backend == "mlx"
    assert cfg.model.quant_bits == 4
    assert cfg.model.quant_group_size == 64
    assert cfg.save_selected_tokens is True
    for method in (
        "attention",
        "snapkv",
        "pyramidkv",
        "attention+l1",
        "attention+l2",
        "l1_decode_only",
        "l2_decode_only",
        "l2_key_prefill_only",
        "compactor",
    ):
        assert canonical_method(method) in SUPPORTED_MLX_METHODS
    for method in ("attention", "snapkv", "pyramidkv", "attention_l1", "attention_l2", "compactor"):
        assert method in cfg.methods
    assert canonical_method("snap") == "snapkv"
    assert MLXRunner._attention_score_source("attention") == (
        "all_prefill_queries_plus_decode_accumulated_causal_attention"
    )
    from src.eviction.registry import unsupported_reason
    assert unsupported_reason("h2o", "mlx") is None


def test_registry_metadata_and_aliases():
    from src.eviction.registry import get_method_spec, method_requires_attention, unsupported_reason

    assert get_method_spec("sink_recency").name == "sink_recent"
    assert get_method_spec("h2o_style").family == "attention"
    assert method_requires_attention("accumulated_attention") is True
    assert get_method_spec("oracle_evidence").oracle is True
    assert unsupported_reason("hidden_l2_norm", "mlx")
    assert get_method_spec("l1_decode_only").supports_mlx is True
    assert unsupported_reason("l1_decode_only", "torch")
    assert get_method_spec("compactor_style").name == "compactor"
    assert get_method_spec("l2_key_prefill_only").score_source == "key"


def test_requested_strategy_panel_and_paper_fidelity():
    from src.eviction.registry import (
        REQUESTED_STRATEGY_METHODS,
        get_method_spec,
        method_supports_backend,
    )

    expected = {
        "full", "random", "streamingllm", "h2o", "snapkv", "tova",
        "knorm", "keydiff", "compactor", "vnorml1", "vnorml2", "vatp",
        "curdkv", "adakv",
    }
    assert set(REQUESTED_STRATEGY_METHODS) == expected
    for method in expected:
        spec = get_method_spec(method)
        assert spec.implementation_fidelity != "unreviewed"
        assert method_supports_backend(method, "mlx")

    assert get_method_spec("TOVA").name == "tova"
    assert get_method_spec("KNorm").score_source == "negative_key_l2_norm"
    assert get_method_spec("KeyDiff").name == "keydiff"
    assert get_method_spec("VNorml1").paper_method is False
    assert get_method_spec("VNorml2").paper_method is False
    assert get_method_spec("CurDKV").name == "curdkv"
    assert get_method_spec("AdaKV").score_source.startswith("adaptive_head_budget")


def test_knorm_retains_lowest_key_norm_not_highest():
    eviction = create_eviction("knorm", cache_size=2, sink_size=0, recent_size=0)
    norms = torch.tensor([1.0, 4.0, 2.0, 3.0])
    keys = torch.zeros(1, 1, 4, 2)
    keys[0, 0, :, 0] = norms
    values = torch.zeros_like(keys)
    scores = eviction.compute_scores(keys, values, 0)
    selected = eviction.select_indices(scores, 4, 2, keys.device)
    assert selected.tolist() == [0, 2]


def test_benchmark_instantiation_niah():
    cfg = ExperimentConfig.from_yaml("configs/experiments/dev/tiny_niah_cpu.yaml")
    bench = instantiate_benchmark(cfg)
    assert isinstance(bench, NIAHBenchmark)


def test_ruler_prompt_answer_boundary():
    from src.benchmarks.ruler import RULERBenchmark

    tokenizer = CharTokenizer()
    bench = RULERBenchmark(
        tasks=["variable_tracking"],
        n_samples_per_task=1,
        seq_words=300,
        seed=42,
    )
    sample = bench.load_samples(tokenizer, 1)[0]
    prompt = sample["prompt"]
    answer_text = sample["answer_text"]
    answer_positions = sample["answer_positions"]

    assert sample["full_text"] == prompt + answer_text
    assert prompt.endswith("The value is")
    assert not prompt.endswith(answer_text)
    assert answer_positions[0] == len(prompt)
    assert answer_positions[-1] == len(sample["full_text"]) - 1
    assert sample["metadata"]["answer_token_start"] == answer_positions[0]


def test_eviction_constructor_filter():
    eviction = create_eviction(
        "recency",
        cache_size=8,
        score_source="v",
        sketch_dim=16,
        seed=0,
        k_seq_dim=2,
        v_seq_dim=2,
    )
    assert eviction.cache_size == 8


def test_budget_validity_for_all_methods():
    methods = [
        "recency",
        "sink_recent",
        "sink_recency",
        "random",
        "sink_recent_random",
        "uniform",
        "attention",
        "last_token_attention",
        "windowed_attention",
        "attention_decay",
        "h2o",
        "snapkv",
        "tova",
        "knorm",
        "keydiff",
        "vnorml1",
        "vnorml2",
        "vatp",
        "curdkv",
        "l1_leverage",
        "l2_leverage",
        "ridge_leverage",
        "approximate_l2_leverage",
        "key_norm",
        "value_norm",
        "key_l1_norm",
        "value_l1_norm",
        "kv_norm",
        "farthest_point",
        "kmeans_medoid",
        "facility_location_greedy",
        "pca_residual",
        "mahalanobis_distance",
        "zscore_outlier",
        "random_projection_outlier",
        "attention+l1",
        "attention_l2",
        "attention_recency",
        "sink_recent_l1",
        "sink_recent_l2",
        "oracle_evidence",
    ]
    for method in methods:
        eviction = create_eviction(
            method,
            cache_size=8,
            k_seq_dim=2,
            v_seq_dim=2,
            sink_size=2,
            recent_size=3,
            sketch_dim=16,
            n_clusters=4,
            debug_budget=True,
        )
        eviction.set_sample_metadata(
            {"evidence_positions": [3, 4], "metadata": {"answer_token_start": 10, "answer_token_end": 12}}
        )
        k = torch.randn(1, 2, 20, 4)
        v = torch.randn(1, 2, 20, 4)
        if method in {
            "attention", "last_token_attention", "windowed_attention",
            "attention_decay", "h2o", "tova", "vatp",
        }:
            eviction.update_attention(0, torch.softmax(torch.randn(1, 2, 1, 20), dim=-1))
        eviction(((k, v),))
        selected = eviction.last_selected[0]
        validate_selected_indices(selected, seq_len=20, budget=8)


def test_l2_leverage_not_equal_norm():
    rows = torch.randn(12, 4)
    scores = l2_row_leverage_scores(rows)
    rank = torch.linalg.matrix_rank(rows.float()).item()
    assert bool((scores >= -1e-6).all())
    assert abs(float(scores.sum()) - rank) < 1e-4
    assert not torch.allclose(scores, torch.norm(rows.float(), dim=1), atol=1e-4)

    q, _ = torch.linalg.qr(torch.randn(10, 3), mode="reduced")
    q_scores = l2_row_leverage_scores(q)
    assert torch.allclose(q_scores, q.pow(2).sum(dim=1), atol=1e-5)

    low_rank = torch.ones(8, 3)
    low_scores = l2_row_leverage_scores(low_rank)
    assert torch.isfinite(low_scores).all()


def test_niah_evidence_span_decode():
    tokenizer = CharTokenizer()
    bench = NIAHBenchmark(depths=[0.2, 0.8], max_words=60, needles_per_depth=1)
    samples = bench.load_samples(tokenizer, 2)
    for sample in samples:
        meta = sample["metadata"]
        ids = sample["input_ids"][0]
        decoded = tokenizer.decode(ids[meta["needle_token_start"] : meta["needle_token_end"]])
        assert meta["value"] in decoded
        assert len(sample["evidence_positions"]) == meta["needle_token_end"] - meta["needle_token_start"]


def test_analysis_loads_json_selected_and_scores(tmp_path):
    from scripts.run_analysis import _load_score_dict, _load_selected

    selected_path = tmp_path / "selected.json"
    scores_path = tmp_path / "scores.json"
    selected_path.write_text(json.dumps({"0": [1, 2, 3]}), encoding="utf-8")
    scores_path.write_text(json.dumps({"0": [0.1, 0.2, 0.3]}), encoding="utf-8")

    selected = _load_selected({"selected_tokens_path": str(selected_path)})
    scores = _load_score_dict({"scores_path": str(scores_path)})

    assert selected[0].dtype == torch.long
    assert selected[0].tolist() == [1, 2, 3]
    assert torch.allclose(scores[0], torch.tensor([0.1, 0.2, 0.3]))


def test_official_metric_helpers():
    from src.evaluation.official_metrics import longbench_score, ruler_score

    assert longbench_score("narrativeqa", "the red door", ["red door"]) == 100.0
    assert longbench_score("hotpotqa", "Barack Obama was born in Hawaii.", ["Hawaii"]) > 0
    assert ruler_score("vt", "AAA BBB", ["AAA", "BBB", "CCC"]) == 66.6667
    assert ruler_score("niah_single_1", "The number is 12345.", ["12345"]) == 100.0


def test_longbench_official_prompt_metadata():
    from src.benchmarks.longbench import _build_qa_sample

    sample = _build_qa_sample(
        {
            "_task": "hotpotqa",
            "context": "Passage A.",
            "input": "Where?",
            "answers": ["Hawaii"],
            "length": 10,
            "all_classes": None,
        },
        max_words=0,
        use_official_prompt=True,
    )
    assert sample["dataset_official"] is True
    assert sample["official_prompt"] is True
    assert sample["official_metric_name"] == "qa_f1"
    assert "Only give me the answer" in sample["prefix_text"]


def test_mlx_prefill_headwise_shape_smoke():
    mx = pytest.importorskip("mlx.core")
    from src.runners.mlx_runner import MLXCacheEvictor, _cache_head_valid_attention_mask

    class Cache:
        pass

    def make_cache(seq_len=24, heads=2, dim=8):
        c = Cache()
        c.keys = mx.random.normal((1, heads, seq_len, dim))
        c.values = mx.random.normal((1, heads, seq_len, dim))
        c.offset = seq_len
        c.logical_offset = seq_len
        return [c]

    cfg = ExperimentConfig()
    cfg.eviction.window_size = 4
    cfg.eviction.pooling_kernel = 5
    cfg.eviction.pooling_method = "avgpool"
    cfg.eviction.compactor_sketch_dim = 4
    cfg.eviction.compactor_chunk_size = 8
    cfg.eviction.compactor_attention_chunk_size = 8
    cfg.eviction.compactor_protected_first_tokens = 2
    cfg.eviction.compactor_protected_last_tokens = 3
    cfg.seed = 7

    for method in ("snapkv", "pyramidkv", "adakv"):
        cache = make_cache()
        state = {
            "observe_heads": {0: [mx.random.uniform(shape=(2, 24)) for _ in range(4)]},
            "observe": {},
            "last": {},
            "accumulated": {},
            "decayed": {},
            "hook_errors": 0,
        }
        evictor = MLXCacheEvictor(method, 8, cfg, 1, attention_state=state)
        evictor.set_phase("prefill")
        evictor.prefill_compress(cache, 8)
        c = cache[0]
        mask = _cache_head_valid_attention_mask(c, 4, c.offset)
        assert c.logical_offset == 24
        assert c.head_valid_mask.shape == (2, c.offset)
        assert mask.shape == (4, 1, c.offset)
        assert evictor.last_selected_by_head[0]

    cache = make_cache(seq_len=24, heads=2, dim=8)
    state = {
        "observe_heads": {},
        "observe": {},
        "last": {},
        "accumulated": {},
        "decayed": {},
        "hook_errors": 0,
        "prefill_q_post": {0: [mx.random.normal((1, 4, 24, 8))]},
        "prefill_k_post": {0: [mx.random.normal((1, 2, 24, 8))]},
        "prefill_k_pre": {0: [mx.random.normal((1, 2, 24, 8))]},
    }
    evictor = MLXCacheEvictor("compactor", 8, cfg, 1, attention_state=state)
    evictor.set_phase("prefill")
    evictor.prefill_compress(cache, 8)
    c = cache[0]
    assert c.logical_offset == 24
    assert c.head_valid_mask.shape[1] == c.offset
    assert sum(len(v) for v in evictor.last_selected_by_head[0].values()) <= 16


def test_mlx_novel_geometry_prefill_methods_are_finite_and_budget_valid():
    mx = pytest.importorskip("mlx.core")
    from src.runners.mlx_runner import MLXCacheEvictor

    class Cache:
        pass

    methods = (
        "conditional_v_leverage",
        "conditional_k_leverage",
        "attention_residual_v_leverage",
        "attention_weighted_v_leverage",
        "window_weighted_v_leverage",
        "joint_kv_leverage",
        "ridge_v_allocation",
        "ridge_v_fixed",
        "ridge_v_shared",
        "diversity_v_leverage",
    )
    cfg = ExperimentConfig()
    cfg.seed = 7
    cfg.eviction.diversity_candidate_multiplier = 2
    for method in methods:
        cache = Cache()
        cache.keys = mx.random.normal((1, 2, 24, 8))
        cache.values = mx.random.normal((1, 2, 24, 8))
        cache.offset = 24
        cache.logical_offset = 24
        state = {
            "accumulated_heads": {0: mx.random.uniform(shape=(2, 24))},
            "accumulated": {},
            "last": {},
            "last_heads": {},
            "decayed": {},
            "observe": {},
            "observe_heads": {
                0: [mx.random.uniform(shape=(2, 24)) for _ in range(2)]
            },
            "hook_errors": 0,
        }
        evictor = MLXCacheEvictor(method, 8, cfg, 1, attention_state=state)
        evictor.set_phase("prefill")
        evictor.prefill_compress([cache], 8)
        counts = [
            len(values) for values in evictor.last_selected_by_head[0].values()
        ]
        assert sum(counts) == 16
        assert cache.logical_offset == 24
        assert all(
            all(math.isfinite(value) for value in values)
            for values in evictor.last_scores_by_head[0].values()
        )


def test_attention_residual_core_and_geometry_budgets_are_separate():
    mx = pytest.importorskip("mlx.core")
    from src.runners.mlx_runner import MLXCacheEvictor

    class Cache:
        pass

    cache = Cache()
    cache.keys = mx.random.normal((1, 1, 12, 4))
    cache.values = mx.random.normal((1, 1, 12, 4))
    cache.offset = 12
    cache.logical_offset = 12
    attention = mx.array(
        [[12.0, 11.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    )
    cfg = ExperimentConfig()
    cfg.eviction.attention_residual_budget_ratio = 0.5
    evictor = MLXCacheEvictor(
        "attention_residual_v_leverage",
        6,
        cfg,
        1,
        attention_state={"accumulated_heads": {0: attention}},
    )
    evictor.prefill_compress([cache], 6)
    event = evictor.estimator_events[0]
    assert event["attention_budget"] == 3
    assert event["geometry_budget"] == 3


def test_window_residual_uses_bounded_observation_attention():
    mx = pytest.importorskip("mlx.core")
    from src.eviction.registry import get_method_spec
    from src.runners.mlx_runner import MLXCacheEvictor, MLXRunner

    class Cache:
        pass

    cache = Cache()
    cache.keys = mx.random.normal((1, 1, 12, 4))
    cache.values = mx.random.normal((1, 1, 12, 4))
    cache.offset = 12
    cache.logical_offset = 12
    attention_rows = [
        mx.array([[12.0, 11.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        mx.array([[11.0, 10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
    ]
    cfg = ExperimentConfig()
    cfg.eviction.observation_window = 2
    cfg.eviction.pooling_kernel = 1
    cfg.eviction.attention_residual_budget_ratio = 0.5
    evictor = MLXCacheEvictor(
        "window_residual_v_leverage",
        6,
        cfg,
        1,
        attention_state={"observe_heads": {0: attention_rows}},
    )
    evictor.prefill_compress([cache], 6)
    event = evictor.estimator_events[0]
    assert event["calculation"] == "window_attention_core_residual_v_leverage"
    assert event["attention_query_window"] == 2
    assert event["attention_budget"] == 3
    assert event["geometry_budget"] == 3
    assert len(evictor.last_selected[0]) == 6

    runner = MLXRunner(cfg)
    runner.reset_attention_state()
    runner.configure_attention_recording("window_residual_v_leverage")
    assert runner.attention_state["record_all_queries"] is False
    assert get_method_spec("bounded_residual_v_leverage").name == "window_residual_v_leverage"


def test_window_weighted_uses_bounded_observation_attention():
    mx = pytest.importorskip("mlx.core")
    from src.eviction.registry import get_method_spec
    from src.runners.mlx_runner import MLXCacheEvictor, MLXRunner

    class Cache:
        pass

    cache = Cache()
    cache.keys = mx.random.normal((1, 1, 12, 4))
    cache.values = mx.random.normal((1, 1, 12, 4))
    cache.offset = 12
    cache.logical_offset = 12
    attention_rows = [
        mx.array([[12.0, 11.0, 10.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        mx.array([[11.0, 10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]),
    ]
    cfg = ExperimentConfig()
    cfg.eviction.observation_window = 2
    cfg.eviction.pooling_kernel = 1
    evictor = MLXCacheEvictor(
        "window_weighted_v_leverage",
        6,
        cfg,
        1,
        attention_state={"observe_heads": {0: attention_rows}},
    )
    evictor.prefill_compress([cache], 6)
    event = evictor.estimator_events[0]
    assert event["calculation"] == "window_attention_weighted_v_ridge_leverage"
    assert event["attention_query_window"] == 2
    assert len(evictor.last_selected[0]) == 6

    runner = MLXRunner(cfg)
    runner.reset_attention_state()
    runner.configure_attention_recording("window_weighted_v_leverage")
    assert runner.attention_state["record_all_queries"] is False
    assert get_method_spec("window_query_weighted_v_leverage").name == "window_weighted_v_leverage"
