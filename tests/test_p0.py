import json
import subprocess
import sys
from pathlib import Path

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
    text = Path("configs/experiment/quick.yaml").read_text(encoding="utf-8")
    path.write_text(text.replace("quick_test", "quick_test_中文"), encoding="utf-8")
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.experiment_name == "quick_test_中文"


def test_benchmark_instantiation_niah():
    cfg = ExperimentConfig.from_yaml("configs/experiment/quick.yaml")
    bench = instantiate_benchmark(cfg)
    assert isinstance(bench, NIAHBenchmark)


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
        "random",
        "uniform",
        "attention",
        "h2o",
        "snapkv",
        "l1_leverage",
        "l2_leverage",
        "key_norm",
        "value_norm",
        "kv_norm",
        "farthest_point",
        "kmeans_medoid",
        "pca_residual",
        "attention+l1",
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
        k = torch.randn(1, 2, 20, 4)
        v = torch.randn(1, 2, 20, 4)
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

