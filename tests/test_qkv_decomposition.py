"""QK-V decomposition machinery tests (discovery protocol Phase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from statekv.qkv_decomposition import (
    add_future_targets,
    classify_token,
    exact_removal_delta,
    kv_head_for_query,
    rank_and_margin,
    swap_selection_all_layers,
)
from statekv.selectors import CoreSelection, LayerSelection


def _gram_from_weight(weight: torch.Tensor, heads: int, dim: int) -> torch.Tensor:
    slices = weight[:, : heads * dim].reshape(weight.shape[0], heads, dim)
    return torch.einsum("ohd,ohj->hdj", slices, slices).float()


def test_exact_removal_delta_matches_direct_projection() -> None:
    torch.manual_seed(0)
    heads, kv_heads, count, dim, out = 3, 2, 6, 4, 8
    attn = torch.softmax(torch.randn(heads, count), dim=-1)
    values = torch.randn(kv_heads, count, dim)
    weight = torch.randn(out, heads * dim)
    gram = _gram_from_weight(weight, heads, dim)
    kv_map = [0, 1, 1]
    delta, pv, outputs = exact_removal_delta(attn, values, gram, kv_map)
    for head in range(heads):
        kv = kv_map[head]
        w_slice = weight[:, head * dim : (head + 1) * dim]  # [out, dim]
        o = attn[head].double() @ values[kv].double()
        assert torch.allclose(outputs[head].double(), o, atol=1e-5)
        for token in range(count):
            a = float(attn[head, token])
            o_removed = (o - a * values[kv, token].double()) / (1.0 - a)
            direct = ((o_removed - o).float() @ w_slice.T).norm()
            assert float(delta[head, token]) == pytest.approx(
                float(direct), rel=1e-4
            )
            direct_pv = (values[kv, token] @ w_slice.T).norm()
            assert float(pv[head, token]) == pytest.approx(
                float(direct_pv), rel=1e-4
            )


def test_projected_norm_via_gram_matches_direct() -> None:
    torch.manual_seed(1)
    heads, dim, out = 4, 8, 16
    weight = torch.randn(out, heads * dim)
    gram = _gram_from_weight(weight, heads, dim)
    x = torch.randn(heads, 5, dim)
    for head in range(heads):
        w_slice = weight[:, head * dim : (head + 1) * dim]
        direct = (x[head] @ w_slice.T).norm(dim=-1) ** 2
        via_gram = torch.einsum("ni,ij,nj->n", x[head], gram[head], x[head])
        assert torch.allclose(via_gram, direct, atol=1e-3)


def test_rank_margin_and_core() -> None:
    positions = [10, 11, 12, 13, 14]
    eligible = [11, 12, 13, 14]
    attn = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    rank, margin, core = rank_and_margin(attn, positions, eligible, 2)
    assert core == (11, 12)
    assert rank[11] == 1 and rank[12] == 2 and rank[13] == 3
    assert margin[11] == pytest.approx(0.3 - 0.4)
    assert margin[12] == pytest.approx(0.0)
    assert margin[13] == pytest.approx(0.3 - 0.2)
    assert margin[14] > margin[13]
    # tie-break by position
    attn_tie = np.array([0.5, 0.3, 0.3, 0.3, 0.1])
    _, _, core_tie = rank_and_margin(attn_tie, positions, eligible, 2)
    assert core_tie == (11, 12)


def test_kv_head_mapping_gqa() -> None:
    assert [kv_head_for_query(h, 32, 8) for h in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert kv_head_for_query(31, 32, 8) == 7


def _selection(cores):
    return CoreSelection(
        strategy="qk_pool",
        horizon_condition=None,
        by_layer={
            int(layer): LayerSelection(
                layer=int(layer),
                selected_positions=list(core),
                eligible_positions=list(core),
                aggregate_scores=[1.0] * len(core),
            )
            for layer, core in cores.items()
        },
    )


def test_swap_selection_all_layers_budget_preserved() -> None:
    selection = _selection({0: [1, 2, 3], 1: [1, 2, 9], 2: [4, 5, 6]})
    swapped, count = swap_selection_all_layers(selection, 2, 7)
    assert count == 2  # layers 0 and 1 contain 2 and lack 7
    assert swapped.by_layer[0].selected_positions == [1, 3, 7]
    assert swapped.by_layer[1].selected_positions == [1, 7, 9]
    assert swapped.by_layer[2].selected_positions == [4, 5, 6]
    for layer, current in swapped.by_layer.items():
        assert len(current.selected_positions) == 3
    # original untouched
    assert selection.by_layer[0].selected_positions == [1, 2, 3]


def test_add_future_targets_forward_looking() -> None:
    rows = []
    attn_seq = [0.1, 0.2, 0.4, 0.8]
    core_seq = [True, False, False, True]
    for cycle in range(4):
        rows.append(
            {
                "sample_id": "s",
                "layer": 0,
                "position": 7,
                "cycle": cycle,
                "attn": attn_seq[cycle],
                "rank": float(cycle + 1),
                "in_core": core_seq[cycle],
            }
    )
    frame = add_future_targets(pd.DataFrame(rows), horizons=[2], core_budget=1)
    assert frame.loc[0, "fut_attn_2"] == pytest.approx(np.mean([0.2, 0.4]))
    assert frame.loc[2, "fut_attn_2"] == pytest.approx(0.8)
    assert np.isnan(frame.loc[3, "fut_attn_2"])
    assert frame.loc[0, "fut_min_rank_2"] == pytest.approx(2.0)
    # revival: outside core now, inside within 2 cycles
    assert bool(frame.loc[1, "revival_2"]) is True  # in core at cycle 3
    assert bool(frame.loc[0, "revival_2"]) is False  # currently inside
    assert bool(frame.loc[2, "revival_2"]) is True  # in core at cycle 3


def test_classify_token() -> None:
    assert classify_token("Ġ17", 5) == "numeric"
    assert classify_token(",", 100) == "punctuation"
    assert classify_token("Ġzebra", 1) == "rare"
    assert classify_token("ĠApple", 10) == "capitalized"
    assert classify_token("Ġthe", 500) == "common"
    assert classify_token(" ", 3) == "structural"


def test_headwise_rows_own_topk_dominates_shared() -> None:
    from types import SimpleNamespace

    from statekv.qkv_decomposition import _headwise_rows

    # 2 KV heads with disjoint preferences over 8 eligible positions.
    n = 10
    positions = list(range(n))
    eligible = list(range(2, n))  # 0,1 mandatory (sink/recent)
    attn = torch.zeros(2, n, dtype=torch.float64)
    attn[0, [0, 1, 2, 3, 4, 5]] = torch.tensor([0.2, 0.2, 0.2, 0.15, 0.15, 0.1], dtype=torch.float64)
    attn[1, [0, 1, 6, 7, 8, 9]] = torch.tensor([0.2, 0.2, 0.2, 0.15, 0.15, 0.1], dtype=torch.float64)
    core = [2, 3, 4, 5]  # shared core favours head 0
    sample = SimpleNamespace(sample_id="s", task="t")
    rows = _headwise_rows(sample, 0, 0, attn, positions, eligible, core, 4)
    assert len(rows) == 2
    by_head = {r["head"]: r for r in rows}
    # own top-k must capture at least as much mass as the shared core
    assert by_head[0]["own_mass"] >= by_head[0]["shared_mass"] - 1e-12
    assert by_head[1]["own_mass"] > by_head[1]["shared_mass"]
    # head 1 prefers a disjoint set -> zero overlap with the shared core
    assert by_head[1]["own_shared_overlap"] == 0.0
    assert by_head[0]["own_shared_overlap"] == 1.0
    # mandatory positions are always included
    assert by_head[0]["shared_mass"] > 0.4 - 1e-12
