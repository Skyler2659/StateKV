import pytest
import torch

from kvbench.config import BudgetConfig, MethodConfig, ProtocolConfig
from kvbench.errors import ProtocolError
from kvbench.methods.policy import EvictionPolicy
from kvbench.protocols.base import Protocol
from kvbench.types import AttentionSignals, CacheSnapshot


def _snapshot(length=12):
    generator = torch.Generator().manual_seed(4)
    key = torch.randn(1, 2, length, 4, generator=generator)
    value = torch.randn(1, 2, length, 4, generator=generator)
    attention = torch.stack(
        [torch.arange(length, dtype=torch.float32), torch.arange(length, 0, -1.0)]
    )
    return CacheSnapshot(
        sample_id="s0",
        snapshot_id="s0:prefill",
        phase="pre_answer",
        decode_step=0,
        logical_length=length,
        keys=[key],
        values=[value],
        position_maps={0: torch.arange(length)},
        attention=AttentionSignals(
            accumulated_by_layer={0: attention},
            observation_by_layer={0: attention},
            last_query_by_layer={0: attention},
        ),
    )


@pytest.mark.parametrize(
    "method",
    [
        "random", "recency", "sink_recent", "streamingllm", "attention",
        "h2o", "snapkv", "k_norm", "v_norm_l1", "v_norm_l2",
        "k_leverage", "v_leverage", "joint_kv_leverage", "curdkv",
        "independent_hybrid", "score_fusion", "product", "residual_v",
    ],
)
def test_every_compressed_method_obeys_same_total_budget(method):
    policy = EvictionPolicy(
        MethodConfig(name=method, pooling_kernel=3),
        BudgetConfig(cache_budget=6, sink_size=1, recent_size=1),
        seed=9,
    )
    decisions, _ = policy.decide(_snapshot())
    assert len(decisions) == 1
    assert decisions[0].effective_budget == 6
    assert len(set(decisions[0].selected_positions)) == 6
    assert set(decisions[0].mandatory_positions).issubset(
        decisions[0].selected_positions
    )


def test_random_policy_is_reproducible():
    cfg = MethodConfig(name="random")
    budget = BudgetConfig(cache_budget=5, sink_size=0, recent_size=0)
    first, _ = EvictionPolicy(cfg, budget, seed=91).decide(_snapshot())
    second, _ = EvictionPolicy(cfg, budget, seed=91).decide(_snapshot())
    assert first[0].selected_positions == second[0].selected_positions


def test_residual_policy_records_core_and_both_components():
    policy = EvictionPolicy(
        MethodConfig(name="residual_v", attention_ratio=0.5),
        BudgetConfig(cache_budget=6, sink_size=1, recent_size=1),
        seed=0,
    )
    decisions, bundle = policy.decide(_snapshot())
    assert set(bundle.components) >= {"attention", "v_leverage", "residual_v"}
    assert decisions[0].metadata["attention_core_rows"]
    assert set(decisions[0].metadata["attention_core_rows"]).issubset(
        decisions[0].metadata["projection_core_rows"]
    )


def test_attention_method_is_forbidden_when_future_query_is_hidden():
    policy = EvictionPolicy(
        MethodConfig(name="attention"), BudgetConfig(), seed=0
    )
    protocol = Protocol(ProtocolConfig(visibility="query_agnostic"))
    with pytest.raises(ProtocolError, match="query-visible"):
        protocol.validate_method(policy.spec.requires_visible_query, policy.name)


def test_attention_and_h2o_use_distinct_visible_signals():
    snapshot = _snapshot()
    last = torch.zeros_like(snapshot.attention.accumulated_by_layer[0])
    last[:, 3] = 10.0
    snapshot.attention.last_query_by_layer[0] = last
    budget = BudgetConfig(cache_budget=6, sink_size=0, recent_size=0)
    _, attention_bundle = EvictionPolicy(
        MethodConfig(name="attention"), budget, seed=0
    ).decide(snapshot)
    _, h2o_bundle = EvictionPolicy(
        MethodConfig(name="h2o"), budget, seed=0
    ).decide(snapshot)
    assert torch.equal(attention_bundle.components["attention"][0], last.mean(0))
    assert not torch.equal(
        attention_bundle.components["attention"][0],
        h2o_bundle.components["attention"][0],
    )
