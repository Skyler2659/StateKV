"""Eviction method registry, metadata, and safe constructor filtering."""
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from src.eviction.base import BaseEviction

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvictionMethodSpec:
    """Static metadata for an eviction method."""

    name: str
    module_path: Optional[str]
    class_name: Optional[str]
    family: str
    supports_backends: Tuple[str, ...] = ("torch",)
    requires_attention: bool = False
    requires_scores: bool = False
    supports_layerwise: bool = True
    supports_headwise: bool = False
    score_source: Optional[str] = None
    score_normalization: str = "none"
    approximate: bool = False
    experimental: bool = False
    oracle: bool = False
    paper_method: Optional[bool] = None
    paper_title: Optional[str] = None
    paper_url: Optional[str] = None
    reference_implementation_url: Optional[str] = None
    implementation_fidelity: str = "unreviewed"
    fidelity_notes: Optional[str] = None
    aliases: Tuple[str, ...] = ()
    unsupported_reason: Optional[str] = None
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def supports_mlx(self) -> bool:
        return "mlx" in self.supports_backends

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["supports_mlx"] = self.supports_mlx
        return data


def _spec(
    name: str,
    module_path: Optional[str],
    class_name: Optional[str],
    family: str,
    *,
    supports: Iterable[str] = ("torch",),
    aliases: Iterable[str] = (),
    default_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> EvictionMethodSpec:
    return EvictionMethodSpec(
        name=name,
        module_path=module_path,
        class_name=class_name,
        family=family,
        supports_backends=tuple(supports),
        aliases=tuple(aliases),
        default_kwargs=dict(default_kwargs or {}),
        **kwargs,
    )


_SPECS: Dict[str, EvictionMethodSpec] = {}
_ALIASES: Dict[str, Tuple[str, Dict[str, Any]]] = {}


def _register(spec: EvictionMethodSpec) -> None:
    _SPECS[spec.name] = spec
    for alias in spec.aliases:
        _ALIASES[_clean(alias)] = (spec.name, dict(spec.default_kwargs))


def _clean(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def _init_registry() -> None:
    # Recency/locality
    _register(_spec("full", "src.eviction.recency", "FullKVCache", "recency",
                    supports=("torch", "mlx"), aliases=("full_cache",),
                    paper_method=False, implementation_fidelity="control",
                    fidelity_notes="No-eviction upper-bound control; not an originating paper method."))
    _register(_spec("recency", "src.eviction.recency", "RecencyEviction", "recency",
                    supports=("torch", "mlx"), aliases=("sliding_window",)))
    _register(_spec("sink_recent", "src.eviction.sink_recent", "SinkRecentEviction", "recency",
                    supports=("torch", "mlx"), aliases=("sink_recency",)))
    _register(_spec("streamingllm", "src.eviction.sink_recent", "SinkRecentEviction", "recency",
                    supports=("torch", "mlx"), aliases=("streaming_llm",),
                    paper_method=True,
                    paper_title="Efficient Streaming Language Models with Attention Sinks",
                    paper_url="https://arxiv.org/abs/2309.17453",
                    reference_implementation_url="https://github.com/mit-han-lab/streaming-llm",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Retains attention sinks plus a recent rotating window. The MLX cache "
                        "preserves a logical RoPE offset, but this is not the paper's dedicated "
                        "position-shift/re-rotation implementation."
                    )))
    _register(_spec("random", "src.eviction.random_eviction", "RandomEviction", "random",
                    supports=("torch", "mlx"), aliases=("random_only",),
                    paper_method=False, implementation_fidelity="control",
                    fidelity_notes="Seeded random token-selection control; not an originating paper method."))
    _register(_spec("sink_recent_random", "src.eviction.random_eviction", "RandomEviction", "random",
                    aliases=("sink_recency_random",), supports=("torch", "mlx")))
    _register(_spec("uniform", "src.eviction.uniform", "UniformEviction", "geometry"))

    # Attention
    _register(_spec("attention", "src.eviction.attention", "AttentionEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="accumulated_attention", aliases=("accumulated_attention",)))
    _register(_spec("last_token_attention", "src.eviction.attention", "LastTokenAttentionEviction", "attention",
                    requires_attention=True, requires_scores=True, score_source="last_token_attention"))
    _register(_spec("windowed_attention", "src.eviction.attention", "WindowedAttentionEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="windowed_attention"))
    _register(_spec("attention_decay", "src.eviction.attention", "AttentionDecayEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="decayed_attention"))
    _register(_spec("h2o", "src.eviction.h2o", "H2OEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="accumulated_attention", aliases=("h2o_style",),
                    paper_method=True,
                    paper_title="H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models",
                    paper_url="https://arxiv.org/abs/2306.14048",
                    reference_implementation_url="https://github.com/FMInference/H2O",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Uses accumulated causal attention and a heavy-hitter/recent split. "
                        "Attention is pooled to the physical KV-head/token layout rather than "
                        "using the original CUDA implementation."
                    )))
    _register(_spec("snapkv", "src.eviction.snapkv", "SnapKVEviction", "attention",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True,
                    experimental=True,
                    score_source="observation_window_attention",
                    aliases=("snap", "snapkv_style", "approximate_snapkv"),
                    paper_method=True,
                    paper_title="SnapKV: LLM Knows What You are Looking for Before Generation",
                    paper_url="https://arxiv.org/abs/2404.14469",
                    reference_implementation_url="https://github.com/FasterDecoding/SnapKV",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "MLX path performs one per-KV-head prefill selection from the final "
                        "observation window with pooling and retains the window. It is a Qwen/MLX "
                        "reimplementation, not a line-for-line port of the official backend."
                    )))
    _register(_spec("tova", "src.eviction.paper_methods", "TOVAEviction", "attention",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="last_query_attention",
                    paper_method=True,
                    paper_title="Transformers are Multi-State RNNs",
                    paper_url="https://aclanthology.org/2024.emnlp-main.1043/",
                    reference_implementation_url="https://github.com/schwartz-lab-NLP/TOVA",
                    implementation_fidelity="faithful_core",
                    fidelity_notes=(
                        "Drops the lowest latest-query attention after averaging heads, as in "
                        "Algorithm 1. The optional logarithmic position-gap compression used only "
                        "for extrapolation beyond training length is not implemented."
                    )))
    _register(_spec("pyramidkv", "src.eviction.pyramidkv", "PyramidKVEviction", "attention",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_layerwise=True, supports_headwise=True, score_source="layer_budget_observation_attention",
                    approximate=True, experimental=True,
                    aliases=("pyramidkv_style", "layer_budget_attention")))

    # Geometry / norms
    _register(_spec("key_l2_norm", "src.eviction.norm_based", "KeyL2NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="key",
                    aliases=("key_norm",)))
    _register(_spec("value_l2_norm", "src.eviction.norm_based", "ValueL2NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    aliases=("value_norm", "value_l2")))
    _register(_spec("key_l1_norm", "src.eviction.norm_based", "KeyL1NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="key"))
    _register(_spec("value_l1_norm", "src.eviction.norm_based", "ValueL1NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value"))
    _register(_spec("knorm", "src.eviction.paper_methods", "KNormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True,
                    score_source="negative_key_l2_norm", approximate=True,
                    paper_method=True,
                    paper_title="A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression",
                    paper_url="https://aclanthology.org/2024.emnlp-main.1027/",
                    reference_implementation_url="https://github.com/alessiodevoto/l2compress",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Correctly retains the lowest key-L2-norm tokens. Unlike the paper's "
                        "default experiment, the shared-budget runner does not automatically leave "
                        "the first two transformer layers uncompressed."
                    )))
    _register(_spec("keydiff", "src.eviction.paper_methods", "KeyDiffEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True,
                    score_source="negative_cosine_to_mean_key", approximate=True,
                    paper_method=True,
                    paper_title="KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments",
                    paper_url="https://arxiv.org/abs/2504.15364",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Uses the paper's efficient negative cosine-to-mean-key score. Current "
                        "prefill sees the complete prompt instead of enforcing the paper's default "
                        "128-token block-wise memory bound."
                    )))
    _register(_spec("vnorml1", "src.eviction.norm_based", "ValueL1NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value_l1_norm",
                    aliases=("vnorm_l1", "v_norm_l1"), paper_method=False,
                    implementation_fidelity="control",
                    fidelity_notes=(
                        "Standalone high-value-L1-norm retention ablation. VATP uses this norm as "
                        "a factor, but VNormL1 is not itself the named VATP paper method."
                    )))
    _register(_spec("vnorml2", "src.eviction.norm_based", "ValueL2NormEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value_l2_norm",
                    aliases=("vnorm_l2", "v_norm_l2"), paper_method=False,
                    implementation_fidelity="control",
                    fidelity_notes="Standalone high-value-L2-norm retention ablation; no single originating paper method."))
    _register(_spec("vatp", "src.eviction.paper_methods", "VATPEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="accumulated_attention_times_value_l1_norm", approximate=True,
                    paper_method=True,
                    paper_title="Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters",
                    paper_url="https://aclanthology.org/2024.emnlp-main.1178/",
                    reference_implementation_url="https://github.com/guozhiyu/vatp",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Implements the paper's H2O-with-VATP variant: accumulated attention times "
                        "per-token value L1 norm, followed by heavy-hitter/recent selection."
                    )))
    _register(_spec("curdkv", "src.eviction.paper_methods", "CurDKVEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True,
                    score_source="gaussian_projected_key_value_row_norm_product", approximate=True,
                    paper_method=True,
                    paper_title="Value-Guided KV Compression for LLMs via Approximated CUR Decomposition",
                    paper_url="https://arxiv.org/abs/2509.15038",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "Uses Algorithm 1's Gaussian projection (r=20), product of projected key "
                        "and value squared row norms, normalization, and protected initial sinks. "
                        "The live runner aggregates KV-head scores to a shared token set."
                    ),
                    default_kwargs={"curdkv_projection_dim": 20, "curdkv_num_sink": 4}))
    _register(_spec("kv_norm", "src.eviction.norm_based", "KVNormEviction", "geometry",
                    requires_scores=True, score_source="key_value_concat", aliases=("norm",)))
    _register(_spec("hidden_l2_norm", None, None, "geometry",
                    requires_scores=True, score_source="hidden",
                    unsupported_reason="Hidden states are not available through the current eviction cache interface."))

    # Leverage / subspace
    _register(_spec("l1_leverage", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    approximate=True, aliases=("l1", "l1_mixed", "l1_only")))
    _register(_spec("l2_leverage", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    aliases=("l2",)))
    _register(_spec("key_l2_leverage", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="key"))
    _register(_spec("value_l2_leverage", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="value"))
    _register(_spec("kv_l2_leverage", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="key_value_concat"))
    _register(_spec("l1_prefill_only", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    approximate=True,
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("l2_prefill_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("torch", "mlx"), requires_scores=True, score_source="value",
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("l2_key_prefill_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="key",
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0, "score_source": "k"}))
    _register(_spec("conditional_v_leverage", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="ridge_residual_v_given_k_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: ridge-regress V on K independently in each KV head, "
                        "then select by ridge leverage of the unexplained V residual."
                    ), aliases=("k_conditioned_v_leverage", "v_given_k_leverage"),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("conditional_k_leverage", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="ridge_residual_k_given_v_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Reverse project ablation: ridge-regress K on V, then select by ridge "
                        "leverage of the unexplained K residual."
                    ), aliases=("v_conditioned_k_leverage", "k_given_v_leverage"),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("attention_residual_v_leverage", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True, experimental=True, paper_method=False,
                    score_source="attention_core_plus_residual_value_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: attention selects a per-head core; ridge V leverage fills "
                        "directions outside the value row space covered by that core."
                    ), aliases=("residual_attention_v_leverage",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("window_residual_v_leverage", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True, experimental=True, paper_method=False,
                    score_source="observation_window_attention_core_plus_residual_value_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: the final fixed prefill query window selects an attention "
                        "core using SnapKV-style aggregation and pooling; ridge V leverage then "
                        "fills directions outside the value row space covered by that core. "
                        "Unlike the all-query residual prototype, attention recording is O(Wn)."
                    ), aliases=(
                        "window_attention_residual_v_leverage",
                        "bounded_residual_v_leverage",
                    ),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("attention_weighted_v_leverage", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True, experimental=True, paper_method=False,
                    score_source="attention_weighted_value_ridge_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: compute ridge leverage on diag(epsilon+attention)^1/2 V; "
                        "this changes the fitted geometry rather than multiplying two final scores."
                    ), aliases=("query_weighted_v_leverage",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("window_weighted_v_leverage", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True, experimental=True, paper_method=False,
                    score_source="observation_window_attention_weighted_value_ridge_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: compute ridge leverage on "
                        "diag(epsilon+window_attention)^1/2 V using the same bounded "
                        "SnapKV-style observation attention as window residual leverage."
                    ), aliases=("window_query_weighted_v_leverage",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("joint_kv_leverage", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="scale_normalized_joint_key_value_leverage",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: concatenate separately RMS-normalized K and V blocks, "
                        "weighted by gamma, before fitting ridge leverage."
                    ), aliases=("joint_block_leverage",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("ridge_v_allocation", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="budget_adaptive_ridge_value_leverage+effective_dimension_head_budget",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: budget-linked ridge V leverage selects tokens and its "
                        "per-head effective dimension allocates the layer's token-head budget."
                    ), aliases=("ridge_v_leverage_allocation", "geometry_adakv"),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("ridge_v_fixed", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="budget_adaptive_ridge_value_leverage+uniform_head_budget",
                    implementation_fidelity="ablation",
                    fidelity_notes=(
                        "Ablation for ridge_v_allocation: identical budget-linked ridge scores "
                        "and per-head token selection, but a uniform budget for every KV head."
                    ), aliases=("ridge_v_uniform_allocation",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("ridge_v_shared", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="budget_adaptive_ridge_value_leverage+shared_token_selection",
                    implementation_fidelity="ablation",
                    fidelity_notes=(
                        "Shared-selection control: average the same per-head budget-linked "
                        "ridge V scores used by ridge_v_fixed, then retain one common token set."
                    ), aliases=("shared_ridge_v_leverage",),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("diversity_v_leverage", None, None, "geometry",
                    supports=("mlx",), requires_scores=True, supports_headwise=True,
                    experimental=True, paper_method=False,
                    score_source="ridge_value_leverage_candidates+pivoted_qr",
                    implementation_fidelity="prototype",
                    fidelity_notes=(
                        "Project method: top-cB ridge-leverage candidates followed by pivoted QR "
                        "on whitened value rows to discourage redundant selections."
                    ), aliases=("diversity_aware_leverage", "qr_v_leverage"),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True,
                    approximate=True, experimental=True,
                    score_source="prefill_key_leverage+non_causal_attention",
                    unsupported_reason=None,
                    aliases=("compactor_style", "compactor_l2_attention"),
                    paper_method=True,
                    paper_title="Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores",
                    paper_url="https://arxiv.org/abs/2507.08143",
                    reference_implementation_url="https://github.com/vnchari/compactor-vllm",
                    implementation_fidelity="approximate",
                    fidelity_notes=(
                        "Implements pre-RoPE approximate key leverage plus chunked non-causal "
                        "attention and z-score blending. Context-calibrated retention and the "
                        "official vLLM/Triton sparse-cache kernels are not implemented."
                    ),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0, "score_source": "k"}))
    _register(_spec("adakv", None, None, "attention",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    supports_headwise=True, experimental=True,
                    score_source="adaptive_head_budget+snapkv_observation_attention",
                    paper_method=True,
                    paper_title="Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference",
                    paper_url="https://arxiv.org/abs/2407.11550",
                    reference_implementation_url="https://github.com/FFY0/AdaKV",
                    implementation_fidelity="core",
                    fidelity_notes=(
                        "AdaKV is a budget allocator rather than a standalone scorer. The registered "
                        "strategy is Ada-SnapKV: pooled observation-window scores, global per-layer "
                        "top-B allocation, and the paper's safeguard interpolation (alpha=0.2)."
                    ),
                    default_kwargs={"update_policy": "prefill_only", "update_interval": 0}))
    _register(_spec("l1_decode_only", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="value",
                    approximate=True,
                    default_kwargs={"update_policy": "decode_only"}))
    _register(_spec("l2_decode_only", "src.eviction.l2_leverage", "L2LeverageEviction", "geometry",
                    supports=("mlx",), requires_scores=True, score_source="value",
                    default_kwargs={"update_policy": "decode_only"}))
    _register(_spec("ridge_leverage", "src.eviction.l2_leverage", "RidgeLeverageEviction", "geometry",
                    requires_scores=True, score_source="value"))
    _register(_spec("approximate_l2_leverage", "src.eviction.l2_leverage", "ApproximateL2LeverageEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("approximate_l1_leverage", "src.eviction.l1_leverage", "L1LeverageEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True,
                    default_kwargs={"use_reweight": True}))

    # Diversity / coverage
    _register(_spec("farthest_point_sampling", "src.eviction.clustering", "FarthestPointEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("farthest_point", "cosine_diversity", "k_center", "clustering")))
    _register(_spec("kmeans_centroid", "src.eviction.clustering", "KMeansMedoidEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("kmeans_medoid",)))
    _register(_spec("facility_location_greedy", "src.eviction.clustering", "ApproxFacilityLocationEviction", "geometry",
                    requires_scores=True, score_source="key", approximate=True,
                    aliases=("approximate_facility_location",)))
    _register(_spec("pca_residual", "src.eviction.pca_residual", "PCAResidualEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))

    # Outlier / rarity
    _register(_spec("mahalanobis_distance", "src.eviction.outlier", "MahalanobisDistanceEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("zscore_outlier", "src.eviction.outlier", "ZScoreOutlierEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))
    _register(_spec("random_projection_outlier", "src.eviction.outlier", "RandomProjectionOutlierEviction", "geometry",
                    requires_scores=True, score_source="value", approximate=True))

    # Hybrid
    _register(_spec("attention_l1", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+l1_leverage",
                    approximate=True,
                    aliases=("attention+l1", "attn_l1"),
                    default_kwargs={"geometry_method": "l1"}))
    _register(_spec("attention_l2", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+l2_leverage",
                    aliases=("attention+l2", "attn_l2"),
                    default_kwargs={"geometry_method": "l2"}))
    _register(_spec("attention_l1_compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    score_source="rank_attention+rank_l1_leverage_score_fusion",
                    approximate=True,
                    aliases=("attention+l1_compactor", "attn_l1_compactor", "compactorlike_l1_attention"),
                    default_kwargs={"geometry_method": "l1", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_l2_compactor", None, None, "hybrid",
                    supports=("mlx",), requires_attention=True, requires_scores=True,
                    score_source="rank_attention+rank_l2_leverage_score_fusion",
                    aliases=("attention+l2_compactor", "attn_l2_compactor", "compactorlike_l2_attention"),
                    default_kwargs={"geometry_method": "l2", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_recency", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+recency",
                    aliases=("attn_recency",),
                    default_kwargs={"geometry_method": "recency", "hybrid_mode": "interpolation"}))
    _register(_spec("attention_sink_recency", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+sink+recency",
                    default_kwargs={"geometry_method": "recency", "hybrid_mode": "budget_split"}))
    _register(_spec("attention_norm", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="attention+norm",
                    default_kwargs={"geometry_method": "value_norm"}))
    _register(_spec("weighted_score_hybrid", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    requires_attention=True, requires_scores=True,
                    score_source="weighted_components",
                    default_kwargs={"hybrid_mode": "interpolation"}))
    _register(_spec("budget_split_hybrid", "src.eviction.hybrid", "HybridEviction", "hybrid",
                    supports=("torch", "mlx"), requires_attention=True, requires_scores=True,
                    score_source="budget_split_components",
                    aliases=("sink_recent_attention_l1",),
                    default_kwargs={"hybrid_mode": "budget_split", "geometry_method": "l1"}))
    _register(_spec("recency_l1", "src.eviction.l1_leverage", "L1LeverageEviction", "hybrid",
                    requires_scores=True, score_source="recency+l1_leverage"))
    _register(_spec("sink_recent_l1", "src.eviction.l1_leverage", "L1LeverageEviction", "hybrid",
                    supports=("torch", "mlx"), requires_scores=True, score_source="sink+recent+l1_leverage"))
    _register(_spec("sink_recent_l2", "src.eviction.l2_leverage", "L2LeverageEviction", "hybrid",
                    supports=("torch", "mlx"), requires_scores=True, score_source="sink+recent+l2_leverage"))

    # Oracle
    _register(_spec("oracle_evidence", "src.eviction.oracle", "OracleEvidenceEviction", "oracle",
                    supports=("torch", "mlx"), oracle=True, score_source="evidence_positions"))
    _register(_spec("oracle_answer_region", "src.eviction.oracle", "OracleAnswerRegionEviction", "oracle",
                    supports=("torch", "mlx"), oracle=True, score_source="answer_region"))


_init_registry()


def canonicalize_method(method: str, kwargs: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    """Resolve aliases and return canonical method plus alias defaults."""
    key = _clean(method)
    kwargs = dict(kwargs or {})
    if key == "clustering":
        key = _clean(kwargs.get("clustering_method", "farthest_point_sampling"))
    if key in _SPECS:
        spec = _SPECS[key]
        return spec.name, dict(spec.default_kwargs)
    canonical, defaults = _ALIASES.get(key, (key, {}))
    if canonical in _SPECS:
        merged = dict(_SPECS[canonical].default_kwargs)
        merged.update(defaults)
        return canonical, merged
    return canonical, dict(defaults)


def list_methods(include_aliases: bool = True) -> List[str]:
    values = set(_SPECS)
    if include_aliases:
        values.update(_ALIASES)
    return sorted(values)


def get_method_spec(method: str) -> EvictionMethodSpec:
    canonical, _ = canonicalize_method(method)
    if canonical not in _SPECS:
        raise ValueError(f"Unknown eviction method: {method!r}. Available: {list_methods()}")
    return _SPECS[canonical]


def method_metadata(method: str) -> Dict[str, Any]:
    return get_method_spec(method).to_dict()


def method_requires_attention(method: str) -> bool:
    return bool(get_method_spec(method).requires_attention)


def method_supports_backend(method: str, backend: str) -> bool:
    spec = get_method_spec(method)
    return str(backend).lower() in spec.supports_backends and spec.unsupported_reason is None


def unsupported_reason(method: str, backend: str) -> Optional[str]:
    spec = get_method_spec(method)
    if spec.unsupported_reason:
        return spec.unsupported_reason
    if str(backend).lower() not in spec.supports_backends:
        return f"{backend} backend does not support method={method}"
    return None


def get_eviction_class(method: str) -> Type[BaseEviction]:
    spec = get_method_spec(method)
    if spec.module_path is None or spec.class_name is None:
        raise NotImplementedError(spec.unsupported_reason or f"Method {method} is not implemented")
    mod = importlib.import_module(spec.module_path)
    return getattr(mod, spec.class_name)


def _accepted_constructor_params(cls: Type) -> set:
    accepted = set()
    for klass in cls.mro():
        if klass is object:
            continue
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == "self" or param.kind == param.VAR_KEYWORD:
                continue
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                accepted.add(name)
    return accepted


def create_eviction(
    method: str,
    cache_size: int,
    k_seq_dim: int = 2,
    v_seq_dim: int = 2,
    sink_size: int = 0,
    recent_size: int = 0,
    **kwargs: Any,
) -> BaseEviction:
    """Create an eviction instance and drop irrelevant kwargs with a warning."""
    canonical, alias_defaults = canonicalize_method(method, kwargs)
    spec = get_method_spec(canonical)
    cls = get_eviction_class(canonical)

    if _clean(method) == "l1_only":
        sink_size = 0
        recent_size = 0
    if canonical == "random":
        sink_size = 0
        recent_size = 0

    if "geom_budget_ratio" not in kwargs and "l1_budget_ratio" in kwargs:
        kwargs["geom_budget_ratio"] = kwargs["l1_budget_ratio"]

    base_kwargs = {
        "cache_size": cache_size,
        "k_seq_dim": k_seq_dim,
        "v_seq_dim": v_seq_dim,
        "sink_size": sink_size,
        "recent_size": recent_size,
    }
    merged = {**kwargs, **alias_defaults, **base_kwargs}
    accepted = _accepted_constructor_params(cls)
    filtered = {k: v for k, v in merged.items() if k in accepted and v is not None}
    ignored = sorted(k for k in merged if k not in accepted)
    if ignored:
        logger.debug(
            "Eviction method %s (%s) ignored unsupported parameter(s): %s",
            method,
            cls.__name__,
            ignored,
        )
    instance = cls(**filtered)
    instance.name = canonical
    instance.method_family = spec.family
    instance.supports_backends = spec.supports_backends
    instance.requires_attention = spec.requires_attention
    instance.requires_scores = spec.requires_scores
    instance.supports_layerwise = spec.supports_layerwise
    instance.supports_headwise = spec.supports_headwise
    instance.score_source = getattr(instance, "score_source", None) or spec.score_source
    instance.score_normalization = str(
        filtered.get("score_normalization", spec.score_normalization)
        or spec.score_normalization
    )
    instance.approximate = spec.approximate
    instance.experimental = spec.experimental
    instance.oracle = spec.oracle
    return instance


BASIC_METHODS = ["full", "recency", "sink_recent", "streamingllm", "random", "uniform"]
ATTENTION_METHODS = [
    "attention",
    "last_token_attention",
    "windowed_attention",
    "attention_decay",
    "h2o",
    "snapkv",
    "tova",
    "vatp",
    "adakv",
    "pyramidkv",
]
GEOMETRY_METHODS = [
    "knorm",
    "keydiff",
    "vnorml1",
    "vnorml2",
    "curdkv",
    "key_l2_norm",
    "value_l2_norm",
    "key_l1_norm",
    "value_l1_norm",
    "l1_leverage",
    "l2_leverage",
    "l1_prefill_only",
    "l2_prefill_only",
    "l2_key_prefill_only",
    "conditional_v_leverage",
    "conditional_k_leverage",
    "joint_kv_leverage",
    "ridge_v_allocation",
    "ridge_v_fixed",
    "ridge_v_shared",
    "diversity_v_leverage",
    "l1_decode_only",
    "l2_decode_only",
    "compactor",
    "ridge_leverage",
    "approximate_l2_leverage",
    "approximate_l1_leverage",
    "farthest_point_sampling",
    "kmeans_centroid",
    "facility_location_greedy",
    "mahalanobis_distance",
    "zscore_outlier",
    "random_projection_outlier",
]
L1_METHODS = ["l1_leverage", "approximate_l1_leverage"]
HYBRID_METHODS = [
    "attention_residual_v_leverage",
    "window_residual_v_leverage",
    "attention_weighted_v_leverage",
    "window_weighted_v_leverage",
    "attention_l1",
    "attention_l2",
    "attention_l1_compactor",
    "attention_l2_compactor",
    "attention_norm",
    "attention_recency",
    "attention_sink_recency",
    "recency_l1",
    "sink_recent_l1",
    "sink_recent_l2",
    "sink_recent_attention_l1",
    "weighted_score_hybrid",
    "budget_split_hybrid",
]
ORACLE_METHODS = ["oracle_evidence", "oracle_answer_region"]

PAPER_BASELINES = [
    "full",
    "random",
    "streamingllm",
    "tova",
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "snapkv",
    "knorm",
    "keydiff",
    "compactor",
    "vnorml1",
    "vnorml2",
    "vatp",
    "curdkv",
    "adakv",
    "key_l2_norm",
    "value_l2_norm",
    "l2_leverage",
    "l1_leverage",
    "attention_l1",
]

# The exact strategy panel requested for paper-method comparison.  Names are
# canonical registry identifiers; input remains case-insensitive, so e.g.
# ``TOVA``, ``KeyDiff``, ``VNorml1`` and ``CurDKV`` resolve to these entries.
REQUESTED_STRATEGY_METHODS = [
    "full",
    "random",
    "streamingllm",
    "h2o",
    "snapkv",
    "tova",
    "knorm",
    "keydiff",
    "compactor",
    "vnorml1",
    "vnorml2",
    "vatp",
    "curdkv",
    "adakv",
]

AGGRESSIVE_COMPARISON = [
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "l2_leverage",
    "l1_leverage",
    "attention_l1",
]
