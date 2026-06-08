"""Eviction method registry and safe constructor filtering."""
from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Dict, List, Tuple, Type

from src.eviction.base import BaseEviction

logger = logging.getLogger(__name__)


_REGISTRY: Dict[str, Tuple[str, str]] = {
    # Basic baselines
    "full": ("src.eviction.recency", "FullKVCache"),
    "recency": ("src.eviction.recency", "RecencyEviction"),
    "sliding_window": ("src.eviction.recency", "RecencyEviction"),
    "sink_recent": ("src.eviction.sink_recent", "SinkRecentEviction"),
    "random": ("src.eviction.random_eviction", "RandomEviction"),
    "uniform": ("src.eviction.uniform", "UniformEviction"),
    # Attention-based
    "attention": ("src.eviction.attention", "AttentionEviction"),
    "h2o": ("src.eviction.h2o", "H2OEviction"),
    "snapkv": ("src.eviction.snapkv", "SnapKVEviction"),
    "pyramidkv": ("src.eviction.pyramidkv", "PyramidKVEviction"),
    # Geometry-based
    "key_norm": ("src.eviction.norm_based", "KeyNormEviction"),
    "value_norm": ("src.eviction.norm_based", "ValueNormEviction"),
    "kv_norm": ("src.eviction.norm_based", "KVNormEviction"),
    "norm": ("src.eviction.norm_based", "KVNormEviction"),
    "l2_leverage": ("src.eviction.l2_leverage", "L2LeverageEviction"),
    "farthest_point": ("src.eviction.clustering", "FarthestPointEviction"),
    "k_center": ("src.eviction.clustering", "KCenterEviction"),
    "kmeans_medoid": ("src.eviction.clustering", "KMeansMedoidEviction"),
    "pca_residual": ("src.eviction.pca_residual", "PCAResidualEviction"),
    # L1
    "l1_leverage": ("src.eviction.l1_leverage", "L1LeverageEviction"),
    "l1_mixed": ("src.eviction.l1_leverage", "L1LeverageEviction"),
    "l1_only": ("src.eviction.l1_leverage", "L1LeverageEviction"),
    # Hybrid
    "hybrid": ("src.eviction.hybrid", "HybridEviction"),
    "attention+l1": ("src.eviction.hybrid", "HybridEviction"),
    "attn_l1": ("src.eviction.hybrid", "HybridEviction"),
    "attn_l2": ("src.eviction.hybrid", "HybridEviction"),
    "attn_recency": ("src.eviction.hybrid", "HybridEviction"),
}


_ALIASES: Dict[str, Tuple[str, Dict[str, Any]]] = {
    "l1": ("l1_leverage", {}),
    "l2": ("l2_leverage", {}),
    "value_l2": ("l2_leverage", {"score_source": "v"}),
    "attention_l1": ("attn_l1", {"geometry_method": "l1"}),
    "attention+l1": ("attn_l1", {"geometry_method": "l1"}),
    "attn_l1": ("attn_l1", {"geometry_method": "l1"}),
    "attn_l2": ("attn_l2", {"geometry_method": "l2"}),
    "attn_recency": ("attn_recency", {"geometry_method": "recency"}),
    "clustering": ("farthest_point", {}),
}


def canonicalize_method(method: str, kwargs: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any]]:
    """Resolve aliases and return method plus default kwargs for that alias."""
    key = str(method).strip().lower().replace("-", "_")
    kwargs = dict(kwargs or {})
    if key == "clustering":
        key = str(kwargs.get("clustering_method", "farthest_point")).lower()
    canonical, defaults = _ALIASES.get(key, (key, {}))
    merged_defaults = dict(defaults)
    if canonical == "attn_l1":
        canonical = "hybrid"
        merged_defaults.setdefault("geometry_method", "l1")
    elif canonical == "attn_l2":
        canonical = "hybrid"
        merged_defaults.setdefault("geometry_method", "l2")
    elif canonical == "attn_recency":
        canonical = "hybrid"
        merged_defaults.setdefault("geometry_method", "recency")
    return canonical, merged_defaults


def list_methods() -> List[str]:
    return sorted(set(_REGISTRY) | set(_ALIASES))


def get_eviction_class(method: str) -> Type[BaseEviction]:
    canonical, _ = canonicalize_method(method)
    if canonical not in _REGISTRY:
        raise ValueError(
            f"Unknown eviction method: {method!r}. Available: {list_methods()}"
        )
    module_path, class_name = _REGISTRY[canonical]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


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
    cls = get_eviction_class(canonical)

    if canonical == "l1_only":
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
        logger.warning(
            "Eviction method %s (%s) ignored unsupported parameter(s): %s",
            method,
            cls.__name__,
            ignored,
        )
    return cls(**filtered)


BASIC_METHODS = ["full", "recency", "sink_recent", "random", "uniform"]
ATTENTION_METHODS = ["attention", "h2o", "snapkv", "pyramidkv"]
GEOMETRY_METHODS = [
    "key_norm",
    "value_norm",
    "kv_norm",
    "l2_leverage",
    "farthest_point",
    "pca_residual",
]
L1_METHODS = ["l1_leverage", "l1_mixed", "l1_only"]
HYBRID_METHODS = ["hybrid", "attn_l1", "attn_l2", "attn_recency", "attention+l1"]

PAPER_BASELINES = [
    "full",
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "snapkv",
    "key_norm",
    "value_norm",
    "l2_leverage",
    "l1_leverage",
    "attention+l1",
]

AGGRESSIVE_COMPARISON = [
    "recency",
    "sink_recent",
    "attention",
    "h2o",
    "l2_leverage",
    "l1_leverage",
    "attention+l1",
]
