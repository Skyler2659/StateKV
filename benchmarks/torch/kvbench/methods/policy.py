"""One policy implementation composed from canonical signals and budget rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from kvbench.cache.budget import BudgetAllocator, BudgetSelection, stable_topk
from kvbench.config import BudgetConfig, MethodConfig
from kvbench.errors import SignalUnavailableError, UnsupportedMethodError
from kvbench.scoring.core import ScoreEngine, normalize_scores
from kvbench.types import CacheSnapshot, ScoreBundle, SelectionDecision


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    requires_visible_query: bool
    requires_attention: bool
    fidelity: str
    notes: str


_SPECS = {
    "full": MethodSpec("full", "control", False, False, "control", "No compression."),
    "random": MethodSpec("random", "control", False, False, "control", "Seeded random control."),
    "recency": MethodSpec("recency", "locality", False, False, "core", "Strict sliding window."),
    "sink_recent": MethodSpec("sink_recent", "locality", False, False, "core", "Shared sink/recent policy."),
    "streamingllm": MethodSpec(
        "streamingllm", "locality", False, False, "reproduction",
        "Attention-sink token selection; not the official position-shift kernel.",
    ),
    "attention": MethodSpec(
        "attention", "attention", True, True, "project_definition",
        "Final visible prompt-query attention pooled to KV heads.",
    ),
    "h2o": MethodSpec(
        "h2o", "attention", True, True, "reproduction",
        "Accumulated attention heavy hitters plus the shared recent policy.",
    ),
    "snapkv": MethodSpec(
        "snapkv", "attention", True, True, "shared_cache_reproduction",
        "Observation-window attention with pooling and one shared token set per layer.",
    ),
    "k_norm": MethodSpec("k_norm", "geometry", False, False, "core", "Negative key L2 norm."),
    "v_norm_l1": MethodSpec("v_norm_l1", "geometry", False, False, "core", "Value row L1 norm."),
    "v_norm_l2": MethodSpec("v_norm_l2", "geometry", False, False, "core", "Value row L2 norm."),
    "k_leverage": MethodSpec("k_leverage", "geometry", False, False, "project_definition", "Per-head K leverage."),
    "v_leverage": MethodSpec("v_leverage", "geometry", False, False, "project_definition", "Per-head V leverage."),
    "joint_kv_leverage": MethodSpec(
        "joint_kv_leverage", "geometry", False, False, "project_definition",
        "Per-head leverage of concatenated K/V rows.",
    ),
    "curdkv": MethodSpec(
        "curdkv", "geometry", False, False, "shared_cache_reproduction",
        "CurDKV Gaussian-projected K/V row-norm product, averaged to one shared token set.",
    ),
    "independent_hybrid": MethodSpec(
        "independent_hybrid", "hybrid", True, True, "project_definition",
        "Independent attention and V-leverage budget split with deterministic backfill.",
    ),
    "score_fusion": MethodSpec(
        "score_fusion", "hybrid", True, True, "project_definition",
        "Normalized linear fusion of attention and V leverage.",
    ),
    "product": MethodSpec(
        "product", "hybrid", True, True, "project_definition",
        "Product of normalized attention and V leverage.",
    ),
    "residual_v": MethodSpec(
        "residual_v", "hybrid", True, True, "project_definition",
        "Attention core followed by ridge-residual V-space leverage.",
    ),
}

_ALIASES = {
    "full_cache": "full",
    "sliding_window": "recency",
    "attention_only": "attention",
    "v_leverage_only": "v_leverage",
    "independent_50_50": "independent_hybrid",
    "fusion": "score_fusion",
    "residual_hybrid": "residual_v",
    "residual_v_leverage": "residual_v",
    "knorm": "k_norm",
    "vnorml1": "v_norm_l1",
    "vnorml2": "v_norm_l2",
}


def canonical_method(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


def get_method_spec(name: str) -> MethodSpec:
    canonical = canonical_method(name)
    if canonical not in _SPECS:
        raise UnsupportedMethodError("unknown method: %s" % name)
    return _SPECS[canonical]


def list_methods() -> List[Dict[str, Any]]:
    return [spec.__dict__.copy() for spec in _SPECS.values()]


class EvictionPolicy:
    """Compute decisions without directly mutating a backend cache."""

    def __init__(
        self,
        method_cfg: MethodConfig,
        budget_cfg: BudgetConfig,
        seed: int,
    ):
        self.cfg = method_cfg
        self.name = canonical_method(method_cfg.name)
        self.spec = get_method_spec(self.name)
        self.budget_cfg = budget_cfg
        self.allocator = BudgetAllocator(budget_cfg)
        self.scorer = ScoreEngine(method_cfg, seed)
        self.seed = int(seed)
        self.decision_count = 0
        self._cached_priority: Dict[int, Dict[int, float]] = {}

    @property
    def variant(self) -> str:
        if self.name in {"k_leverage", "v_leverage", "joint_kv_leverage"}:
            return "%s_%s" % (self.name, self.cfg.leverage_estimator)
        if self.name == "independent_hybrid":
            return "%s_r%.2f_%s" % (
                self.name, self.cfg.attention_ratio, self.cfg.leverage_estimator
            )
        if self.name in {"score_fusion", "product"}:
            return "%s_a%.2f_%s_%s" % (
                self.name,
                self.cfg.alpha,
                self.cfg.normalization,
                self.cfg.leverage_estimator,
            )
        if self.name == "residual_v":
            return "%s_r%.2f_lam%.1e_%s" % (
                self.name,
                self.cfg.attention_ratio,
                self.cfg.residual_lambda,
                self.cfg.residual_lambda_mode,
            )
        if self.name == "curdkv":
            return "%s_r%d" % (self.name, self.cfg.curdkv_projection_dim)
        return self.name

    def decide(
        self,
        snapshot: CacheSnapshot,
        recompute: bool = True,
    ) -> Tuple[List[SelectionDecision], ScoreBundle]:
        if snapshot.num_layers == 0:
            return [], ScoreBundle()
        if not recompute and self.name not in {"full", "random", "recency", "sink_recent", "streamingllm"}:
            cached = self._decide_from_cached_priority(snapshot)
            if cached is not None:
                return cached
        decisions: List[SelectionDecision] = []
        bundle = ScoreBundle(diagnostics={"method": self.name, "fidelity": self.spec.fidelity})
        for layer, (key, value) in enumerate(zip(snapshot.keys, snapshot.values)):
            seq_len = int(key.shape[2])
            if int(value.shape[2]) != seq_len:
                raise SignalUnavailableError("K/V token axes do not align")
            position_map = snapshot.position_maps[layer].detach().cpu().tolist()
            if len(position_map) != seq_len:
                raise SignalUnavailableError("position map does not align with cache rows")
            decision, aggregate, by_head, components, diagnostics = self._decide_layer(
                snapshot, layer, key, value, seq_len
            )
            component_heads = diagnostics.pop("_components_by_head", {})
            decisions.append(
                self._selection_decision(
                    layer, position_map, decision, aggregate, components, diagnostics
                )
            )
            if aggregate is not None:
                bundle.aggregate[layer] = aggregate.detach().cpu()
            if by_head is not None:
                bundle.by_head[layer] = by_head.detach().cpu()
            for name, score in components.items():
                bundle.components.setdefault(name, {})[layer] = score.detach().cpu()
            for name, score in component_heads.items():
                bundle.components_by_head.setdefault(name, {})[layer] = (
                    score.detach().cpu()
                )
            bundle.diagnostics[str(layer)] = diagnostics
            if aggregate is not None:
                positions = snapshot.position_maps[layer].detach().cpu().tolist()
                values = aggregate.detach().float().cpu().tolist()
                self._cached_priority[layer] = {
                    int(position): float(score)
                    for position, score in zip(positions, values)
                }
        self.decision_count += 1
        return decisions, bundle

    def _decide_from_cached_priority(
        self, snapshot: CacheSnapshot
    ) -> Optional[Tuple[List[SelectionDecision], ScoreBundle]]:
        if not self._cached_priority:
            return None
        decisions: List[SelectionDecision] = []
        bundle = ScoreBundle(
            diagnostics={
                "method": self.name,
                "fidelity": self.spec.fidelity,
                "score_refresh": "cached",
            }
        )
        for layer, key in enumerate(snapshot.keys):
            cache = self._cached_priority.get(layer)
            if cache is None:
                return None
            positions = [
                int(value) for value in snapshot.position_maps[layer].detach().cpu().tolist()
            ]
            known = list(cache.values())
            floor = min(known) - max(1.0, abs(min(known))) if known else -1.0
            score = torch.tensor(
                [cache.get(position, floor) for position in positions],
                device=key.device,
                dtype=torch.float32,
            )
            selection = self.allocator.select(score, "cached_priority")
            decision = self._selection_decision(
                layer,
                positions,
                selection,
                score,
                {"cached_priority": score},
                {"score_refresh": "cached"},
            )
            decisions.append(decision)
            bundle.aggregate[layer] = score.detach().cpu()
            bundle.components.setdefault("cached_priority", {})[layer] = score.detach().cpu()
        self.decision_count += 1
        return decisions, bundle

    def _decide_layer(
        self,
        snapshot: CacheSnapshot,
        layer: int,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_len: int,
    ) -> Tuple[
        BudgetSelection,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, Any],
    ]:
        method = self.name
        if method == "full":
            rows = list(range(seq_len))
            return (
                BudgetSelection(rows, [], seq_len, {row: ["full"] for row in rows}),
                None,
                None,
                {},
                {"control": "full_cache"},
            )
        if seq_len <= int(self.budget_cfg.cache_budget):
            rows = list(range(seq_len))
            return (
                BudgetSelection(rows, self.allocator.mandatory_rows(seq_len), 0, {}),
                None,
                None,
                {},
                {"no_compression_needed": True},
            )

        if method in {"recency", "sink_recent", "streamingllm"}:
            score = torch.arange(seq_len, device=key.device, dtype=torch.float32)
            return self.allocator.select(score, method), score, None, {method: score}, {}

        if method == "random":
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                self.seed
                + int(self.cfg.random_seed_offset)
                + layer * 1009
                + self.decision_count * 1_000_003
            )
            score = torch.rand(seq_len, generator=generator).to(key.device)
            return self.allocator.select(score, "random"), score, None, {"random": score}, {
                "seed": int(generator.initial_seed())
            }

        if method in {"attention", "h2o"}:
            signal = (
                snapshot.attention.last_query_by_layer
                if method == "attention"
                else snapshot.attention.accumulated_by_layer
            )
            score, by_head = self.scorer.attention_by_layer(
                signal, layer, seq_len
            )
            return self.allocator.select(score, method), score, by_head, {"attention": score}, {
                "attention_signal": (
                    "final_visible_query_token" if method == "attention" else "cumulative_causal"
                ),
                "_components_by_head": {"attention": by_head}
            }

        if method == "snapkv":
            score, by_head = self.scorer.attention_by_layer(
                snapshot.attention.observation_by_layer, layer, seq_len
            )
            kernel = min(int(self.cfg.pooling_kernel), seq_len)
            if kernel > 1:
                if kernel % 2 == 0:
                    kernel -= 1
                pooled = F.avg_pool1d(
                    by_head.unsqueeze(1), kernel_size=kernel, stride=1, padding=kernel // 2
                ).squeeze(1)
                by_head = pooled
                score = by_head.mean(dim=0)
            return self.allocator.select(score, "snapkv"), score, by_head, {"observation_attention": score}, {
                "fidelity": self.spec.fidelity,
                "pooling": "avg_pool1d",
                "pooling_kernel": kernel,
                "_components_by_head": {"observation_attention": by_head},
            }

        if method in {"k_leverage", "v_leverage", "joint_kv_leverage"}:
            source = {"k_leverage": "k", "v_leverage": "v", "joint_kv_leverage": "kv"}[method]
            score, by_head, diagnostics = self.scorer.geometry(
                key, value, layer, source=source
            )
            diagnostics["_components_by_head"] = {method: by_head}
            return self.allocator.select(score, method), score, by_head, {method: score}, diagnostics

        if method == "curdkv":
            score, by_head, diagnostics = self.scorer.curdkv(key, value, layer)
            return (
                self.allocator.select(score, "curdkv"),
                score,
                by_head,
                {"curdkv": score},
                {**diagnostics, "_components_by_head": {"curdkv": by_head}},
            )

        if method in {"k_norm", "v_norm_l1", "v_norm_l2"}:
            if method == "k_norm":
                score, by_head, diagnostics = self.scorer.geometry(
                    key, value, layer, source="k", estimator="l2_norm"
                )
                score, by_head = -score, -by_head
            elif method == "v_norm_l1":
                score, by_head, diagnostics = self.scorer.geometry(
                    key, value, layer, source="v", estimator="l1_norm"
                )
            else:
                score, by_head, diagnostics = self.scorer.geometry(
                    key, value, layer, source="v", estimator="l2_norm"
                )
            diagnostics["_components_by_head"] = {method: by_head}
            return self.allocator.select(score, method), score, by_head, {method: score}, diagnostics

        attention, attention_heads = self.scorer.attention_by_layer(
            snapshot.attention.last_query_by_layer, layer, seq_len
        )
        leverage, leverage_heads, leverage_diag = self.scorer.geometry(
            key, value, layer, source="v"
        )
        attn_norm = normalize_scores(attention, self.cfg.normalization)
        leverage_norm = normalize_scores(leverage, self.cfg.normalization)
        components = {"attention": attention, "v_leverage": leverage}

        if method == "independent_hybrid":
            mandatory = self.allocator.mandatory_rows(seq_len)
            selectable = int(self.budget_cfg.cache_budget) - len(mandatory)
            attention_count = int(round(selectable * float(self.cfg.attention_ratio)))
            attention_count = min(selectable, max(0, attention_count))
            leverage_count = selectable - attention_count
            fused = float(self.cfg.attention_ratio) * attn_norm + (
                1.0 - float(self.cfg.attention_ratio)
            ) * leverage_norm
            selection = self.allocator.select_partitioned(
                {"attention": attention, "v_leverage": leverage},
                {"attention": attention_count, "v_leverage": leverage_count},
                fused,
            )
            return selection, fused, None, components, {
                "attention_count": attention_count,
                "leverage_count": leverage_count,
                "normalization": self.cfg.normalization,
                "leverage": leverage_diag,
                "_components_by_head": {
                    "attention": attention_heads,
                    "v_leverage": leverage_heads,
                },
            }

        if method == "score_fusion":
            fused = float(self.cfg.alpha) * attn_norm + (1.0 - float(self.cfg.alpha)) * leverage_norm
            return self.allocator.select(fused, "score_fusion"), fused, None, components, {
                "alpha": float(self.cfg.alpha),
                "normalization": self.cfg.normalization,
                "leverage": leverage_diag,
                "_components_by_head": {
                    "attention": attention_heads,
                    "v_leverage": leverage_heads,
                },
            }

        if method == "product":
            product = attn_norm * leverage_norm
            return self.allocator.select(product, "product"), product, None, components, {
                "normalization": self.cfg.normalization,
                "leverage": leverage_diag,
                "_components_by_head": {
                    "attention": attention_heads,
                    "v_leverage": leverage_heads,
                },
            }

        if method == "residual_v":
            mandatory = self.allocator.mandatory_rows(seq_len)
            selectable = int(self.budget_cfg.cache_budget) - len(mandatory)
            attention_count = int(round(selectable * float(self.cfg.attention_ratio)))
            candidates = torch.tensor(
                [row for row in range(seq_len) if row not in set(mandatory)],
                device=attention.device,
                dtype=torch.long,
            )
            attention_rows = stable_topk(attention, attention_count, candidates).tolist()
            projection_core = sorted(set(mandatory + [int(row) for row in attention_rows]))
            residual, residual_heads, residual_diag = self.scorer.residual_v(
                value, projection_core, layer
            )
            selection = self.allocator.select_partitioned(
                {"attention": attention, "residual_v": residual},
                {"attention": attention_count, "residual_v": selectable - attention_count},
                normalize_scores(attention, self.cfg.normalization)
                + normalize_scores(residual, self.cfg.normalization),
            )
            components["residual_v"] = residual
            priority = normalize_scores(attention, self.cfg.normalization) + normalize_scores(
                residual, self.cfg.normalization
            )
            return selection, priority, residual_heads, components, {
                "attention_core_rows": [int(row) for row in attention_rows],
                "projection_core_rows": projection_core,
                "attention_count": attention_count,
                "residual_count": selectable - attention_count,
                "residual": residual_diag,
                "_components_by_head": {
                    "attention": attention_heads,
                    "v_leverage": leverage_heads,
                    "residual_v": residual_heads,
                },
            }

        raise UnsupportedMethodError("method has no decision implementation: %s" % method)

    def _selection_decision(
        self,
        layer: int,
        position_map: List[int],
        selection: BudgetSelection,
        aggregate: Optional[torch.Tensor],
        components: Dict[str, torch.Tensor],
        diagnostics: Dict[str, Any],
    ) -> SelectionDecision:
        selected_positions = [int(position_map[row]) for row in selection.rows]
        mandatory_positions = [int(position_map[row]) for row in selection.mandatory_rows]
        source_by_position = {
            str(int(position_map[row])): list(sources)
            for row, sources in selection.sources.items()
        }
        seq_len = len(position_map)
        for row in selection.rows:
            labels = source_by_position.setdefault(str(int(position_map[row])), [])
            if row < int(self.budget_cfg.sink_size) and "sink" not in labels:
                labels.append("sink")
            if (
                int(self.budget_cfg.recent_size) > 0
                and row >= max(0, seq_len - int(self.budget_cfg.recent_size))
                and "recent" not in labels
            ):
                labels.append("recent")
            if self.budget_cfg.protect_current and row == seq_len - 1 and "current" not in labels:
                labels.append("current")
        scores: Dict[str, List[float]] = {}
        if aggregate is not None:
            scores["aggregate"] = [float(value) for value in aggregate.detach().cpu().tolist()]
        for name, values in components.items():
            scores[name] = [float(value) for value in values.detach().cpu().tolist()]
        return SelectionDecision(
            layer=layer,
            universe_positions=[int(value) for value in position_map],
            selected_rows=list(selection.rows),
            selected_positions=selected_positions,
            requested_budget=(
                seq_len if self.name == "full" else int(self.budget_cfg.cache_budget)
            ),
            effective_budget=len(selection.rows),
            mandatory_positions=mandatory_positions,
            selectable_budget=int(selection.selectable_budget),
            budget_scope=self.budget_cfg.scope,
            budget_unit=self.budget_cfg.unit,
            selected_sources=source_by_position,
            scores=scores,
            metadata={
                "method": self.name,
                "family": self.spec.family,
                "fidelity": self.spec.fidelity,
                "fidelity_notes": self.spec.notes,
                **diagnostics,
            },
        )
