"""Phase-aligned complementarity and representation diagnostics."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Set

import numpy as np
import torch
from scipy.stats import kendalltau, rankdata, spearmanr

from kvbench.cache.budget import stable_topk
from kvbench.config import DiagnosticsConfig
from kvbench.types import CacheSnapshot, ScoreBundle, SelectionDecision


def find_subsequence_positions(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    if not pattern or len(pattern) > len(sequence):
        return []
    positions: List[int] = []
    width = len(pattern)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start : start + width]) == list(pattern):
            positions.extend(range(start, start + width))
    return sorted(set(positions))


def evidence_token_positions(backend, prompt_ids: List[int], evidence_texts: Iterable[str]) -> List[int]:
    positions: Set[int] = set()
    for text in evidence_texts:
        token_ids = backend.encode_text(str(text), add_special_tokens=False)
        positions.update(find_subsequence_positions(prompt_ids, token_ids))
    return sorted(positions)


def _safe_correlation(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    a = left.detach().float().cpu().numpy()
    b = right.detach().float().cpu().numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return {"spearman": None, "kendall_tau_b": None, "n": int(a.size)}
    return {
        "spearman": float(spearmanr(a, b).statistic),
        "kendall_tau_b": float(kendalltau(a, b, variant="b").statistic),
        "n": int(a.size),
    }


def _gini(values: torch.Tensor) -> float:
    x = values.detach().float().flatten().clamp_min(0).cpu()
    if x.numel() == 0 or float(x.sum().item()) == 0.0:
        return 0.0
    x = torch.sort(x).values
    n = int(x.numel())
    index = torch.arange(1, n + 1, dtype=x.dtype)
    return float(((2 * index - n - 1) * x).sum().item() / (n * x.sum().item()))


def _attention_stats(score: torch.Tensor) -> Dict[str, float]:
    values = score.detach().float().clamp_min(0)
    total = values.sum()
    if values.numel() == 0 or float(total.item()) <= 0:
        return {"entropy": 0.0, "gini": 0.0, "effective_support": 0.0, "topk_mass": 0.0}
    probability = values / total
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    effective = torch.exp(entropy)
    k = min(32, int(values.numel()))
    topk_mass = torch.topk(probability, k).values.sum()
    return {
        "entropy": float(entropy.item()),
        "gini": _gini(values),
        "effective_support": float(effective.item()),
        "topk_mass": float(topk_mass.item()),
    }


def _quadrants(
    attention: torch.Tensor,
    leverage: torch.Tensor,
    evidence_rows: Set[int],
    selected_rows: Set[int],
    quantile: float = 0.8,
) -> Dict[str, Any]:
    att_threshold = torch.quantile(attention.float(), quantile)
    lev_threshold = torch.quantile(leverage.float(), quantile)
    labels = {
        "high_attention_high_leverage": [],
        "high_attention_low_leverage": [],
        "low_attention_high_leverage": [],
        "low_attention_low_leverage": [],
    }
    for row in range(int(attention.numel())):
        high_a = bool(attention[row] >= att_threshold)
        high_l = bool(leverage[row] >= lev_threshold)
        if high_a and high_l:
            key = "high_attention_high_leverage"
        elif high_a:
            key = "high_attention_low_leverage"
        elif high_l:
            key = "low_attention_high_leverage"
        else:
            key = "low_attention_low_leverage"
        labels[key].append(row)
    return {
        key: {
            "token_count": len(rows),
            "evidence_count": len(set(rows) & evidence_rows),
            "selected_count": len(set(rows) & selected_rows),
            "evidence_fraction": (
                len(set(rows) & evidence_rows) / len(rows) if rows else 0.0
            ),
            "selection_rate": (
                len(set(rows) & selected_rows) / len(rows) if rows else 0.0
            ),
        }
        for key, rows in labels.items()
    }


def _quadrants_topk(
    attention: torch.Tensor,
    leverage: torch.Tensor,
    evidence_rows: Set[int],
    selected_rows: Set[int],
    k: int,
) -> Dict[str, Any]:
    candidates = torch.arange(attention.numel(), dtype=torch.long)
    count = min(max(0, int(k)), int(candidates.numel()))
    high_attention = set(stable_topk(attention, count, candidates).tolist())
    high_leverage = set(stable_topk(leverage, count, candidates).tolist())
    labels = {
        "high_attention_high_leverage": high_attention & high_leverage,
        "high_attention_low_leverage": high_attention - high_leverage,
        "low_attention_high_leverage": high_leverage - high_attention,
        "low_attention_low_leverage": set(candidates.tolist())
        - high_attention
        - high_leverage,
    }
    return {
        name: {
            "token_count": len(rows),
            "evidence_count": len(rows & evidence_rows),
            "selected_count": len(rows & selected_rows),
            "evidence_fraction": len(rows & evidence_rows) / len(rows) if rows else 0.0,
            "selection_rate": len(rows & selected_rows) / len(rows) if rows else 0.0,
        }
        for name, rows in labels.items()
    }


def _reconstruction_error(value: torch.Tensor, selected_rows: List[int]) -> Dict[str, float]:
    if not selected_rows:
        return {"relative_frobenius_error": 1.0, "effective_rank_preserved": 0.0}
    rows_by_head = value[0].float()
    selected = torch.tensor(selected_rows, device=value.device, dtype=torch.long)
    errors, rank_ratios = [], []
    for rows in rows_by_head:
        chosen = rows.index_select(0, selected)
        _, singular, vh = torch.linalg.svd(chosen, full_matrices=False)
        largest = float(singular.max().item()) if singular.numel() else 0.0
        tolerance = max(chosen.shape) * torch.finfo(rows.dtype).eps * largest
        keep = singular > tolerance
        if keep.any():
            basis = vh[keep]
            reconstructed = rows @ basis.T @ basis
            selected_rank = int(keep.sum().item())
        else:
            reconstructed = torch.zeros_like(rows)
            selected_rank = 0
        error = torch.linalg.vector_norm(rows - reconstructed) / torch.linalg.vector_norm(rows).clamp_min(1e-12)
        full_rank = int(torch.linalg.matrix_rank(rows).item())
        errors.append(float(error.item()))
        rank_ratios.append(selected_rank / max(1, full_rank))
    return {
        "relative_frobenius_error": float(np.mean(errors)),
        "effective_rank_preserved": float(np.mean(rank_ratios)),
    }


def compute_decision_diagnostics(
    snapshot: CacheSnapshot,
    decisions: List[SelectionDecision],
    bundle: ScoreBundle,
    evidence_positions: List[int],
    cfg: DiagnosticsConfig,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "snapshot_id": snapshot.snapshot_id,
        "phase": snapshot.phase,
        "decode_step": snapshot.decode_step,
        "evidence_positions": list(evidence_positions),
        "layers": {},
    }
    evidence_set = set(int(value) for value in evidence_positions)
    layer_recalls: List[float] = []
    for decision in decisions:
        layer = int(decision.layer)
        universe = list(decision.universe_positions)
        row_by_position = {position: row for row, position in enumerate(universe)}
        evidence_rows = {
            row_by_position[position]
            for position in evidence_set
            if position in row_by_position
        }
        selected = set(decision.selected_positions)
        retained = evidence_set & selected
        source_counts: Dict[str, int] = {}
        for sources in decision.selected_sources.values():
            for source in set(sources):
                source_counts[source] = source_counts.get(source, 0) + 1
        layer_result: Dict[str, Any] = {
            "actual_retained_count": decision.effective_budget,
            "requested_budget": decision.requested_budget,
            "mandatory_count": len(decision.mandatory_positions),
            "selectable_budget": decision.selectable_budget,
            "evidence_total": len(evidence_set),
            "evidence_retained": len(retained),
            "evidence_recall": len(retained) / len(evidence_set) if evidence_set else None,
            "any_evidence_recall": bool(retained) if evidence_set else None,
            "complete_evidence_recall": retained == evidence_set if evidence_set else None,
            "selection_precision_evidence": (
                len(retained) / len(selected) if selected and evidence_set else None
            ),
            "selection_source_counts": source_counts,
        }
        if evidence_set:
            layer_recalls.append(float(layer_result["evidence_recall"]))

        components = {
            name: values.get(layer)
            for name, values in bundle.components.items()
            if values.get(layer) is not None
        }
        attention = components.get("attention")
        leverage = components.get("v_leverage")
        if attention is not None and cfg.attention_statistics:
            layer_result["attention"] = _attention_stats(attention)
        if attention is not None and leverage is not None:
            mandatory_rows = {
                row_by_position[position]
                for position in decision.mandatory_positions
                if position in row_by_position
            }
            eligible = torch.tensor(
                [row for row in range(len(universe)) if row not in mandatory_rows],
                dtype=torch.long,
            )
            k = min(decision.selectable_budget, int(eligible.numel()))
            att_top = set(stable_topk(attention, k, eligible).tolist())
            lev_top = set(stable_topk(leverage, k, eligible).tolist())
            intersection = att_top & lev_top
            union = att_top | lev_top
            if cfg.overlap:
                layer_result["overlap"] = {
                    "overlap_at_k": len(intersection) / max(1, k),
                    "jaccard": len(intersection) / max(1, len(union)),
                    "k": k,
                }
            if cfg.rank_correlation and eligible.numel() > 0:
                layer_result["rank_correlation"] = _safe_correlation(
                    attention.index_select(0, eligible.to(attention.device)),
                    leverage.index_select(0, eligible.to(leverage.device)),
                )
            attention_heads = bundle.components_by_head.get("attention", {}).get(layer)
            leverage_heads = bundle.components_by_head.get("v_leverage", {}).get(layer)
            if (
                attention_heads is not None
                and leverage_heads is not None
                and attention_heads.shape == leverage_heads.shape
            ):
                per_head = []
                for head in range(int(attention_heads.shape[0])):
                    head_attention = attention_heads[head]
                    head_leverage = leverage_heads[head]
                    head_att_top = set(
                        stable_topk(head_attention, k, eligible).tolist()
                    )
                    head_lev_top = set(
                        stable_topk(head_leverage, k, eligible).tolist()
                    )
                    head_union = head_att_top | head_lev_top
                    head_intersection = head_att_top & head_lev_top
                    per_head.append(
                        {
                            "head": head,
                            "overlap_at_k": len(head_intersection) / max(1, k),
                            "jaccard": len(head_intersection)
                            / max(1, len(head_union)),
                            **_safe_correlation(
                                head_attention.index_select(0, eligible),
                                head_leverage.index_select(0, eligible),
                            ),
                        }
                    )
                layer_result["per_head_complementarity"] = per_head
            if cfg.quadrants:
                selected_rows = set(decision.selected_rows)
                layer_result["quadrants"] = {
                    "top5pct": _quadrants(
                        attention, leverage, evidence_rows, selected_rows, 0.95
                    ),
                    "top10pct": _quadrants(
                        attention, leverage, evidence_rows, selected_rows, 0.90
                    ),
                    "top20pct": _quadrants(
                        attention, leverage, evidence_rows, selected_rows, 0.80
                    ),
                    "median": _quadrants(
                        attention, leverage, evidence_rows, selected_rows, 0.50
                    ),
                    "budget_aligned": _quadrants_topk(
                        attention,
                        leverage,
                        evidence_rows,
                        selected_rows,
                        decision.selectable_budget,
                    ),
                }
        if cfg.reconstruction:
            layer_result["reconstruction"] = _reconstruction_error(
                snapshot.values[layer], decision.selected_rows
            )
        result["layers"][str(layer)] = layer_result
    result["mean_layer_evidence_recall"] = (
        float(np.mean(layer_recalls)) if layer_recalls else None
    )
    result["any_layer_complete_evidence"] = (
        any(
            value.get("complete_evidence_recall") is True
            for value in result["layers"].values()
        )
        if evidence_set
        else None
    )
    return result
