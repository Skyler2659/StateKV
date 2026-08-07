"""Gauge-aware full-vocabulary output geometry collection.

The collector deliberately does not persist full vocabulary logits.  It computes
all registered G0--G7 statistics while the full and compressed logits are in
memory, and stores the layer-27 drift/direct vectors needed by a gated Stage B.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.functional_probe import _condition_cache
from statekv.output_sensitivity import OutputSensitivityRunner
from statekv.runner import _sample_slug
from statekv.selectors import CoreSelection, LayerSelection
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame


GAUGE_RAW_TABLES = ("oracle_geometry_rows", "gauge_vector_index")


def stable_logsumexp(values: np.ndarray) -> float:
    """Stable scalar log-sum-exp in float64."""

    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(array))
    return maximum + float(np.log(np.exp(array - maximum).sum()))


def exact_kl_cumulant_identity(
    full_logits: np.ndarray, delta_logits: np.ndarray
) -> float:
    """KL(p||softmax(z+delta)) as log E_p exp(delta) - E_p delta."""

    z = np.asarray(full_logits, dtype=np.float64)
    delta = np.asarray(delta_logits, dtype=np.float64)
    log_p = z - stable_logsumexp(z)
    p = np.exp(log_p)
    return float(
        stable_logsumexp(log_p + delta) - float(np.dot(p, delta))
    )


def fisher_variance(probability: np.ndarray, direction: np.ndarray) -> float:
    """Compute v^T(Diag(p)-pp^T)v without forming the Fisher matrix."""

    p = np.asarray(probability, dtype=np.float64)
    value = np.asarray(direction, dtype=np.float64)
    mean = float(np.dot(p, value))
    return max(float(np.dot(p, value * value) - mean * mean), 0.0)


def fisher_pairwise_gap(probability: np.ndarray, direction: np.ndarray) -> float:
    """Pairwise-gap identity evaluated in O(V), not O(V^2)."""

    p = np.asarray(probability, dtype=np.float64)
    value = np.asarray(direction, dtype=np.float64)
    first = float(np.dot(p, value * value))
    mean = float(np.dot(p, value))
    # 1/2 sum_ij p_i p_j (v_i-v_j)^2.
    return max(first - mean * mean, 0.0)


def center_uniform(direction: np.ndarray) -> np.ndarray:
    value = np.asarray(direction, dtype=np.float64)
    return value - float(value.mean())


def gauss_legendre_path_fisher(
    full_logits: np.ndarray, delta_logits: np.ndarray, order: int
) -> float:
    """Integrate (1-s) Var_{softmax(z+s delta)}(delta) on [0, 1]."""

    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    result = 0.0
    for node, weight in zip(nodes, weights):
        s = 0.5 * (float(node) + 1.0)
        logits = full_logits + s * delta_logits
        log_p = logits - stable_logsumexp(logits)
        variance = fisher_variance(np.exp(log_p), delta_logits)
        result += 0.5 * float(weight) * (1.0 - s) * variance
    return float(result)


def simpson_path_fisher(
    full_logits: np.ndarray, delta_logits: np.ndarray, points: int = 9
) -> float:
    if int(points) < 3 or int(points) % 2 != 1:
        raise ValueError("Simpson path reference requires an odd point count")
    grid = np.linspace(0.0, 1.0, int(points), dtype=np.float64)
    values = []
    for s in grid:
        logits = full_logits + float(s) * delta_logits
        log_p = logits - stable_logsumexp(logits)
        values.append(
            (1.0 - float(s))
            * fisher_variance(np.exp(log_p), delta_logits)
        )
    weights = np.ones(int(points), dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    step = 1.0 / float(int(points) - 1)
    return float(step / 3.0 * np.dot(weights, np.asarray(values)))


def centered_cumulants(
    probability: np.ndarray, direction: np.ndarray
) -> Tuple[float, float, float]:
    p = np.asarray(probability, dtype=np.float64)
    value = np.asarray(direction, dtype=np.float64)
    centered = value - float(np.dot(p, value))
    second = float(np.dot(p, centered**2))
    third = float(np.dot(p, centered**3))
    fourth_cumulant = float(np.dot(p, centered**4) - 3.0 * second * second)
    return second, third, fourth_cumulant


def topk_geometry(
    probability: np.ndarray,
    full_logits: np.ndarray,
    delta_logits: np.ndarray,
    ordered_top_indices: np.ndarray,
    k: int,
    epsilon: float = 1.0e-12,
) -> Dict[str, float]:
    indices = np.asarray(ordered_top_indices[: int(k)], dtype=np.int64)
    p_top = probability[indices]
    delta_top = delta_logits[indices]
    mass = float(p_top.sum())
    normalized = p_top / max(mass, epsilon)
    mean_top = float(np.dot(normalized, delta_top))
    variance_top = max(
        float(np.dot(normalized, delta_top * delta_top) - mean_top**2),
        0.0,
    )
    total_mean = float(np.dot(probability, delta_logits))
    tail_mass = max(1.0 - mass, epsilon)
    tail_mean = float(
        (total_mean - float(np.dot(p_top, delta_top))) / tail_mass
    )
    g5a = 0.5 * mass * variance_top
    g5b = 0.5 * mass * mass * variance_top
    cross = 0.5 * mass * max(1.0 - mass, 0.0) * float(
        np.dot(normalized, (delta_top - tail_mean) ** 2)
    )
    return {
        "mass": mass,
        "g5a": float(g5a),
        "g5b": float(g5b),
        "g5c": float(g5b + cross),
    }


def top_margin_geometry(
    probability: np.ndarray,
    full_logits: np.ndarray,
    delta_logits: np.ndarray,
    ordered_top_indices: np.ndarray,
    k: int,
    epsilon: float = 1.0e-9,
) -> Dict[str, float]:
    indices = np.asarray(ordered_top_indices[: int(k)], dtype=np.int64)
    top = int(indices[0])
    competitors = indices[1:]
    delta_margin = delta_logits[top] - delta_logits[competitors]
    original_margin = np.maximum(
        full_logits[top] - full_logits[competitors], epsilon
    )
    weight_map = {
        "p": probability[competitors],
        "p_pair": probability[top] * probability[competitors],
        "uniform": np.full(
            len(competitors), 1.0 / max(len(competitors), 1)
        ),
        "inverse_margin": probability[competitors] / original_margin,
    }
    result: Dict[str, float] = {}
    collapse = np.maximum(-delta_margin, 0.0)
    for name, weight in weight_map.items():
        result["g6_two_%s" % name] = float(
            np.dot(weight, delta_margin**2)
        )
        result["g6_collapse_%s" % name] = float(
            np.dot(weight, collapse**2)
        )
    return result


def gauge_geometry_metrics(
    full_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    topk_values: Sequence[int],
    range_quantiles: Sequence[float],
    dense_path_points: int = 9,
    near_null_probability: float = 1.0e-6,
) -> Dict[str, Any]:
    """Compute every registered Stage-A geometry from full vocabulary logits."""

    z = full_logits.detach().float().cpu().numpy().astype(np.float64)
    z_compressed = (
        compressed_logits.detach().float().cpu().numpy().astype(np.float64)
    )
    delta = z_compressed - z
    log_z = stable_logsumexp(z)
    log_p = z - log_z
    probability = np.exp(log_p)
    mean_p = float(np.dot(probability, delta))
    exact_kl = exact_kl_cumulant_identity(z, delta)
    exact_kl_partition = float(
        stable_logsumexp(z_compressed) - log_z - mean_p
    )
    raw_l2_sq = float(np.dot(delta, delta))
    uniform_mean = float(delta.mean())
    shift_energy = float(delta.size * uniform_mean * uniform_mean)
    centered = delta - uniform_mean
    centered_l2_sq = max(float(np.dot(centered, centered)), 0.0)
    variance = fisher_variance(probability, delta)
    pairwise_variance = fisher_pairwise_gap(probability, delta)

    midpoint_logits = z + 0.5 * delta
    midpoint_probability = np.exp(
        midpoint_logits - stable_logsumexp(midpoint_logits)
    )
    midpoint_variance = fisher_variance(midpoint_probability, delta)

    maximum_k = min(max(int(value) for value in topk_values), int(z.size))
    unordered = np.argpartition(-z, maximum_k - 1)[:maximum_k]
    ordered_top = unordered[np.argsort(-z[unordered], kind="stable")]

    result: Dict[str, Any] = {
        "vocab_size": int(z.size),
        "exact_kl": exact_kl,
        "exact_kl_partition": exact_kl_partition,
        "kl_cumulant_identity_abs_error": abs(
            exact_kl - exact_kl_partition
        ),
        "g0_raw_l2_sq": raw_l2_sq,
        "g0_global_bound": 0.25 * raw_l2_sq,
        "g1_centered_l2_sq": centered_l2_sq,
        "g1_centered_global_bound": 0.25 * centered_l2_sq,
        "g1p_probability_variance": variance,
        "g2_base_fisher": 0.5 * variance,
        "g3_midpoint_fisher": 0.5 * midpoint_variance,
        "fisher_variance_identity_abs_error": abs(
            variance - pairwise_variance
        ),
        "uniform_shift_mean": uniform_mean,
        "common_shift_energy": shift_energy,
        "common_shift_energy_fraction": shift_energy
        / max(raw_l2_sq, 1.0e-30),
        "output_entropy": float(-np.dot(probability, log_p)),
        "top1_margin": float(z[ordered_top[0]] - z[ordered_top[1]]),
        "fisher_effective_rank": float(
            1.0 / max(float(np.dot(probability, probability)), 1.0e-30)
        ),
        "g4_gl2": gauss_legendre_path_fisher(z, delta, 2),
        "g4_gl3": gauss_legendre_path_fisher(z, delta, 3),
        "g4_gl5": gauss_legendre_path_fisher(z, delta, 5),
        "g4_simpson9": simpson_path_fisher(
            z, delta, int(dense_path_points)
        ),
        "full_logits_stored": False,
        "compressed_logits_stored": False,
        "full_vocabulary_streamed": True,
        "stable_logsumexp": True,
    }

    delta_range = float(np.max(delta) - np.min(delta))
    log_range_bound = (
        math.log(max(0.5 * variance, 1.0e-300)) + delta_range
    )
    result.update(
        {
            "logit_oscillation": delta_range,
            "g4b_range_log_bound": log_range_bound,
            "g4b_range_bound": float(
                math.exp(min(log_range_bound, 700.0))
            ),
            "g4b_range_overflow": bool(log_range_bound > 700.0),
            "g4b_range_covered": bool(
                math.log(max(exact_kl, 1.0e-300))
                <= log_range_bound + 1.0e-12
            ),
        }
    )
    for tau in range_quantiles:
        lower, upper = np.quantile(
            delta, [float(tau), 1.0 - float(tau)], method="linear"
        )
        truncated = float(upper - lower)
        key = ("%.0e" % float(tau)).replace("-", "m")
        log_bound = (
            math.log(max(0.5 * variance, 1.0e-300)) + truncated
        )
        result["range_tau_%s" % key] = truncated
        result["g4b_truncated_%s" % key] = float(
            math.exp(min(log_bound, 700.0))
        )

    for k in topk_values:
        actual_k = min(int(k), int(z.size))
        top_result = topk_geometry(
            probability, z, delta, ordered_top, actual_k
        )
        result["topk_mass_%d" % actual_k] = top_result["mass"]
        for family in ("g5a", "g5b", "g5c"):
            result["%s_k%d" % (family, actual_k)] = top_result[family]
        margin = top_margin_geometry(
            probability, z, delta, ordered_top, actual_k
        )
        for name, value in margin.items():
            result["%s_k%d" % (name, actual_k)] = value

    second, third, fourth = centered_cumulants(probability, delta)
    result.update(
        {
            "cumulant_2": second,
            "cumulant_3": third,
            "cumulant_4": fourth,
            "g7_order2": 0.5 * second,
            "g7_order3": 0.5 * second + third / 6.0,
            "g7_order4": (
                0.5 * second + third / 6.0 + fourth / 24.0
            ),
        }
    )
    result["g7_order3_negative"] = bool(result["g7_order3"] < 0.0)
    result["g7_order4_negative"] = bool(result["g7_order4"] < 0.0)

    top256 = ordered_top[: min(256, len(ordered_top))]
    top_mask = np.zeros(delta.size, dtype=bool)
    top_mask[top256] = True
    near_null = probability < float(near_null_probability)
    result.update(
        {
            "top256_centered_energy_fraction": float(
                np.dot(centered[top_mask], centered[top_mask])
                / max(centered_l2_sq, 1.0e-30)
            ),
            "tail_centered_energy_fraction": float(
                np.dot(centered[~top_mask], centered[~top_mask])
                / max(centered_l2_sq, 1.0e-30)
            ),
            "fisher_near_null_euclidean_fraction": float(
                np.dot(centered[near_null], centered[near_null])
                / max(centered_l2_sq, 1.0e-30)
            ),
            "fisher_near_null_vocab_fraction": float(near_null.mean()),
        }
    )
    return result


class GaugeGeometryRunner(OutputSensitivityRunner):
    """Replay inherited candidates and collect Stage-A geometry statistics."""

    @property
    def output_cfg(self) -> Any:
        return self.cfg.gauge_geometry

    @property
    def source_run(self) -> Path:
        return self.store.run_dir.parent / str(self.output_cfg.source_run_id)

    def _fragment(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "gauge_geometry"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _vector_fragment(self, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "gauge_geometry"
            / "vectors"
            / ("%s.npz" % _sample_slug(sample_id))
        )

    def _source_inventory(self) -> pd.DataFrame:
        path = self.source_run / "output_candidate_inventory.parquet"
        if not path.exists():
            raise FileNotFoundError("missing inherited inventory: %s" % path)
        return pd.read_parquet(path)

    def _source_metrics(self) -> pd.DataFrame:
        path = self.source_run / "output_candidate_rows.parquet"
        if not path.exists():
            raise FileNotFoundError("missing inherited candidate rows: %s" % path)
        rows = pd.read_parquet(path)
        rows = rows[
            (rows["trajectory_kind"] == "candidate")
            & (rows["layer"] == int(self.model.selected_layers[0]))
        ].copy()
        return rows[
            [
                "sample_id",
                "candidate_id",
                "anchor",
                "horizon_offset",
                "exact_kl",
                "logit_l2_sq",
            ]
        ].rename(
            columns={
                "exact_kl": "source_exact_kl",
                "logit_l2_sq": "source_logit_l2_sq",
            }
        )

    @staticmethod
    def _selection_from_inventory(row: Mapping[str, Any]) -> CoreSelection:
        masks = json.loads(str(row["selected_positions_json"]))
        by_layer: Dict[int, LayerSelection] = {}
        for layer_text, positions in masks.items():
            layer = int(layer_text)
            selected = [int(value) for value in positions]
            by_layer[layer] = LayerSelection(
                layer=layer,
                selected_positions=selected,
                eligible_positions=[],
                aggregate_scores=[0.0] * len(selected),
                metadata={"inherited_physical_mask": True},
            )
        return CoreSelection(
            strategy=str(row["candidate_source"]),
            horizon_condition=None,
            by_layer=by_layer,
            metadata={
                "source_run_inherited": True,
                "mask_hash": str(row["mask_hash"]),
            },
        )

    def _replay_geometry_candidate(
        self,
        sample: Any,
        reference: Any,
        inventory_row: Mapping[str, Any],
        source_lookup: Mapping[Tuple[str, str, int, int], Tuple[float, float]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
        anchor = int(inventory_row["anchor"])
        selection = self._selection_from_inventory(inventory_row)
        cache_cfg = _condition_cache(
            self.cfg,
            int(self.output_cfg.total_budget),
            int(self.output_cfg.protected_recent),
        )
        state, fixed = self.model.state_from_anchor(
            reference.anchors[anchor],
            selection,
            cache_config=cache_cfg,
        )
        full_values = self._initial_values(reference, anchor)
        current_token = int(reference.anchors[anchor].query_token_id)
        rows: List[Dict[str, Any]] = []
        vector_index: List[Dict[str, Any]] = []
        actual_vectors: List[np.ndarray] = []
        direct_vectors: List[np.ndarray] = []
        full_vectors: List[np.ndarray] = []
        candidate_id = str(inventory_row["candidate_id"])
        try:
            for offset in range(1, int(self.output_cfg.segment_horizon) + 1):
                target_index = int(anchor + offset - 1)
                reference_record = reference.query_records[target_index]
                if offset > 1:
                    self.model.prune_recent_before_query(
                        state, fixed, cache_config=cache_cfg
                    )
                    self._append_reference_value(full_values, reference_record)
                current_position = int(reference_record.query_position)
                retained = set(
                    int(value)
                    for value in state.position_maps[27].tolist()
                )
                retained.add(current_position)
                direct = self._direct_at_step(
                    reference_record,
                    27,
                    full_values[27],
                    sorted(retained),
                )
                self._clear_controls()
                logits, record, forward_s = self.model.forward_one(
                    state, current_token, capture_attention=True
                )
                self.model.validate_active_budget(
                    state, cache_config=cache_cfg
                )
                full_logits = reference.probe_logits[target_index]
                geometry = gauge_geometry_metrics(
                    full_logits,
                    logits,
                    self.output_cfg.topk_values,
                    self.output_cfg.truncated_range_quantiles,
                    dense_path_points=int(
                        self.output_cfg.dense_path_points
                    ),
                    near_null_probability=float(
                        self.output_cfg.topk_tail_probability
                    ),
                )
                source_key = (
                    str(sample.sample_id),
                    candidate_id,
                    anchor,
                    offset,
                )
                source_exact, source_l2 = source_lookup[source_key]
                row_id = len(actual_vectors)
                full_h = reference_record.residual_inputs[27].float()
                actual_delta = record.residual_inputs[27].float() - full_h
                actual_vectors.append(
                    actual_delta.detach().cpu().numpy().astype(np.float16)
                )
                direct_vectors.append(
                    direct["projected"].detach().cpu().numpy().astype(np.float16)
                )
                full_vectors.append(
                    full_h.detach().cpu().numpy().astype(np.float16)
                )
                row = {
                    **self._base(sample),
                    "candidate_id": candidate_id,
                    "candidate_index": int(inventory_row["candidate_index"]),
                    "candidate_source": str(
                        inventory_row["candidate_source"]
                    ),
                    "anchor": anchor,
                    "horizon_offset": offset,
                    "target_index": target_index,
                    "mask_hash": str(inventory_row["mask_hash"]),
                    "total_budget": int(self.output_cfg.total_budget),
                    "active_cache_tokens": int(
                        self.model.active_cache_tokens(state)
                    ),
                    "token_position_aligned": bool(
                        int(record.query_position)
                        == int(reference_record.query_position)
                    ),
                    "uses_future_compressed_truth": False,
                    "task_feature_used": False,
                    "forward_time_s": float(forward_s),
                    "source_exact_kl": float(source_exact),
                    "source_logit_l2_sq": float(source_l2),
                    "source_exact_kl_abs_error": abs(
                        float(geometry["exact_kl"]) - float(source_exact)
                    ),
                    "source_logit_l2_sq_abs_error": abs(
                        float(geometry["g0_raw_l2_sq"]) - float(source_l2)
                    ),
                    "layer27_actual_residual_norm": float(
                        actual_delta.norm().item()
                    ),
                    "layer27_direct_norm": float(direct["coordinate"]),
                    "layer27_direct_deleted_attention_mass": float(
                        direct["deleted_attention_mass"]
                    ),
                    **geometry,
                }
                rows.append(row)
                vector_index.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "candidate_id": candidate_id,
                        "anchor": anchor,
                        "horizon_offset": offset,
                        "target_index": target_index,
                        "vector_row": row_id,
                        "vector_fragment": str(
                            self._vector_fragment(sample.sample_id)
                        ),
                        "storage_dtype": "float16",
                        "hidden_dimension": int(actual_delta.numel()),
                        "actual_residual_vector_saved": True,
                        "direct_projected_vector_saved": True,
                        "full_residual_vector_saved": True,
                    }
                )
                current_token = int(
                    reference.generated_token_ids[target_index]
                )
        finally:
            self._clear_controls()
            self.model.release(state)
        return (
            rows,
            vector_index,
            {
                "actual_residual_delta": np.stack(actual_vectors),
                "direct_projected_l27": np.stack(direct_vectors),
                "full_residual_l27": np.stack(full_vectors),
            },
        )

    def _completed_fragment_valid(self, sample_id: str) -> bool:
        rows = self._fragment("oracle_geometry_rows", sample_id)
        index = self._fragment("gauge_vector_index", sample_id)
        vectors = self._vector_fragment(sample_id)
        if not rows.exists() or not index.exists() or not vectors.exists():
            return False
        try:
            frame = pd.read_parquet(rows)
            vector_frame = pd.read_parquet(index)
        except Exception:
            return False
        expected = (
            len(self.output_cfg.anchors)
            * int(self.output_cfg.candidate_count)
            * int(self.output_cfg.segment_horizon)
        )
        return bool(len(frame) == expected and len(vector_frame) == expected)

    @staticmethod
    def _write_vector_fragment(
        path: Path, arrays: Dict[str, np.ndarray]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)

    def _run_gauge_sample(
        self,
        sample: Any,
        inventory: pd.DataFrame,
        source_lookup: Mapping[Tuple[str, str, int, int], Tuple[float, float]],
    ) -> None:
        key = "gauge_geometry:%s" % _sample_slug(sample.sample_id)
        if (
            self.cfg.runtime.resume
            and self.store.is_complete(key)
            and self._completed_fragment_valid(sample.sample_id)
        ):
            return
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            sample_inventory = inventory[
                (
                    inventory["sample_id"].astype(str)
                    == str(sample.sample_id)
                )
                & inventory["anchor"].isin(
                    [int(value) for value in self.output_cfg.anchors]
                )
            ].sort_values(["anchor", "candidate_index"])
            expected = (
                len(self.output_cfg.anchors)
                * int(self.output_cfg.candidate_count)
            )
            if len(sample_inventory) != expected:
                raise RuntimeError(
                    "inherited inventory count mismatch for %s: %d != %d"
                    % (sample.sample_id, len(sample_inventory), expected)
                )
            all_rows: List[Dict[str, Any]] = []
            all_index: List[Dict[str, Any]] = []
            vector_blocks: Dict[str, List[np.ndarray]] = {
                "actual_residual_delta": [],
                "direct_projected_l27": [],
                "full_residual_l27": [],
            }
            base_row = 0
            for row in sample_inventory.to_dict("records"):
                rows, vector_index, arrays = self._replay_geometry_candidate(
                    sample, reference, row, source_lookup
                )
                for item in vector_index:
                    item["vector_row"] = int(item["vector_row"]) + base_row
                base_row += len(rows)
                all_rows.extend(rows)
                all_index.extend(vector_index)
                for name, value in arrays.items():
                    vector_blocks[name].append(value)
            combined = {
                name: np.concatenate(blocks, axis=0)
                for name, blocks in vector_blocks.items()
            }
            _atomic_frame(
                pd.DataFrame(all_rows),
                self._fragment("oracle_geometry_rows", sample.sample_id),
            )
            _atomic_frame(
                pd.DataFrame(all_index),
                self._fragment("gauge_vector_index", sample.sample_id),
            )
            self._write_vector_fragment(
                self._vector_fragment(sample.sample_id), combined
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(time.perf_counter() - started),
                    "oracle_geometry_rows": len(all_rows),
                    "vector_rows": len(all_index),
                    "candidate_decode_forwards": len(all_rows),
                    "full_vocabulary_metrics_streamed": True,
                },
            )
            self.model.release(reference)
        except Exception as exc:
            self.store.append_error(
                {
                    "key": key,
                    "sample_id": sample.sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            self.store.mark_failed(
                key, "%s: %s" % (type(exc).__name__, exc)
            )
            if self.cfg.runtime.fail_on_error:
                raise

    def _consolidate_gauge(self) -> None:
        for table in GAUGE_RAW_TABLES:
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "gauge_geometry"
                    / table
                ).glob("*.parquet")
            )
            frames = [pd.read_parquet(path) for path in paths]
            frame = (
                pd.concat(frames, ignore_index=True, sort=False)
                if frames
                else pd.DataFrame()
            )
            _atomic_frame(frame, self.store.run_dir / ("%s.parquet" % table))

    def run(self) -> Path:
        if not self.cfg.gauge_geometry.enabled:
            raise ValueError("gauge_geometry.enabled must be true")
        self.store.status["state"] = "running"
        self.store.status["protocol"] = "gauge_aware_output_geometry_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        inventory = self._source_inventory()
        source = self._source_metrics()
        source_lookup = {
            (
                str(row.sample_id),
                str(row.candidate_id),
                int(row.anchor),
                int(row.horizon_offset),
            ): (float(row.source_exact_kl), float(row.source_logit_l2_sq))
            for row in source.itertuples(index=False)
        }
        for table in GAUGE_RAW_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "gauge_geometry"
                / table
            ).mkdir(parents=True, exist_ok=True)
        (
            self.store.run_dir
            / "fragments"
            / "gauge_geometry"
            / "vectors"
        ).mkdir(parents=True, exist_ok=True)
        try:
            for sample in samples:
                self._run_gauge_sample(
                    sample, inventory, source_lookup
                )
            self._consolidate_gauge()
            self.store.status["state"] = "stage_a_data_complete"
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir
