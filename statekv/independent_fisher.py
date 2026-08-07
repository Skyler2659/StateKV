"""Independent midpoint-Fisher replication and adaptive curvature collector."""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.integrate import IntegrationWarning, quad

from statekv.functional_probe import _condition_cache
from statekv.gauge_geometry import (
    exact_kl_cumulant_identity,
    fisher_variance,
    stable_logsumexp,
)
from statekv.output_sensitivity import OutputSensitivityRunner
from statekv.runner import _sample_slug
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame


INDEPENDENT_TABLES = (
    "independent_fisher_geometry_rows",
    "adaptive_curvature_rows",
    "independent_candidate_inventory",
    "independent_vector_index",
    "new_sequence_manifest",
)


def _probability(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return np.exp(values - stable_logsumexp(values))


def _quadrature_from_curvature(
    curvature: Any, order: int
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    value = 0.0
    for node, weight in zip(nodes, weights):
        point = 0.5 * (float(node) + 1.0)
        value += (
            0.5
            * float(weight)
            * (1.0 - point)
            * float(curvature(point))
        )
    return float(value)


def adaptive_fisher_geometry(
    full_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    trust_top_k: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    subdivision_limit: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return frozen G0/G2/G3 statistics and adaptive path diagnostics."""

    z = full_logits.detach().float().cpu().numpy().astype(np.float64)
    z_comp = (
        compressed_logits.detach().float().cpu().numpy().astype(np.float64)
    )
    delta = z_comp - z
    p0 = _probability(z)
    exact_kl = exact_kl_cumulant_identity(z, delta)
    partition_kl = float(
        stable_logsumexp(z_comp)
        - stable_logsumexp(z)
        - float(np.dot(p0, delta))
    )
    variance0 = fisher_variance(p0, delta)
    midpoint_probability = _probability(z + 0.5 * delta)
    variance_midpoint = fisher_variance(midpoint_probability, delta)
    raw = float(np.dot(delta, delta))
    g2 = 0.5 * variance0
    g3 = 0.5 * variance_midpoint
    epsilon = 1.0e-12

    top_count = min(max(int(trust_top_k), 2), len(z))
    unordered = np.argpartition(-z, top_count - 1)[:top_count]
    ordered = unordered[np.argsort(-z[unordered], kind="stable")]
    base_top = int(ordered[0])
    second = int(ordered[1])
    initial_margin = float(z[base_top] - z[second])
    top_indices = ordered[: min(int(trust_top_k), len(ordered))]
    competitor = top_indices[top_indices != base_top]
    top_switch_effect = (
        float(
            np.max(np.abs(delta[base_top] - delta[competitor]))
        )
        if len(competitor)
        else 0.0
    )
    fisher_distance = float(np.sqrt(max(variance0, 0.0)))

    curvature_cache: Dict[float, float] = {}
    top_cache: Dict[float, int] = {}
    margin_cache: Dict[float, float] = {}
    base_margin_cache: Dict[float, float] = {}

    def curvature(point: float) -> float:
        key = float(point)
        if key not in curvature_cache:
            path_logits = z + key * delta
            probability = _probability(path_logits)
            curvature_cache[key] = fisher_variance(probability, delta)
            path_two = np.argpartition(-path_logits, 1)[:2]
            path_order = path_two[
                np.argsort(-path_logits[path_two], kind="stable")
            ]
            top_cache[key] = int(path_order[0])
            margin_cache[key] = float(
                path_logits[int(path_order[0])]
                - path_logits[int(path_order[1])]
            )
            competitor_max = max(
                float(np.max(path_logits[:base_top]))
                if base_top > 0
                else -np.inf,
                float(np.max(path_logits[base_top + 1 :]))
                if base_top + 1 < len(path_logits)
                else -np.inf,
            )
            base_margin_cache[key] = float(
                path_logits[base_top] - competitor_max
            )
        return curvature_cache[key]

    for point in (0.0, 0.25, 0.5, 0.75, 1.0):
        curvature(point)
    gl3 = _quadrature_from_curvature(curvature, 3)
    gl5 = _quadrature_from_curvature(curvature, 5)

    integration_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        adaptive_weighted, weighted_error = quad(
            lambda point: (1.0 - float(point)) * curvature(float(point)),
            0.0,
            1.0,
            epsrel=float(relative_tolerance),
            epsabs=float(absolute_tolerance),
            limit=int(subdivision_limit),
        )
        adaptive_unweighted, unweighted_error = quad(
            lambda point: curvature(float(point)),
            0.0,
            1.0,
            epsrel=float(relative_tolerance),
            epsabs=float(absolute_tolerance),
            limit=int(subdivision_limit),
        )
        integration_warnings = [str(item.message) for item in caught]

    points = np.asarray(sorted(curvature_cache), dtype=np.float64)
    curvature_values = np.asarray(
        [curvature_cache[float(point)] for point in points],
        dtype=np.float64,
    )
    peak_index = int(np.argmax(curvature_values))
    peak_value = float(curvature_values[peak_index])
    peak_location = float(points[peak_index])
    half = 0.5 * peak_value
    above = points[curvature_values >= half]
    half_width = float(above[-1] - above[0]) if len(above) else 0.0
    effective_width = float(
        adaptive_unweighted / max(peak_value, epsilon)
    )
    sampled_mean = float(np.mean(curvature_values))
    ordered_points = [float(value) for value in points]
    top_path = [top_cache[value] for value in ordered_points]
    switch_count = int(
        sum(
            int(left != right)
            for left, right in zip(top_path[:-1], top_path[1:])
        )
    )
    final_margin = float(base_margin_cache[1.0])
    midpoint_margin = float(base_margin_cache[0.5])
    exact_scale = max(abs(exact_kl), epsilon)

    geometry = {
        "vocab_size": int(z.size),
        "exact_kl": float(exact_kl),
        "exact_kl_partition": float(partition_kl),
        "kl_cumulant_identity_abs_error": float(
            abs(exact_kl - partition_kl)
        ),
        "g0_raw_l2_sq": raw,
        "g2_base_fisher": float(g2),
        "g3_midpoint_fisher": float(g3),
        "g0_symmetric_ratio": float(
            max(
                (raw + epsilon) / (exact_kl + epsilon),
                (exact_kl + epsilon) / (raw + epsilon),
            )
        ),
        "g2_symmetric_ratio": float(
            max(
                (g2 + epsilon) / (exact_kl + epsilon),
                (exact_kl + epsilon) / (g2 + epsilon),
            )
        ),
        "g3_symmetric_ratio": float(
            max(
                (g3 + epsilon) / (exact_kl + epsilon),
                (exact_kl + epsilon) / (g3 + epsilon),
            )
        ),
        "g2_relative_error": float(abs(g2 - exact_kl) / exact_scale),
        "g3_relative_error": float(abs(g3 - exact_kl) / exact_scale),
        "gl3_path_fisher": float(gl3),
        "gl5_path_fisher": float(gl5),
        "gl3_relative_error": float(abs(gl3 - exact_kl) / exact_scale),
        "gl5_relative_error": float(abs(gl5 - exact_kl) / exact_scale),
        "delta_logit_l2": float(np.sqrt(max(raw, 0.0))),
        "base_fisher_distance": fisher_distance,
        "initial_top1_margin": initial_margin,
        "trust_t0_base_fisher_distance": fisher_distance,
        "trust_t1_fisher_margin_ratio": float(
            fisher_distance / max(initial_margin, epsilon)
        ),
        "trust_t2_top_switch_margin_ratio": float(
            top_switch_effect / max(initial_margin, epsilon)
        ),
        "trust_t3_g2_g3_disagreement": float(
            abs(g3 - g2) / max(g3, epsilon)
        ),
    }
    curvature_row = {
        "exact_kl": float(exact_kl),
        "base_fisher_norm": float(g2),
        "midpoint_fisher_norm": float(g3),
        "adaptive_weighted_integral": float(adaptive_weighted),
        "adaptive_weighted_error_estimate": float(weighted_error),
        "adaptive_weighted_abs_error_vs_exact": float(
            abs(adaptive_weighted - exact_kl)
        ),
        "adaptive_weighted_relative_error_vs_exact": float(
            abs(adaptive_weighted - exact_kl) / exact_scale
        ),
        "adaptive_unweighted_curvature_integral": float(
            adaptive_unweighted
        ),
        "adaptive_unweighted_error_estimate": float(unweighted_error),
        "curvature_max": peak_value,
        "curvature_peak_location": peak_location,
        "curvature_half_max_width": half_width,
        "effective_curvature_width": effective_width,
        "curvature_concentration": float(
            peak_value / max(adaptive_unweighted, epsilon)
        ),
        "curvature_peak_ratio": float(
            peak_value / max(sampled_mean, epsilon)
        ),
        "adaptive_evaluation_count": int(len(points)),
        "adaptive_warning_count": int(len(integration_warnings)),
        "adaptive_warnings_json": json.dumps(integration_warnings),
        "top1_changed_along_path": bool(switch_count > 0),
        "top1_change_count": switch_count,
        "top1_top2_margin_crossed_zero": bool(
            midpoint_margin <= 0.0 or final_margin <= 0.0
        ),
        "initial_top1_margin": initial_margin,
        "midpoint_top1_margin": midpoint_margin,
        "final_top1_margin": final_margin,
        "g2_relative_error": float(abs(g2 - exact_kl) / exact_scale),
        "g3_relative_error": float(abs(g3 - exact_kl) / exact_scale),
        "gl3_relative_error": float(abs(gl3 - exact_kl) / exact_scale),
        "gl5_relative_error": float(abs(gl5 - exact_kl) / exact_scale),
    }
    return geometry, curvature_row


class IndependentFisherRunner(OutputSensitivityRunner):
    """Create a new physical pool and collect independent Stage-A′ evidence."""

    @property
    def output_cfg(self) -> Any:
        return self.cfg.independent_fisher

    def _fragment(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "independent_fisher"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _vector_fragment(self, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "independent_fisher"
            / "vectors"
            / ("%s.npz" % _sample_slug(sample_id))
        )

    def _prior_sample_ids(self) -> Sequence[str]:
        result = set()
        for run_id in self.output_cfg.prior_run_ids:
            run_dir = self.store.run_dir.parent / str(run_id)
            for name in (
                "oracle_geometry_rows.parquet",
                "output_candidate_inventory.parquet",
            ):
                path = run_dir / name
                if path.exists():
                    frame = pd.read_parquet(path, columns=["sample_id"])
                    result.update(frame["sample_id"].astype(str).unique())
                    break
        return sorted(result)

    @staticmethod
    def _write_vector_fragment(
        path: Path, arrays: Mapping[str, np.ndarray]
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)

    def _replay_candidate_geometry(
        self,
        sample: Any,
        reference: Any,
        inventory_row: Mapping[str, Any],
        selection: Any,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        Dict[str, np.ndarray],
    ]:
        anchor = int(inventory_row["anchor"])
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
        geometry_rows: List[Dict[str, Any]] = []
        curvature_rows: List[Dict[str, Any]] = []
        vector_rows: List[Dict[str, Any]] = []
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
                geometry, curvature = adaptive_fisher_geometry(
                    reference.probe_logits[target_index],
                    logits,
                    trust_top_k=int(self.output_cfg.trust_top_k),
                    relative_tolerance=float(
                        self.output_cfg.adaptive_relative_tolerance
                    ),
                    absolute_tolerance=float(
                        self.output_cfg.adaptive_absolute_tolerance
                    ),
                    subdivision_limit=int(
                        self.output_cfg.adaptive_subdivision_limit
                    ),
                )
                full_h = reference_record.residual_inputs[27].float()
                actual_delta = record.residual_inputs[27].float() - full_h
                vector_index = len(actual_vectors)
                actual_vectors.append(
                    actual_delta.detach().cpu().numpy().astype(np.float16)
                )
                direct_vectors.append(
                    direct["projected"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                full_vectors.append(
                    full_h.detach().cpu().numpy().astype(np.float16)
                )
                common = {
                    **self._base(sample),
                    "candidate_id": candidate_id,
                    "candidate_index": int(
                        inventory_row["candidate_index"]
                    ),
                    "candidate_source": str(
                        inventory_row["candidate_source"]
                    ),
                    "anchor": anchor,
                    "horizon_offset": int(offset),
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
                    "uses_task_feature": False,
                    "forward_time_s": float(forward_s),
                    "layer27_actual_residual_norm": float(
                        actual_delta.norm().item()
                    ),
                    "layer27_direct_norm": float(direct["coordinate"]),
                    "layer27_direct_deleted_attention_mass": float(
                        direct["deleted_attention_mass"]
                    ),
                }
                geometry_rows.append({**common, **geometry})
                curvature_rows.append(
                    {
                        **common,
                        **curvature,
                        "large_direct_perturbation": bool(
                            float(direct["coordinate"]) >= 1.0
                        ),
                        "large_accumulated_state_drift": bool(
                            float(actual_delta.norm().item()) >= 1.0
                        ),
                    }
                )
                vector_rows.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "candidate_id": candidate_id,
                        "anchor": anchor,
                        "horizon_offset": int(offset),
                        "target_index": target_index,
                        "vector_row": vector_index,
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
            geometry_rows,
            curvature_rows,
            vector_rows,
            {
                "actual_residual_delta": np.stack(actual_vectors),
                "direct_projected_l27": np.stack(direct_vectors),
                "full_residual_l27": np.stack(full_vectors),
            },
        )

    def _completed_fragment_valid(self, sample_id: str) -> bool:
        expected = (
            len(self.output_cfg.anchors)
            * int(self.output_cfg.candidate_count)
            * int(self.output_cfg.segment_horizon)
        )
        for table in (
            "independent_fisher_geometry_rows",
            "adaptive_curvature_rows",
            "independent_vector_index",
        ):
            path = self._fragment(table, sample_id)
            if not path.exists():
                return False
            try:
                if len(pd.read_parquet(path)) != expected:
                    return False
            except Exception:
                return False
        return self._vector_fragment(sample_id).exists()

    def _run_independent_sample(
        self, sample: Any, sample_index: int
    ) -> None:
        key = "independent_fisher:%s" % _sample_slug(sample.sample_id)
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
            geometry_rows: List[Dict[str, Any]] = []
            curvature_rows: List[Dict[str, Any]] = []
            vector_rows: List[Dict[str, Any]] = []
            inventory_rows: List[Dict[str, Any]] = []
            vector_blocks: Dict[str, List[np.ndarray]] = {
                "actual_residual_delta": [],
                "direct_projected_l27": [],
                "full_residual_l27": [],
            }
            base_vector_row = 0
            for anchor in self.output_cfg.anchors:
                candidates = self._candidate_selections(
                    reference, int(anchor), sample_index
                )
                for candidate_index, (source, selection) in enumerate(
                    candidates
                ):
                    inventory = self._inventory_row(
                        sample,
                        reference,
                        int(anchor),
                        candidate_index,
                        source,
                        selection,
                    )
                    inventory_rows.append(inventory)
                    geometry, curvature, vectors, arrays = (
                        self._replay_candidate_geometry(
                            sample, reference, inventory, selection
                        )
                    )
                    for row in vectors:
                        row["vector_row"] = (
                            int(row["vector_row"]) + base_vector_row
                        )
                    base_vector_row += len(vectors)
                    geometry_rows.extend(geometry)
                    curvature_rows.extend(curvature)
                    vector_rows.extend(vectors)
                    for name, values in arrays.items():
                        vector_blocks[name].append(values)
            manifest = {
                **self._base(sample),
                "sample_id": str(sample.sample_id),
                "task": str(sample.task),
                "prompt_sha256": hashlib.sha256(
                    sample.prompt.encode("utf-8")
                ).hexdigest(),
                "dataset_official": bool(
                    sample.metadata.get("dataset_official", False)
                ),
                "official_dataset_index": sample.metadata.get(
                    "official_dataset_index"
                ),
                "sample_offset_preregistered": (
                    12 if str(sample.task).startswith("niah") else None
                ),
                "prior_id_overlap": False,
                "generation_rule_preregistered": True,
            }
            _atomic_frame(
                pd.DataFrame(geometry_rows),
                self._fragment(
                    "independent_fisher_geometry_rows", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(curvature_rows),
                self._fragment("adaptive_curvature_rows", sample.sample_id),
            )
            _atomic_frame(
                pd.DataFrame(inventory_rows),
                self._fragment(
                    "independent_candidate_inventory", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(vector_rows),
                self._fragment(
                    "independent_vector_index", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame([manifest]),
                self._fragment("new_sequence_manifest", sample.sample_id),
            )
            self._write_vector_fragment(
                self._vector_fragment(sample.sample_id),
                {
                    name: np.concatenate(values, axis=0)
                    for name, values in vector_blocks.items()
                },
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(time.perf_counter() - started),
                    "geometry_rows": len(geometry_rows),
                    "curvature_rows": len(curvature_rows),
                    "inventory_rows": len(inventory_rows),
                    "vector_rows": len(vector_rows),
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

    def _consolidate_independent(self) -> None:
        for table in INDEPENDENT_TABLES:
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "independent_fisher"
                    / table
                ).glob("*.parquet")
            )
            frame = (
                pd.concat(
                    [pd.read_parquet(path) for path in paths],
                    ignore_index=True,
                    sort=False,
                )
                if paths
                else pd.DataFrame()
            )
            _atomic_frame(frame, self.store.run_dir / ("%s.parquet" % table))

    def run(self) -> Path:
        if not self.cfg.independent_fisher.enabled:
            raise ValueError("independent_fisher.enabled must be true")
        self.store.status["state"] = "running_stage_a_prime"
        self.store.status["protocol"] = "independent_fisher_validation_v1"
        self.store.save_status()
        samples, task_events = load_discovery_tasks(self.cfg)
        prior_ids = set(self._prior_sample_ids())
        current_ids = {str(sample.sample_id) for sample in samples}
        overlap = sorted(prior_ids & current_ids)
        if overlap:
            raise RuntimeError(
                "new/old sequence IDs overlap: %s" % overlap[:10]
            )
        if len(samples) != 24 or len(current_ids) != 24:
            raise RuntimeError("formal independent run requires 24 unique sequences")
        official_gov = [
            sample
            for sample in samples
            if sample.task == "gov_report"
            and bool(sample.metadata.get("dataset_official", False))
        ]
        if len(official_gov) != 12:
            raise RuntimeError(
                "formal run requires 12 official GovReport sequences"
            )
        model_info = self.model.load()
        self.metadata = self.store.write_metadata(model_info, task_events)
        self.store.status["new_old_sequence_overlap"] = overlap
        self.store.status["formal_sequence_count"] = len(samples)
        for table in INDEPENDENT_TABLES:
            (
                self.store.run_dir
                / "fragments"
                / "independent_fisher"
                / table
            ).mkdir(parents=True, exist_ok=True)
        (
            self.store.run_dir
            / "fragments"
            / "independent_fisher"
            / "vectors"
        ).mkdir(parents=True, exist_ok=True)
        try:
            for sample_index, sample in enumerate(samples):
                self._run_independent_sample(sample, sample_index)
            self._consolidate_independent()
            self.store.status["state"] = (
                "stage_a_prime_data_complete_analysis_pending"
            )
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir
