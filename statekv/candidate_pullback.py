"""Model-backed final-boundary pullback for the independent Fisher protocol."""
from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from statekv.functional_probe import _condition_cache
from statekv.gauge_geometry import fisher_variance
from statekv.independent_fisher import IndependentFisherRunner
from statekv.runner import _sample_slug
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import _atomic_frame


def covariance_energy(
    probability: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    p = np.asarray(probability, dtype=np.float64)
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mean_a = float(np.dot(p, a))
    mean_b = float(np.dot(p, b))
    return float(np.dot(p, a * b) - mean_a * mean_b)


def state_action_energy_decomposition(
    probability: np.ndarray,
    state_output: np.ndarray,
    direct_output: np.ndarray,
) -> Dict[str, float]:
    state = np.asarray(state_output, dtype=np.float64)
    direct = np.asarray(direct_output, dtype=np.float64)
    state_energy = 0.5 * fisher_variance(probability, state)
    direct_energy = 0.5 * fisher_variance(probability, direct)
    cross = covariance_energy(probability, state, direct)
    total = 0.5 * fisher_variance(probability, state + direct)
    scalar_bound = 0.5 * (
        math.sqrt(max(2.0 * state_energy, 0.0))
        + math.sqrt(max(2.0 * direct_energy, 0.0))
    ) ** 2
    return {
        "state_energy": float(state_energy),
        "direct_energy": float(direct_energy),
        "cross_energy": float(cross),
        "total_energy": float(total),
        "decomposition_abs_error": float(
            abs(total - (state_energy + direct_energy + cross))
        ),
        "cauchy_schwarz_rhs": float(
            2.0 * math.sqrt(max(state_energy * direct_energy, 0.0))
        ),
        "cauchy_schwarz_holds": bool(
            abs(cross)
            <= 2.0
            * math.sqrt(max(state_energy * direct_energy, 0.0))
            + 1.0e-10
        ),
        "scalar_safe_bound": float(scalar_bound),
        "scalar_bound_holds": bool(total <= scalar_bound + 1.0e-10),
        "scalar_bound_looseness": float(
            scalar_bound / max(total, 1.0e-12)
        ),
        "cross_ratio": float(abs(cross) / max(total, 1.0e-12)),
        "cross_sign": int(np.sign(cross)),
    }


def psd_interaction_parameters(
    alpha: float, beta: float, eta: float
) -> Tuple[float, float, float]:
    a = float(alpha) ** 2
    b = float(beta) ** 2
    gamma = float(alpha) * float(beta) * math.tanh(float(eta))
    return a, b, gamma


def principal_angles(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    q_left, _ = np.linalg.qr(np.asarray(left, dtype=np.float64))
    q_right, _ = np.linalg.qr(np.asarray(right, dtype=np.float64))
    singular = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


class PureFinalBoundaryMap:
    """Pure layer-27 residual-to-logits map with immutable cached K/V."""

    def __init__(self, backend: Any, layer_cache: Any):
        import mlx.core as mx

        self.backend = backend
        self.layer_index = 27
        self.layer = backend.runner.model.model.layers[self.layer_index]
        self.attention = self.layer.self_attn
        self.final_norm = backend.runner.model.model.norm
        self.output_model = backend.runner.model
        offset = int(layer_cache.offset)
        self.keys = mx.array(layer_cache.keys[:, :, :offset, :])
        self.values = mx.array(layer_cache.values[:, :, :offset, :])
        self.rope_offset = int(
            getattr(layer_cache, "logical_offset", offset)
        )
        mx.eval(self.keys, self.values)

    def cache_fingerprint(self) -> str:
        import mlx.core as mx

        mx.eval(self.keys, self.values)
        digest = hashlib.sha256()
        digest.update(np.asarray(self.keys).tobytes())
        digest.update(np.asarray(self.values).tobytes())
        digest.update(str(self.rope_offset).encode("utf-8"))
        return digest.hexdigest()

    def __call__(self, residual: Any) -> Any:
        import mlx.core as mx
        from mlx_lm.models.base import scaled_dot_product_attention

        hidden_size = int(self.backend.model_info["hidden_size"])
        x = residual.reshape(1, 1, hidden_size)
        normalized = self.layer.input_layernorm(x)
        queries = self.attention.q_proj(normalized)
        keys = self.attention.k_proj(normalized)
        values = self.attention.v_proj(normalized)
        queries = queries.reshape(
            1, 1, self.attention.n_heads, -1
        )
        keys = keys.reshape(1, 1, self.attention.n_kv_heads, -1)
        values = values.reshape(1, 1, self.attention.n_kv_heads, -1)
        if hasattr(self.attention, "q_norm"):
            queries = self.attention.q_norm(queries)
        if hasattr(self.attention, "k_norm"):
            keys = self.attention.k_norm(keys)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.attention.rope(
            queries, offset=int(self.rope_offset)
        )
        keys = self.attention.rope(keys, offset=int(self.rope_offset))
        all_keys = mx.concatenate([self.keys, keys], axis=2)
        all_values = mx.concatenate([self.values, values], axis=2)
        attention_output = scaled_dot_product_attention(
            queries,
            all_keys,
            all_values,
            cache=None,
            scale=self.attention.scale,
            mask=None,
        )
        attention_output = attention_output.transpose(
            0, 2, 1, 3
        ).reshape(1, 1, -1)
        projected = self.attention.o_proj(attention_output)
        hidden = x + projected
        output = hidden + self.layer.mlp(
            self.layer.post_attention_layernorm(hidden)
        )
        output = self.final_norm(output)
        if self.output_model.args.tie_word_embeddings:
            logits = self.output_model.model.embed_tokens.as_linear(output)
        else:
            logits = self.output_model.lm_head(output)
        return logits.reshape(-1)

    def evaluate(self, point: np.ndarray) -> np.ndarray:
        import mlx.core as mx

        value = self(mx.array(np.asarray(point, dtype=np.float32)))
        mx.eval(value)
        return np.asarray(value).astype(np.float64)

    def jvp(
        self, point: np.ndarray, direction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        import mlx.core as mx

        primal = mx.array(np.asarray(point, dtype=np.float32))
        tangent = mx.array(np.asarray(direction, dtype=np.float32))
        output, derivative = mx.jvp(self, [primal], [tangent])
        if isinstance(output, (list, tuple)):
            output = output[0]
        if isinstance(derivative, (list, tuple)):
            derivative = derivative[0]
        mx.eval(output, derivative)
        return (
            np.asarray(output).astype(np.float64),
            np.asarray(derivative).astype(np.float64),
        )

    def symmetric_fd(
        self,
        point: np.ndarray,
        direction: np.ndarray,
        relative_radius: float,
        center_output: np.ndarray = None,
    ) -> Dict[str, Any]:
        base = np.asarray(point, dtype=np.float64)
        vector = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        unit = vector / max(norm, 1.0e-12)
        epsilon = float(relative_radius) * max(
            float(np.linalg.norm(base)), 1.0e-12
        )
        center = (
            np.asarray(center_output, dtype=np.float64)
            if center_output is not None
            else self.evaluate(base)
        )
        plus = self.evaluate(base + epsilon * unit)
        minus = self.evaluate(base - epsilon * unit)
        derivative_unit = (plus - minus) / max(2.0 * epsilon, 1.0e-30)
        derivative = derivative_unit * norm
        plus_delta = plus - center
        minus_delta = center - minus
        return {
            "derivative": derivative,
            "epsilon": epsilon,
            "plus_minus_asymmetry": float(
                np.linalg.norm(plus_delta - minus_delta)
                / max(
                    np.linalg.norm(plus_delta)
                    + np.linalg.norm(minus_delta),
                    1.0e-12,
                )
            ),
        }


def jvp_fd_diagnostics(
    pure_map: PureFinalBoundaryMap,
    point: np.ndarray,
    direction: np.ndarray,
    probability: np.ndarray,
    radii: Sequence[float],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    center, derivative = pure_map.jvp(point, direction)
    derivative_norm = float(np.linalg.norm(derivative))
    jvp_energy = fisher_variance(probability, derivative)
    rows: List[Dict[str, Any]] = []
    for radius in radii:
        finite = pure_map.symmetric_fd(
            point,
            direction,
            float(radius),
            center_output=center,
        )
        fd = np.asarray(finite["derivative"], dtype=np.float64)
        fd_norm = float(np.linalg.norm(fd))
        cosine = float(
            np.dot(derivative, fd)
            / max(derivative_norm * fd_norm, 1.0e-30)
        )
        fd_energy = fisher_variance(probability, fd)
        rows.append(
            {
                "relative_radius": float(radius),
                "absolute_epsilon": float(finite["epsilon"]),
                "jvp_fd_cosine": cosine,
                "relative_norm_error": float(
                    abs(fd_norm - derivative_norm)
                    / max(derivative_norm, 1.0e-12)
                ),
                "fisher_energy_relative_error": float(
                    abs(fd_energy - jvp_energy)
                    / max(jvp_energy, 1.0e-12)
                ),
                "plus_minus_asymmetry": float(
                    finite["plus_minus_asymmetry"]
                ),
            }
        )
    return derivative, rows


class CandidatePullbackRunner(IndependentFisherRunner):
    """Use saved physical stateful vectors and replay only the full reference."""

    def _pullback_fragment(self, table: str, sample_id: str) -> Path:
        return (
            self.store.run_dir
            / "fragments"
            / "candidate_pullback"
            / table
            / ("%s.parquet" % _sample_slug(sample_id))
        )

    def _stage_a_frames(
        self, sample_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Mapping[str, np.ndarray]]:
        geometry = pd.read_parquet(
            self.store.run_dir
            / "independent_fisher_geometry_rows.parquet"
        )
        geometry = geometry[
            geometry["sample_id"].astype(str) == str(sample_id)
        ].copy()
        index = pd.read_parquet(
            self.store.run_dir / "independent_vector_index.parquet"
        )
        index = index[
            index["sample_id"].astype(str) == str(sample_id)
        ].copy()
        vector_path = Path(str(index.iloc[0]["vector_fragment"]))
        arrays = np.load(vector_path)
        return geometry, index, arrays

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        maximum = float(np.max(values))
        weights = np.exp(values - maximum)
        return weights / weights.sum()

    @staticmethod
    def _vector_lookup(
        index: pd.DataFrame,
        arrays: Mapping[str, np.ndarray],
        candidate_id: str,
        anchor: int,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = index[
            (index["candidate_id"].astype(str) == str(candidate_id))
            & (index["anchor"] == int(anchor))
            & (index["horizon_offset"] == int(offset))
        ]
        if len(row) != 1:
            raise RuntimeError("vector index must identify exactly one row")
        vector_row = int(row.iloc[0]["vector_row"])
        return (
            np.asarray(arrays["full_residual_l27"][vector_row]).astype(
                np.float64
            ),
            np.asarray(
                arrays["actual_residual_delta"][vector_row]
            ).astype(np.float64),
            np.asarray(
                arrays["direct_projected_l27"][vector_row]
            ).astype(np.float64),
        )

    def _selected_geometry(self, geometry: pd.DataFrame) -> pd.DataFrame:
        sources = set(
            str(value)
            for value in self.output_cfg.stage_b_candidate_sources
        )
        selected = geometry[
            geometry["candidate_source"].astype(str).isin(sources)
        ].copy()
        counts = selected.groupby(
            ["anchor", "horizon_offset"]
        )["candidate_id"].nunique()
        if not counts.eq(int(self.output_cfg.stage_b_candidate_count)).all():
            raise RuntimeError("Stage-B fixed candidate pool is incomplete")
        return selected

    def _predicted_response_scale(self, heldout_sample_id: str) -> float:
        """Fit a one-step nonnegative scalar response on training sequences only."""

        geometry = pd.read_parquet(
            self.store.run_dir
            / "independent_fisher_geometry_rows.parquet",
            columns=[
                "sample_id",
                "candidate_id",
                "anchor",
                "horizon_offset",
                "candidate_source",
            ],
        )
        sources = set(
            str(value)
            for value in self.output_cfg.stage_b_candidate_sources
        )
        training = geometry[
            (geometry["sample_id"].astype(str) != str(heldout_sample_id))
            & (geometry["horizon_offset"] == 1)
            & geometry["candidate_source"].astype(str).isin(sources)
        ]
        index = pd.read_parquet(
            self.store.run_dir / "independent_vector_index.parquet"
        )
        training = training.merge(
            index,
            on=[
                "sample_id",
                "candidate_id",
                "anchor",
                "horizon_offset",
            ],
            how="left",
            validate="one_to_one",
        )
        numerator = 0.0
        denominator = 0.0
        for vector_path, current in training.groupby("vector_fragment"):
            arrays = np.load(Path(str(vector_path)))
            for vector_row in current["vector_row"].to_numpy(dtype=np.int64):
                actual = np.asarray(
                    arrays["actual_residual_delta"][int(vector_row)]
                ).astype(np.float64)
                direct = np.asarray(
                    arrays["direct_projected_l27"][int(vector_row)]
                ).astype(np.float64)
                numerator += float(np.dot(actual, direct))
                denominator += float(np.dot(direct, direct))
        coefficient = max(numerator / max(denominator, 1.0e-12), 0.0)
        # The inherited local-response evidence is registered only for beta <= 1.
        return float(min(coefficient, 1.0))

    def _run_pullback_sample(self, sample: Any) -> None:
        key = "candidate_pullback:%s" % _sample_slug(sample.sample_id)
        if self.cfg.runtime.resume and self.store.is_complete(key):
            return
        started = time.perf_counter()
        try:
            reference = self.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            geometry, index, arrays = self._stage_a_frames(sample.sample_id)
            selected = self._selected_geometry(geometry)
            predicted_response_scale = self._predicted_response_scale(
                str(sample.sample_id)
            )
            pullback_rows: List[Dict[str, Any]] = []
            fd_rows: List[Dict[str, Any]] = []
            cross_rows: List[Dict[str, Any]] = []
            for anchor in self.output_cfg.anchors:
                anchor_state = reference.anchors[int(anchor)]
                full_selection = self._all_history_selection(
                    reference, int(anchor)
                )
                full_cache = _condition_cache(
                    self.cfg,
                    int(anchor_state.logical_length)
                    + int(self.output_cfg.segment_horizon)
                    + 2,
                    1,
                )
                full_state, _ = self.model.state_from_anchor(
                    anchor_state,
                    full_selection,
                    cache_config=full_cache,
                )
                current_token = int(anchor_state.query_token_id)
                try:
                    for offset in range(
                        1, int(self.output_cfg.segment_horizon) + 1
                    ):
                        target_index = int(anchor + offset - 1)
                        current = selected[
                            (selected["anchor"] == int(anchor))
                            & (
                                selected["horizon_offset"]
                                == int(offset)
                            )
                        ].sort_values("candidate_index")
                        pure_map = PureFinalBoundaryMap(
                            self.model, full_state.cache[27]
                        )
                        pullback_group_start = len(pullback_rows)
                        fingerprint_before = pure_map.cache_fingerprint()
                        reference_logits = (
                            reference.probe_logits[target_index]
                            .float()
                            .numpy()
                            .astype(np.float64)
                        )
                        repeated_a = pure_map.evaluate(
                            current.iloc[0][
                                "layer27_actual_residual_norm"
                            ]
                            * 0.0
                            + self._vector_lookup(
                                index,
                                arrays,
                                str(current.iloc[0]["candidate_id"]),
                                int(anchor),
                                int(offset),
                            )[0]
                        )
                        repeated_b = pure_map.evaluate(
                            self._vector_lookup(
                                index,
                                arrays,
                                str(current.iloc[0]["candidate_id"]),
                                int(anchor),
                                int(offset),
                            )[0]
                        )
                        repeated_equal = bool(
                            np.array_equal(repeated_a, repeated_b)
                        )
                        map_reference_relative_error = float(
                            np.linalg.norm(repeated_a - reference_logits)
                            / max(
                                np.linalg.norm(reference_logits), 1.0e-12
                            )
                        )
                        for row in current.to_dict("records"):
                            full_h, actual_delta, direct = self._vector_lookup(
                                index,
                                arrays,
                                str(row["candidate_id"]),
                                int(anchor),
                                int(offset),
                            )
                            base_logits, base_jvp = pure_map.jvp(
                                full_h, actual_delta
                            )
                            _, base_direct_jvp = pure_map.jvp(
                                full_h, direct
                            )
                            base_probability = self._softmax(base_logits)
                            _, diagnostics = jvp_fd_diagnostics(
                                pure_map,
                                full_h,
                                actual_delta,
                                base_probability,
                                self.output_cfg.pullback_radii,
                            )
                            oracle_point = full_h + 0.5 * actual_delta
                            oracle_logits, oracle_jvp = pure_map.jvp(
                                oracle_point, actual_delta
                            )
                            _, oracle_direct_jvp = pure_map.jvp(
                                oracle_point, direct
                            )
                            oracle_probability = self._softmax(oracle_logits)
                            direct_point = full_h + 0.5 * direct
                            direct_logits, direct_actual_jvp = pure_map.jvp(
                                direct_point, actual_delta
                            )
                            _, direct_jvp = pure_map.jvp(
                                direct_point, direct
                            )
                            direct_probability = self._softmax(direct_logits)
                            predicted_point = (
                                full_h
                                + 0.5 * predicted_response_scale * direct
                            )
                            (
                                predicted_logits,
                                predicted_actual_jvp,
                            ) = pure_map.jvp(
                                predicted_point, actual_delta
                            )
                            _, predicted_direct_jvp = pure_map.jvp(
                                predicted_point, direct
                            )
                            predicted_probability = self._softmax(
                                predicted_logits
                            )
                            modes = (
                                (
                                    "B0_BASE",
                                    base_probability,
                                    base_jvp,
                                    base_direct_jvp,
                                    full_h,
                                ),
                                (
                                    "B1_ORACLE_MIDPOINT",
                                    oracle_probability,
                                    oracle_jvp,
                                    oracle_direct_jvp,
                                    oracle_point,
                                ),
                                (
                                    "B2_CANDIDATE_DIRECT_MIDPOINT",
                                    direct_probability,
                                    direct_actual_jvp,
                                    direct_jvp,
                                    direct_point,
                                ),
                                (
                                    "B3_PREDICTED_MIDPOINT",
                                    predicted_probability,
                                    predicted_actual_jvp,
                                    predicted_direct_jvp,
                                    predicted_point,
                                ),
                            )
                            for (
                                mode,
                                probability,
                                actual_jvp,
                                current_direct_jvp,
                                operating_point,
                            ) in modes:
                                pullback_rows.append(
                                    {
                                        **self._base(sample),
                                        "anchor": int(anchor),
                                        "horizon_offset": int(offset),
                                        "target_index": target_index,
                                        "candidate_id": str(
                                            row["candidate_id"]
                                        ),
                                        "candidate_source": str(
                                            row["candidate_source"]
                                        ),
                                        "pullback_mode": mode,
                                        "exact_kl": float(row["exact_kl"]),
                                        "true_g2": float(
                                            row["g2_base_fisher"]
                                        ),
                                        "true_g3": float(
                                            row["g3_midpoint_fisher"]
                                        ),
                                        "actual_state_energy": float(
                                            0.5
                                            * fisher_variance(
                                                probability, actual_jvp
                                            )
                                        ),
                                        "direct_q_energy": float(
                                            0.5
                                            * fisher_variance(
                                                probability,
                                                current_direct_jvp,
                                            )
                                        ),
                                        "operating_point_norm": float(
                                            np.linalg.norm(operating_point)
                                        ),
                                        "actual_direction_norm": float(
                                            np.linalg.norm(actual_delta)
                                        ),
                                        "direct_direction_norm": float(
                                            np.linalg.norm(direct)
                                        ),
                                        "pure_map_repeated_equal": (
                                            repeated_equal
                                        ),
                                        "pure_map_cache_unchanged": bool(
                                            True
                                        ),
                                        "pure_map_reference_relative_error": (
                                            map_reference_relative_error
                                        ),
                                        "uses_future_compressed_truth": bool(
                                            mode
                                            == "B1_ORACLE_MIDPOINT"
                                        ),
                                        "deployable": bool(
                                            mode
                                            in {
                                                "B0_BASE",
                                                "B2_CANDIDATE_DIRECT_MIDPOINT",
                                                "B3_PREDICTED_MIDPOINT",
                                            }
                                        ),
                                        "predicted_midpoint_response": (
                                            "outer-fold_nonnegative_scalar_one_step"
                                            if mode.startswith("B3_")
                                            else None
                                        ),
                                        "predicted_response_scale": (
                                            predicted_response_scale
                                            if mode.startswith("B3_")
                                            else None
                                        ),
                                        "predicted_response_training_excludes_test_sequence": bool(
                                            mode.startswith("B3_")
                                        ),
                                    }
                                )
                            for diagnostic in diagnostics:
                                fd_rows.append(
                                    {
                                        **self._base(sample),
                                        "anchor": int(anchor),
                                        "horizon_offset": int(offset),
                                        "candidate_id": str(
                                            row["candidate_id"]
                                        ),
                                        "candidate_source": str(
                                            row["candidate_source"]
                                        ),
                                        "directions_per_operating_point": int(
                                            self.output_cfg.stage_b_candidate_count
                                        ),
                                        "pullback_mode": "B0_BASE",
                                        **diagnostic,
                                    }
                                )
                            cross = state_action_energy_decomposition(
                                direct_probability,
                                direct_actual_jvp,
                                direct_jvp,
                            )
                            cross_rows.append(
                                {
                                    **self._base(sample),
                                    "anchor": int(anchor),
                                    "horizon_offset": int(offset),
                                    "candidate_id": str(
                                        row["candidate_id"]
                                    ),
                                    "candidate_source": str(
                                        row["candidate_source"]
                                    ),
                                    "exact_kl": float(row["exact_kl"]),
                                    "operating_point": (
                                        "candidate_direct_midpoint"
                                    ),
                                    **cross,
                                }
                            )
                        cache_unchanged = bool(
                            pure_map.cache_fingerprint()
                            == fingerprint_before
                        )
                        for emitted in pullback_rows[
                            pullback_group_start:
                        ]:
                            emitted["pure_map_cache_unchanged"] = (
                                cache_unchanged
                            )
                        _, full_record, _ = self.model.forward_one(
                            full_state,
                            current_token,
                            capture_attention=True,
                        )
                        if (
                            int(full_record.query_position)
                            != int(
                                reference.query_records[
                                    target_index
                                ].query_position
                            )
                        ):
                            raise RuntimeError(
                                "full replay query position is misaligned"
                            )
                        current_token = int(
                            reference.generated_token_ids[target_index]
                        )
                finally:
                    self.model.release(full_state)
            _atomic_frame(
                pd.DataFrame(pullback_rows),
                self._pullback_fragment(
                    "pullback_operating_point_rows", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(fd_rows),
                self._pullback_fragment(
                    "pullback_jvp_validation_rows", sample.sample_id
                ),
            )
            _atomic_frame(
                pd.DataFrame(cross_rows),
                self._pullback_fragment(
                    "state_action_cross_term_rows", sample.sample_id
                ),
            )
            self.store.mark_complete(
                key,
                {
                    "elapsed_s": float(time.perf_counter() - started),
                    "pullback_rows": len(pullback_rows),
                    "jvp_fd_rows": len(fd_rows),
                    "cross_rows": len(cross_rows),
                },
            )
            self.model.release(reference)
        except Exception as exc:
            self.store.append_error(
                {
                    "key": key,
                    "sample_id": str(sample.sample_id),
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

    def _consolidate_pullback(self) -> None:
        for table in (
            "pullback_operating_point_rows",
            "pullback_jvp_validation_rows",
            "state_action_cross_term_rows",
        ):
            paths = sorted(
                (
                    self.store.run_dir
                    / "fragments"
                    / "candidate_pullback"
                    / table
                ).glob("*.parquet")
            )
            frame = pd.concat(
                [pd.read_parquet(path) for path in paths],
                ignore_index=True,
                sort=False,
            )
            _atomic_frame(frame, self.store.run_dir / ("%s.parquet" % table))

    def run_pullback(self) -> Path:
        gate_path = self.store.run_dir / "independent_fisher_gate_decision.json"
        if not gate_path.exists():
            raise RuntimeError("Stage-A′ gate decision is missing")
        gate = json.loads(gate_path.read_text())
        if not bool(gate.get("stage_a_prime_replication_passed", False)):
            raise RuntimeError("Stage B′ is not authorized")
        self.store.status["state"] = "running_stage_b_prime"
        self.store.save_status()
        samples, _ = load_discovery_tasks(self.cfg)
        self.model.load()
        for table in (
            "pullback_operating_point_rows",
            "pullback_jvp_validation_rows",
            "state_action_cross_term_rows",
        ):
            (
                self.store.run_dir
                / "fragments"
                / "candidate_pullback"
                / table
            ).mkdir(parents=True, exist_ok=True)
        try:
            for sample in samples:
                self._run_pullback_sample(sample)
            self._consolidate_pullback()
            self.store.status["state"] = (
                "stage_b_prime_data_complete_analysis_pending"
            )
            self.store.save_status()
        finally:
            self.model.close()
        return self.store.run_dir
