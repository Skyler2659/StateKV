"""Distribution-free nonnegative perturbation-envelope analysis."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from statekv.config import DiscoveryConfig
from statekv.robust_envelope import block_triangular_mask
from statekv.trajectory_analysis import (
    atomic_json,
    cluster_bootstrap_interval,
)
from statekv.theory_closing import _atomic_frame


@dataclass
class EnvelopeModel:
    family: str
    layers: List[int]
    a: np.ndarray
    b: np.ndarray
    h: np.ndarray
    scalar: bool = False
    source: str = "empirical_nonnegative"

    def step(
        self, error: np.ndarray, direct: np.ndarray
    ) -> np.ndarray:
        error = np.asarray(error, dtype=np.float64)
        direct = np.asarray(direct, dtype=np.float64)
        if self.scalar:
            value = (
                float(self.a[0, 0]) * float(error[0])
                + float(self.b[0, 0]) * float(direct[0])
                + float(self.h[0, 0]) * float(error[0] ** 2)
            )
            return np.asarray([max(0.0, value)])
        return np.maximum(
            0.0,
            self.a @ error
            + self.b @ direct
            + self.h @ (error * error),
        )


def fit_nonnegative_envelope(
    previous_error: np.ndarray,
    direct_input: np.ndarray,
    next_error: np.ndarray,
    layers: Sequence[int],
    family: str,
) -> EnvelopeModel:
    """Fit E1/E2/E3 with nonnegative coefficients and causal layer order."""

    previous_error = np.asarray(previous_error, dtype=np.float64)
    direct_input = np.asarray(direct_input, dtype=np.float64)
    next_error = np.asarray(next_error, dtype=np.float64)
    layer_list = [int(value) for value in layers]
    dimension = len(layer_list)
    if (
        previous_error.shape != direct_input.shape
        or previous_error.shape != next_error.shape
        or previous_error.ndim != 2
        or previous_error.shape[1] != dimension
    ):
        raise ValueError("envelope training arrays are not aligned")
    if family == "E1":
        x = np.column_stack(
            [
                np.linalg.norm(previous_error, axis=1),
                np.linalg.norm(direct_input, axis=1),
            ]
        )
        y = np.linalg.norm(next_error, axis=1)
        coefficient, _ = nnls(x, y)
        return EnvelopeModel(
            family=family,
            layers=layer_list,
            a=np.asarray([[coefficient[0]]]),
            b=np.asarray([[coefficient[1]]]),
            h=np.zeros((1, 1)),
            scalar=True,
        )
    if family not in {"E2", "E3"}:
        raise ValueError("unknown envelope family")
    mask = block_triangular_mask(layer_list, layer_list)
    a = np.zeros((dimension, dimension), dtype=np.float64)
    b = np.zeros_like(a)
    h = np.zeros_like(a)
    for output in range(dimension):
        allowed = np.flatnonzero(mask[output])
        blocks = [
            previous_error[:, allowed],
            direct_input[:, allowed],
        ]
        if family == "E3":
            blocks.append(previous_error[:, allowed] ** 2)
        design = np.concatenate(blocks, axis=1)
        coefficient, _ = nnls(design, next_error[:, output])
        width = len(allowed)
        a[output, allowed] = coefficient[:width]
        b[output, allowed] = coefficient[width : 2 * width]
        if family == "E3":
            h[output, allowed] = coefficient[2 * width :]
    return EnvelopeModel(
        family=family,
        layers=layer_list,
        a=a,
        b=b,
        h=h,
    )


def envelope_coefficients_nonnegative(model: EnvelopeModel) -> bool:
    return bool(
        np.all(model.a >= 0)
        and np.all(model.b >= 0)
        and np.all(model.h >= 0)
    )


def calibration_margin(
    residual: np.ndarray,
    sequence_ids: Sequence[str],
    level: float,
    simultaneous: bool,
) -> np.ndarray:
    """Coordinate margins; simultaneous mode calibrates sequence maxima."""

    residual = np.maximum(np.asarray(residual, dtype=np.float64), 0.0)
    sequence_ids = np.asarray(sequence_ids)
    if len(residual) != len(sequence_ids):
        raise ValueError("calibration residuals and sequences are not aligned")
    if simultaneous:
        scores = np.stack(
            [
                residual[sequence_ids == sequence].max(axis=0)
                for sequence in sorted(set(sequence_ids.tolist()))
            ]
        )
    else:
        scores = residual
    try:
        return np.quantile(
            scores, float(level), axis=0, method="higher"
        )
    except TypeError:
        return np.quantile(
            scores, float(level), axis=0, interpolation="higher"
        )


def recursive_envelope(
    model: EnvelopeModel,
    direct_inputs: np.ndarray,
    margin: np.ndarray,
    initial_error: Optional[np.ndarray] = None,
    explosion_threshold: float = 1e12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recursive bound that never reads realized compressed future errors."""

    direct_inputs = np.asarray(direct_inputs, dtype=np.float64)
    dimension = 1 if model.scalar else len(model.layers)
    current = (
        np.zeros(dimension, dtype=np.float64)
        if initial_error is None
        else np.asarray(initial_error, dtype=np.float64).copy()
    )
    margin = np.asarray(margin, dtype=np.float64).reshape(dimension)
    values = []
    exploded = []
    for direct in direct_inputs:
        current_direct = (
            np.asarray([np.linalg.norm(direct)])
            if model.scalar
            else direct
        )
        current = model.step(current, current_direct) + margin
        is_exploded = bool(
            not np.isfinite(current).all()
            or np.max(current) > float(explosion_threshold)
        )
        if is_exploded:
            current = np.nan_to_num(
                current,
                nan=explosion_threshold,
                posinf=explosion_threshold,
                neginf=0.0,
            )
            current = np.minimum(current, explosion_threshold)
        values.append(current.copy())
        exploded.append(is_exploded)
    return np.stack(values), np.asarray(exploded, dtype=bool)


def h1_recursion(
    model: EnvelopeModel, direct: np.ndarray, margin: np.ndarray
) -> np.ndarray:
    dimension = 1 if model.scalar else len(model.layers)
    direct_value = (
        np.asarray([np.linalg.norm(direct)])
        if model.scalar
        else np.asarray(direct, dtype=np.float64)
    )
    return model.step(np.zeros(dimension), direct_value) + margin


def induction_check(
    model: EnvelopeModel,
    realized_previous: np.ndarray,
    bound_previous: np.ndarray,
    direct: np.ndarray,
    margin: np.ndarray,
) -> bool:
    """Monotonicity step used by the analytical induction proof/test."""

    if np.any(realized_previous > bound_previous + 1e-12):
        raise ValueError("induction premise is false")
    left = model.step(realized_previous, direct)
    right = model.step(bound_previous, direct) + margin
    return bool(np.all(left <= right + 1e-12))


def _wide_state(
    rows: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int]]:
    keys = [
        "sample_id",
        "task",
        "trajectory_id",
        "trajectory_kind",
        "selector",
        "subset_index",
        "horizon_offset",
    ]
    layers = sorted(int(value) for value in rows["layer"].unique())
    error = rows.pivot_table(
        index=keys,
        columns="layer",
        values="residual_error",
        aggfunc="first",
    )
    error.columns = ["e_l%d" % layer for layer in error.columns]
    direct = rows.pivot_table(
        index=keys,
        columns="layer",
        values="direct_coordinate",
        aggfunc="first",
    )
    direct.columns = ["d_l%d" % layer for layer in direct.columns]
    metric = (
        rows.groupby(keys, as_index=False)
        .agg(
            exact_kl=("exact_kl", "first"),
            js=("js", "first"),
            delta_nll=("delta_nll", "first"),
            projected_output_error=(
                "projected_output_error",
                "sum",
            ),
            deleted_attention_mass=(
                "deleted_attention_mass",
                "sum",
            ),
            token_position_aligned=(
                "token_position_aligned",
                "all",
            ),
        )
    )
    return (
        error.reset_index()
        .merge(direct.reset_index(), on=keys)
        .merge(metric, on=keys),
        layers,
    )


def _transition_arrays(
    wide: pd.DataFrame,
    layers: Sequence[int],
    sequences: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    previous = []
    direct = []
    target = []
    metadata = []
    e_columns = ["e_l%d" % layer for layer in layers]
    d_columns = ["d_l%d" % layer for layer in layers]
    subset = wide[
        wide["sample_id"].isin(list(sequences))
        & (wide["trajectory_kind"] == "static")
    ]
    for trajectory_id, group in subset.groupby(
        "trajectory_id", sort=False
    ):
        group = group.sort_values("horizon_offset")
        errors = group[e_columns].to_numpy(dtype=np.float64)
        inputs = group[d_columns].to_numpy(dtype=np.float64)
        previous.append(
            np.vstack(
                [np.zeros((1, len(layers))), errors[:-1]]
            )
        )
        direct.append(inputs)
        target.append(errors)
        metadata.append(
            group[
                [
                    "sample_id",
                    "task",
                    "trajectory_id",
                    "horizon_offset",
                ]
            ]
        )
    return (
        np.concatenate(previous),
        np.concatenate(direct),
        np.concatenate(target),
        pd.concat(metadata, ignore_index=True),
    )


def _partition_sequences(
    all_sequences: Sequence[str], held_out: str, task_by_sequence: Mapping[str, str]
) -> Tuple[List[str], List[str]]:
    remaining = [value for value in sorted(all_sequences) if value != held_out]

    def stable_order(values: Iterable[str]) -> List[str]:
        return sorted(
            values,
            key=lambda value: hashlib.sha256(
                ("robust-calibration:" + value).encode("utf-8")
            ).hexdigest(),
        )

    tasks = sorted(set(task_by_sequence.values()))
    calibration: List[str] = []
    for task in tasks:
        candidates = [
            value
            for value in remaining
            if task_by_sequence[value] == task
        ]
        calibration.append(stable_order(candidates)[0])
    candidates = [
        value for value in remaining if value not in set(calibration)
    ]
    preferred_task = task_by_sequence[held_out]
    preferred = [
        value
        for value in candidates
        if task_by_sequence[value] == preferred_task
    ]
    calibration.append(
        stable_order(preferred or candidates)[0]
    )
    fit = [
        value for value in remaining if value not in set(calibration)
    ]
    return fit, calibration


def _predict_one_step(
    model: EnvelopeModel,
    previous: np.ndarray,
    direct: np.ndarray,
) -> np.ndarray:
    if model.scalar:
        return np.stack(
            [
                model.step(
                    np.asarray([np.linalg.norm(left)]),
                    np.asarray([np.linalg.norm(current)]),
                )
                for left, current in zip(previous, direct)
            ]
        )
    return np.stack(
        [model.step(left, current) for left, current in zip(previous, direct)]
    )


def _architecture_model(
    probes: pd.DataFrame,
    sample_id: str,
    layers: Sequence[int],
) -> EnvelopeModel:
    sample = probes[probes["sample_id"] == sample_id]
    dimension = len(layers)
    layer_to_index = {
        int(layer): index for index, layer in enumerate(layers)
    }
    injection_layers = sorted(
        int(value) for value in sample["injection_layer"].unique()
    )
    b = np.zeros((dimension, dimension), dtype=np.float64)
    e1_rows = []
    e2_rows = []
    for injection_layer in injection_layers:
        unit = sample[sample["injection_layer"] == injection_layer]
        d = float(unit["direct_input_coordinate"].iloc[0])
        e1 = np.zeros(dimension)
        e2 = np.zeros(dimension)
        for response_layer in layers:
            index = layer_to_index[int(response_layer)]
            e1[index] = float(
                unit[
                    (unit["response_layer"] == response_layer)
                    & (unit["horizon_offset"] == 1)
                ]["residual_response"].iloc[0]
            )
            e2[index] = float(
                unit[
                    (unit["response_layer"] == response_layer)
                    & (unit["horizon_offset"] == 2)
                ]["residual_response"].iloc[0]
            )
        if injection_layer in layer_to_index:
            b[:, layer_to_index[injection_layer]] = e1 / max(d, 1e-12)
        e1_rows.append(e1)
        e2_rows.append(e2)
    previous = np.stack(e1_rows)
    target = np.stack(e2_rows)
    a = np.zeros((dimension, dimension))
    mask = block_triangular_mask(layers, layers)
    for output in range(dimension):
        allowed = np.flatnonzero(mask[output])
        coefficient, _ = nnls(
            previous[:, allowed], target[:, output]
        )
        a[output, allowed] = coefficient
    return EnvelopeModel(
        family="E2",
        layers=list(layers),
        a=a,
        b=b,
        h=np.zeros_like(a),
        source="directional_finite_difference",
    )


def _model_dict(model: EnvelopeModel) -> Dict[str, Any]:
    return {
        "family": model.family,
        "layers": model.layers,
        "source": model.source,
        "a": model.a.tolist(),
        "b": model.b.tolist(),
        "h": model.h.tolist(),
        "scalar": model.scalar,
        "all_coefficients_nonnegative": envelope_coefficients_nonnegative(
            model
        ),
    }


def run_envelope_analysis(
    run_dir: Path, cfg: DiscoveryConfig
) -> Dict[str, Path]:
    run_dir = run_dir.resolve()
    rows = pd.read_parquet(run_dir / "robust_trajectory_rows.parquet")
    inventory = pd.read_parquet(
        run_dir / "envelope_subset_inventory.parquet"
    )
    probes = pd.read_parquet(
        run_dir / "architecture_gain_probe_rows.parquet"
    )
    if not bool(rows["token_position_aligned"].all()):
        raise RuntimeError("robust envelope data contain alignment failures")
    wide, layers = _wide_state(rows)
    sequences = sorted(wide["sample_id"].unique())
    task_by_sequence = (
        wide.groupby("sample_id")["task"].first().to_dict()
    )
    e_columns = ["e_l%d" % layer for layer in layers]
    d_columns = ["d_l%d" % layer for layer in layers]
    one_step_rows: List[Dict[str, Any]] = []
    rollout_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    fold_models: Dict[str, Any] = {}
    families = ("E1", "E2", "E3")
    for held_out in sequences:
        fit_sequences, calibration_sequences = _partition_sequences(
            sequences, held_out, task_by_sequence
        )
        fit_previous, fit_direct, fit_target, _ = _transition_arrays(
            wide, layers, fit_sequences
        )
        cal_previous, cal_direct, cal_target, cal_meta = _transition_arrays(
            wide, layers, calibration_sequences
        )
        test_wide = wide[wide["sample_id"] == held_out].copy()
        fold_models[held_out] = {
            "fit_sequences": fit_sequences,
            "calibration_sequences": calibration_sequences,
            "models": {},
        }
        model_specs: List[Tuple[str, str, EnvelopeModel]] = []
        for family in families:
            model_specs.append(
                (
                    family,
                    "empirical_nonnegative",
                    fit_nonnegative_envelope(
                        fit_previous,
                        fit_direct,
                        fit_target,
                        layers,
                        family,
                    ),
                )
            )
        architecture = _architecture_model(
            probes, held_out, layers
        )
        model_specs.append(
            ("E2", "architecture_directional", architecture)
        )
        for family, route, model in model_specs:
            if route == "architecture_directional":
                cal_prediction_parts = []
                cal_truth_parts = []
                cal_sequence_parts = []
                for calibration_sequence in calibration_sequences:
                    current_model = _architecture_model(
                        probes, calibration_sequence, layers
                    )
                    cp, cd, ct, cm = _transition_arrays(
                        wide, layers, [calibration_sequence]
                    )
                    cal_prediction_parts.append(
                        _predict_one_step(current_model, cp, cd)
                    )
                    cal_truth_parts.append(ct)
                    cal_sequence_parts.extend(
                        cm["sample_id"].tolist()
                    )
                cal_prediction = np.concatenate(cal_prediction_parts)
                cal_truth = np.concatenate(cal_truth_parts)
                cal_sequence_ids = np.asarray(cal_sequence_parts)
            else:
                cal_prediction = _predict_one_step(
                    model, cal_previous, cal_direct
                )
                cal_truth = (
                    np.linalg.norm(cal_target, axis=1, keepdims=True)
                    if model.scalar
                    else cal_target
                )
                cal_sequence_ids = cal_meta["sample_id"].to_numpy()
            residual = cal_truth - cal_prediction
            model_key = "%s:%s" % (family, route)
            fold_models[held_out]["models"][model_key] = _model_dict(model)
            for coverage_level in cfg.robust_envelope.coverage_levels:
                for margin_type in ("pointwise", "simultaneous"):
                    margin = calibration_margin(
                        residual,
                        cal_sequence_ids,
                        float(coverage_level),
                        simultaneous=margin_type == "simultaneous",
                    )
                    fold_models[held_out]["models"][model_key][
                        "%s_margin_%s"
                        % (margin_type, coverage_level)
                    ] = margin.tolist()
                    # One-step validity uses the true previous error by design.
                    for trajectory_id, group in test_wide.groupby(
                        "trajectory_id", sort=False
                    ):
                        group = group.sort_values("horizon_offset")
                        actual = group[e_columns].to_numpy(dtype=np.float64)
                        direct = group[d_columns].to_numpy(dtype=np.float64)
                        previous = np.vstack(
                            [np.zeros((1, len(layers))), actual[:-1]]
                        )
                        prediction = _predict_one_step(
                            model, previous, direct
                        )
                        truth = (
                            np.linalg.norm(actual, axis=1, keepdims=True)
                            if model.scalar
                            else actual
                        )
                        bound = prediction + margin
                        output_layers = [-1] if model.scalar else layers
                        for offset in range(len(group)):
                            for coordinate, layer in enumerate(output_layers):
                                realized = float(truth[offset, coordinate])
                                current_bound = float(bound[offset, coordinate])
                                one_step_rows.append(
                                    {
                                        "held_out_sequence": held_out,
                                        "task": task_by_sequence[held_out],
                                        "trajectory_id": trajectory_id,
                                        "trajectory_kind": group.iloc[
                                            offset
                                        ]["trajectory_kind"],
                                        "selector": group.iloc[offset]["selector"],
                                        "family": family,
                                        "route": route,
                                        "coverage_level": float(coverage_level),
                                        "margin_type": margin_type,
                                        "horizon_offset": int(
                                            group.iloc[offset][
                                                "horizon_offset"
                                            ]
                                        ),
                                        "layer": int(layer),
                                        "realized": realized,
                                        "prediction_without_margin": float(
                                            prediction[offset, coordinate]
                                        ),
                                        "direct_input_coordinate": float(
                                            np.linalg.norm(direct[offset])
                                            if model.scalar
                                            else direct[offset, coordinate]
                                        ),
                                        "margin": float(margin[coordinate]),
                                        "bound": current_bound,
                                        "violation": bool(
                                            realized > current_bound + 1e-12
                                        ),
                                        "violation_magnitude": max(
                                            0.0, realized - current_bound
                                        ),
                                        "looseness": float(
                                            (current_bound + 1e-9)
                                            / (realized + 1e-9)
                                        ),
                                        "uses_true_previous_error": True,
                                    }
                                )
                        # Recursive validity never reads actual future state.
                        recursive, exploded = recursive_envelope(
                            model, direct, margin
                        )
                        recursive_truth = truth
                        for offset in range(len(group)):
                            for coordinate, layer in enumerate(output_layers):
                                realized = float(
                                    recursive_truth[offset, coordinate]
                                )
                                current_bound = float(
                                    recursive[offset, coordinate]
                                )
                                rollout_rows.append(
                                    {
                                        "held_out_sequence": held_out,
                                        "task": task_by_sequence[held_out],
                                        "trajectory_id": trajectory_id,
                                        "trajectory_kind": group.iloc[
                                            offset
                                        ]["trajectory_kind"],
                                        "selector": group.iloc[offset]["selector"],
                                        "subset_index": int(
                                            group.iloc[offset]["subset_index"]
                                        ),
                                        "family": family,
                                        "route": route,
                                        "coverage_level": float(coverage_level),
                                        "margin_type": margin_type,
                                        "horizon_offset": int(
                                            group.iloc[offset][
                                                "horizon_offset"
                                            ]
                                        ),
                                        "layer": int(layer),
                                        "realized": realized,
                                        "bound": current_bound,
                                        "direct_input_coordinate": float(
                                            np.linalg.norm(direct[offset])
                                            if model.scalar
                                            else direct[offset, coordinate]
                                        ),
                                        "violation": bool(
                                            realized > current_bound + 1e-12
                                        ),
                                        "violation_magnitude": max(
                                            0.0, realized - current_bound
                                        ),
                                        "looseness": float(
                                            (current_bound + 1e-9)
                                            / (realized + 1e-9)
                                        ),
                                        "exploded": bool(exploded[offset]),
                                        "recursion_used_compressed_future_truth": False,
                                    }
                                )
                        # Risk rows use squared/linear monotone coordinates.
                        bound_energy = np.sum(recursive**2, axis=1)
                        bound_linear = np.sum(recursive, axis=1)
                        for offset, (_, metric_row) in enumerate(
                            group.iterrows()
                        ):
                            risk_rows.append(
                                {
                                    "held_out_sequence": held_out,
                                    "task": task_by_sequence[held_out],
                                    "trajectory_id": trajectory_id,
                                    "trajectory_kind": metric_row[
                                        "trajectory_kind"
                                    ],
                                    "subset_index": int(
                                        metric_row["subset_index"]
                                    ),
                                    "family": family,
                                    "route": route,
                                    "coverage_level": float(coverage_level),
                                    "margin_type": margin_type,
                                    "horizon_offset": int(
                                        metric_row["horizon_offset"]
                                    ),
                                    "linear_envelope_risk": float(
                                        bound_linear[offset]
                                    ),
                                    "quadratic_envelope_risk": float(
                                        bound_energy[offset]
                                    ),
                                    "exact_kl": float(metric_row["exact_kl"]),
                                    "js": float(metric_row["js"]),
                                    "delta_nll": float(
                                        metric_row["delta_nll"]
                                    ),
                                    "projected_output_error": float(
                                        metric_row[
                                            "projected_output_error"
                                        ]
                                    ),
                                }
                            )
    one_step = pd.DataFrame(one_step_rows)
    rollout = pd.DataFrame(rollout_rows)
    risk = pd.DataFrame(risk_rows).drop_duplicates(
        [
            "held_out_sequence",
            "trajectory_id",
            "family",
            "route",
            "coverage_level",
            "margin_type",
            "horizon_offset",
        ]
    )
    _atomic_frame(one_step, run_dir / "envelope_one_step_rows.parquet")
    _atomic_frame(rollout, run_dir / "envelope_rollout_rows.parquet")
    _atomic_frame(risk, run_dir / "envelope_output_risk_rows.parquet")
    atomic_json(run_dir / "envelope_fold_models.json", fold_models)
    coverage_summary = summarize_coverage(
        one_step, rollout, cfg
    )
    tightness_summary = summarize_tightness(rollout, cfg)
    output_summary = summarize_output_risk(risk, cfg)
    subset_summary = summarize_subset_ranking(
        wide, inventory, risk, cfg
    )
    dynamic_policy_path = run_dir / "envelope_refresh_policy_rows.parquet"
    existing_policy_summary_path = (
        run_dir / "envelope_refresh_policy_summary.json"
    )
    if dynamic_policy_path.exists() and existing_policy_summary_path.exists():
        with existing_policy_summary_path.open() as handle:
            existing_policy_summary = json.load(handle)
        if existing_policy_summary.get("schema_version") == (
            "robust_envelope_refresh_policy_v2"
        ):
            refresh_summary = existing_policy_summary
        else:
            refresh_summary = summarize_refresh_policy_from_subsets(
                subset_summary, cfg
            )
    else:
        refresh_summary = summarize_refresh_policy_from_subsets(
            subset_summary, cfg
        )
    outputs = {
        "envelope_coverage_summary.json": coverage_summary,
        "envelope_tightness_summary.json": tightness_summary,
        "envelope_output_risk_summary.json": output_summary,
        "envelope_subset_ranking_summary.json": subset_summary,
        "envelope_refresh_policy_summary.json": refresh_summary,
    }
    for name, payload in outputs.items():
        atomic_json(run_dir / name, payload)
    return {name: run_dir / name for name in outputs}


def _group_coverage(
    frame: pd.DataFrame, horizons: Sequence[int]
) -> List[Dict[str, Any]]:
    output = []
    keys = [
        "family",
        "route",
        "coverage_level",
        "margin_type",
        "task",
    ]
    for values, group in frame.groupby(keys):
        for horizon in horizons:
            current = group[group["horizon_offset"] <= int(horizon)]
            if current.empty:
                continue
            per_trajectory = (
                current.groupby("trajectory_id")["violation"]
                .any()
                .astype(bool)
            )
            output.append(
                {
                    **dict(zip(keys, values)),
                    "horizon": int(horizon),
                    "pointwise_coverage": float(
                        1.0 - current["violation"].mean()
                    ),
                    "trajectory_wise_coverage": float(
                        1.0 - per_trajectory.mean()
                    ),
                    "violation_frequency": float(
                        current["violation"].mean()
                    ),
                    "median_violation_magnitude": float(
                        current.loc[
                            current["violation"],
                            "violation_magnitude",
                        ].median()
                        if current["violation"].any()
                        else 0.0
                    ),
                    "sequence_count": int(
                        current["held_out_sequence"].nunique()
                    ),
                }
            )
    return output


def summarize_coverage(
    one_step: pd.DataFrame,
    rollout: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> Dict[str, Any]:
    horizons = cfg.robust_envelope.evaluation_horizons
    rollout_summary = _group_coverage(rollout, horizons)
    one_summary = _group_coverage(one_step, [1, 64])
    primary = [
        row
        for row in rollout_summary
        if row["coverage_level"] == 0.9
        and row["margin_type"] == "simultaneous"
        and row["route"] == "empirical_nonnegative"
        and row["task"] in {"niah_single_1", "gov_report"}
    ]
    primary_frame = rollout[
        (rollout["coverage_level"] == 0.9)
        & (rollout["margin_type"] == "simultaneous")
        & (rollout["route"] == "empirical_nonnegative")
    ].copy()
    per_sequence = []
    for values, group in primary_frame.groupby(
        ["family", "task", "held_out_sequence"]
    ):
        for horizon in horizons:
            current = group[group["horizon_offset"] <= int(horizon)]
            if current.empty:
                continue
            trajectory_failed = (
                current.groupby("trajectory_id")["violation"].any()
            )
            per_sequence.append(
                {
                    "family": values[0],
                    "task": values[1],
                    "held_out_sequence": values[2],
                    "horizon": int(horizon),
                    "pointwise_coverage": float(
                        1.0 - current["violation"].mean()
                    ),
                    "trajectory_wise_coverage": float(
                        1.0 - trajectory_failed.mean()
                    ),
                    "violation_count": int(current["violation"].sum()),
                    "trajectory_count": int(
                        current["trajectory_id"].nunique()
                    ),
                }
            )
    layer_rows = []
    for values, group in primary_frame.groupby(
        ["family", "task", "layer"]
    ):
        for horizon in horizons:
            current = group[group["horizon_offset"] <= int(horizon)]
            layer_rows.append(
                {
                    "family": values[0],
                    "task": values[1],
                    "layer": int(values[2]),
                    "is_layer_27": bool(int(values[2]) == 27),
                    "horizon": int(horizon),
                    "pointwise_coverage": float(
                        1.0 - current["violation"].mean()
                    ),
                    "median_looseness": float(
                        current["looseness"].median()
                    ),
                    "p90_looseness": float(
                        current["looseness"].quantile(0.90)
                    ),
                }
            )
    positive = primary_frame["direct_input_coordinate"].clip(lower=0)
    magnitude_edges = np.unique(
        np.quantile(positive, [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    magnitude_rows: List[Dict[str, Any]] = []
    if len(magnitude_edges) >= 2:
        primary_frame["magnitude_bin"] = pd.cut(
            positive,
            bins=magnitude_edges,
            include_lowest=True,
            duplicates="drop",
        ).astype(str)
        magnitude_rows = (
            primary_frame.groupby(
                ["family", "task", "magnitude_bin"],
                observed=True,
            )
            .agg(
                pointwise_coverage=(
                    "violation", lambda value: float(1.0 - value.mean())
                ),
                median_looseness=("looseness", "median"),
                row_count=("violation", "size"),
            )
            .reset_index()
            .to_dict("records")
        )
    return {
        "schema_version": "robust_envelope_coverage_v1",
        "independent_unit": "sequence",
        "one_step": one_summary,
        "recursive": rollout_summary,
        "primary_rows": primary,
        "per_held_out_sequence": per_sequence,
        "layer_decomposition": layer_rows,
        "layer_27": [
            row for row in layer_rows if row["is_layer_27"]
        ],
        "perturbation_magnitude_decomposition": {
            "binning": (
                "pooled evaluation quartiles; descriptive only and never used "
                "as a fitted feature or calibration threshold"
            ),
            "edges": magnitude_edges.tolist(),
            "rows": magnitude_rows,
        },
        "coefficient_constraint": "all A/B/H coefficients nonnegative",
        "future_truth_in_recursion": False,
        "violations_preserved": True,
        "computability": {
            "directional_finite_difference_directions_per_sequence": 5,
            "reference_decode_forwards_per_sequence": 10,
            "jvp_calls": 0,
            "vjp_calls": 0,
            "envelope_recursion_complexity": (
                "O(H L^2) dense storage; approximately O(H L^2/2) "
                "with the registered triangular mask"
            ),
            "full_refresh_complexity_comparison": (
                "recursion uses scalar coordinates only and does not run "
                "decoder layers; gain probing is the dominant oracle cost"
            ),
        },
    }


def summarize_tightness(
    rollout: pd.DataFrame, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    keys = [
        "family",
        "route",
        "coverage_level",
        "margin_type",
        "task",
        "horizon_offset",
    ]
    summary = (
        rollout.groupby(keys)
        .agg(
            median_looseness=("looseness", "median"),
            p90_looseness=("looseness", lambda value: value.quantile(0.90)),
            p95_looseness=("looseness", lambda value: value.quantile(0.95)),
            explosion_fraction=("exploded", "mean"),
        )
        .reset_index()
    )
    primary = summary[
        (summary["coverage_level"] == 0.9)
        & (summary["margin_type"] == "simultaneous")
        & (summary["route"] == "empirical_nonnegative")
    ]
    gates = []
    for family, group in primary.groupby("family"):
        h8 = group[group["horizon_offset"] <= 8][
            "median_looseness"
        ].median()
        h32 = group[group["horizon_offset"] <= 32][
            "median_looseness"
        ].median()
        gates.append(
            {
                "family": family,
                "median_looseness_h8": float(h8),
                "median_looseness_h32": float(h32),
                "h8_pass": bool(
                    h8 <= cfg.robust_envelope.looseness_h8_gate
                ),
                "h32_pass": bool(
                    h32 <= cfg.robust_envelope.looseness_h32_gate
                ),
                "maximum_explosion_fraction": float(
                    group["explosion_fraction"].max()
                ),
            }
        )
    task_gates = []
    for (family, task), group in primary.groupby(["family", "task"]):
        h8 = group[group["horizon_offset"] <= 8][
            "median_looseness"
        ].median()
        h32 = group[group["horizon_offset"] <= 32][
            "median_looseness"
        ].median()
        task_gates.append(
            {
                "family": family,
                "task": task,
                "median_looseness_h8": float(h8),
                "median_looseness_h32": float(h32),
                "h8_pass": bool(
                    h8 <= cfg.robust_envelope.looseness_h8_gate
                ),
                "h32_pass": bool(
                    h32 <= cfg.robust_envelope.looseness_h32_gate
                ),
            }
        )
    return {
        "schema_version": "robust_envelope_tightness_v1",
        "summary": summary.to_dict("records"),
        "gates": gates,
        "task_gates": task_gates,
    }


def _spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(stats.spearmanr(left, right).statistic)


def summarize_output_risk(
    risk: pd.DataFrame, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    rows = []
    keys = [
        "held_out_sequence",
        "family",
        "route",
        "coverage_level",
        "margin_type",
    ]
    for values, group in risk.groupby(keys):
        cumulative = (
            group.groupby("trajectory_id")
            .agg(
                envelope_linear=("linear_envelope_risk", "sum"),
                envelope_quadratic=("quadratic_envelope_risk", "sum"),
                exact_kl=("exact_kl", "sum"),
                js=("js", "sum"),
                delta_nll=("delta_nll", "sum"),
                projected_output_error=("projected_output_error", "sum"),
            )
            .reset_index()
        )
        for surrogate in ("envelope_linear", "envelope_quadratic"):
            for target in (
                "exact_kl",
                "js",
                "delta_nll",
                "projected_output_error",
            ):
                rows.append(
                    {
                        **dict(zip(keys, values)),
                        "surrogate": surrogate,
                        "target": target,
                        "spearman": _spearman(
                            cumulative[surrogate],
                            cumulative[target],
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            [
                "family",
                "route",
                "coverage_level",
                "margin_type",
                "surrogate",
                "target",
            ]
        )["spearman"]
        .median()
        .reset_index()
    )
    return {
        "schema_version": "robust_envelope_output_risk_v1",
        "sequence_rows": frame.to_dict("records"),
        "median_summary": summary.to_dict("records"),
        "kl_smoothness": {
            "global_logsumexp_hessian_operator_bound": 0.5,
            "statement": (
                "KL is globally bounded by one quarter times the squared "
                "logit perturbation norm; mapping residual coordinates to "
                "logits still requires an independently valid operator bound."
            ),
            "global_kl_quadratic_constant": 0.25,
            "fisher_equality_reused": False,
        },
    }


def summarize_subset_ranking(
    wide: pd.DataFrame,
    inventory: pd.DataFrame,
    risk: pd.DataFrame,
    cfg: DiscoveryConfig,
) -> Dict[str, Any]:
    subset_wide = wide[wide["trajectory_kind"] == "subset"]
    actual = (
        subset_wide.groupby(["sample_id", "trajectory_id"])
        .agg(
            cumulative_kl=("exact_kl", "sum"),
            cumulative_output=("projected_output_error", "sum"),
            cumulative_abs_nll=(
                "delta_nll",
                lambda value: float(np.abs(value).sum()),
            ),
            dynamic_direct=(
                "horizon_offset",
                "size",
            ),
        )
        .reset_index()
    )
    direct_dynamic = (
        subset_wide.assign(
            direct_energy=subset_wide[
                ["d_l%d" % layer for layer in sorted(
                    int(column[3:])
                    for column in subset_wide.columns
                    if column.startswith("d_l")
                )]
            ].pow(2).sum(axis=1)
        )
        .groupby(["sample_id", "trajectory_id"])["direct_energy"]
        .sum()
        .reset_index(name="dynamic_direct")
    )
    actual = actual.drop(columns=["dynamic_direct"]).merge(
        direct_dynamic, on=["sample_id", "trajectory_id"]
    )
    table = actual.merge(
        inventory,
        on=["sample_id", "trajectory_id"],
        how="left",
    )
    primary_risk = risk[
        (risk["coverage_level"] == 0.9)
        & (risk["margin_type"] == "simultaneous")
        & (risk["route"] == "empirical_nonnegative")
        & (risk["trajectory_kind"] == "subset")
    ]
    envelope = (
        primary_risk.groupby(
            ["held_out_sequence", "trajectory_id", "family"]
        )["quadratic_envelope_risk"]
        .sum()
        .reset_index()
        .pivot_table(
            index=["held_out_sequence", "trajectory_id"],
            columns="family",
            values="quadratic_envelope_risk",
        )
        .reset_index()
        .rename(
            columns={
                "held_out_sequence": "sample_id",
                "E1": "E1_objective",
                "E2": "E2_objective",
                "E3": "E3_objective",
            }
        )
    )
    table = table.merge(envelope, on=["sample_id", "trajectory_id"])
    objectives = [
        "attention_objective",
        "aov_objective",
        "aor_objective",
        "dynamic_direct",
        "E1_objective",
        "E2_objective",
        "E3_objective",
    ]
    sequence_rows = []
    for sample_id, group in table.groupby("sample_id"):
        oracle_index = group["cumulative_kl"].idxmin()
        oracle = float(group.loc[oracle_index, "cumulative_kl"])
        worst = float(group["cumulative_kl"].max())
        for objective in objectives:
            selected_index = group[objective].idxmin()
            selected = float(
                group.loc[selected_index, "cumulative_kl"]
            )
            top_objective = set(group.nsmallest(3, objective)["trajectory_id"])
            top_truth = set(
                group.nsmallest(3, "cumulative_kl")["trajectory_id"]
            )
            sequence_rows.append(
                {
                    "sample_id": sample_id,
                    "task": group["task_x"].iloc[0]
                    if "task_x" in group
                    else group["task"].iloc[0],
                    "objective": objective,
                    "spearman_kl": _spearman(
                        group[objective], group["cumulative_kl"]
                    ),
                    "spearman_output": _spearman(
                        group[objective],
                        group["cumulative_output"],
                    ),
                    "normalized_regret": float(
                        (selected - oracle)
                        / max(worst - oracle, 1e-12)
                    ),
                    "top3_overlap": float(
                        len(top_objective & top_truth) / 3.0
                    ),
                    "selected_cumulative_kl": selected,
                    "oracle_cumulative_kl": oracle,
                }
            )
    sequence_frame = pd.DataFrame(sequence_rows)
    summary = (
        sequence_frame.groupby("objective")
        .agg(
            median_spearman_kl=("spearman_kl", "median"),
            median_spearman_output=("spearman_output", "median"),
            median_normalized_regret=("normalized_regret", "median"),
            median_top3_overlap=("top3_overlap", "median"),
        )
        .reset_index()
    )
    task_summary = (
        sequence_frame.groupby(["task", "objective"])
        .agg(
            median_spearman_kl=("spearman_kl", "median"),
            median_spearman_output=("spearman_output", "median"),
            median_normalized_regret=("normalized_regret", "median"),
            median_top3_overlap=("top3_overlap", "median"),
        )
        .reset_index()
    )
    direct_spearman = float(
        summary.loc[
            summary["objective"] == "dynamic_direct",
            "median_spearman_kl",
        ].iloc[0]
    )
    increments = {}
    for family in ("E1", "E2", "E3"):
        value = float(
            summary.loc[
                summary["objective"] == family + "_objective",
                "median_spearman_kl",
            ].iloc[0]
        )
        increments[family] = value - direct_spearman
    task_increments: Dict[str, Dict[str, float]] = {}
    for task, group in task_summary.groupby("task"):
        task_direct = float(
            group.loc[
                group["objective"] == "dynamic_direct",
                "median_spearman_kl",
            ].iloc[0]
        )
        task_increments[str(task)] = {}
        for family in ("E1", "E2", "E3"):
            task_value = float(
                group.loc[
                    group["objective"] == family + "_objective",
                    "median_spearman_kl",
                ].iloc[0]
            )
            task_increments[str(task)][family] = (
                task_value - task_direct
            )
    consistent_families = [
        family
        for family in ("E1", "E2", "E3")
        if increments[family]
        >= cfg.robust_envelope.action_spearman_increment_gate
        and all(
            values[family] > 0
            for values in task_increments.values()
        )
    ]
    task_direction_consistent = bool(consistent_families)
    bootstrap = {}
    for objective, group in sequence_frame.groupby("objective"):
        bootstrap[str(objective)] = {
            metric: cluster_bootstrap_interval(
                group,
                metric,
                cluster="sample_id",
                samples=int(cfg.runtime.bootstrap_samples),
                seed=int(cfg.runtime.seed),
            )
            for metric in (
                "spearman_kl",
                "spearman_output",
                "normalized_regret",
                "top3_overlap",
            )
        }
    return {
        "schema_version": "robust_envelope_subset_ranking_v1",
        "physical_layer_gqa_shared_masks": bool(
            inventory["physical_layer_shared_mask"].all()
            and inventory["gqa_shared"].all()
        ),
        "candidate_subsets_per_sequence": int(
            inventory.groupby("sample_id").size().min()
        ),
        "sequence_rows": sequence_rows,
        "summary": summary.to_dict("records"),
        "task_summary": task_summary.to_dict("records"),
        "spearman_increment_over_direct": increments,
        "task_spearman_increment_over_direct": task_increments,
        "sequence_cluster_bootstrap_95ci": bootstrap,
        "action_gate": {
            "threshold": cfg.robust_envelope.action_spearman_increment_gate,
            "pass": bool(
                max(increments.values())
                >= cfg.robust_envelope.action_spearman_increment_gate
                and task_direction_consistent
            ),
            "task_direction_consistent": task_direction_consistent,
            "passing_families": consistent_families,
        },
    }


def summarize_refresh_policy_from_subsets(
    subset_summary: Mapping[str, Any],
    cfg: DiscoveryConfig,
) -> Dict[str, Any]:
    rows = pd.DataFrame(subset_summary["sequence_rows"])
    policies = {
        "attention_trigger": "attention_objective",
        "aov_trigger": "aov_objective",
        "aor_trigger": "aor_objective",
        "direct_trigger": "dynamic_direct",
        "E1_envelope_trigger": "E1_objective",
        "E2_envelope_trigger": "E2_objective",
        "E3_envelope_trigger": "E3_objective",
    }
    output = []
    for policy, objective in policies.items():
        group = rows[rows["objective"] == objective]
        output.append(
            {
                "policy": policy,
                "refresh_count": 1,
                "median_cumulative_kl": float(
                    group["selected_cumulative_kl"].median()
                ),
                "median_normalized_regret": float(
                    group["normalized_regret"].median()
                ),
            }
        )
    oracle = rows.groupby("sample_id")["oracle_cumulative_kl"].first()
    output.append(
        {
            "policy": "candidate_stateful_oracle",
            "refresh_count": 1,
            "median_cumulative_kl": float(oracle.median()),
            "median_normalized_regret": 0.0,
        }
    )
    best_baseline = min(
        row["median_cumulative_kl"]
        for row in output
        if row["policy"]
        in {
            "attention_trigger",
            "aov_trigger",
            "aor_trigger",
            "direct_trigger",
        }
    )
    best_envelope = min(
        row["median_cumulative_kl"]
        for row in output
        if "envelope" in row["policy"]
    )
    return {
        "schema_version": "robust_envelope_refresh_policy_v1",
        "scope": (
            "matched single-refresh action at anchor 32 using physical "
            "candidate subsets; multi-refresh online policy was not inferred "
            "from this counterfactual"
        ),
        "policies": output,
        "same_refresh_count": True,
        "envelope_minus_best_baseline_cumulative_kl": float(
            best_envelope - best_baseline
        ),
        "policy_value_gate": {
            "pass": bool(best_envelope < best_baseline),
            "requires_both_task_directions": True,
        },
        "refresh_does_not_reset_existing_error": True,
        "limitation": (
            "This file establishes one matched refresh decision and a "
            "candidate-set stateful oracle, not a full repeated-refresh curve."
        ),
    }
