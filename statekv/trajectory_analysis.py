"""Sequence-cluster analysis for trajectory stochastic-model experiments."""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import subspace_angles
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from statekv.config import DiscoveryConfig


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_native(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                _native(dict(payload)),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cluster_bootstrap_interval(
    frame: pd.DataFrame,
    value: str,
    cluster: str = "sample_id",
    samples: int = 2000,
    seed: int = 42,
    statistic: str = "median",
) -> Dict[str, float]:
    grouped = frame.groupby(cluster, sort=True)[value].agg(statistic)
    values = grouped.to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "clusters": 0,
        }
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[index] = (
            float(np.median(sample))
            if statistic == "median"
            else float(np.mean(sample))
        )
    estimate = (
        float(np.median(values))
        if statistic == "median"
        else float(np.mean(values))
    )
    return {
        "estimate": estimate,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "clusters": int(len(values)),
    }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.sum((true - true.mean(axis=0)) ** 2))
    return 1.0 - float(np.sum((true - pred) ** 2)) / max(
        denominator, 1e-30
    )


def _origin_r2(actual: Sequence[np.ndarray], predicted: Sequence[np.ndarray]) -> float:
    numerator = sum(
        float(np.sum((left - right) ** 2))
        for left, right in zip(actual, predicted)
    )
    denominator = sum(float(np.sum(left**2)) for left in actual)
    return 1.0 - numerator / max(denominator, 1e-30)


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3 or np.unique(left[finite]).size < 2 or np.unique(
        right[finite]
    ).size < 2:
        return float("nan")
    return float(stats.spearmanr(left[finite], right[finite]).statistic)


def _npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def analyze_scaling_superposition(
    run_dir: Path, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    inventory = pd.read_parquet(
        run_dir / "trajectory_intervention_inventory.parquet"
    )
    scaling = inventory[
        inventory["intervention_type"].isin(
            ["scaling_single", "scaling_multi"]
        )
    ].copy()
    rows: List[Dict[str, Any]] = []
    modalities = (
        "layer_output_drift",
        "residual_drift",
        "query_drift",
        "projected_attention_output_drift",
        "logit_top128_drift",
    )
    for (_, unit), group in scaling.groupby(
        ["sample_id", "unit_key"], sort=True
    ):
        by_beta = {
            float(row.beta): _npz(str(row.state_npz_path))
            for row in group.itertuples()
        }
        if 1.0 not in by_beta:
            continue
        metadata = group.iloc[0]
        positive_betas = sorted(
            beta for beta in by_beta if 0.0 < beta <= 1.5
        )
        for modality in modalities:
            reference = by_beta[1.0][modality].astype(np.float64)
            horizon = int(reference.shape[0])
            for offset in range(horizon):
                actual = [
                    by_beta[beta][modality][offset].astype(np.float64)
                    for beta in positive_betas
                ]
                predicted = [
                    beta * reference[offset] for beta in positive_betas
                ]
                relative = [
                    float(
                        np.linalg.norm(left - right)
                        / max(np.linalg.norm(right), 1e-12)
                    )
                    for left, right in zip(actual, predicted)
                    if not np.isclose(
                        np.linalg.norm(right), 0.0, atol=1e-12
                    )
                ]
                rows.append(
                    {
                        "sample_id": metadata.sample_id,
                        "task": metadata.task,
                        "unit_key": unit,
                        "intervention_type": metadata.intervention_type,
                        "anchor": int(metadata.anchor),
                        "protected_recent_size": int(
                            metadata.protected_recent_size
                        ),
                        "mask_type": metadata.mask_type,
                        "modality": modality,
                        "layer": -1,
                        "horizon_offset": offset + 1,
                        "origin_r2": _origin_r2(actual, predicted),
                        "median_relative_deviation": (
                            float(np.median(relative))
                            if relative
                            else 0.0
                        ),
                    }
                )
                if (
                    modality == "layer_output_drift"
                    and reference.ndim == 3
                ):
                    layers = by_beta[1.0]["selected_layers"].tolist()
                    for layer_index, layer in enumerate(layers):
                        layer_actual = [
                            by_beta[beta][modality][
                                offset, layer_index
                            ].astype(np.float64)
                            for beta in positive_betas
                        ]
                        layer_prediction = [
                            beta
                            * reference[offset, layer_index].astype(
                                np.float64
                            )
                            for beta in positive_betas
                        ]
                        deviations = [
                            float(
                                np.linalg.norm(left - right)
                                / max(np.linalg.norm(right), 1e-12)
                            )
                            for left, right in zip(
                                layer_actual, layer_prediction
                            )
                            if np.linalg.norm(right) > 1e-12
                        ]
                        rows.append(
                            {
                                "sample_id": metadata.sample_id,
                                "task": metadata.task,
                                "unit_key": unit,
                                "intervention_type": metadata.intervention_type,
                                "anchor": int(metadata.anchor),
                                "protected_recent_size": int(
                                    metadata.protected_recent_size
                                ),
                                "mask_type": metadata.mask_type,
                                "modality": modality,
                                "layer": int(layer),
                                "horizon_offset": offset + 1,
                                "origin_r2": _origin_r2(
                                    layer_actual, layer_prediction
                                ),
                                "median_relative_deviation": (
                                    float(np.median(deviations))
                                    if deviations
                                    else 0.0
                                ),
                            }
                        )
    scaling_rows = pd.DataFrame(rows)
    scaling_rows.to_parquet(
        run_dir / "scaling_diagnostics.parquet", index=False
    )

    aggregate_rows = scaling_rows[
        (scaling_rows["layer"] == -1)
        & (scaling_rows["modality"] == "layer_output_drift")
    ]
    horizon_summary = (
        aggregate_rows.groupby("horizon_offset", sort=True)
        .agg(
            median_origin_r2=("origin_r2", "median"),
            median_relative_deviation=(
                "median_relative_deviation",
                "median",
            ),
        )
        .reset_index()
    )
    valid = horizon_summary[
        (horizon_summary["median_origin_r2"] >= cfg.trajectory_model.scaling_r2_gate)
        & (
            horizon_summary["median_relative_deviation"]
            <= cfg.trajectory_model.scaling_relative_tolerance
        )
    ]
    valid_horizon = int(valid["horizon_offset"].max()) if len(valid) else 0

    radius_rows: List[Dict[str, Any]] = []
    for (_, unit), group in scaling.groupby(
        ["sample_id", "unit_key"], sort=True
    ):
        by_beta = {
            float(row.beta): _npz(str(row.state_npz_path))[
                "layer_output_drift"
            ].astype(np.float64)
            for row in group.itertuples()
        }
        reference = by_beta.get(1.0)
        if reference is None:
            continue
        for beta, actual in by_beta.items():
            if beta <= 0:
                continue
            deviations = np.linalg.norm(
                (actual - beta * reference).reshape(len(actual), -1), axis=1
            ) / np.maximum(
                np.linalg.norm(
                    (beta * reference).reshape(len(reference), -1), axis=1
                ),
                1e-12,
            )
            radius_rows.append(
                {
                    "sample_id": group.iloc[0].sample_id,
                    "beta": beta,
                    "median_relative_deviation": float(
                        np.median(deviations)
                    ),
                }
            )
    radius_frame = pd.DataFrame(radius_rows)
    radius = (
        radius_frame.groupby("beta")["median_relative_deviation"]
        .median()
        .reset_index()
    )
    valid_radius_rows = radius[
        radius["median_relative_deviation"]
        <= cfg.trajectory_model.scaling_relative_tolerance
    ]
    valid_radius = (
        float(valid_radius_rows["beta"].max())
        if len(valid_radius_rows)
        else 0.0
    )

    super_rows: List[Dict[str, Any]] = []
    superposition = inventory[
        inventory["intervention_type"] == "superposition"
    ]
    for (_, unit), group in superposition.groupby(
        ["sample_id", "unit_key"], sort=True
    ):
        by_arm = {
            str(row.superposition_arm): _npz(str(row.state_npz_path))
            for row in group.itertuples()
        }
        if set(by_arm) != {"D1", "D2", "union"}:
            continue
        metadata = group.iloc[0]
        for modality in (
            "layer_output_drift",
            "projected_attention_output_drift",
            "logit_top128_drift",
        ):
            union = by_arm["union"][modality].astype(np.float64)
            additive = (
                by_arm["D1"][modality].astype(np.float64)
                + by_arm["D2"][modality].astype(np.float64)
            )
            for offset in range(len(union)):
                super_rows.append(
                    {
                        "sample_id": metadata.sample_id,
                        "task": metadata.task,
                        "unit_key": unit,
                        "category": metadata.superposition_category,
                        "modality": modality,
                        "horizon_offset": offset + 1,
                        "relative_superposition_error": float(
                            np.linalg.norm(union[offset] - additive[offset])
                            / max(np.linalg.norm(union[offset]), 1e-12)
                        ),
                        "direct_cross_interaction_relative": float(
                            metadata.direct_cross_interaction_relative
                        ),
                    }
                )
    super_frame = pd.DataFrame(super_rows)
    super_frame.to_parquet(
        run_dir / "superposition_diagnostics.parquet", index=False
    )

    breakdown_columns = [
        "task",
        "intervention_type",
        "protected_recent_size",
        "mask_type",
        "layer",
    ]
    breakdown = (
        scaling_rows[
            scaling_rows["modality"] == "layer_output_drift"
        ]
        .groupby(breakdown_columns, dropna=False)
        .agg(
            median_origin_r2=("origin_r2", "median"),
            median_relative_deviation=(
                "median_relative_deviation",
                "median",
            ),
            rows=("origin_r2", "size"),
        )
        .reset_index()
        .to_dict("records")
    )
    beta0_max = float(
        inventory.loc[
            inventory["beta"] == 0, "beta0_max_abs_drift"
        ].max()
    )
    beta1_error = float(
        inventory.loc[
            inventory["intervention_type"] == "scaling_single",
            "beta1_injection_l2_error",
        ]
        .dropna()
        .max()
    )
    super_aggregate = super_frame[
        super_frame["modality"] == "layer_output_drift"
    ]
    return {
        "schema_version": "trajectory_scaling_superposition_v1",
        "independent_unit": "sequence",
        "beta0_exact_identity": {
            "maximum_drift": beta0_max,
            "pass": bool(beta0_max == 0.0),
            "implementation": "immutable_full_reference_exact_alias",
        },
        "beta1_exact_branch_equivalence": {
            "maximum_single_layer_l2_error": beta1_error,
            "tolerance": 1e-5,
            "pass": bool(beta1_error <= 1e-5),
        },
        "scaling": {
            "aggregate_cluster_bootstrap": {
                "origin_r2": cluster_bootstrap_interval(
                    aggregate_rows,
                    "origin_r2",
                    samples=cfg.runtime.bootstrap_samples,
                    seed=cfg.runtime.seed,
                ),
                "relative_deviation": cluster_bootstrap_interval(
                    aggregate_rows,
                    "median_relative_deviation",
                    samples=cfg.runtime.bootstrap_samples,
                    seed=cfg.runtime.seed,
                ),
            },
            "validity_gate": {
                "r2_gate": cfg.trajectory_model.scaling_r2_gate,
                "relative_tolerance": (
                    cfg.trajectory_model.scaling_relative_tolerance
                ),
                "maximum_valid_beta": valid_radius,
                "maximum_valid_horizon": valid_horizon,
            },
            "horizon_summary": horizon_summary.to_dict("records"),
            "beta_radius_summary": radius.to_dict("records"),
            "breakdown": breakdown,
        },
        "superposition": {
            "aggregate_cluster_bootstrap": cluster_bootstrap_interval(
                super_aggregate,
                "relative_superposition_error",
                samples=cfg.runtime.bootstrap_samples,
                seed=cfg.runtime.seed,
            ),
            "tolerance": cfg.trajectory_model.superposition_relative_tolerance,
            "by_category": (
                super_aggregate.groupby("category")
                .agg(
                    median_relative_error=(
                        "relative_superposition_error",
                        "median",
                    ),
                    median_direct_cross_interaction=(
                        "direct_cross_interaction_relative",
                        "median",
                    ),
                )
                .reset_index()
                .to_dict("records")
            ),
            "by_horizon": (
                super_aggregate.groupby("horizon_offset")
                ["relative_superposition_error"]
                .median()
                .reset_index()
                .to_dict("records")
            ),
        },
    }


def _latent_records(
    inventory: pd.DataFrame,
    modality: str,
    horizons: Sequence[int],
) -> Tuple[np.ndarray, pd.DataFrame]:
    eligible = inventory[
        (
            inventory["intervention_type"].isin(
                ["scaling_single", "scaling_multi", "superposition"]
            )
            & (inventory["beta"] == 1.0)
        )
        | (inventory["intervention_type"] == "hybrid_stateful")
    ]
    vectors: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for row in eligible.itertuples():
        arrays = _npz(str(row.state_npz_path))
        if modality == "new_kv_concat":
            values = np.concatenate(
                [arrays["new_key_drift"], arrays["new_value_drift"]],
                axis=2,
            )
        elif modality == "multi_layer_concat":
            values = arrays["layer_output_drift"]
        else:
            values = arrays[modality]
        for horizon in horizons:
            vector = values[int(horizon) - 1].reshape(-1)
            vectors.append(vector.astype(np.float32))
            metadata.append(
                {
                    "sample_id": row.sample_id,
                    "task": row.task,
                    "protected_recent_size": int(
                        row.protected_recent_size
                    ),
                    "horizon_offset": int(horizon),
                    "trajectory_id": row.trajectory_id,
                }
            )
    return np.stack(vectors), pd.DataFrame(metadata)


def _fit_pca_basis(
    values: np.ndarray, components: int, seed: int
) -> PCA:
    count = min(
        int(components),
        int(values.shape[0]) - 1,
        int(values.shape[1]),
    )
    return PCA(
        n_components=max(1, count),
        svd_solver="randomized",
        random_state=int(seed),
        iterated_power=4,
    ).fit(values)


def analyze_latent_rank(
    run_dir: Path, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    inventory = pd.read_parquet(
        run_dir / "trajectory_intervention_inventory.parquet"
    )
    horizons = [1, 2, 4, 8, 16, 32, 64]
    modalities = (
        "residual_drift",
        "query_drift",
        "new_kv_concat",
        "projected_attention_output_drift",
        "multi_layer_concat",
    )
    rank_rows: List[Dict[str, Any]] = []
    stability: Dict[str, Any] = {}
    task_angles: Dict[str, Any] = {}
    maximum_rank = max(int(value) for value in cfg.trajectory_model.latent_ranks)
    for modality in modalities:
        values, metadata = _latent_records(
            inventory, modality, horizons
        )
        sequences = sorted(metadata["sample_id"].unique())
        fold_bases: List[np.ndarray] = []
        for held_out in sequences:
            train = metadata["sample_id"].to_numpy() != held_out
            test = ~train
            pca = _fit_pca_basis(
                values[train], maximum_rank, cfg.runtime.seed
            )
            fold_bases.append(pca.components_)
            transformed = pca.transform(values[test])
            centered = values[test] - pca.mean_
            denominator = float(np.sum(centered**2))
            cumulative_train = np.cumsum(pca.explained_variance_ratio_)
            for rank in cfg.trajectory_model.latent_ranks:
                use = min(int(rank), int(pca.n_components_))
                reconstruction = (
                    transformed[:, :use] @ pca.components_[:use]
                )
                heldout = 1.0 - float(
                    np.sum((centered - reconstruction) ** 2)
                ) / max(denominator, 1e-30)
                rank_rows.append(
                    {
                        "modality": modality,
                        "held_out_sequence": held_out,
                        "held_out_task": metadata.loc[
                            test, "task"
                        ].iloc[0],
                        "rank": int(rank),
                        "train_explained_variance": float(
                            cumulative_train[use - 1]
                        ),
                        "heldout_reconstruction_fraction": heldout,
                    }
                )
        angles = []
        use = min(8, *(basis.shape[0] for basis in fold_bases))
        for left in range(len(fold_bases)):
            for right in range(left + 1, len(fold_bases)):
                current = np.degrees(
                    subspace_angles(
                        fold_bases[left][:use].T,
                        fold_bases[right][:use].T,
                    )
                )
                angles.extend(current.tolist())
        stability[modality] = {
            "rank": int(use),
            "mean_principal_angle_deg": float(np.mean(angles)),
            "maximum_principal_angle_deg": float(np.max(angles)),
        }
        task_bases = {}
        for task in sorted(metadata["task"].unique()):
            task_bases[task] = _fit_pca_basis(
                values[metadata["task"].to_numpy() == task],
                min(8, maximum_rank),
                cfg.runtime.seed,
            ).components_
        if len(task_bases) == 2:
            names = sorted(task_bases)
            use_task = min(
                8,
                task_bases[names[0]].shape[0],
                task_bases[names[1]].shape[0],
            )
            angles_task = np.degrees(
                subspace_angles(
                    task_bases[names[0]][:use_task].T,
                    task_bases[names[1]][:use_task].T,
                )
            )
            task_angles[modality] = {
                "tasks": names,
                "rank": int(use_task),
                "mean_principal_angle_deg": float(
                    np.mean(angles_task)
                ),
                "maximum_principal_angle_deg": float(
                    np.max(angles_task)
                ),
            }
    rank_frame = pd.DataFrame(rank_rows)
    rank_frame.to_parquet(
        run_dir / "latent_rank_loso_rows.parquet", index=False
    )
    aggregate = (
        rank_frame.groupby(["modality", "rank"])
        .agg(
            median_train_explained=(
                "train_explained_variance",
                "median",
            ),
            median_heldout_reconstruction=(
                "heldout_reconstruction_fraction",
                "median",
            ),
            minimum_heldout_reconstruction=(
                "heldout_reconstruction_fraction",
                "min",
            ),
        )
        .reset_index()
    )
    dimension90 = {}
    for modality, group in aggregate.groupby("modality"):
        passing = group[
            group["median_heldout_reconstruction"]
            >= cfg.trajectory_model.latent_variance_gate
        ]
        dimension90[modality] = (
            int(passing["rank"].min()) if len(passing) else ">%d" % maximum_rank
        )

    # Descriptive layer-27 separation uses equal-dimensional layer outputs.
    sampled = inventory[
        (inventory["intervention_type"] == "scaling_single")
        & (inventory["beta"] == 1.0)
    ]
    layer_values: Dict[int, List[np.ndarray]] = defaultdict(list)
    for row in sampled.itertuples():
        arrays = _npz(str(row.state_npz_path))
        for layer_index, layer in enumerate(
            arrays["selected_layers"].tolist()
        ):
            layer_values[int(layer)].append(
                arrays["layer_output_drift"][
                    [0, 7, 15, 31, 63], layer_index
                ].astype(np.float32)
            )
    layer27_angles = []
    if 27 in layer_values:
        basis27 = _fit_pca_basis(
            np.concatenate(layer_values[27]), 8, cfg.runtime.seed
        ).components_
        for layer in sorted(layer_values):
            if layer == 27:
                continue
            basis = _fit_pca_basis(
                np.concatenate(layer_values[layer]), 8, cfg.runtime.seed
            ).components_
            angle = np.degrees(
                subspace_angles(basis27.T, basis.T)
            )
            layer27_angles.append(
                {
                    "other_layer": int(layer),
                    "mean_principal_angle_deg": float(np.mean(angle)),
                    "maximum_principal_angle_deg": float(np.max(angle)),
                }
            )
    concat_rows = aggregate[
        aggregate["modality"] == "multi_layer_concat"
    ]
    low_dimensional = bool(
        len(
            concat_rows[
                concat_rows["median_heldout_reconstruction"]
                >= cfg.trajectory_model.latent_variance_gate
            ]
        )
    )
    return {
        "schema_version": "trajectory_latent_rank_v1",
        "independent_unit": "sequence",
        "pca_fit_scope": "training_sequences_only_in_each_LOSO_fold",
        "sampled_horizons": horizons,
        "rank_summary": aggregate.to_dict("records"),
        "dimension_for_90pct_heldout": dimension90,
        "fold_basis_stability": stability,
        "task_subspace_angles": task_angles,
        "layer27_vs_other_principal_angles": layer27_angles,
        "gate": {
            "variance_target": cfg.trajectory_model.latent_variance_gate,
            "maximum_tested_rank": maximum_rank,
            "multi_layer_low_dimensional_pass": low_dimensional,
        },
    }


NORM_FIELDS = (
    "projected_attention_output_l2",
    "residual_l2",
    "query_l2",
    "new_key_l2",
    "new_value_l2",
    "attention_input_l2",
    "layer_output_l2",
)


def build_state_frame(run_dir: Path) -> Tuple[pd.DataFrame, List[int]]:
    rows = pd.read_parquet(run_dir / "trajectory_state_rows.parquet")
    rows = rows[rows["beta"] > 0].copy()
    layers = sorted(int(value) for value in rows["layer"].unique())
    keys = [
        "sample_id",
        "task",
        "trajectory_id",
        "intervention_type",
        "anchor",
        "protected_recent_size",
        "mask_type",
        "beta",
        "direct_input_l2",
        "horizon_offset",
    ]
    wide: Optional[pd.DataFrame] = None
    for field in NORM_FIELDS:
        current = rows.pivot_table(
            index=keys,
            columns="layer",
            values=field,
            aggfunc="first",
        )
        current.columns = ["%s_l%d" % (field, layer) for layer in current.columns]
        current = current.reset_index()
        wide = current if wide is None else wide.merge(current, on=keys)
    metric = (
        rows.groupby(keys, as_index=False)
        .first()[
            keys
            + [
                "exact_kl",
                "js",
                "delta_nll",
                "fisher_quadratic",
                "logit_l2_sq",
                "post_recent_exit",
            ]
        ]
    )
    assert wide is not None
    return wide.merge(metric, on=keys), layers


def _candidate_columns(
    frame: pd.DataFrame, layers: Sequence[int]
) -> Dict[str, List[str]]:
    def columns(fields: Sequence[str]) -> List[str]:
        return [
            "%s_l%d" % (field, layer)
            for field in fields
            for layer in layers
        ]

    return {
        "A_direct_error": columns(["projected_attention_output_l2"]),
        "B_residual": columns(["residual_l2"]),
        "C_residual_query": columns(["residual_l2", "query_l2"]),
        "D_residual_query_newkv": columns(
            ["residual_l2", "query_l2", "new_key_l2", "new_value_l2"]
        ),
        "E_plus_regime": columns(
            [
                "residual_l2",
                "query_l2",
                "new_key_l2",
                "new_value_l2",
                "attention_input_l2",
                "layer_output_l2",
            ]
        ),
    }


def _transition_arrays(
    frame: pd.DataFrame, columns: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    left: List[np.ndarray] = []
    right: List[np.ndarray] = []
    metadata: List[Dict[str, Any]] = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=False):
        group = group.sort_values("horizon_offset")
        values = np.log1p(
            group[list(columns)].to_numpy(dtype=np.float64)
        )
        if len(values) < 2:
            continue
        left.append(values[:-1])
        right.append(values[1:])
        for row in group.iloc[1:].itertuples():
            metadata.append(
                {
                    "sample_id": row.sample_id,
                    "task": row.task,
                    "trajectory_id": trajectory_id,
                    "intervention_type": row.intervention_type,
                    "horizon_offset": int(row.horizon_offset),
                    "protected_recent_size": int(
                        row.protected_recent_size
                    ),
                    "direct_input_l2": float(row.direct_input_l2),
                    "post_recent_exit": bool(row.post_recent_exit),
                }
            )
    return np.concatenate(left), np.concatenate(right), pd.DataFrame(metadata)


def _autocorrelation(
    residual: np.ndarray, metadata: pd.DataFrame
) -> float:
    values = []
    for _, indices in metadata.groupby("trajectory_id").groups.items():
        ordered = np.asarray(sorted(indices))
        if len(ordered) < 2:
            continue
        first = residual[ordered[:-1]].reshape(-1)
        second = residual[ordered[1:]].reshape(-1)
        if np.std(first) > 0 and np.std(second) > 0:
            values.append(float(np.corrcoef(first, second)[0, 1]))
    return float(np.median(values)) if values else float("nan")


def _fit_transition_fold(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    alpha: float,
) -> Dict[str, Any]:
    x_scaler = StandardScaler().fit(x[train])
    y_scaler = StandardScaler().fit(y[train])
    train_x = x_scaler.transform(x[train])
    train_y = y_scaler.transform(y[train])
    test_x = x_scaler.transform(x[test])
    test_y = y_scaler.transform(y[test])
    model = Ridge(alpha=float(alpha)).fit(train_x, train_y)
    prediction = model.predict(test_x)
    train_residual = train_y - model.predict(train_x)
    variance = np.var(train_residual, axis=0) + 1e-6
    nll = 0.5 * np.mean(
        np.sum(
            np.log(2.0 * np.pi * variance)
            + (test_y - prediction) ** 2 / variance,
            axis=1,
        )
    )
    coefficient = np.asarray(model.coef_)
    raw_coefficient = (
        np.diag(y_scaler.scale_)
        @ coefficient
        @ np.diag(1.0 / x_scaler.scale_)
    )
    spectral_radius = (
        float(np.max(np.abs(np.linalg.eigvals(raw_coefficient))))
        if raw_coefficient.shape[0] == raw_coefficient.shape[1]
        else float("nan")
    )
    return {
        "r2": _r2(test_y, prediction),
        "nll": float(nll),
        "spectral_radius": spectral_radius,
        "prediction": prediction,
        "truth": test_y,
        "residual": test_y - prediction,
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }


def analyze_markov(
    run_dir: Path, cfg: DiscoveryConfig
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    complete_frame, layers = build_state_frame(run_dir)
    # The six real recent-FIFO trajectories are a separate mechanism arm with
    # a much larger persistent cache intervention.  Pooling them with pulse
    # trajectories makes variance-weighted R2 look artificially excellent.
    # Primary Markov selection therefore uses pulse trajectories; the real
    # recent-exit arm is evaluated separately below.
    frame = complete_frame[
        complete_frame["intervention_type"]
        != "recent_exit_stateful"
    ].copy()
    candidate_columns = _candidate_columns(frame, layers)
    sequence_ids = sorted(frame["sample_id"].unique())
    rows: List[Dict[str, Any]] = []
    residual_cache: Dict[str, List[Tuple[np.ndarray, pd.DataFrame]]] = defaultdict(list)
    for candidate, columns in candidate_columns.items():
        x, y, metadata = _transition_arrays(frame, columns)
        for held_out in sequence_ids:
            train = metadata["sample_id"].to_numpy() != held_out
            test = ~train
            result = _fit_transition_fold(
                x, y, train, test, cfg.trajectory_model.ridge_alpha
            )
            # Same-target history increment, evaluated only where t-1 exists.
            history_x: List[np.ndarray] = []
            history_y: List[np.ndarray] = []
            history_meta: List[Dict[str, Any]] = []
            for trajectory_id, group in frame.groupby(
                "trajectory_id", sort=False
            ):
                group = group.sort_values("horizon_offset")
                values = np.log1p(
                    group[columns].to_numpy(dtype=np.float64)
                )
                if len(values) < 3:
                    continue
                history_x.append(
                    np.concatenate([values[1:-1], values[:-2]], axis=1)
                )
                history_y.append(values[2:])
                for row in group.iloc[2:].itertuples():
                    history_meta.append(
                        {
                            "sample_id": row.sample_id,
                            "trajectory_id": trajectory_id,
                        }
                    )
            hx = np.concatenate(history_x)
            hy = np.concatenate(history_y)
            hm = pd.DataFrame(history_meta)
            htrain = hm["sample_id"].to_numpy() != held_out
            htest = ~htrain
            augmented = _fit_transition_fold(
                hx,
                hy,
                htrain,
                htest,
                cfg.trajectory_model.ridge_alpha,
            )
            base_history = _fit_transition_fold(
                hx[:, : len(columns)],
                hy,
                htrain,
                htest,
                cfg.trajectory_model.ridge_alpha,
            )

            # Recursive rollout starts from the first held-out observation.
            rollout_truth: List[np.ndarray] = []
            rollout_prediction: List[np.ndarray] = []
            by_trajectory = frame[
                frame["sample_id"] == held_out
            ].groupby("trajectory_id", sort=False)
            for _, group in by_trajectory:
                group = group.sort_values("horizon_offset")
                raw = np.log1p(
                    group[columns].to_numpy(dtype=np.float64)
                )
                if len(raw) < 2:
                    continue
                current_raw = raw[0].copy()
                for index in range(1, len(raw)):
                    current_x = result["x_scaler"].transform(
                        current_raw.reshape(1, -1)
                    )[0]
                    prediction_y = result["model"].predict(
                        current_x.reshape(1, -1)
                    )[0]
                    current_raw = result[
                        "y_scaler"
                    ].inverse_transform(
                        prediction_y.reshape(1, -1)
                    )[0]
                    if int(group.iloc[index]["horizon_offset"]) in {
                        8,
                        16,
                        32,
                        64,
                    }:
                        rollout_prediction.append(prediction_y.copy())
                        rollout_truth.append(
                            result["y_scaler"].transform(
                                raw[index : index + 1]
                            )[0]
                        )
            rollout_r2 = (
                _r2(
                    np.asarray(rollout_truth),
                    np.asarray(rollout_prediction),
                )
                if rollout_truth
                else float("nan")
            )
            rows.append(
                {
                    "candidate": candidate,
                    "held_out_sequence": held_out,
                    "held_out_task": metadata.loc[
                        test, "task"
                    ].iloc[0],
                    "one_step_r2": result["r2"],
                    "prediction_nll": result["nll"],
                    "residual_autocorrelation": _autocorrelation(
                        result["residual"],
                        metadata.loc[test].reset_index(drop=True),
                    ),
                    "history_base_r2": base_history["r2"],
                    "history_augmented_r2": augmented["r2"],
                    "history_delta_r2": (
                        augmented["r2"] - base_history["r2"]
                    ),
                    "rollout_r2_8_16_32_64": rollout_r2,
                    "spectral_radius": result["spectral_radius"],
                }
            )
            if candidate == "D_residual_query_newkv":
                residual_cache[candidate].append(
                    (
                        result["residual"],
                        metadata.loc[test].reset_index(drop=True),
                    )
                )
    markov_rows = pd.DataFrame(rows)
    markov_rows.to_parquet(
        run_dir / "markov_loso_rows.parquet", index=False
    )
    summary = (
        markov_rows.groupby("candidate")
        .agg(
            median_one_step_r2=("one_step_r2", "median"),
            minimum_one_step_r2=("one_step_r2", "min"),
            median_prediction_nll=("prediction_nll", "median"),
            median_residual_autocorrelation=(
                "residual_autocorrelation",
                "median",
            ),
            median_history_delta_r2=("history_delta_r2", "median"),
            median_rollout_r2=(
                "rollout_r2_8_16_32_64",
                "median",
            ),
            maximum_spectral_radius=("spectral_radius", "max"),
        )
        .reset_index()
    )
    bootstrap = {}
    for candidate, group in markov_rows.groupby("candidate"):
        bootstrap[str(candidate)] = {
            "one_step_r2": cluster_bootstrap_interval(
                group,
                "one_step_r2",
                cluster="held_out_sequence",
                samples=cfg.runtime.bootstrap_samples,
                seed=cfg.runtime.seed,
            ),
            "rollout_r2": cluster_bootstrap_interval(
                group,
                "rollout_r2_8_16_32_64",
                cluster="held_out_sequence",
                samples=cfg.runtime.bootstrap_samples,
                seed=cfg.runtime.seed + 1,
            ),
        }
    cross_task = []
    for candidate, columns in candidate_columns.items():
        x, y, metadata = _transition_arrays(frame, columns)
        for target_task in sorted(metadata["task"].unique()):
            test = metadata["task"].to_numpy() == target_task
            train = ~test
            result = _fit_transition_fold(
                x,
                y,
                train,
                test,
                cfg.trajectory_model.ridge_alpha,
            )
            cross_task.append(
                {
                    "candidate": candidate,
                    "train_task": str(
                        metadata.loc[train, "task"].iloc[0]
                    ),
                    "test_task": str(target_task),
                    "one_step_r2": result["r2"],
                    "prediction_nll": result["nll"],
                }
            )
    best = summary.sort_values(
        ["median_rollout_r2", "median_one_step_r2"],
        ascending=False,
    ).iloc[0]
    markov_pass = bool(
        best.median_history_delta_r2
        <= cfg.trajectory_model.markov_history_delta_r2_gate
        and best.median_rollout_r2 >= cfg.trajectory_model.rollout_r2_gate
        and abs(best.maximum_spectral_radius) < 1.2
    )
    markov_payload = {
        "schema_version": "trajectory_markov_sufficiency_v1",
        "independent_unit": "sequence",
        "state_implementation": (
            "log1p layerwise observable drift energies; all transforms fit "
            "inside each LOSO training fold"
        ),
        "primary_scope": (
            "controlled pulse trajectories; persistent recent-FIFO arm "
            "reported separately to avoid variance-dominance pooling"
        ),
        "candidate_columns": candidate_columns,
        "summary": summary.to_dict("records"),
        "sequence_cluster_bootstrap": bootstrap,
        "cross_task_transfer": cross_task,
        "best_candidate": str(best.candidate),
        "gate": {
            "history_delta_r2_threshold": (
                cfg.trajectory_model.markov_history_delta_r2_gate
            ),
            "rollout_r2_threshold": cfg.trajectory_model.rollout_r2_gate,
            "first_order_markov_pass": markov_pass,
        },
    }

    transition_payload = analyze_transition_structures(
        frame, layers, cfg, recent_frame=complete_frame
    )
    noise_payload = analyze_noise(
        residual_cache["D_residual_query_newkv"], cfg
    )
    return markov_payload, transition_payload, noise_payload


def _transition_model_predictions(
    x: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
    model_name: str,
    layers: Sequence[int],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scaler_x = StandardScaler().fit(x[train])
    scaler_y = StandardScaler().fit(y[train])
    train_x = scaler_x.transform(x[train])
    test_x = scaler_x.transform(x[test])
    train_y = scaler_y.transform(y[train])
    test_y = scaler_y.transform(y[test])
    dimension = int(x.shape[1])
    if model_name == "M0_direct_only":
        features_train = np.column_stack(
            [
                metadata.loc[train, "direct_input_l2"].to_numpy(),
                metadata.loc[train, "horizon_offset"].to_numpy(),
            ]
        )
        features_test = np.column_stack(
            [
                metadata.loc[test, "direct_input_l2"].to_numpy(),
                metadata.loc[test, "horizon_offset"].to_numpy(),
            ]
        )
        model = Ridge(alpha=alpha).fit(features_train, train_y)
        prediction = model.predict(features_test)
        parameters = int(2 * train_y.shape[1])
    elif model_name == "M1_scalar_ar":
        features_train = np.column_stack(
            [np.linalg.norm(train_x, axis=1)]
        )
        features_test = np.column_stack(
            [np.linalg.norm(test_x, axis=1)]
        )
        model = Ridge(alpha=alpha).fit(features_train, train_y)
        prediction = model.predict(features_test)
        parameters = int(train_y.shape[1])
    elif model_name == "M2_global_linear":
        model = Ridge(alpha=alpha).fit(train_x, train_y)
        prediction = model.predict(test_x)
        parameters = int(dimension * dimension)
    elif model_name in {"M3_layer_group", "M4_block_triangular"}:
        modalities = dimension // len(layers)
        prediction = np.zeros_like(test_y)
        groups = {
            "low": [index for index, layer in enumerate(layers) if layer <= 7],
            "middle": [
                index
                for index, layer in enumerate(layers)
                if 7 < layer <= 21
            ],
            "high": [
                index for index, layer in enumerate(layers) if layer > 21
            ],
        }
        parameters = 0
        for output_layer_index, output_layer in enumerate(layers):
            output_columns = [
                modality * len(layers) + output_layer_index
                for modality in range(modalities)
            ]
            if model_name == "M3_layer_group":
                group_name = next(
                    name
                    for name, indices in groups.items()
                    if output_layer_index in indices
                )
                source_layers = groups[group_name]
            else:
                source_layers = [
                    index
                    for index, layer in enumerate(layers)
                    if layer <= output_layer
                ]
            input_columns = [
                modality * len(layers) + layer_index
                for modality in range(modalities)
                for layer_index in source_layers
            ]
            model = Ridge(alpha=alpha).fit(
                train_x[:, input_columns],
                train_y[:, output_columns],
            )
            prediction[:, output_columns] = model.predict(
                test_x[:, input_columns]
            )
            parameters += len(input_columns) * len(output_columns)
    elif model_name == "M5_switching":
        prediction = np.zeros_like(test_y)
        magnitude_threshold = float(
            np.median(metadata.loc[train, "direct_input_l2"])
        )
        train_regime = (
            metadata.loc[train, "task"].astype(str)
            + "|r"
            + metadata.loc[train, "protected_recent_size"].astype(str)
            + "|m"
            + (
                metadata.loc[train, "direct_input_l2"]
                > magnitude_threshold
            ).astype(str)
            + "|e"
            + metadata.loc[train, "post_recent_exit"].astype(str)
        ).to_numpy()
        test_regime = (
            metadata.loc[test, "task"].astype(str)
            + "|r"
            + metadata.loc[test, "protected_recent_size"].astype(str)
            + "|m"
            + (
                metadata.loc[test, "direct_input_l2"]
                > magnitude_threshold
            ).astype(str)
            + "|e"
            + metadata.loc[test, "post_recent_exit"].astype(str)
        ).to_numpy()
        global_model = Ridge(alpha=alpha).fit(train_x, train_y)
        parameters = dimension * dimension
        for regime in np.unique(test_regime):
            train_rows = train_regime == regime
            test_rows = test_regime == regime
            if train_rows.sum() >= max(32, dimension + 1):
                model = Ridge(alpha=alpha).fit(
                    train_x[train_rows], train_y[train_rows]
                )
                parameters += dimension * dimension
            else:
                model = global_model
            prediction[test_rows] = model.predict(test_x[test_rows])
    elif model_name == "M6_local_quadratic":
        pca = PCA(
            n_components=min(8, dimension),
            random_state=42,
        ).fit(train_x)
        train_latent = pca.transform(train_x)
        test_latent = pca.transform(test_x)
        polynomial = PolynomialFeatures(
            degree=2, include_bias=False
        ).fit(train_latent)
        model = Ridge(alpha=alpha).fit(
            polynomial.transform(train_latent), train_y
        )
        prediction = model.predict(
            polynomial.transform(test_latent)
        )
        parameters = int(
            polynomial.n_output_features_ * train_y.shape[1]
        )
    else:
        raise ValueError(model_name)
    return test_y, prediction, parameters


def analyze_transition_structures(
    frame: pd.DataFrame,
    layers: Sequence[int],
    cfg: DiscoveryConfig,
    recent_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    columns = _candidate_columns(frame, layers)[
        "D_residual_query_newkv"
    ]
    x, y, metadata = _transition_arrays(frame, columns)
    models = (
        "M0_direct_only",
        "M1_scalar_ar",
        "M2_global_linear",
        "M3_layer_group",
        "M4_block_triangular",
        "M5_switching",
        "M6_local_quadratic",
    )
    rows = []
    for held_out in sorted(metadata["sample_id"].unique()):
        train = metadata["sample_id"].to_numpy() != held_out
        test = ~train
        for model_name in models:
            truth, prediction, parameters = _transition_model_predictions(
                x,
                y,
                metadata,
                train,
                test,
                model_name,
                layers,
                cfg.trajectory_model.ridge_alpha,
            )
            rows.append(
                {
                    "model": model_name,
                    "held_out_sequence": held_out,
                    "held_out_task": metadata.loc[
                        test, "task"
                    ].iloc[0],
                    "one_step_r2": _r2(truth, prediction),
                    "rmse": float(
                        np.sqrt(np.mean((truth - prediction) ** 2))
                    ),
                    "parameter_count_proxy": int(parameters),
                }
            )
    result = pd.DataFrame(rows)
    summary = (
        result.groupby("model")
        .agg(
            median_one_step_r2=("one_step_r2", "median"),
            minimum_one_step_r2=("one_step_r2", "min"),
            median_rmse=("rmse", "median"),
            parameter_count_proxy=("parameter_count_proxy", "median"),
        )
        .reset_index()
    )
    best_score = float(summary["median_one_step_r2"].max())
    eligible = summary[
        summary["median_one_step_r2"] >= best_score - 0.02
    ].sort_values("parameter_count_proxy")
    selected = eligible.iloc[0]

    # Hybrid causal arms: lower cumulative KL means that restoring this source
    # blocks a larger share of the feedback.
    global_rows = (
        frame.groupby(
            [
                "sample_id",
                "trajectory_id",
                "intervention_type",
                "horizon_offset",
            ],
            as_index=False,
        )
        .first()
    )
    hybrid = global_rows[
        global_rows["intervention_type"].str.startswith("hybrid_")
    ].copy()
    hybrid["arm"] = hybrid["intervention_type"].str.replace(
        "hybrid_", "", regex=False
    )
    cumulative = (
        hybrid.groupby(["sample_id", "arm"])["exact_kl"]
        .sum()
        .reset_index()
    )
    pivot = cumulative.pivot(
        index="sample_id", columns="arm", values="exact_kl"
    )
    causal = []
    if "stateful" in pivot:
        for arm in (
            "query_restore",
            "new_kv_restore",
            "attention_input_restore",
            "next_layer_hidden_restore",
        ):
            if arm not in pivot:
                continue
            reduction = (
                pivot["stateful"] - pivot[arm]
            ) / pivot["stateful"].abs().clip(lower=1e-12)
            causal.append(
                {
                    "arm": arm,
                    "median_relative_kl_reduction": float(
                        reduction.median()
                    ),
                    "sequence_values": {
                        str(index): float(value)
                        for index, value in reduction.items()
                    },
                }
            )
    recent_rows = result[result["model"].isin(["M2_global_linear", "M5_switching"])]
    switching_gain = (
        recent_rows.groupby("model")["one_step_r2"].median().to_dict()
    )
    recent_exit_rows = []
    if recent_frame is not None:
        recent_columns = _candidate_columns(recent_frame, layers)[
            "D_residual_query_newkv"
        ]
        recent_x, recent_y, recent_metadata = _transition_arrays(
            recent_frame, recent_columns
        )
        dedicated = (
            recent_metadata["intervention_type"].to_numpy()
            == "recent_exit_stateful"
        )
    else:
        recent_x, recent_y, recent_metadata = x, y, metadata
        dedicated = (
            recent_metadata["intervention_type"].to_numpy()
            == "recent_exit_stateful"
        )
    if dedicated.any():
        for held_out in sorted(
            recent_metadata.loc[dedicated, "sample_id"].unique()
        ):
            train = dedicated & (
                recent_metadata["sample_id"].to_numpy() != held_out
            )
            test = dedicated & (
                recent_metadata["sample_id"].to_numpy() == held_out
            )
            global_result = _fit_transition_fold(
                recent_x,
                recent_y,
                train,
                test,
                cfg.trajectory_model.ridge_alpha,
            )
            x_scaler = StandardScaler().fit(recent_x[train])
            y_scaler = StandardScaler().fit(recent_y[train])
            train_x = x_scaler.transform(recent_x[train])
            train_y = y_scaler.transform(recent_y[train])
            test_x = x_scaler.transform(recent_x[test])
            test_y = y_scaler.transform(recent_y[test])
            train_exit = recent_metadata.loc[
                train, "post_recent_exit"
            ].to_numpy(dtype=bool)
            test_exit = recent_metadata.loc[
                test, "post_recent_exit"
            ].to_numpy(dtype=bool)
            prediction = np.zeros_like(test_y)
            for regime in (False, True):
                train_regime = train_exit == regime
                test_regime = test_exit == regime
                model = Ridge(
                    alpha=cfg.trajectory_model.ridge_alpha
                ).fit(train_x[train_regime], train_y[train_regime])
                prediction[test_regime] = model.predict(
                    test_x[test_regime]
                )
            switched_r2 = _r2(test_y, prediction)
            recent_exit_rows.append(
                {
                    "held_out_sequence": held_out,
                    "pooled_r2": global_result["r2"],
                    "pre_post_switching_r2": switched_r2,
                    "switching_gain": (
                        switched_r2 - global_result["r2"]
                    ),
                }
            )
    return {
        "schema_version": "trajectory_transition_comparison_v1",
        "independent_unit": "sequence",
        "model_summary": summary.to_dict("records"),
        "selection_rule": (
            "simplest parameter-count proxy within 0.02 held-out R2 of best"
        ),
        "selected_model": str(selected.model),
        "best_heldout_r2": best_score,
        "switching_minus_global_r2": float(
            switching_gain.get("M5_switching", float("nan"))
            - switching_gain.get("M2_global_linear", float("nan"))
        ),
        "recent_exit_real_fifo": {
            "actual_exit_offset": 33,
            "rows": recent_exit_rows,
            "median_switching_gain": (
                float(
                    np.median(
                        [
                            row["switching_gain"]
                            for row in recent_exit_rows
                        ]
                    )
                )
                if recent_exit_rows
                else float("nan")
            ),
        },
        "hybrid_causal_source_summary": causal,
        "layer27_reported_separately": True,
        "model_rows": result.to_dict("records"),
    }


def analyze_noise(
    residual_folds: Sequence[Tuple[np.ndarray, pd.DataFrame]],
    cfg: DiscoveryConfig,
) -> Dict[str, Any]:
    likelihood_rows = []
    all_residual = []
    all_metadata = []
    for fold_index, (test_residual, test_metadata) in enumerate(
        residual_folds
    ):
        train_residual = np.concatenate(
            [
                residual
                for index, (residual, _) in enumerate(residual_folds)
                if index != fold_index
            ]
        )
        pca = PCA(
            n_components=min(8, train_residual.shape[1]),
            random_state=cfg.runtime.seed,
        ).fit(train_residual)
        train = pca.transform(train_residual)
        test = pca.transform(test_residual)
        mean = train.mean(axis=0)
        diagonal_variance = train.var(axis=0) + 1e-6
        diagonal_ll = -0.5 * np.sum(
            np.log(2 * np.pi * diagonal_variance)
            + (test - mean) ** 2 / diagonal_variance,
            axis=1,
        )
        covariance = LedoitWolf().fit(train)
        full_distribution = stats.multivariate_normal(
            mean=covariance.location_,
            cov=covariance.covariance_,
            allow_singular=False,
        )
        full_ll = full_distribution.logpdf(test)
        # Low-rank plus isotropic diagonal covariance.
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.cov(train, rowvar=False)
        )
        order = np.argsort(eigenvalues)[::-1]
        rank = min(3, len(order))
        basis = eigenvectors[:, order[:rank]]
        retained = np.maximum(eigenvalues[order[:rank]], 1e-6)
        diagonal = max(
            float(np.mean(eigenvalues[order[rank:]]))
            if rank < len(order)
            else 1e-6,
            1e-6,
        )
        lowrank_covariance = (
            basis @ np.diag(np.maximum(retained - diagonal, 0)) @ basis.T
            + diagonal * np.eye(train.shape[1])
        )
        lowrank_distribution = stats.multivariate_normal(
            mean=mean, cov=lowrank_covariance, allow_singular=False
        )
        lowrank_ll = lowrank_distribution.logpdf(test)
        excess = np.nanmean(stats.kurtosis(train, axis=0, fisher=True))
        degrees = float(np.clip(6.0 / max(excess, 1e-3) + 4.0, 3.0, 50.0))
        scale = np.sqrt(diagonal_variance * (degrees - 2.0) / degrees)
        student_ll = np.sum(
            stats.t.logpdf(
                (test - mean) / scale,
                df=degrees,
            )
            - np.log(scale),
            axis=1,
        )
        mixture = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=cfg.runtime.seed,
        ).fit(train)
        mixture_ll = mixture.score_samples(test)
        task = test_metadata["task"].astype(str).to_numpy()
        regime_ll = np.empty(len(test))
        for regime in np.unique(task):
            # Training folds do not retain row-level task labels here; use a
            # conservative diagonal regime proxy by variance scale estimated
            # from the held-in residual mixture. This is reported as
            # task-scale-conditioned, not a fully supervised SLDS likelihood.
            scale_factor = (
                1.0
                if "gov" in regime
                else max(float(np.median(np.var(train, axis=0))), 1e-3)
            )
            variance = diagonal_variance * scale_factor
            index = task == regime
            regime_ll[index] = -0.5 * np.sum(
                np.log(2 * np.pi * variance)
                + (test[index] - mean) ** 2 / variance,
                axis=1,
            )
        models = {
            "diagonal_gaussian": diagonal_ll,
            "full_shrinkage_gaussian": full_ll,
            "lowrank_plus_diagonal_gaussian": lowrank_ll,
            "student_t_diagonal": student_ll,
            "two_component_gaussian_mixture": mixture_ll,
            "task_scale_conditioned_gaussian": regime_ll,
        }
        for name, values in models.items():
            likelihood_rows.append(
                {
                    "fold": fold_index,
                    "model": name,
                    "mean_log_likelihood": float(np.mean(values)),
                }
            )
        all_residual.append(test)
        all_metadata.append(test_metadata)
    residual = np.concatenate(all_residual)
    likelihood = pd.DataFrame(likelihood_rows)
    summary = (
        likelihood.groupby("model")["mean_log_likelihood"]
        .median()
        .sort_values(ascending=False)
    )
    covariance = LedoitWolf().fit(residual)
    centered = residual - covariance.location_
    mahalanobis = np.einsum(
        "ni,ij,nj->n",
        centered,
        covariance.precision_,
        centered,
    )
    expected_tail = 0.01
    threshold = stats.chi2.ppf(1.0 - expected_tail, residual.shape[1])
    actual_tail = float(np.mean(mahalanobis > threshold))
    normality_p = float(
        stats.kstest(
            stats.chi2.cdf(mahalanobis, residual.shape[1]), "uniform"
        ).pvalue
    )
    skewness = stats.skew(residual, axis=0, bias=False)
    excess_kurtosis = stats.kurtosis(
        residual, axis=0, fisher=True, bias=False
    )
    gaussian_pass = bool(
        normality_p >= 0.05
        and actual_tail <= 2.0 * expected_tail
        and np.nanmax(np.abs(excess_kurtosis)) <= 2.0
    )
    return {
        "schema_version": "trajectory_noise_diagnostics_v1",
        "transition_source": "M2 global D-state LOSO residuals",
        "latent_noise_dimension": int(residual.shape[1]),
        "heldout_log_likelihood": summary.reset_index()
        .rename(columns={"mean_log_likelihood": "median_log_likelihood"})
        .to_dict("records"),
        "selected_noise_model": str(summary.index[0]),
        "diagnostics": {
            "conditional_mean_l2": float(
                np.linalg.norm(residual.mean(axis=0))
            ),
            "median_absolute_skewness": float(
                np.nanmedian(np.abs(skewness))
            ),
            "maximum_absolute_skewness": float(
                np.nanmax(np.abs(skewness))
            ),
            "median_excess_kurtosis": float(
                np.nanmedian(excess_kurtosis)
            ),
            "maximum_absolute_excess_kurtosis": float(
                np.nanmax(np.abs(excess_kurtosis))
            ),
            "mahalanobis_uniform_ks_p": normality_p,
            "nominal_1pct_tail_exceedance": actual_tail,
            "gaussian_pass": gaussian_pass,
        },
        "second_order_note": (
            "Gaussian rejection does not by itself reject mean/covariance "
            "closure; that is gated separately by quadratic-risk calibration."
        ),
    }


def analyze_quadratic_loss(
    run_dir: Path, cfg: DiscoveryConfig
) -> Dict[str, Any]:
    rows = pd.read_parquet(run_dir / "trajectory_state_rows.parquet")
    rows = rows[rows["beta"] > 0].copy()
    keys = [
        "sample_id",
        "task",
        "trajectory_id",
        "horizon_offset",
        "protected_recent_size",
        "direct_input_l2",
    ]
    aggregate = (
        rows.groupby(keys)
        .agg(
            hidden_quadratic=("layer_output_l2", lambda value: float(np.sum(np.square(value)))),
            residual_quadratic=("residual_l2", lambda value: float(np.sum(np.square(value)))),
            query_quadratic=("query_l2", lambda value: float(np.sum(np.square(value)))),
            kv_quadratic=(
                "new_key_l2",
                lambda value: float(np.sum(np.square(value))),
            ),
            value_quadratic=(
                "new_value_l2",
                lambda value: float(np.sum(np.square(value))),
            ),
            projected_output_error=(
                "projected_attention_output_l2",
                lambda value: float(np.sum(np.square(value))),
            ),
            exact_kl=("exact_kl", "first"),
            js=("js", "first"),
            delta_nll=("delta_nll", "first"),
            fisher_quadratic=("fisher_quadratic", "first"),
        )
        .reset_index()
    )
    aggregate["newkv_quadratic"] = (
        aggregate["kv_quadratic"] + aggregate["value_quadratic"]
    )
    predictors = {
        "hidden_only": ["hidden_quadratic"],
        "query_only": ["query_quadratic"],
        "newkv_only": ["newkv_quadratic"],
        "combined_state": [
            "hidden_quadratic",
            "residual_quadratic",
            "query_quadratic",
            "newkv_quadratic",
        ],
        "fisher_logit": ["fisher_quadratic"],
    }
    targets = (
        "projected_output_error",
        "exact_kl",
        "js",
        "delta_nll",
    )
    cv_rows = []
    for held_out in sorted(aggregate["sample_id"].unique()):
        train = aggregate["sample_id"].to_numpy() != held_out
        test = ~train
        for predictor, columns in predictors.items():
            scaler = StandardScaler().fit(
                aggregate.loc[train, columns]
            )
            train_x = scaler.transform(aggregate.loc[train, columns])
            test_x = scaler.transform(aggregate.loc[test, columns])
            for target in targets:
                train_y = aggregate.loc[train, target].to_numpy(
                    dtype=np.float64
                )
                test_y = aggregate.loc[test, target].to_numpy(
                    dtype=np.float64
                )
                model = Ridge(
                    alpha=cfg.trajectory_model.ridge_alpha
                ).fit(train_x, train_y)
                prediction = model.predict(test_x)
                cv_rows.append(
                    {
                        "held_out_sequence": held_out,
                        "held_out_task": aggregate.loc[
                            test, "task"
                        ].iloc[0],
                        "predictor": predictor,
                        "target": target,
                        "r2": _r2(test_y[:, None], prediction[:, None]),
                        "spearman": _spearman(test_y, prediction),
                    }
                )
    cv = pd.DataFrame(cv_rows)
    summary = (
        cv.groupby(["predictor", "target"])
        .agg(
            median_r2=("r2", "median"),
            minimum_r2=("r2", "min"),
            median_spearman=("spearman", "median"),
        )
        .reset_index()
    )
    median_direct = float(aggregate["direct_input_l2"].median())
    aggregate["perturbation_regime"] = np.where(
        aggregate["direct_input_l2"] <= median_direct,
        "small",
        "large",
    )
    breakdown = []
    for columns, group in aggregate.groupby(
        ["task", "protected_recent_size", "perturbation_regime"]
    ):
        breakdown.append(
            {
                "task": columns[0],
                "protected_recent_size": int(columns[1]),
                "perturbation_regime": columns[2],
                "fisher_vs_kl_spearman": _spearman(
                    group["fisher_quadratic"], group["exact_kl"]
                ),
                "hidden_vs_kl_spearman": _spearman(
                    group["hidden_quadratic"], group["exact_kl"]
                ),
                "hidden_vs_nll_spearman": _spearman(
                    group["hidden_quadratic"],
                    group["delta_nll"].abs(),
                ),
            }
        )
    fisher_kl = summary[
        (summary["predictor"] == "fisher_logit")
        & (summary["target"] == "exact_kl")
    ].iloc[0]
    combined_kl = summary[
        (summary["predictor"] == "combined_state")
        & (summary["target"] == "exact_kl")
    ].iloc[0]
    quadratic_pass = bool(
        max(fisher_kl.median_r2, combined_kl.median_r2) >= 0.5
        and max(
            fisher_kl.median_spearman, combined_kl.median_spearman
        )
        >= 0.5
    )
    return {
        "schema_version": "trajectory_quadratic_loss_v1",
        "independent_unit": "sequence",
        "loso_summary": summary.to_dict("records"),
        "regime_breakdown": breakdown,
        "small_large_threshold_train_independent_descriptive": median_direct,
        "recommended_quadratic_target": (
            "exact_KL"
            if fisher_kl.median_r2
            >= summary[
                summary["target"] == "delta_nll"
            ]["median_r2"].max()
            else "none"
        ),
        "gate": {
            "quadratic_loss_pass": quadratic_pass,
            "criterion": "median LOSO R2 and Spearman both at least 0.5",
        },
    }


def closed_form_gate(
    scaling: Mapping[str, Any],
    latent: Mapping[str, Any],
    markov: Mapping[str, Any],
    noise: Mapping[str, Any],
    quadratic: Mapping[str, Any],
) -> Dict[str, Any]:
    scaling_gate = (
        scaling["scaling"]["validity_gate"]["maximum_valid_horizon"] >= 16
        and scaling["scaling"]["validity_gate"]["maximum_valid_beta"] >= 1.0
        and scaling["superposition"]["aggregate_cluster_bootstrap"][
            "estimate"
        ]
        <= scaling["superposition"]["tolerance"]
    )
    latent_gate = bool(
        latent["gate"]["multi_layer_low_dimensional_pass"]
    )
    markov_gate = bool(markov["gate"]["first_order_markov_pass"])
    quadratic_gate = bool(quadratic["gate"]["quadratic_loss_pass"])
    second_order_gate = bool(
        noise["diagnostics"]["nominal_1pct_tail_exceedance"] <= 0.05
    )
    all_pass = (
        scaling_gate
        and latent_gate
        and markov_gate
        and quadratic_gate
        and second_order_gate
    )
    failed = [
        name
        for name, passed in {
            "local_linearity_and_superposition": scaling_gate,
            "low_dimensionality": latent_gate,
            "first_order_markov": markov_gate,
            "quadratic_loss": quadratic_gate,
            "second_order_noise_calibration": second_order_gate,
        }.items()
        if not passed
    ]
    return {
        "schema_version": "trajectory_closed_form_validation_v1",
        "executed": bool(all_pass),
        "pre_registered_gate_results": {
            "local_linearity_and_superposition": scaling_gate,
            "low_dimensionality": latent_gate,
            "first_order_markov": markov_gate,
            "quadratic_loss": quadratic_gate,
            "second_order_noise_calibration": second_order_gate,
        },
        "failed_gates": failed,
        "closed_form_ranking_metrics": None,
        "decision": (
            "execute held-out closed-form objective validation"
            if all_pass
            else (
                "not executed: enforcing the instruction that closed-form "
                "validation is conditional on preceding mechanism gates"
            )
        ),
    }


def run_trajectory_analysis(
    run_dir: Path, cfg: DiscoveryConfig
) -> Dict[str, Path]:
    run_dir = run_dir.resolve()
    scaling_path = run_dir / "scaling_superposition_summary.json"
    if scaling_path.exists():
        with open(scaling_path, "r", encoding="utf-8") as handle:
            scaling = json.load(handle)
    else:
        scaling = analyze_scaling_superposition(run_dir, cfg)
        atomic_json(scaling_path, scaling)
    latent_path = run_dir / "latent_rank_summary.json"
    if latent_path.exists():
        with open(latent_path, "r", encoding="utf-8") as handle:
            latent = json.load(handle)
    else:
        latent = analyze_latent_rank(run_dir, cfg)
        atomic_json(latent_path, latent)
    markov, transition, noise = analyze_markov(run_dir, cfg)
    atomic_json(run_dir / "markov_sufficiency_summary.json", markov)
    atomic_json(run_dir / "transition_model_comparison.json", transition)
    atomic_json(run_dir / "noise_diagnostics.json", noise)
    quadratic = analyze_quadratic_loss(run_dir, cfg)
    atomic_json(run_dir / "quadratic_loss_validation.json", quadratic)
    closed = closed_form_gate(
        scaling, latent, markov, noise, quadratic
    )
    atomic_json(
        run_dir / "closed_form_objective_validation.json", closed
    )
    return {
        path.name: path
        for path in (
            run_dir / "scaling_superposition_summary.json",
            run_dir / "latent_rank_summary.json",
            run_dir / "markov_sufficiency_summary.json",
            run_dir / "transition_model_comparison.json",
            run_dir / "noise_diagnostics.json",
            run_dir / "quadratic_loss_validation.json",
            run_dir / "closed_form_objective_validation.json",
        )
    }
