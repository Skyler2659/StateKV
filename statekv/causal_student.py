"""Deployable R2 students for the StateKV counterfactual-utility study.

This module turns the expensive R2 prefix-recomputation teacher into a cheap
runtime scorer:

- :func:`dump_teacher_scores` repackages the per-token R2 scores already
  persisted by ``run_causal_rollout_study`` into the counterfactual results
  tree and documents the (sample_id, cycle, position) join to the collected
  full-cache artifacts.
- :func:`train_students` fits a histogram GBDT and a small MLP that map the
  existing ``artifact_boundary`` causal feature vector (120 dimensions, built
  only from current/past observations) onto the R2 teacher score.
- :class:`RuntimeStudentScorer` rebuilds the exact same feature vector inside
  the strict closed loop from runtime-causal observations only, so the
  ``STRICT_STATEKV_STUDENT`` policy is a pure-eviction policy with no future
  information and no rollout at deployment time.

Feature construction is deliberately NOT reimplemented here: both the
training path and the runtime path call
``statekv.causal_predictors.artifact_boundary``.  The runtime side only
assembles an artifact-shaped mapping from its own observation history, which
keeps train/serve feature drift at zero by construction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from statekv.causal_distillation import _teacher_arrays
from statekv.causal_existence import (
    _atomic_npz,
    _global_logit_features,
    _safe_sample_id,
    sample_id_for,
)
from statekv.causal_existence_analysis import boundary_metrics, topk_indices
from statekv.causal_predictors import (
    FixedProjector,
    MultiHorizonMLP,
    _load_npz,
    artifact_boundary,
)
from statekv.selectors import mandatory_and_eligible
from statekv.storage import atomic_frame, atomic_json


STUDENT_FORMAT = "statekv_r2_student/v1"
FEATURE_WIDTH = 120
R2_TEACHER = "CAUSAL_EXPENSIVE_ROLLOUT_R2_PREFIX_RECOMPUTE"
JOIN_KEYS = ("sample_id", "cycle", "position")

# Documented layout of the 120-dimensional artifact_boundary feature vector.
# Every segment is computed from runtime-causal sources: current query,
# current hidden state, cached K/V, QK geometry, age/position, and
# current/cumulative attention history.  No segment reads rollout, future,
# or oracle data.
FEATURE_SEGMENTS: Dict[str, Tuple[int, int, str]] = {
    "attention_history_scalars": (0, 16, "current/lag/EMA/rank statistics of per-cycle attention"),
    "qk_geometry": (16, 20, "cached keys x current query dot statistics"),
    "token_norms_position": (20, 24, "key/value norms, relative position, age"),
    "key_projection": (24, 32, "cached keys x fixed public projection"),
    "value_projection": (32, 40, "cached values x fixed public projection"),
    "current_state": (40, 120, "current query/hidden-state projections, query trajectory, global logit features, cycle/layer/head scalars"),
}

# Keys that exist in the collected artifacts but must never enter the runtime
# feature path because they encode generated futures or label bookkeeping.
FORBIDDEN_RUNTIME_KEYS = (
    "generated_token_ids",
    "current_token_ids",
    "label_source",
)


# --------------------------------------------------------------------- dump


def dump_teacher_scores(
    source_run: Path,
    dest_root: Path,
    splits: Sequence[str] = ("train", "validation"),
) -> Path:
    """Repackage persisted per-token R2 teacher scores with join metadata.

    The causal rollout study already writes per-token scores to
    ``<source_run>/rollout/<split>/teacher_scores/<sample>.npz``.  The
    full-cache artifacts under ``<source_run>/artifacts/<split>/`` carry every
    feature the student needs (``artifact_boundary`` derives the 120-dim
    vector from them), so only the scores plus join keys are copied.  The
    source tree is opened read-only and is never modified.
    """

    dest_root = Path(dest_root)
    manifest_rows: List[Dict[str, Any]] = []
    for split in splits:
        source_dir = Path(source_run) / "rollout" / str(split) / "teacher_scores"
        if not source_dir.is_dir():
            raise RuntimeError(f"missing teacher score directory: {source_dir}")
        artifact_dir = Path(source_run) / "artifacts" / str(split)
        dest_dir = dest_root / str(split)
        paths = sorted(source_dir.glob("*.npz"))
        if not paths:
            raise RuntimeError(f"no teacher score files in {source_dir}")
        for path in paths:
            teacher = _load_npz(path)
            required = {
                "cycles",
                "horizons",
                "position_ids",
                "position_lengths",
                "scores",
                "sample_id",
                "task",
                "split",
                "teacher",
                "runtime_future_access",
            }
            missing = required - set(teacher)
            if missing:
                raise RuntimeError(f"{path.name} lacks teacher keys: {sorted(missing)}")
            if str(teacher["teacher"].item()) != R2_TEACHER:
                raise RuntimeError(f"{path.name} is not an R2 teacher dump")
            if bool(teacher["runtime_future_access"].item()):
                raise RuntimeError(f"{path.name} violates the causal teacher contract")
            cycles = [int(value) for value in teacher["cycles"]]
            horizons = [int(value) for value in teacher["horizons"]]
            lengths = [int(value) for value in teacher["position_lengths"]]
            scores = np.asarray(teacher["scores"], dtype=np.float32)
            expected = (
                len(cycles),
                len(horizons),
                scores.shape[2],
                scores.shape[3],
                teacher["position_ids"].shape[1],
            )
            if tuple(scores.shape) != expected:
                raise RuntimeError(f"{path.name} has inconsistent score shape")
            for cycle_index, length in enumerate(lengths):
                valid = scores[cycle_index, :, :, :, :length]
                if not np.isfinite(valid).all():
                    raise RuntimeError(f"{path.name} has non-finite scores inside the valid region")
            sample_id = str(teacher["sample_id"].item())
            artifact_path = artifact_dir / path.name
            if not artifact_path.exists():
                raise RuntimeError(f"{path.name} lacks a feature artifact: {artifact_path}")
            _atomic_npz(
                dest_dir / path.name,
                **teacher,
                join_keys=np.asarray(list(JOIN_KEYS)),
                feature_artifact=np.asarray(
                    f"artifacts/{split}/{path.name}"
                ),
                feature_source_run=np.asarray(str(Path(source_run).name)),
            )
            manifest_rows.append(
                {
                    "split": str(split),
                    "sample_id": sample_id,
                    "file": f"{split}/{path.name}",
                    "feature_artifact": f"artifacts/{split}/{path.name}",
                    "cycles": len(cycles),
                    "horizons": horizons,
                    "max_positions": int(teacher["position_ids"].shape[1]),
                }
            )
    atomic_json(
        dest_root / "manifest.json",
        {
            "format": STUDENT_FORMAT,
            "teacher": R2_TEACHER,
            "source_run": str(source_run),
            "source_read_only": True,
            "join_keys": list(JOIN_KEYS),
            "join_semantics": (
                "teacher scores[cycle_index, horizon, layer, head, j] join the "
                "feature artifact on sample_id, cycle=cycles[cycle_index], and "
                "position=position_ids[cycle_index, j]; artifact_boundary(artifact, "
                "cycle, layer_index, head) row j is the same position"
            ),
            "score_semantics": "summed future attention over the model's own greedy rollout",
            "runtime_future_access": False,
            "splits": [str(split) for split in splits],
            "files": manifest_rows,
        },
    )
    return dest_root


# ------------------------------------------------------------------ scoring


def save_student_checkpoint(
    path: Path,
    *,
    kind: str,
    models: Any,
    scaler: StandardScaler,
    horizons: Sequence[int],
    projector_seed: int,
    score_channel: int = 1,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    if kind not in {"hist_gbdt", "mlp"}:
        raise ValueError(f"unknown student kind: {kind}")
    payload = {
        "format": STUDENT_FORMAT,
        "kind": str(kind),
        "models": models,
        "scaler": scaler,
        "horizons": [int(value) for value in horizons],
        "feature_width": FEATURE_WIDTH,
        "projector_seed": int(projector_seed),
        "score_channel": int(score_channel),
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save(payload, path)
    else:
        joblib.dump(payload, path)
    return path


def load_student_checkpoint(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"student checkpoint is missing: {path}")
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
    else:
        payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != STUDENT_FORMAT:
        raise RuntimeError(f"unrecognized student checkpoint format: {path}")
    if int(payload["feature_width"]) != FEATURE_WIDTH:
        raise RuntimeError(
            f"student checkpoint feature width {payload['feature_width']} "
            f"does not match the runtime contract {FEATURE_WIDTH}"
        )
    return payload


class StudentScorer:
    """Checkpoint-backed per-token scorer over artifact_boundary features."""

    def __init__(self, checkpoint: Mapping[str, Any]):
        self.kind = str(checkpoint["kind"])
        self.horizons = [int(value) for value in checkpoint["horizons"]]
        self.scaler = checkpoint["scaler"]
        self.score_channel = int(checkpoint.get("score_channel", 1))
        if self.kind == "hist_gbdt":
            self.models = {
                int(horizon): checkpoint["models"][int(horizon)]
                if int(horizon) in checkpoint["models"]
                else checkpoint["models"][horizon]
                for horizon in self.horizons
            }
        elif self.kind == "mlp":
            self.model = MultiHorizonMLP(FEATURE_WIDTH, len(self.horizons))
            self.model.load_state_dict(checkpoint["models"])
            self.model.eval()
        else:
            raise RuntimeError(f"unknown student kind: {self.kind}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FEATURE_WIDTH:
            raise RuntimeError(
                f"student features must be (n_tokens, {FEATURE_WIDTH}), "
                f"got {tuple(features.shape)}"
            )
        normalized = self.scaler.transform(features).astype(np.float32)
        if self.kind == "hist_gbdt":
            return np.stack(
                [
                    self.models[horizon].predict(normalized)
                    for horizon in self.horizons
                ],
                axis=1,
            ).astype(np.float32)
        with torch.no_grad():
            output = self.model(
                torch.from_numpy(normalized),
                torch.zeros((len(normalized), 1, 2), dtype=torch.float32),
            ).numpy()
        return output[:, :, self.score_channel].astype(np.float32)

    def horizon_column(self, horizon: int) -> int:
        if int(horizon) not in self.horizons:
            raise RuntimeError(
                f"student checkpoint horizons {self.horizons} do not cover "
                f"the requested deployment horizon {horizon}"
            )
        return self.horizons.index(int(horizon))


# ------------------------------------------------------- runtime feature log


class RuntimeFeatureHistory:
    """Accumulates runtime-causal observations and exposes artifact views.

    Only whitelisted runtime observations enter: per-cycle attention of the
    current query over the active pool, current post-RoPE queries, cached K/V,
    current residual/attention-input hidden states, and global logit
    features.  Generated token ids, future attention, and rollout data never
    appear here.
    """

    def __init__(
        self,
        *,
        score_layers: Sequence[int],
        kv_heads: int,
        query_heads: int,
        sink_size: int,
        recent_size: int,
        total_cycles: int,
        hidden_width: int = 4096,
        head_width: int = 128,
    ):
        self.score_layers = [int(value) for value in score_layers]
        self.kv_heads = int(kv_heads)
        self.query_heads = int(query_heads)
        self.sink_size = int(sink_size)
        self.recent_size = int(recent_size)
        self.total_cycles = int(total_cycles)
        self.hidden_width = int(hidden_width)
        self.head_width = int(head_width)
        self._positions: List[np.ndarray] = []
        self._attention: List[np.ndarray] = []
        self._queries: List[np.ndarray] = []
        self._cycle = -1

    def observe(
        self,
        *,
        cycle: int,
        positions: Sequence[int],
        per_head_attention: Mapping[int, np.ndarray],
        post_rope_queries: Mapping[int, np.ndarray],
        residual: Mapping[int, np.ndarray],
        attention_input: Mapping[int, np.ndarray],
        global_features: np.ndarray,
    ) -> None:
        """Record one cycle of runtime observations (arrays copied)."""

        cycle = int(cycle)
        if cycle != self._cycle + 1:
            raise RuntimeError(
                f"runtime feature history expects consecutive cycles, got "
                f"{cycle} after {self._cycle}"
            )
        positions_array = np.asarray(
            [int(value) for value in positions], dtype=np.int32
        )
        attention = np.stack(
            [
                np.asarray(per_head_attention[layer], dtype=np.float32)
                for layer in self.score_layers
            ],
            axis=0,
        )
        if attention.shape != (
            len(self.score_layers),
            self.kv_heads,
            len(positions_array),
        ):
            raise RuntimeError(
                f"attention observation shape {attention.shape} does not match "
                f"(layers, kv_heads, positions) = "
                f"{(len(self.score_layers), self.kv_heads, len(positions_array))}"
            )
        queries = np.stack(
            [
                np.asarray(post_rope_queries[layer], dtype=np.float32)
                for layer in self.score_layers
            ],
            axis=0,
        )
        if queries.shape != (len(self.score_layers), self.query_heads, self.head_width):
            raise RuntimeError(
                f"query observation shape {queries.shape} does not match "
                f"(layers, query_heads, head_width)"
            )
        for name, mapping, width in (
            ("residual", residual, self.hidden_width),
            ("attention_input", attention_input, self.hidden_width),
        ):
            for layer in self.score_layers:
                value = np.asarray(mapping[layer], dtype=np.float32)
                if value.shape != (width,):
                    raise RuntimeError(
                        f"{name} observation at layer {layer} has shape "
                        f"{value.shape}, expected ({width},)"
                    )
        if np.asarray(global_features).shape != (4,):
            raise RuntimeError("global logit features must have width 4")
        self._positions.append(positions_array)
        self._attention.append(attention)
        self._queries.append(queries)
        self._last_residual = {
            int(layer): np.asarray(residual[layer], dtype=np.float32).copy()
            for layer in self.score_layers
        }
        self._last_attention_input = {
            int(layer): np.asarray(attention_input[layer], dtype=np.float32).copy()
            for layer in self.score_layers
        }
        self._last_global = np.asarray(global_features, dtype=np.float32).copy()
        self._cycle = cycle

    def artifact_view(
        self,
        *,
        cycle: int,
        positions: Sequence[int],
        keys: Mapping[int, np.ndarray],
        values: Mapping[int, np.ndarray],
    ) -> Dict[str, Any]:
        """Materialize an artifact-shaped mapping for ``artifact_boundary``.

        History rows are repacked into the current cycle's column order, which
        mirrors the stable-column packing of the collected full-cache
        artifacts.  Rows beyond ``cycle`` stay NaN; reading them would poison
        the features, which the causality tests check for.
        """

        cycle = int(cycle)
        if cycle > self._cycle:
            raise RuntimeError("artifact views cannot exceed the observed cycles")
        current = np.asarray([int(value) for value in positions], dtype=np.int32)
        count = int(current.size)
        layers = len(self.score_layers)
        row_by_position = {
            int(position): row for row, position in enumerate(current.tolist())
        }
        attention = np.full(
            (self.total_cycles, layers, self.kv_heads, count),
            np.nan,
            dtype=np.float32,
        )
        for step in range(cycle + 1):
            step_positions = self._positions[step]
            keep = np.asarray(
                [int(position) in row_by_position for position in step_positions.tolist()]
            )
            columns = np.asarray(
                [
                    row_by_position[int(position)]
                    for position in step_positions[keep].tolist()
                ],
                dtype=np.int64,
            )
            attention[step, :, :, columns] = self._attention[step][:, :, keep].transpose(
                2, 0, 1
            )
        position_ids = np.full((self.total_cycles, count), -1, dtype=np.int32)
        position_ids[cycle] = current
        position_lengths = np.zeros(self.total_cycles, dtype=np.int32)
        position_lengths[cycle] = count
        query_post = np.zeros(
            (self.total_cycles, layers, self.query_heads, self.head_width),
            dtype=np.float32,
        )
        first = max(0, cycle + 1 - len(self._queries))
        for offset, queries in enumerate(self._queries):
            query_post[first + offset] = queries
        residual = np.zeros(
            (self.total_cycles, layers, self.hidden_width), dtype=np.float32
        )
        attention_input = np.zeros_like(residual)
        global_features = np.zeros((self.total_cycles, 4), dtype=np.float32)
        for layer_index, layer in enumerate(self.score_layers):
            residual[cycle, layer_index] = self._last_residual[layer]
            attention_input[cycle, layer_index] = self._last_attention_input[layer]
        global_features[cycle] = self._last_global
        key_array = np.stack(
            [np.asarray(keys[layer], dtype=np.float32) for layer in self.score_layers],
            axis=0,
        )
        value_array = np.stack(
            [np.asarray(values[layer], dtype=np.float32) for layer in self.score_layers],
            axis=0,
        )
        expected = (layers, self.kv_heads, count, self.head_width)
        if key_array.shape != expected or value_array.shape != expected:
            raise RuntimeError(
                f"cached K/V observation shape {key_array.shape} does not match {expected}"
            )
        return {
            "attention": attention,
            "position_ids": position_ids,
            "position_lengths": position_lengths,
            "query_post": query_post,
            "residual": residual,
            "attention_input": attention_input,
            "global_features": global_features,
            "keys": key_array,
            "values": value_array,
            "kv_position_ids": current,
            "layers": np.asarray(self.score_layers, dtype=np.int16),
            "sample_id": np.asarray("runtime"),
            "task": np.asarray("runtime"),
            "split": np.asarray("runtime"),
        }


class RuntimeStudentScorer:
    """Scores eligible tokens in the strict loop with a trained student."""

    def __init__(
        self,
        checkpoint: Mapping[str, Any],
        *,
        score_layers: Sequence[int],
        kv_heads: int,
        query_heads: int,
        sink_size: int,
        recent_size: int,
        horizon: int,
    ):
        self.student = StudentScorer(checkpoint)
        self.projector = FixedProjector(int(checkpoint["projector_seed"]))
        self.horizon = int(horizon)
        self.horizon_column = self.student.horizon_column(self.horizon)
        self.score_layers = [int(value) for value in score_layers]
        self.kv_heads = int(kv_heads)
        self.query_heads = int(query_heads)
        self.sink_size = int(sink_size)
        self.recent_size = int(recent_size)
        self.history: Optional[RuntimeFeatureHistory] = None

    def reset(self, total_cycles: int) -> None:
        self.history = RuntimeFeatureHistory(
            score_layers=self.score_layers,
            kv_heads=self.kv_heads,
            query_heads=self.query_heads,
            sink_size=self.sink_size,
            recent_size=self.recent_size,
            total_cycles=int(total_cycles),
        )

    def observe_and_score(
        self,
        *,
        cycle: int,
        positions: Sequence[int],
        per_head_attention: Mapping[int, np.ndarray],
        post_rope_queries: Mapping[int, np.ndarray],
        residual: Mapping[int, np.ndarray],
        attention_input: Mapping[int, np.ndarray],
        global_features: np.ndarray,
        keys: Mapping[int, np.ndarray],
        values: Mapping[int, np.ndarray],
    ) -> Dict[int, float]:
        """Score every currently eligible position with the student.

        Returns {position: mean student score across diagnostic layers and KV
        heads at the deployment horizon}.  Only runtime-causal observations
        are consumed; the sink/recent mandatory set is never scored because it
        is never eligible for eviction.
        """

        if self.history is None:
            raise RuntimeError("RuntimeStudentScorer.reset() must precede scoring")
        self.history.observe(
            cycle=cycle,
            positions=positions,
            per_head_attention=per_head_attention,
            post_rope_queries=post_rope_queries,
            residual=residual,
            attention_input=attention_input,
            global_features=global_features,
        )
        view = self.history.artifact_view(
            cycle=cycle, positions=positions, keys=keys, values=values
        )
        _, _, eligible = mandatory_and_eligible(
            positions, self.sink_size, self.recent_size
        )
        totals = np.zeros(len(eligible), dtype=np.float64)
        for layer_index in range(len(self.score_layers)):
            for head in range(self.kv_heads):
                boundary = artifact_boundary(
                    view,
                    int(cycle),
                    layer_index,
                    head,
                    self.student.horizons,
                    self.sink_size,
                    self.recent_size,
                    1,
                    self.projector,
                    feature_only=True,
                )
                prediction = self.student.predict(boundary.features)
                totals += prediction[:, self.horizon_column].astype(np.float64)
        means = totals / float(len(self.score_layers) * self.kv_heads)
        return {
            int(position): float(value)
            for position, value in zip(eligible, means.tolist())
        }


def runtime_observation_from_record(
    record: Any,
    logits: torch.Tensor,
    backing: Any,
    score_layers: Sequence[int],
    query_heads: int,
) -> Dict[str, Any]:
    """Extract RuntimeStudentScorer inputs from a scoring-forward record."""

    def _tensor(value: Any) -> np.ndarray:
        return value.detach().float().cpu().numpy()

    post_rope = getattr(record, "post_rope_queries")
    residuals = getattr(record, "residual_inputs")
    attention_inputs = getattr(record, "attention_inputs")
    observation: Dict[str, Any] = {
        "post_rope_queries": {},
        "residual": {},
        "attention_input": {},
        "keys": {},
        "values": {},
        "global_features": _global_logit_features(logits),
    }
    for layer in score_layers:
        layer = int(layer)
        missing = [
            f"{layer}:{head}"
            for head in range(int(query_heads))
            if f"{layer}:{head}" not in post_rope
        ]
        if missing:
            raise RuntimeError(
                f"scoring record lacks post-RoPE queries for layer {layer}; "
                "the student policy requires all query heads captured"
            )
        observation["post_rope_queries"][layer] = np.stack(
            [
                _tensor(post_rope[f"{layer}:{head}"])
                for head in range(int(query_heads))
            ],
            axis=0,
        )
        observation["residual"][layer] = _tensor(residuals[layer]).reshape(-1)
        observation["attention_input"][layer] = _tensor(
            attention_inputs[layer]
        ).reshape(-1)
        layer_keys, layer_values = backing.layer_arrays(layer)
        observation["keys"][layer] = _tensor(layer_keys[0])
        observation["values"][layer] = _tensor(layer_values[0])
    return observation


# ------------------------------------------------------------------ training


def _train_student_mlp(
    features: np.ndarray,
    truth: np.ndarray,
    binary: np.ndarray,
    boundary_ids: np.ndarray,
    horizons: int,
    seed: int,
    epochs: int,
    device: str = "cpu",
) -> MultiHorizonMLP:
    """Same objective as causal_predictors._train_neural, device-pinned."""
    torch.manual_seed(int(seed))
    device_obj = torch.device(str(device))
    model = MultiHorizonMLP(int(features.shape[1]), int(horizons)).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    smooth = torch.nn.SmoothL1Loss()
    regression = np.log1p(truth / np.maximum(truth.mean(axis=0), 1.0e-9))
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        groups = [
            np.flatnonzero(boundary_ids == boundary_id)
            for boundary_id in rng.permutation(np.unique(boundary_ids))
        ]
        for rows in groups:
            x = torch.from_numpy(features[rows]).to(device_obj)
            y_class = torch.from_numpy(binary[rows]).to(device_obj)
            y_reg = torch.from_numpy(regression[rows].astype(np.float32)).to(device_obj)
            output = model(
                x, torch.zeros((len(rows), 1, 2), dtype=torch.float32, device=device_obj)
            )
            loss = bce(output[:, :, 0], y_class) + 0.25 * smooth(
                output[:, :, 1], y_reg
            )
            if len(rows) > 1:
                paired = torch.roll(torch.arange(len(rows), device=device_obj), 1)
                true_difference = y_reg - y_reg[paired]
                predicted_difference = output[:, :, 1] - output[paired, :, 1]
                informative = true_difference.abs() > 1.0e-6
                if bool(informative.any()):
                    pairwise = torch.nn.functional.softplus(
                        -true_difference.sign() * predicted_difference
                    )[informative].mean()
                    loss = loss + 0.10 * pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"[causal-student] mlp epoch {epoch + 1}/{epochs}", flush=True)
    return model.cpu()


def _v2_boundary_targets(
    truth: np.ndarray,
    boundary_ids: np.ndarray,
    cutoff_budgets: Sequence[int],
) -> Tuple[np.ndarray, Dict[int, Dict[Tuple[int, int], np.ndarray]]]:
    """Percentile targets and cutoff-straddling pairs per boundary.

    The v1 objective scores "is this token in the teacher top-220" and pairs
    tokens by arbitrary cyclic shift.  Deployment, however, only cares about
    the eviction ordering near the retention cutoff, at core budgets 92
    (budget 128) and 220 (budget 256).  v2 therefore trains channel 0 on the
    within-boundary teacher percentile (a scale-free soft label) and builds
    pair lists that straddle each deployment cutoff.  Pairs are keyed by
    (horizon column, budget): a pair straddling the H=32 cutoff does not
    necessarily straddle the H=1 cutoff, so each horizon gets its own lists.
    """

    horizons = int(truth.shape[1])
    percentile = np.zeros_like(truth, dtype=np.float32)
    pairs: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
    for boundary_id in np.unique(boundary_ids):
        rows = np.flatnonzero(boundary_ids == boundary_id)
        boundary_pairs: Dict[Tuple[int, int], np.ndarray] = {}
        for column in range(horizons):
            values = truth[rows, column]
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(rows), dtype=np.float32)
            ranks[order] = np.arange(len(rows), dtype=np.float32)
            percentile[rows, column] = ranks / max(float(len(rows) - 1), 1.0)
            for budget in cutoff_budgets:
                keep = min(int(budget), max(len(rows) - 1, 1))
                top_rows = rows[order[-keep:]]
                bottom_rows = rows[order[:-keep]]
                if not len(top_rows) or not len(bottom_rows):
                    continue
                # Every retained token against its nearest evicted neighbours
                # dominates the eviction decision; cap the pair count so the
                # per-boundary loss stays cheap.  Iterating highs from the
                # cutoff upward concentrates pairs on the near-cutoff region.
                closest = bottom_rows[
                    np.argsort(values[order[:-keep]])[ -min(keep, len(bottom_rows)): ]
                ]
                pair_list = [
                    (int(high), int(low))
                    for high in top_rows
                    for low in closest
                ]
                pair_list = pair_list[: min(len(pair_list), 4 * int(budget))]
                boundary_pairs[(column, int(budget))] = np.asarray(
                    pair_list, dtype=np.int64
                ).reshape(-1, 2)
        pairs[int(boundary_id)] = boundary_pairs
    return percentile, pairs


def _train_student_mlp_v2(
    features: np.ndarray,
    truth: np.ndarray,
    boundary_ids: np.ndarray,
    horizons: int,
    seed: int,
    epochs: int,
    cutoff_budgets: Sequence[int],
    pairwise_weight: float,
    device: str = "cpu",
) -> MultiHorizonMLP:
    """Cutoff-aware ranking objective on top of the v1 architecture.

    Channel 0 regresses the within-boundary teacher percentile (BCE on soft
    labels); channel 1 keeps the v1 log-utility regression.  A pairwise
    logistic loss on cutoff-straddling pairs at each deployment core budget
    directly optimizes the retention/eviction ordering the closed loop uses.
    """

    torch.manual_seed(int(seed))
    device_obj = torch.device(str(device))
    model = MultiHorizonMLP(int(features.shape[1]), int(horizons)).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    smooth = torch.nn.SmoothL1Loss()
    regression = np.log1p(truth / np.maximum(truth.mean(axis=0), 1.0e-9))
    percentile, pairs = _v2_boundary_targets(truth, boundary_ids, cutoff_budgets)
    rng = np.random.default_rng(int(seed))
    for epoch in range(int(epochs)):
        groups = [
            (int(boundary_id), np.flatnonzero(boundary_ids == boundary_id))
            for boundary_id in rng.permutation(np.unique(boundary_ids))
        ]
        for boundary_id, rows in groups:
            x = torch.from_numpy(features[rows]).to(device_obj)
            y_soft = torch.from_numpy(percentile[rows]).to(device_obj)
            y_reg = torch.from_numpy(regression[rows].astype(np.float32)).to(device_obj)
            output = model(
                x, torch.zeros((len(rows), 1, 2), dtype=torch.float32, device=device_obj)
            )
            loss = bce(output[:, :, 0], y_soft) + 0.25 * smooth(
                output[:, :, 1], y_reg
            )
            row_lookup = {
                int(global_row): local_row
                for local_row, global_row in enumerate(rows.tolist())
            }
            boundary_pairs = pairs.get(boundary_id, {})
            n_pair_terms = 0
            pairwise_total = None
            for (column, budget), pair_array in boundary_pairs.items():
                if pair_array is None or not len(pair_array):
                    continue
                local_pairs = np.asarray(
                    [
                        (row_lookup[int(high)], row_lookup[int(low)])
                        for high, low in pair_array.tolist()
                    ],
                    dtype=np.int64,
                )
                high = torch.from_numpy(local_pairs[:, 0]).to(device_obj)
                low = torch.from_numpy(local_pairs[:, 1]).to(device_obj)
                difference = output[high, column, 0] - output[low, column, 0]
                term = torch.nn.functional.softplus(-difference).mean()
                pairwise_total = (
                    term if pairwise_total is None else pairwise_total + term
                )
                n_pair_terms += 1
            if pairwise_total is not None:
                loss = loss + float(pairwise_weight) * pairwise_total / max(
                    n_pair_terms, 1
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(
            f"[causal-student] mlp-v2 epoch {epoch + 1}/{epochs}", flush=True
        )
    return model.cpu()


# --------------------------------------------------------------- v3 ranking


def cutoff_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    k: int,
    n_pairs: int = 4096,
) -> Dict[str, float]:
    """Selection-fidelity metrics at one retention budget.

    - topk recall / Jaccard between the teacher top-k and student top-k sets
    - cutoff pair accuracy: fraction of (teacher-retained, teacher-evicted)
      pairs the student orders correctly
    - band pair accuracy: ordering agreement inside the near-cutoff band
      (teacher ranks k±B), where eviction decisions are actually decided
    """

    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    n = len(truth)
    k = min(int(k), n - 1)
    oracle = set(topk_indices(truth, k).tolist())
    selected = set(topk_indices(prediction, k).tolist())
    inter = len(oracle & selected)
    order = np.argsort(truth, kind="stable")
    cut = n - k
    rng = np.random.default_rng(0)
    hi_all = order[cut:]
    lo_all = order[:cut]
    take_hi = rng.choice(hi_all, size=min(len(hi_all), 64), replace=False)
    take_lo = rng.choice(lo_all, size=min(len(lo_all), 64), replace=False)
    pairs = np.asarray(
        [(i, j) for i in take_hi for j in take_lo], dtype=np.int64
    ).reshape(-1, 2)
    if len(pairs) > n_pairs:
        pairs = pairs[rng.choice(len(pairs), size=n_pairs, replace=False)]
    diff = prediction[pairs[:, 0]] - prediction[pairs[:, 1]]
    cutoff_acc = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
    band = max(16, k // 8)
    band_rows = order[max(0, cut - band) : min(n, cut + band)]
    band_pairs = [
        (band_rows[a + 1], band_rows[a]) for a in range(len(band_rows) - 1)
    ]
    if band_pairs:
        bp = np.asarray(band_pairs, dtype=np.int64)
        bd = prediction[bp[:, 0]] - prediction[bp[:, 1]]
        band_acc = float((bd > 0).mean() + 0.5 * (bd == 0).mean())
    else:
        band_acc = 0.0
    return {
        "topk_recall": float(inter / max(k, 1)),
        "jaccard": float(inter / max(2 * k - inter, 1)),
        "cutoff_pair_accuracy": cutoff_acc,
        "band_pair_accuracy": band_acc,
    }


def _v3_pairs(
    values: np.ndarray,
    k_sub: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cutoff-weighted pair set for one boundary/horizon.

    Heavy weight on pairs straddling the retention cutoff (both near and
    very-near bands), light weight on random pairs with distance-decayed
    weights so easy pairs cannot drown the cutoff signal.
    """

    n = len(values)
    k_sub = min(int(k_sub), n - 1)
    order = np.argsort(values, kind="stable")
    cut = n - k_sub
    pairs: List[Tuple[int, int]] = []
    weights: List[float] = []
    for band, weight in ((max(8, k_sub // 8), 5.0), (max(24, k_sub // 3), 3.0)):
        hi = order[cut : min(n, cut + band)]
        lo = order[max(0, cut - band) : cut]
        for i in hi:
            for j in lo:
                pairs.append((int(i), int(j)))
                weights.append(weight)
    n_random = min(256, n * (n - 1) // 2)
    if n_random:
        ri = rng.integers(0, n, size=n_random)
        rj = rng.integers(0, n, size=n_random)
        valid = ri != rj
        for i, j in zip(ri[valid].tolist(), rj[valid].tolist()):
            if values[i] == values[j]:
                continue
            high, low = (i, j) if values[i] > values[j] else (j, i)
            rank_hi = int(np.flatnonzero(order == high)[0])
            rank_lo = int(np.flatnonzero(order == low)[0])
            proximity = np.exp(-min(abs(rank_hi - cut), abs(rank_lo - cut)) / max(k_sub * 0.3, 1.0))
            straddle = (rank_hi >= cut) != (rank_lo >= cut)
            pairs.append((high, low))
            weights.append(float((1.5 if straddle else 0.4) * (0.5 + proximity)))
    return (
        np.asarray(pairs, dtype=np.int64).reshape(-1, 2),
        np.asarray(weights, dtype=np.float32),
    )


def _train_student_mlp_v3(
    features: np.ndarray,
    truth: np.ndarray,
    boundary_ids: np.ndarray,
    sizes: np.ndarray,
    horizons: int,
    seed: int,
    epochs: int,
    core_budget: int,
    device: str = "cpu",
    init_state: Optional[Mapping[str, Any]] = None,
    extra_pairs: Optional[Mapping[int, np.ndarray]] = None,
    extra_weight: float = 5.0,
) -> MultiHorizonMLP:
    """Pure ranking distillation: cutoff-weighted pairwise logistic loss.

    Channel 0 is the ranking score (deployment channel).  A small soft
    percentile BCE keeps the channel globally calibrated; channel 1 keeps a
    light log-utility regression as a regularizer.  ``extra_pairs`` injects
    hard-negative-mined pairs (global row indices, deployment horizon H1).
    """

    torch.manual_seed(int(seed))
    device_obj = torch.device(str(device))
    model = MultiHorizonMLP(int(features.shape[1]), int(horizons)).to(device_obj)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    bce = torch.nn.BCEWithLogitsLoss()
    smooth = torch.nn.SmoothL1Loss()
    regression = np.log1p(truth / np.maximum(truth.mean(axis=0), 1.0e-9))
    rng = np.random.default_rng(int(seed))

    percentile = np.zeros_like(truth, dtype=np.float32)
    pair_cache: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for boundary_id in np.unique(boundary_ids):
        rows = np.flatnonzero(boundary_ids == boundary_id)
        full = int(sizes[rows[0]])
        k_sub = max(8, int(round(int(core_budget) * len(rows) / max(full, 1))))
        per_h: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for column in range(int(horizons)):
            values = truth[rows, column]
            order = np.argsort(values, kind="stable")
            ranks = np.empty(len(rows), dtype=np.float32)
            ranks[order] = np.arange(len(rows), dtype=np.float32)
            percentile[rows, column] = ranks / max(float(len(rows) - 1), 1.0)
            local_pairs, pair_weights = _v3_pairs(values, k_sub, rng)
            pair_cache.setdefault(int(boundary_id), {})[column] = (
                rows[local_pairs[:, 0]],
                rows[local_pairs[:, 1]],
                pair_weights,
            )
    if extra_pairs:
        for boundary_id, mined in extra_pairs.items():
            hi = np.asarray([int(p[0]) for p in mined], dtype=np.int64)
            lo = np.asarray([int(p[1]) for p in mined], dtype=np.int64)
            w = np.full(len(mined), float(extra_weight), dtype=np.float32)
            pair_cache[int(boundary_id)][0] = (
                np.concatenate([pair_cache[int(boundary_id)][0][0], hi]),
                np.concatenate([pair_cache[int(boundary_id)][0][1], lo]),
                np.concatenate([pair_cache[int(boundary_id)][0][2], w]),
            )

    for epoch in range(int(epochs)):
        groups = [
            (int(boundary_id), np.flatnonzero(boundary_ids == boundary_id))
            for boundary_id in rng.permutation(np.unique(boundary_ids))
        ]
        for boundary_id, rows in groups:
            x = torch.from_numpy(features[rows]).to(device_obj)
            y_soft = torch.from_numpy(percentile[rows]).to(device_obj)
            y_reg = torch.from_numpy(regression[rows].astype(np.float32)).to(device_obj)
            output = model(
                x, torch.zeros((len(rows), 1, 2), dtype=torch.float32, device=device_obj)
            )
            loss = 0.10 * bce(output[:, :, 0], y_soft) + 0.05 * smooth(
                output[:, :, 1], y_reg
            )
            row_lookup = {
                int(global_row): local_row
                for local_row, global_row in enumerate(rows.tolist())
            }
            for column, (hi, lo, w) in pair_cache[boundary_id].items():
                local_hi = torch.from_numpy(
                    np.asarray([row_lookup[int(g)] for g in hi.tolist()], dtype=np.int64)
                ).to(device_obj)
                local_lo = torch.from_numpy(
                    np.asarray([row_lookup[int(g)] for g in lo.tolist()], dtype=np.int64)
                ).to(device_obj)
                weight_t = torch.from_numpy(w).to(device_obj)
                diff = output[local_hi, column, 0] - output[local_lo, column, 0]
                loss = loss + (weight_t * torch.nn.functional.softplus(-diff)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f"[causal-student] mlp-v3 epoch {epoch + 1}/{epochs}", flush=True)
    return model.cpu()


def mine_cutoff_errors(
    scorer: "StudentScorer",
    features: np.ndarray,
    truth: np.ndarray,
    boundary_ids: np.ndarray,
    sizes: np.ndarray,
    core_budget: int,
    horizon_column: int = 0,
) -> Dict[int, np.ndarray]:
    """Hard-negative mining at the deployment horizon.

    For every boundary: FN = teacher top-k the student evicts, FP = student
    top-k the teacher evicts.  Returns {boundary_id: (FN x FP) pair array of
    global row indices} for retraining with elevated weight.
    """

    mined: Dict[int, List[Tuple[int, int]]] = {}
    total_fn = 0
    total_fp = 0
    for boundary_id in np.unique(boundary_ids):
        rows = np.flatnonzero(boundary_ids == boundary_id)
        full = int(sizes[rows[0]])
        k_sub = max(8, int(round(int(core_budget) * len(rows) / max(full, 1))))
        prediction = scorer.predict(features[rows])[:, horizon_column]
        values = truth[rows, horizon_column]
        teacher_top = set(topk_indices(values, k_sub).tolist())
        student_top = set(topk_indices(prediction, k_sub).tolist())
        fn = sorted(teacher_top - student_top)
        fp = sorted(student_top - teacher_top)
        total_fn += len(fn)
        total_fp += len(fp)
        if fn and fp:
            mined[int(boundary_id)] = [
                (int(rows[i]), int(rows[j])) for i in fn for j in fp
            ]
    print(
        f"[causal-student] hard-negative mining: {total_fn} missed-retains, "
        f"{total_fp} false-retains across {len(np.unique(boundary_ids))} boundaries",
        flush=True,
    )
    return mined


def train_students(config_path: Path, repository_root: Path) -> Path:
    """Train the GBDT and MLP R2 students and evaluate them on validation."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repository_root = Path(repository_root)
    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"])
    output_root = repository_root / str(config["output_models"])
    output_root.mkdir(parents=True, exist_ok=True)
    horizons = [int(value) for value in config["future_utility_horizons"]]
    seed = int(config["data_seed"])

    train_ids = [
        sample_id_for(str(family), int(index))
        for family in config["task_families"]
        for index in config["distillation"]["train_indices"]
    ]
    artifact_paths = [
        source_run / "artifacts" / "train" / f"{_safe_sample_id(sample_id)}.npz"
        for sample_id in train_ids
    ]
    expected = int(config["distillation"]["expected_train_sequences"])
    if len(artifact_paths) != expected:
        raise RuntimeError(
            f"expected {expected} train artifacts, found {len(artifact_paths)}"
        )
    missing = [path.name for path in artifact_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"student train artifacts are missing: {missing}")
    started = time.perf_counter()
    features, histories, truth, binary, boundary_ids = _teacher_arrays(
        artifact_paths, teacher_root / "train", config
    )
    del histories
    print(
        f"[causal-student] sampled {len(features)} token rows from "
        f"{len(artifact_paths)} sequences",
        flush=True,
    )
    scaler = StandardScaler().fit(features)
    normalized = scaler.transform(features).astype(np.float32)

    subset = np.linspace(
        0, len(normalized) - 1, num=min(200000, len(normalized)), dtype=np.int64
    )
    gbdt_models: Dict[int, Any] = {}
    for column, horizon in enumerate(horizons):
        print(f"[causal-student] hist_gbdt horizon H={horizon}", flush=True)
        gbdt_models[int(horizon)] = HistGradientBoostingRegressor(
            max_iter=int(config["student"]["gbdt_max_iter"]),
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ).fit(normalized[subset], np.log1p(truth[subset, column]))
    gbdt_checkpoint = save_student_checkpoint(
        output_root / "r2_student_hist_gbdt.joblib",
        kind="hist_gbdt",
        models=gbdt_models,
        scaler=scaler,
        horizons=horizons,
        projector_seed=seed,
        metadata={
            "teacher": R2_TEACHER,
            "target": "log1p(per-token R2 future-utility score)",
            "train_sequences": len(artifact_paths),
            "sampled_token_rows": int(len(features)),
            "runtime_future_access": False,
            "feature_segments": {
                name: [start, stop]
                for name, (start, stop, _) in FEATURE_SEGMENTS.items()
            },
        },
    )

    mlp = _train_student_mlp(
        normalized,
        truth,
        binary,
        boundary_ids,
        len(horizons),
        seed + 41,
        epochs=int(config["student"]["mlp_epochs"]),
        device=str(config["student"].get("device", "cpu")),
    )
    mlp_checkpoint = save_student_checkpoint(
        output_root / "r2_student_mlp.pt",
        kind="mlp",
        models=mlp.state_dict(),
        scaler=scaler,
        horizons=horizons,
        projector_seed=seed,
        score_channel=1,
        metadata={
            "teacher": R2_TEACHER,
            "objective": "topk BCE + log-utility regression + pairwise ranking",
            "train_sequences": len(artifact_paths),
            "sampled_token_rows": int(len(features)),
            "runtime_future_access": False,
        },
    )

    checkpoints = {
        "hist_gbdt": str(gbdt_checkpoint.relative_to(repository_root)),
        "mlp": str(mlp_checkpoint.relative_to(repository_root)),
    }
    scorers = {
        "student_hist_gbdt": StudentScorer(load_student_checkpoint(gbdt_checkpoint)),
        "student_mlp": StudentScorer(load_student_checkpoint(mlp_checkpoint)),
    }
    objective = str(config["student"].get("objective", "v1"))
    if objective == "v2":
        mlp_v2 = _train_student_mlp_v2(
            normalized,
            truth,
            boundary_ids,
            len(horizons),
            seed + 43,
            epochs=int(config["student"]["mlp_epochs"]),
            cutoff_budgets=[
                int(value)
                for value in config["student"].get(
                    "cutoff_budgets", [92, int(config["core_budget"])]
                )
            ],
            pairwise_weight=float(config["student"].get("pairwise_weight", 1.0)),
            device=str(config["student"].get("device", "cpu")),
        )
        mlp_v2_checkpoint = save_student_checkpoint(
            output_root / "r2_student_mlp_v2.pt",
            kind="mlp",
            models=mlp_v2.state_dict(),
            scaler=scaler,
            horizons=horizons,
            projector_seed=seed,
            score_channel=int(config["student"].get("v2_score_channel", 0)),
            metadata={
                "teacher": R2_TEACHER,
                "objective": (
                    "percentile BCE + log-utility regression + "
                    "cutoff-straddling pairwise ranking at "
                    f"{config['student'].get('cutoff_budgets', [92, int(config['core_budget'])])}"
                ),
                "train_sequences": len(artifact_paths),
                "sampled_token_rows": int(len(features)),
                "runtime_future_access": False,
            },
        )
        checkpoints["mlp_v2"] = str(mlp_v2_checkpoint.relative_to(repository_root))
        scorers["student_mlp_v2"] = StudentScorer(
            load_student_checkpoint(mlp_v2_checkpoint)
        )
    elif objective != "v1":
        raise RuntimeError(f"unknown student objective: {objective}")
    rows = evaluate_students(config, repository_root, scorers)
    frame = pd.DataFrame(rows)
    atomic_frame(frame, output_root / "validation_boundary_metrics.parquet")
    summary = (
        frame.groupby(["method", "future_horizon"], as_index=False)
        .agg(
            spearman=("spearman", "mean"),
            pairwise_accuracy=("pairwise_accuracy", "mean"),
            topk_recall=("future_topk_recall", "mean"),
            ndcg=("ndcg", "mean"),
            oracle_gap_recovery=("oracle_gap_recovery", "mean"),
            boundaries=("spearman", "size"),
            sequences=("sample_id", "nunique"),
        )
    )
    atomic_frame(summary, output_root / "validation_summary.csv")
    atomic_json(
        output_root / "training_summary.json",
        {
            "format": STUDENT_FORMAT,
            "teacher": R2_TEACHER,
            "feature_width": FEATURE_WIDTH,
            "feature_segments": {
                name: {"slice": [start, stop], "source": source}
                for name, (start, stop, source) in FEATURE_SEGMENTS.items()
            },
            "dropped_features_no_runtime_source": [],
            "horizons": horizons,
            "objective": objective,
            "checkpoints": checkpoints,
            "elapsed_s": float(time.perf_counter() - started),
        },
    )
    print(summary.to_string(index=False), flush=True)
    return output_root


def evaluate_students(
    config: Mapping[str, Any],
    repository_root: Path,
    scorers: Mapping[str, StudentScorer],
    split: str = "validation",
) -> List[Dict[str, Any]]:
    """Rank-quality metrics of each student against the R2 teacher."""

    repository_root = Path(repository_root)
    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"]) / str(split)
    artifact_dir = source_run / "artifacts" / str(split)
    horizons = [int(value) for value in config["future_utility_horizons"]]
    fixed_rhos = json.loads(
        (source_run / "models" / "fixed_baseline_tuning.json").read_text(
            encoding="utf-8"
        )
    )["per_head"]
    projector = FixedProjector(int(config["data_seed"]))
    rows: List[Dict[str, Any]] = []
    artifact_paths = sorted(artifact_dir.glob("*.npz"))
    if not artifact_paths:
        raise RuntimeError(f"no {split} artifacts in {artifact_dir}")
    for ordinal, artifact_path in enumerate(artifact_paths, start=1):
        teacher_path = teacher_root / artifact_path.name
        if not teacher_path.exists():
            raise RuntimeError(f"missing teacher scores: {teacher_path}")
        artifact = _load_npz(artifact_path)
        teacher = _load_npz(teacher_path)
        teacher_horizons = [int(value) for value in teacher["horizons"]]
        horizon_rows = [teacher_horizons.index(value) for value in horizons]
        for cycle_index, cycle_value in enumerate(teacher["cycles"]):
            cycle = int(cycle_value)
            count = int(teacher["position_lengths"][cycle_index])
            teacher_positions = [
                int(value) for value in teacher["position_ids"][cycle_index, :count]
            ]
            current_count = int(artifact["position_lengths"][cycle])
            positions = [
                int(value)
                for value in artifact["position_ids"][cycle, :current_count]
            ]
            _, _, eligible = mandatory_and_eligible(
                positions, int(config["sink_size"]), int(config["recent_size"])
            )
            if teacher_positions != [int(value) for value in eligible]:
                raise RuntimeError(
                    "teacher positions do not match feature positions: "
                    f"{artifact_path.name} cycle {cycle}"
                )
            for layer_index in range(int(artifact["layers"].size)):
                for head in range(int(artifact["attention"].shape[2])):
                    boundary = artifact_boundary(
                        artifact,
                        cycle,
                        layer_index,
                        head,
                        horizons,
                        int(config["sink_size"]),
                        int(config["recent_size"]),
                        int(config["core_budget"]),
                        projector,
                        fixed_rhos,
                    )
                    truth = np.take(
                        teacher["scores"][cycle_index, :, layer_index, head, :count],
                        horizon_rows,
                        axis=0,
                    ).T.astype(np.float32)
                    predictions = {
                        name: scorer.predict(boundary.features)
                        for name, scorer in scorers.items()
                    }
                    for name, prediction in predictions.items():
                        for column, horizon in enumerate(horizons):
                            rows.append(
                                {
                                    "sample_id": boundary.sample_id,
                                    "task": boundary.task,
                                    "split": str(split),
                                    "cycle": cycle,
                                    "layer": boundary.layer,
                                    "head": head,
                                    "method": name,
                                    "future_horizon": horizon,
                                    **boundary_metrics(
                                        truth[:, column],
                                        prediction[:, column],
                                        boundary.baseline[:, column],
                                        int(config["core_budget"]),
                                    ),
                                }
                            )
        print(
            f"[causal-student] {split} {ordinal}/{len(artifact_paths)} {artifact_path.stem}",
            flush=True,
        )
    return rows


def evaluate_students_cutoff(
    config: Mapping[str, Any],
    repository_root: Path,
    scorers: Mapping[str, StudentScorer],
    ks: Sequence[int] = (220, 476),
    split: str = "validation",
    horizon: int = 1,
) -> List[Dict[str, Any]]:
    """Selection-fidelity battery at the deployment horizon and budgets.

    Unlike :func:`evaluate_students` (rank-quality metrics at the training
    budget), this reports top-B recall / Jaccard / cutoff-pair accuracy at
    the two deployment budgets (k=220 ~ budget 256, k=476 ~ budget 512) on
    the full eligible sets, aggregated per boundary across layers/heads.
    """

    repository_root = Path(repository_root)
    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"]) / str(split)
    artifact_dir = source_run / "artifacts" / str(split)
    horizons = [int(value) for value in config["future_utility_horizons"]]
    if int(horizon) not in horizons:
        raise RuntimeError(f"deployment horizon {horizon} not in {horizons}")
    horizon_col = horizons.index(int(horizon))
    projector = FixedProjector(int(config["data_seed"]))
    rows: List[Dict[str, Any]] = []
    artifact_paths = sorted(artifact_dir.glob("*.npz"))
    if not artifact_paths:
        raise RuntimeError(f"no {split} artifacts in {artifact_dir}")
    for ordinal, artifact_path in enumerate(artifact_paths, start=1):
        teacher_path = teacher_root / artifact_path.name
        if not teacher_path.exists():
            continue
        artifact = _load_npz(artifact_path)
        teacher = _load_npz(teacher_path)
        teacher_h1 = [int(v) for v in teacher["horizons"]].index(int(horizon))
        for cycle_index, cycle_value in enumerate(teacher["cycles"]):
            cycle = int(cycle_value)
            count = int(teacher["position_lengths"][cycle_index])
            teacher_positions = [
                int(value) for value in teacher["position_ids"][cycle_index, :count]
            ]
            current_count = int(artifact["position_lengths"][cycle])
            positions = [
                int(value)
                for value in artifact["position_ids"][cycle, :current_count]
            ]
            _, _, eligible = mandatory_and_eligible(
                positions, int(config["sink_size"]), int(config["recent_size"])
            )
            if teacher_positions != [int(value) for value in eligible]:
                raise RuntimeError(
                    "teacher positions do not match feature positions: "
                    f"{artifact_path.name} cycle {cycle}"
                )
            # mean teacher/student scores across diagnostic layers and KV
            # heads: this is the aggregation the closed loop actually deploys
            layer_count = int(artifact["layers"].size)
            head_count = int(artifact["attention"].shape[2])
            truth_stack = []
            pred_stacks = {name: [] for name in scorers}
            for layer_index in range(layer_count):
                for head in range(head_count):
                    boundary = artifact_boundary(
                        artifact,
                        cycle,
                        layer_index,
                        head,
                        horizons,
                        int(config["sink_size"]),
                        int(config["recent_size"]),
                        int(config["core_budget"]),
                        projector,
                        feature_only=True,
                    )
                    truth_stack.append(
                        teacher["scores"][
                            cycle_index, teacher_h1, layer_index, head, :count
                        ]
                    )
                    for name, scorer in scorers.items():
                        pred_stacks[name].append(
                            scorer.predict(boundary.features)[:, horizon_col]
                        )
            truth_mean = np.mean(np.stack(truth_stack), axis=0)
            for name in scorers:
                pred_mean = np.mean(np.stack(pred_stacks[name]), axis=0)
                row = {
                    "sample_id": str(teacher["sample_id"].item()),
                    "task": str(teacher["task"].item()),
                    "split": str(split),
                    "cycle": cycle,
                    "method": name,
                    "horizon": int(horizon),
                }
                for k in ks:
                    row.update(
                        {
                            f"{key}@{k}": value
                            for key, value in cutoff_metrics(
                                truth_mean, pred_mean, int(k)
                            ).items()
                        }
                    )
                rows.append(row)
        print(
            f"[cutoff-eval] {split} {ordinal}/{len(artifact_paths)} {artifact_path.stem}",
            flush=True,
        )
    return rows


__all__ = [
    "FEATURE_SEGMENTS",
    "FEATURE_WIDTH",
    "FORBIDDEN_RUNTIME_KEYS",
    "JOIN_KEYS",
    "R2_TEACHER",
    "RuntimeFeatureHistory",
    "RuntimeStudentScorer",
    "StudentScorer",
    "cutoff_metrics",
    "dump_teacher_scores",
    "evaluate_students",
    "evaluate_students_cutoff",
    "load_student_checkpoint",
    "mine_cutoff_errors",
    "runtime_observation_from_record",
    "save_student_checkpoint",
    "train_students",
]
