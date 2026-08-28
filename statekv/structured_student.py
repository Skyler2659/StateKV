"""Structure-preserving R2 student (STATEKV_STRUCTURED_STUDENT).

The 120-dim ``artifact_boundary`` student pools every signal over the 6
diagnostic layers x 8 KV heads and scores each token independently; it
plateaus at top-220 recall ~0.71 and near-cutoff band accuracy ~0.51.
This module keeps the per-(layer, head) structure explicit:

- :func:`structured_boundary` builds, for one boundary (sequence, cycle),
  per-(token, layer, head) features (19 dims), per-layer state features
  (40 dims), per-token position features, and global features.  It consumes
  an artifact-shaped mapping, so the training path (real artifact npz) and
  the runtime path (``RuntimeFeatureHistory.artifact_view()``) share one
  implementation and cannot drift.  Every feature uses only cycles <= the
  current cycle.
- :class:`StructuredStudent` is a small deployable network: per-head MLP
  encoder with layer/head embeddings, mean/max/attention pooling over the
  48 (layer, head) pairs, a per-layer state encoder, a DeepSets context
  over eligible tokens, and a final per-token scorer S_i.
- :func:`train_structured_student` distills the R2 teacher H=1 scores with
  a cutoff-weighted pairwise loss on S_i plus a soft-percentile BCE and a
  per-head auxiliary percentile BCE on the per-head readout s_{i,l,h}.
- :class:`RuntimeStructuredScorer` deploys the checkpoint inside the strict
  closed loop with the same interface as ``RuntimeStudentScorer``.

No feature reads rollout, future cycles, or oracle data.  The artifact keys
``generated_token_ids`` / ``current_token_ids`` / ``label_source`` are never
touched.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler

from statekv.causal_existence import sample_id_for
from statekv.causal_existence_analysis import topk_indices
from statekv.causal_predictors import FixedProjector, _history_features, _load_npz
from statekv.storage import safe_path_component
from statekv.causal_student import (
    R2_TEACHER,
    RuntimeFeatureHistory,
    StudentScorer,
    _v3_pairs,
    cutoff_metrics,
    load_student_checkpoint,
    save_student_checkpoint,
)
from statekv.selectors import mandatory_and_eligible


STRUCTURED_KIND = "structured_mlp"
HEAD_FEATURE_WIDTH = 19
# Column layout of the per-(token, layer, head) feature vector.
HEAD_FEATURE_SEGMENTS: Dict[str, Tuple[int, int, str]] = {
    "attention_history_scalars": (0, 10, "current/EMA/mean/max/lag/slope/rank/drift of per-cycle attention"),
    "qk_geometry": (10, 14, "cached keys x current query-group dot statistics"),
    "query_trajectory": (14, 17, "query movement (lags 1/2/4) dotted with cached keys"),
    "kv_norms": (17, 19, "key norm, value norm"),
}
QUERY_TRAJECTORY_COLUMNS = (14, 15, 16)
STATE_FEATURE_WIDTH = 40
TOKEN_FEATURE_WIDTH = 2
GLOBAL_FEATURE_WIDTH = 6
TRAJECTORY_LAGS = (1, 2, 4)

ABLATIONS = (
    "full",
    "no_query_traj",
    "no_head_identity",
    "no_context",
    "no_head_structure",
    "no_perhead_aux",
)


# ------------------------------------------------------------------ features


def structured_boundary(
    artifact: Mapping[str, np.ndarray],
    cycle: int,
    sink_size: int,
    recent_size: int,
    projector: FixedProjector,
    drop_query_traj: bool = False,
) -> Dict[str, np.ndarray]:
    """Structured causal features for every eligible token at ``cycle``.

    Returns ``positions`` (n,), ``head_features`` (n, L, H, 19),
    ``state_features`` (L, 40), ``token_features`` (n, 2) and
    ``global_features`` (6,).  Only artifact rows at or before ``cycle`` are
    read; NaN attention history (positions not yet present) is handled by
    the shared ``_history_features`` helpers.
    """

    cycle = int(cycle)
    attention = artifact["attention"]
    layer_count = int(attention.shape[1])
    kv_heads = int(attention.shape[2])
    count = int(artifact["position_lengths"][cycle])
    positions = [int(value) for value in artifact["position_ids"][cycle, :count]]
    _, _, eligible = mandatory_and_eligible(positions, sink_size, recent_size)
    row_by_position = {position: row for row, position in enumerate(positions)}
    candidate_rows = np.asarray(
        [row_by_position[int(position)] for position in eligible], dtype=np.int64
    )
    kv_row_by_position = {
        int(position): row for row, position in enumerate(artifact["kv_position_ids"])
    }
    kv_rows = np.asarray(
        [kv_row_by_position[int(position)] for position in eligible], dtype=np.int64
    )
    n = int(len(eligible))
    query_heads = int(artifact["query_post"].shape[2])
    group = query_heads // kv_heads

    head_features = np.zeros(
        (n, layer_count, kv_heads, HEAD_FEATURE_WIDTH), dtype=np.float32
    )
    state_features = np.zeros(
        (layer_count, STATE_FEATURE_WIDTH), dtype=np.float32
    )
    history_columns = [0, 8, 9, 10, 2, 3, 4, 6, 12, 13]
    for layer_index in range(layer_count):
        query_layer = np.asarray(
            artifact["query_post"][cycle, layer_index], dtype=np.float32
        )
        residual_raw = np.asarray(
            artifact["residual"][cycle, layer_index], dtype=np.float32
        )
        attention_input_raw = np.asarray(
            artifact["attention_input"][cycle, layer_index], dtype=np.float32
        )
        state_features[layer_index] = np.concatenate(
            [
                query_layer.mean(axis=0) @ projector.q,
                query_layer.std(axis=0) @ projector.q,
                residual_raw @ projector.state,
                attention_input_raw @ projector.state,
                np.asarray(
                    [
                        residual_raw.mean(),
                        residual_raw.std(),
                        np.linalg.norm(residual_raw) / math.sqrt(residual_raw.size),
                        np.max(np.abs(residual_raw)),
                        attention_input_raw.mean(),
                        attention_input_raw.std(),
                        np.linalg.norm(attention_input_raw)
                        / math.sqrt(attention_input_raw.size),
                        np.max(np.abs(attention_input_raw)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        for head in range(kv_heads):
            rows = attention[: cycle + 1, layer_index, head, candidate_rows]
            scalar, _ = _history_features(rows)
            # [current, ema0.9, ema0.5, ema0.99, mean, max, lag1,
            #  slope(current-lag4), current rank percentile, rank drift]
            history = scalar[:, history_columns]
            keys = np.asarray(
                artifact["keys"][layer_index, head, kv_rows], dtype=np.float32
            )
            values = np.asarray(
                artifact["values"][layer_index, head, kv_rows], dtype=np.float32
            )
            query = query_layer[head * group : (head + 1) * group]
            dots = keys @ query.T / math.sqrt(keys.shape[1])
            geometry = np.stack(
                [dots.mean(axis=1), dots.max(axis=1), dots.min(axis=1), dots.std(axis=1)],
                axis=1,
            )
            trajectory = np.zeros((n, len(TRAJECTORY_LAGS)), dtype=np.float32)
            for column, lag in enumerate(TRAJECTORY_LAGS):
                previous = np.asarray(
                    artifact["query_post"][
                        max(0, cycle - int(lag)),
                        layer_index,
                        head * group : (head + 1) * group,
                    ],
                    dtype=np.float32,
                )
                delta = (query - previous).mean(axis=0)
                trajectory[:, column] = keys @ delta
            norms = np.stack(
                [np.linalg.norm(keys, axis=1), np.linalg.norm(values, axis=1)],
                axis=1,
            )
            head_features[:, layer_index, head] = np.concatenate(
                [history, geometry, trajectory, norms], axis=1
            )
    if drop_query_traj:
        keep = [
            index
            for index in range(HEAD_FEATURE_WIDTH)
            if index not in QUERY_TRAJECTORY_COLUMNS
        ]
        head_features = head_features[..., keep]

    position_values = np.asarray(eligible, dtype=np.float32)
    maximum = max(1.0, float(max(positions)))
    token_features = np.stack(
        [position_values / maximum, (maximum - position_values) / maximum],
        axis=1,
    ).astype(np.float32)
    global_features = np.concatenate(
        [
            np.asarray(artifact["global_features"][cycle], dtype=np.float32),
            np.asarray(
                [
                    cycle / max(1, int(attention.shape[0]) - 1),
                    math.log1p(count) / math.log1p(4096),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return {
        "positions": np.asarray(eligible, dtype=np.int64),
        "head_features": head_features,
        "state_features": state_features,
        "token_features": token_features,
        "global_features": global_features,
    }


# --------------------------------------------------------------------- model


class StructuredStudent(nn.Module):
    """Per-(layer, head) structured scorer; no transformer over tokens."""

    def __init__(
        self,
        *,
        layers: int = 6,
        kv_heads: int = 8,
        head_feature_width: int = HEAD_FEATURE_WIDTH,
        state_feature_width: int = STATE_FEATURE_WIDTH,
        misc_width: int = TOKEN_FEATURE_WIDTH + GLOBAL_FEATURE_WIDTH,
        embed_width: int = 16,
        head_hidden: int = 96,
        head_out: int = 64,
        token_width: int = 128,
        state_width: int = 64,
        scorer_hidden: int = 256,
        use_identity: bool = True,
        use_context: bool = True,
        head_structure: bool = True,
    ):
        super().__init__()
        self.layers = int(layers)
        self.kv_heads = int(kv_heads)
        self.head_feature_width = int(head_feature_width)
        self.use_identity = bool(use_identity)
        self.use_context = bool(use_context)
        self.head_structure = bool(head_structure)
        self.layer_embedding = nn.Embedding(self.layers, embed_width)
        self.head_embedding = nn.Embedding(self.kv_heads, embed_width)
        if self.head_structure:
            self.head_encoder = nn.Sequential(
                nn.Linear(self.head_feature_width + 2 * embed_width, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, head_out),
            )
            self.head_readout = nn.Linear(head_out, 1)
            self.pool_query = nn.Parameter(torch.zeros(head_out))
            self.token_projection = nn.Linear(3 * head_out, token_width)
            scorer_input = token_width + state_width + misc_width
            if self.use_context:
                scorer_input += 2 * token_width
        else:
            # Pooled ablation: flatten (l, h) away and score with a plain MLP.
            scorer_input = self.head_feature_width + state_width + misc_width
        self.state_encoder = nn.Sequential(
            nn.Linear(state_feature_width, state_width),
            nn.GELU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input, scorer_hidden),
            nn.GELU(),
            nn.Linear(scorer_hidden, 1),
        )

    def forward(
        self,
        head_features: torch.Tensor,
        state_features: torch.Tensor,
        misc_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """head_features (n, L, H, F), state (L, S), misc (n, M).

        Returns (S_i (n,), s_{i,l,h} (n, L, H) or None for the pooled
        ablation).
        """

        n, layers, heads, _ = head_features.shape
        misc = misc_features
        state = self.state_encoder(state_features).mean(dim=0)  # (state_width,)
        if not self.head_structure:
            pooled = head_features.mean(dim=(1, 2))
            score = self.scorer(
                torch.cat(
                    [pooled, state.expand(n, -1), misc], dim=1
                )
            ).squeeze(-1)
            return score, None
        layer_ids = torch.arange(layers, device=head_features.device)
        head_ids = torch.arange(heads, device=head_features.device)
        if self.use_identity:
            layer_emb = self.layer_embedding(layer_ids)[None, :, None, :]
            head_emb = self.head_embedding(head_ids)[None, None, :, :]
        else:
            layer_emb = torch.zeros(
                (1, layers, 1, self.layer_embedding.embedding_dim),
                device=head_features.device,
                dtype=head_features.dtype,
            )
            head_emb = torch.zeros(
                (1, 1, heads, self.head_embedding.embedding_dim),
                device=head_features.device,
                dtype=head_features.dtype,
            )
        encoded = self.head_encoder(
            torch.cat(
                [
                    head_features,
                    layer_emb.expand(n, -1, heads, -1),
                    head_emb.expand(n, layers, -1, -1),
                ],
                dim=-1,
            )
        )  # (n, L, H, head_out)
        per_head_score = self.head_readout(encoded).squeeze(-1)  # (n, L, H)
        flat = encoded.reshape(n, layers * heads, -1)
        attention_logits = flat @ self.pool_query / math.sqrt(flat.shape[-1])
        attention_pool = (torch.softmax(attention_logits, dim=1)[..., None] * flat).sum(1)
        token = self.token_projection(
            torch.cat([flat.mean(dim=1), flat.max(dim=1).values, attention_pool], dim=1)
        )  # (n, token_width)
        parts = [token, state.expand(n, -1), misc]
        if self.use_context:
            context = torch.cat([token.mean(dim=0), token.max(dim=0).values], dim=0)
            parts.append(context.expand(n, -1))
        score = self.scorer(torch.cat(parts, dim=1)).squeeze(-1)
        return score, per_head_score


# --------------------------------------------------------------- checkpoints


class StructuredStudentScorer:
    """Checkpoint-backed scorer over :func:`structured_boundary` blocks."""

    def __init__(self, checkpoint: Mapping[str, Any]):
        if str(checkpoint.get("kind")) != STRUCTURED_KIND:
            raise RuntimeError(
                f"StructuredStudentScorer requires kind={STRUCTURED_KIND}, "
                f"got {checkpoint.get('kind')}"
            )
        self.architecture = dict(checkpoint["metadata"]["architecture"])
        self.scalers = checkpoint["scaler"]
        self.horizons = [int(value) for value in checkpoint["horizons"]]
        self.model = StructuredStudent(**self.architecture)
        self.model.load_state_dict(checkpoint["models"])
        self.model.eval()

    def normalize(self, boundary: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        head_features = boundary["head_features"]
        expected = int(self.architecture["head_feature_width"])
        if head_features.shape[-1] != expected:
            # Checkpoints trained without the query-trajectory group consume
            # the same boundary with those columns removed.
            keep = [
                index
                for index in range(head_features.shape[-1])
                if index not in QUERY_TRAJECTORY_COLUMNS
            ]
            head_features = head_features[..., keep]
            if head_features.shape[-1] != expected:
                raise RuntimeError(
                    f"boundary head width {boundary['head_features'].shape[-1]} "
                    f"does not match checkpoint width {expected}"
                )
        n = int(head_features.shape[0])
        head = self.scalers["head"].transform(
            head_features.reshape(-1, head_features.shape[-1])
        ).astype(np.float32)
        return {
            "head_features": head.reshape(head_features.shape),
            "state_features": self.scalers["state"]
            .transform(boundary["state_features"])
            .astype(np.float32),
            "token_features": self.scalers["token"]
            .transform(boundary["token_features"])
            .astype(np.float32),
            "global_features": self.scalers["global"]
            .transform(boundary["global_features"][None, :])
            .astype(np.float32)[0],
            "positions": boundary["positions"],
        }

    def predict(self, boundary: Mapping[str, np.ndarray]) -> np.ndarray:
        normalized = self.normalize(boundary)
        n = int(normalized["head_features"].shape[0])
        misc = np.concatenate(
            [
                np.broadcast_to(
                    normalized["global_features"][None, :], (n, GLOBAL_FEATURE_WIDTH)
                ),
                normalized["token_features"],
            ],
            axis=1,
        )
        with torch.no_grad():
            score, _ = self.model(
                torch.from_numpy(normalized["head_features"]),
                torch.from_numpy(normalized["state_features"]),
                torch.from_numpy(misc),
            )
        return score.numpy().astype(np.float32)


class RuntimeStructuredScorer:
    """Runtime-causal deployment of a structured student checkpoint.

    Same interface as ``RuntimeStudentScorer``: ``reset(total_cycles)`` then
    ``observe_and_score(...) -> {position: score}`` per cycle.  One artifact
    view and one model forward per cycle.
    """

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
        self.student = StructuredStudentScorer(checkpoint)
        self.projector = FixedProjector(int(checkpoint["projector_seed"]))
        if int(horizon) not in self.student.horizons:
            raise RuntimeError(
                f"structured student horizons {self.student.horizons} do not "
                f"cover the deployment horizon {horizon}"
            )
        self.horizon = int(horizon)
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
        if self.history is None:
            raise RuntimeError(
                "RuntimeStructuredScorer.reset() must precede scoring"
            )
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
        boundary = structured_boundary(
            view, int(cycle), self.sink_size, self.recent_size, self.projector
        )
        prediction = self.student.predict(boundary)
        return {
            int(position): float(value)
            for position, value in zip(boundary["positions"], prediction.tolist())
        }


# ------------------------------------------------------------ data assembly


def _percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(float(len(values) - 1), 1.0)


def _assemble_boundaries(
    artifact_paths: Sequence[Path],
    teacher_root: Path,
    config: Mapping[str, Any],
    projector: FixedProjector,
    *,
    cache_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build every (sequence x teacher cycle) boundary with H=1 targets.

    Joins are validated exactly like ``_teacher_arrays``: the teacher's
    ``position_ids`` at a cycle must equal the artifact's eligible set.
    """

    if cache_path is not None and Path(cache_path).exists():
        return joblib.load(cache_path)
    sink = int(config["sink_size"])
    recent = int(config["recent_size"])
    boundaries: List[Dict[str, Any]] = []
    for ordinal, artifact_path in enumerate(artifact_paths, start=1):
        teacher_path = Path(teacher_root) / artifact_path.name
        if not teacher_path.exists():
            raise RuntimeError(f"missing causal teacher scores: {teacher_path.name}")
        artifact = _load_npz(artifact_path)
        teacher = _load_npz(teacher_path)
        if str(teacher["teacher"].item()) != R2_TEACHER:
            raise RuntimeError(f"{teacher_path.name} is not an R2 teacher dump")
        teacher_h1 = [int(value) for value in teacher["horizons"]].index(1)
        for cycle_index, cycle_value in enumerate(teacher["cycles"]):
            cycle = int(cycle_value)
            count = int(teacher["position_lengths"][cycle_index])
            teacher_positions = [
                int(value) for value in teacher["position_ids"][cycle_index, :count]
            ]
            boundary = structured_boundary(artifact, cycle, sink, recent, projector)
            if teacher_positions != [int(value) for value in boundary["positions"]]:
                raise RuntimeError(
                    "teacher positions do not match feature positions: "
                    f"{artifact_path.name} cycle {cycle}"
                )
            per_head = np.asarray(
                teacher["scores"][cycle_index, teacher_h1, :, :, :count],
                dtype=np.float32,
            ).transpose(2, 0, 1)  # (L, H, n) -> (n, L, H)
            mean_teacher = per_head.mean(axis=(1, 2))
            boundaries.append(
                {
                    "sample_id": str(teacher["sample_id"].item()),
                    "task": str(teacher["task"].item()),
                    "cycle": cycle,
                    **boundary,
                    "teacher_mean": mean_teacher.astype(np.float32),
                    "teacher_per_head": per_head,
                    "target_mean_percentile": _percentile(mean_teacher),
                    "target_head_percentile": np.stack(
                        [
                            _percentile(per_head[:, layer, head])
                            for layer in range(per_head.shape[1])
                            for head in range(per_head.shape[2])
                        ],
                        axis=1,
                    ).reshape(per_head.shape),
                }
            )
        print(
            f"[structured-student] assembled {ordinal}/{len(artifact_paths)} "
            f"{artifact_path.stem}",
            flush=True,
        )
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(boundaries, cache_path, protocol=4)
    return boundaries


def _fit_scalers(boundaries: Sequence[Mapping[str, Any]]) -> Dict[str, StandardScaler]:
    scalers = {
        "head": StandardScaler(),
        "state": StandardScaler(),
        "token": StandardScaler(),
        "global": StandardScaler(),
    }
    for boundary in boundaries:
        head = boundary["head_features"]
        scalers["head"].partial_fit(head.reshape(-1, head.shape[-1]))
        scalers["state"].partial_fit(boundary["state_features"])
        scalers["token"].partial_fit(boundary["token_features"])
        scalers["global"].partial_fit(boundary["global_features"][None, :])
    return scalers


def _normalize_boundaries(
    boundaries: Sequence[Mapping[str, Any]], scalers: Mapping[str, StandardScaler]
) -> None:
    for boundary in boundaries:
        head = boundary["head_features"]
        boundary["head_features"] = (
            scalers["head"]
            .transform(head.reshape(-1, head.shape[-1]))
            .astype(np.float32)
            .reshape(head.shape)
        )
        boundary["state_features"] = (
            scalers["state"].transform(boundary["state_features"]).astype(np.float32)
        )
        boundary["token_features"] = (
            scalers["token"].transform(boundary["token_features"]).astype(np.float32)
        )
        boundary["global_features"] = (
            scalers["global"]
            .transform(boundary["global_features"][None, :])
            .astype(np.float32)[0]
        )


def _misc(boundary: Mapping[str, np.ndarray]) -> np.ndarray:
    n = int(boundary["head_features"].shape[0])
    return np.concatenate(
        [
            np.broadcast_to(
                boundary["global_features"][None, :], (n, GLOBAL_FEATURE_WIDTH)
            ),
            boundary["token_features"],
        ],
        axis=1,
    ).astype(np.float32)


# ------------------------------------------------------------------ training


def _train_one(
    boundaries: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    epochs: int,
    cutoff_budgets: Sequence[int],
    device: str,
    perhead_aux_weight: float = 0.3,
    soft_weight: float = 0.1,
    use_identity: bool = True,
    use_context: bool = True,
    head_structure: bool = True,
) -> StructuredStudent:
    torch.manual_seed(int(seed))
    device_obj = torch.device(str(device))
    first = boundaries[0]
    model = StructuredStudent(
        layers=int(first["head_features"].shape[1]),
        kv_heads=int(first["head_features"].shape[2]),
        head_feature_width=int(first["head_features"].shape[3]),
        use_identity=use_identity,
        use_context=use_context,
        head_structure=head_structure,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(int(seed))
    pair_cache: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    for boundary in boundaries:
        per_budget: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for budget in cutoff_budgets:
            pairs, weights = _v3_pairs(boundary["teacher_mean"], int(budget), rng)
            per_budget.append(
                (
                    torch.from_numpy(pairs[:, 0]).to(device_obj),
                    torch.from_numpy(pairs[:, 1]).to(device_obj),
                    torch.from_numpy(weights).to(device_obj),
                )
            )
        pair_cache.append(per_budget)
    for epoch in range(int(epochs)):
        total_loss = 0.0
        for boundary_index in rng.permutation(len(boundaries)):
            boundary = boundaries[int(boundary_index)]
            head = torch.from_numpy(boundary["head_features"]).to(device_obj)
            state = torch.from_numpy(boundary["state_features"]).to(device_obj)
            misc = torch.from_numpy(_misc(boundary)).to(device_obj)
            score, per_head_score = model(head, state, misc)
            target_soft = torch.from_numpy(
                boundary["target_mean_percentile"]
            ).to(device_obj)
            loss = float(soft_weight) * bce(score, target_soft)
            for hi, lo, weight in pair_cache[int(boundary_index)]:
                difference = score[hi] - score[lo]
                loss = loss + (weight * torch.nn.functional.softplus(-difference)).mean()
            if per_head_score is not None and float(perhead_aux_weight) > 0.0:
                target_head = torch.from_numpy(
                    boundary["target_head_percentile"]
                ).to(device_obj)
                loss = loss + float(perhead_aux_weight) * bce(
                    per_head_score, target_head
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
        print(
            f"[structured-student] epoch {epoch + 1}/{epochs} "
            f"mean_loss={total_loss / len(boundaries):.4f}",
            flush=True,
        )
    return model.cpu()


def _resolve_device(requested: str) -> str:
    requested = str(requested)
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def train_structured_student(
    config_path: Path,
    repository_root: Path,
    *,
    ablation: str = "full",
    epochs: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Train one structured student variant and save its checkpoint."""

    if str(ablation) not in ABLATIONS:
        raise ValueError(f"unknown structured ablation: {ablation}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    repository_root = Path(repository_root)
    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"])
    output_root = repository_root / str(config["output_models"])
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(config["data_seed"])
    train_ids = [
        sample_id_for(str(family), int(index))
        for family in config["task_families"]
        for index in config["distillation"]["train_indices"]
    ]
    artifact_paths = [
        source_run / "artifacts" / "train" / f"{safe_path_component(sample_id)}.npz"
        for sample_id in train_ids
    ]
    missing = [path.name for path in artifact_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"structured student train artifacts missing: {missing}")
    projector = FixedProjector(seed)
    cache_path = None if cache_dir is None else Path(cache_dir) / "train_boundaries.joblib"
    started = time.perf_counter()
    boundaries = _assemble_boundaries(
        artifact_paths, teacher_root / "train", config, projector,
        cache_path=cache_path,
    )
    if str(ablation) == "no_query_traj":
        keep = [
            index
            for index in range(HEAD_FEATURE_WIDTH)
            if index not in QUERY_TRAJECTORY_COLUMNS
        ]
        for boundary in boundaries:
            boundary["head_features"] = np.ascontiguousarray(
                boundary["head_features"][..., keep]
            )
    scalers = _fit_scalers(boundaries)
    _normalize_boundaries(boundaries, scalers)
    model = _train_one(
        boundaries,
        seed=seed + 101,
        epochs=int(epochs or config["student"]["epochs"]),
        cutoff_budgets=[
            int(value) for value in config["student"]["cutoff_budgets"]
        ],
        device=_resolve_device(str(config["student"].get("device", "auto"))),
        perhead_aux_weight=(
            0.0
            if str(ablation) in {"no_perhead_aux", "no_head_structure"}
            else float(config["student"].get("perhead_aux_weight", 0.3))
        ),
        soft_weight=float(config["student"].get("soft_weight", 0.1)),
        use_identity=str(ablation) != "no_head_identity",
        use_context=str(ablation) != "no_context",
        head_structure=str(ablation) != "no_head_structure",
    )
    first = boundaries[0]
    architecture = {
        "layers": int(first["head_features"].shape[1]),
        "kv_heads": int(first["head_features"].shape[2]),
        "head_feature_width": int(first["head_features"].shape[3]),
        "use_identity": str(ablation) != "no_head_identity",
        "use_context": str(ablation) != "no_context",
        "head_structure": str(ablation) != "no_head_structure",
    }
    path = save_student_checkpoint(
        output_root / f"structured_student_{ablation}.pt",
        kind=STRUCTURED_KIND,
        models=model.state_dict(),
        scaler=scalers,
        horizons=[1],
        projector_seed=seed,
        score_channel=0,
        metadata={
            "teacher": R2_TEACHER,
            "ablation": str(ablation),
            "architecture": architecture,
            "objective": (
                "cutoff-weighted pairwise ranking on S + soft percentile BCE "
                "+ per-head percentile BCE auxiliary"
            ),
            "train_sequences": len(artifact_paths),
            "train_boundaries": len(boundaries),
            "runtime_future_access": False,
        },
    )
    print(
        f"[structured-student] saved {path} "
        f"({time.perf_counter() - started:.0f}s)",
        flush=True,
    )
    return path


# ----------------------------------------------------------------- evaluation


def evaluate_structured_cutoff(
    config: Mapping[str, Any],
    repository_root: Path,
    scorers: Mapping[str, Any],
    ks: Sequence[int] = (220, 476),
    split: str = "validation",
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Selection-fidelity battery for structured and legacy students.

    ``scorers`` maps a method name to either a :class:`StructuredStudentScorer`
    (scored via one :func:`structured_boundary` per boundary) or a legacy
    :class:`StudentScorer` (scored via per-(layer, head)
    ``artifact_boundary`` features, then averaged — exactly the
    ``evaluate_students_cutoff`` protocol, so numbers are comparable).
    Truth is the mean-over-(layer, head) R2 teacher H=1 score.
    """

    from statekv.causal_predictors import artifact_boundary

    repository_root = Path(repository_root)
    source_run = repository_root / str(config["source_run"])
    teacher_root = repository_root / str(config["teacher_scores"]) / str(split)
    artifact_dir = source_run / "artifacts" / str(split)
    artifact_paths = sorted(artifact_dir.glob("*.npz"))
    if not artifact_paths:
        raise RuntimeError(f"no {split} artifacts in {artifact_dir}")
    sink = int(config["sink_size"])
    recent = int(config["recent_size"])
    projector = FixedProjector(int(config["data_seed"]))
    cache_path = (
        None
        if cache_dir is None
        else Path(cache_dir) / f"{split}_boundaries.joblib"
    )
    cached: Optional[List[Dict[str, Any]]] = None
    if cache_path is not None and Path(cache_path).exists():
        cached = joblib.load(cache_path)
    rows: List[Dict[str, Any]] = []
    for ordinal, artifact_path in enumerate(artifact_paths, start=1):
        teacher_path = teacher_root / artifact_path.name
        if not teacher_path.exists():
            continue
        teacher = _load_npz(teacher_path)
        teacher_h1 = [int(value) for value in teacher["horizons"]].index(1)
        artifact: Optional[Dict[str, np.ndarray]] = None
        for cycle_index, cycle_value in enumerate(teacher["cycles"]):
            cycle = int(cycle_value)
            count = int(teacher["position_lengths"][cycle_index])
            teacher_positions = [
                int(value) for value in teacher["position_ids"][cycle_index, :count]
            ]
            if artifact is None:
                artifact = _load_npz(artifact_path)
            current_count = int(artifact["position_lengths"][cycle])
            positions = [
                int(value)
                for value in artifact["position_ids"][cycle, :current_count]
            ]
            _, _, eligible = mandatory_and_eligible(positions, sink, recent)
            if teacher_positions != [int(value) for value in eligible]:
                raise RuntimeError(
                    "teacher positions do not match feature positions: "
                    f"{artifact_path.name} cycle {cycle}"
                )
            truth = np.asarray(
                teacher["scores"][cycle_index, teacher_h1, :, :, :count],
                dtype=np.float32,
            ).mean(axis=(0, 1))
            predictions: Dict[str, np.ndarray] = {}
            boundary: Optional[Dict[str, Any]] = None
            for name, scorer in scorers.items():
                if isinstance(scorer, StructuredStudentScorer):
                    if cached is not None:
                        boundary = next(
                            entry
                            for entry in cached
                            if entry["sample_id"] == artifact_path.stem.replace("__", ":")
                            and int(entry["cycle"]) == cycle
                        )
                    elif boundary is None:
                        boundary = structured_boundary(
                            artifact, cycle, sink, recent, projector
                        )
                    predictions[name] = scorer.predict(boundary)
                elif isinstance(scorer, StudentScorer):
                    horizon_col = scorer.horizons.index(1)
                    stack = []
                    for layer_index in range(int(artifact["layers"].size)):
                        for head in range(int(artifact["attention"].shape[2])):
                            legacy = artifact_boundary(
                                artifact,
                                cycle,
                                layer_index,
                                head,
                                scorer.horizons,
                                sink,
                                recent,
                                int(config["core_budget"]),
                                projector,
                                feature_only=True,
                            )
                            stack.append(
                                scorer.predict(legacy.features)[:, horizon_col]
                            )
                    predictions[name] = np.mean(np.stack(stack), axis=0)
                else:
                    raise RuntimeError(f"unsupported scorer type for {name}")
            for name, prediction in predictions.items():
                row = {
                    "sample_id": str(teacher["sample_id"].item()),
                    "task": str(teacher["task"].item()),
                    "split": str(split),
                    "cycle": cycle,
                    "method": name,
                    "horizon": 1,
                }
                for k in ks:
                    row.update(
                        {
                            f"{key}@{int(k)}": value
                            for key, value in cutoff_metrics(
                                truth, prediction, int(k)
                            ).items()
                        }
                    )
                rows.append(row)
        print(
            f"[structured-eval] {split} {ordinal}/{len(artifact_paths)} "
            f"{artifact_path.stem}",
            flush=True,
        )
    return rows


def summarize_cutoff_rows(
    rows: Sequence[Mapping[str, Any]], ks: Sequence[int] = (220, 476)
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metric_columns = [
        f"{metric}@{int(k)}"
        for k in ks
        for metric in (
            "topk_recall",
            "jaccard",
            "cutoff_pair_accuracy",
            "band_pair_accuracy",
        )
    ]
    summary = frame.groupby("method", as_index=False)[metric_columns].mean()
    summary["boundaries"] = frame.groupby("method")["sample_id"].size().values
    return summary


__all__ = [
    "ABLATIONS",
    "GLOBAL_FEATURE_WIDTH",
    "HEAD_FEATURE_SEGMENTS",
    "HEAD_FEATURE_WIDTH",
    "QUERY_TRAJECTORY_COLUMNS",
    "STATE_FEATURE_WIDTH",
    "STRUCTURED_KIND",
    "TOKEN_FEATURE_WIDTH",
    "RuntimeStructuredScorer",
    "StructuredStudent",
    "StructuredStudentScorer",
    "evaluate_structured_cutoff",
    "structured_boundary",
    "summarize_cutoff_rows",
    "train_structured_student",
]
