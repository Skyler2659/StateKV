"""Pure and state-isolation primitives for P3-Physical-Recovery."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OLD_ROOTS = (
    ROOT / "experiments/p0_v2_fixed_boundary",
    ROOT / "experiments/p1_state_conditioned",
    ROOT / "experiments/p2_state_local_risk",
    ROOT / "experiments/p2_recovery",
    ROOT / "experiments/p3_decision_validity",
)
FORBIDDEN_FEATURE_TOKENS = (
    "exact_physical_kl",
    "physical_endpoint_logits",
    "exact_kl",
    "full_reference",
    "future_token",
    "future_attention",
    "formal_label",
    "task_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                json_safe(value),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".parquet", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def stable_softmax(logits: Any) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = value - float(np.max(value))
    exponential = np.exp(shifted)
    return exponential / max(float(exponential.sum()), 1.0e-300)


def exact_kl(base_logits: Any, changed_logits: Any) -> float:
    left = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    right = np.asarray(changed_logits, dtype=np.float64).reshape(-1)
    probability = stable_softmax(left)
    left_logsum = float(np.max(left)) + float(
        np.log(np.exp(left - float(np.max(left))).sum())
    )
    right_logsum = float(np.max(right)) + float(
        np.log(np.exp(right - float(np.max(right))).sum())
    )
    value = float(
        np.dot(probability, (left - left_logsum) - (right - right_logsum))
    )
    return max(0.0, value)


def fisher_variance(probability: Any, direction: Any) -> float:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    value = np.asarray(direction, dtype=np.float64).reshape(-1)
    centered = value - float(np.dot(p, value))
    return max(0.0, float(np.dot(p, centered * centered)))


def normalized_regret(target: Sequence[float], score: Sequence[float]) -> float:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(score, dtype=np.float64)
    chosen = int(np.argmin(prediction))
    span = float(np.max(truth) - np.min(truth))
    return float((truth[chosen] - np.min(truth)) / max(span, 1.0e-30))


def ranking_spearman(target: Sequence[float], score: Sequence[float]) -> float:
    left = np.asarray(target, dtype=np.float64)
    right = np.asarray(score, dtype=np.float64)
    if len(left) < 2 or np.ptp(left) <= 0 or np.ptp(right) <= 0:
        return 0.0
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def pairwise_accuracy(target: Sequence[float], score: Sequence[float]) -> float:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(score, dtype=np.float64)
    correct = 0.0
    count = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_sign = np.sign(truth[left] - truth[right])
            prediction_sign = np.sign(
                prediction[left] - prediction[right]
            )
            if truth_sign == 0:
                continue
            correct += (
                1.0
                if prediction_sign == truth_sign
                else 0.5 if prediction_sign == 0 else 0.0
            )
            count += 1
    return float(correct / max(count, 1))


def ranking_metrics(target: Sequence[float], score: Sequence[float]) -> Dict[str, Any]:
    truth = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(score, dtype=np.float64)
    return {
        "candidate_count": int(len(truth)),
        "spearman": ranking_spearman(truth, prediction),
        "pairwise_accuracy": pairwise_accuracy(truth, prediction),
        "top1_correct": bool(int(np.argmin(truth)) == int(np.argmin(prediction))),
        "normalized_regret": normalized_regret(truth, prediction),
        "exact_range": float(np.ptp(truth)),
    }


def validate_feature_names(names: Iterable[str]) -> None:
    bad = sorted(
        {
            str(name)
            for name in names
            if any(token in str(name).lower() for token in FORBIDDEN_FEATURE_TOKENS)
        }
    )
    if bad:
        raise ValueError(f"forbidden predictor inputs: {bad}")


def _text_ids(text: str) -> set[int]:
    output: set[int] = set()
    patterns = (
        r"gov_report:(\d+)",
        r"synthetic_niah_(\d+)",
        r"gov_report_indices:\s*\[([^\]]*)\]",
        r"niah_offsets:\s*\[([^\]]*)\]",
        r"sample_indices:\s*\[([^\]]*)\]",
        r"scanned_ids:\s*\[([^\]]*)\]",
    )
    for index, pattern in enumerate(patterns):
        for match in re.findall(pattern, text):
            if index < 2:
                output.add(int(match))
            else:
                output.update(int(value) for value in re.findall(r"\d+", match))
    return output


def discover_old_ids(roots: Sequence[Path] = OLD_ROOTS) -> Dict[str, Any]:
    ids: set[int] = set()
    scanned_files: List[str] = []
    parquet_files = 0
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() in {
                    ".md",
                    ".yaml",
                    ".yml",
                    ".json",
                    ".csv",
                }:
                    ids.update(_text_ids(path.read_text(errors="ignore")))
                    scanned_files.append(str(path.relative_to(ROOT)))
                elif path.suffix.lower() == ".parquet":
                    parquet_files += 1
                    frame = pd.read_parquet(path)
                    if "sample_id" in frame.columns:
                        for value in frame["sample_id"].dropna().astype(str).unique():
                            ids.update(_text_ids(value))
            except Exception as error:
                raise RuntimeError(f"old ID scan failed at {path}: {error}") from error
    return {
        "ids": sorted(ids),
        "count": len(ids),
        "minimum": min(ids) if ids else None,
        "maximum": max(ids) if ids else None,
        "text_file_count": len(scanned_files),
        "parquet_file_count": parquet_files,
        "roots": [str(path.relative_to(ROOT)) for path in roots],
    }


def source_integrity(config: Mapping[str, Any]) -> Dict[str, bool]:
    checks = {}
    for name, payload in config["source"].items():
        if not isinstance(payload, Mapping) or "path" not in payload:
            continue
        checks[name] = (ROOT / str(payload["path"])).is_file()
    if not checks or not all(checks.values()):
        raise RuntimeError(f"required source file is missing: {checks}")
    return checks


def clone_mlx_state(state: Any) -> Any:
    """Deep-copy an MLX replay state so candidates cannot contaminate it."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    from kvbench.temporal.backend_mlx import MLXReplayState

    caches = []
    for cache in state.cache:
        offset = int(cache.offset)
        clone = KVCache()
        clone.state = (
            mx.array(np.asarray(cache.keys[:, :, :offset, :]).copy()),
            mx.array(np.asarray(cache.values[:, :, :offset, :]).copy()),
        )
        clone.logical_offset = int(cache.logical_offset)
        caches.append(clone)
    return MLXReplayState(
        cache=caches,
        position_maps={
            int(layer): positions.detach().clone()
            for layer, positions in state.position_maps.items()
        },
        logical_next_position=int(state.logical_next_position),
    )


def prune_shared_position(state: Any, position: int) -> None:
    """Delete one physical position from every layer of an MLX replay state."""
    import mlx.core as mx

    target = int(position)
    for layer, cache in enumerate(state.cache):
        positions = [
            int(value) for value in state.position_maps[layer].tolist()
        ]
        if target not in positions:
            raise ValueError(
                f"candidate position {target} absent from layer {layer}"
            )
        keep_rows = [
            index for index, value in enumerate(positions) if value != target
        ]
        rows = mx.array(keep_rows)
        offset = int(cache.offset)
        cache.keys = mx.take(cache.keys[:, :, :offset, :], rows, axis=2)
        cache.values = mx.take(cache.values[:, :, :offset, :], rows, axis=2)
        cache.offset = len(keep_rows)
        cache.logical_offset = int(state.logical_next_position)
        state.position_maps[layer] = torch.tensor(
            [positions[index] for index in keep_rows], dtype=torch.long
        )


def state_to_anchor(
    backend: Any,
    state: Any,
    query_token_id: int,
    anchor_step: int,
) -> Any:
    """Freeze a replay state as an AnchorState for pure physical readouts."""
    from kvbench.temporal.backend import AnchorState

    keys = []
    values = []
    for cache in state.cache:
        offset = int(cache.offset)
        keys.append(
            torch.from_numpy(
                np.asarray(cache.keys[:, :, :offset, :]).copy()
            ).float()
        )
        values.append(
            torch.from_numpy(
                np.asarray(cache.values[:, :, :offset, :]).copy()
            ).float()
        )
    return AnchorState(
        anchor_step=int(anchor_step),
        logical_length=int(state.logical_next_position),
        query_token_id=int(query_token_id),
        keys=keys,
        values=values,
        position_maps={
            int(layer): positions.detach().clone()
            for layer, positions in state.position_maps.items()
        },
        attention=None,
        query_head_observation=None,
    )


@dataclass(frozen=True)
class DeletionCandidate:
    candidate_id: str
    source: str
    deleted_position: int
    raw_rank: int
    deduplicated: bool


def unique_deletion_candidates(
    eligible_positions: Sequence[int],
    scores: Mapping[str, Mapping[int, float]],
    source_order: Sequence[str],
) -> Tuple[List[DeletionCandidate], List[Dict[str, Any]]]:
    """Choose one deterministic distinct deletion per frozen selector."""
    eligible = [int(value) for value in eligible_positions]
    if len(eligible) < len(source_order):
        raise ValueError("insufficient eligible positions for candidate pool")
    used: set[int] = set()
    candidates: List[DeletionCandidate] = []
    events: List[Dict[str, Any]] = []
    for source in source_order:
        source_scores = scores[str(source)]
        ranked = sorted(
            eligible,
            key=lambda position: (
                float(source_scores[int(position)]),
                int(position),
            ),
        )
        raw = int(ranked[0])
        selected = next(position for position in ranked if position not in used)
        rank = int(ranked.index(selected))
        dedup = bool(selected != raw)
        used.add(int(selected))
        candidates.append(
            DeletionCandidate(
                candidate_id=f"physical_delete_{source}_{selected}",
                source=str(source),
                deleted_position=int(selected),
                raw_rank=rank,
                deduplicated=dedup,
            )
        )
        events.append(
            {
                "source": str(source),
                "raw_deleted_position": raw,
                "selected_deleted_position": int(selected),
                "selected_rank": rank,
                "deduplicated": dedup,
            }
        )
    return candidates, events


def select_mechanism_disagreement(
    records: Sequence[Mapping[str, Any]],
    count: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Freeze a label-free pool whose action and dense argmins disagree.

    ``action_score`` and ``dense_score`` must both be computed from the
    current physical state before the exact candidate endpoint is executed.
    Exact KL and candidate endpoint logits are therefore neither accepted nor
    inspected by this selector.
    """
    required = {"candidate_id", "action_score", "dense_score"}
    normalized: List[Dict[str, Any]] = []
    for row in records:
        missing = required - set(row)
        if missing:
            raise ValueError(f"mechanism-disagreement row missing {missing}")
        forbidden = {
            name
            for name in row
            if any(
                token in str(name).lower()
                for token in ("exact", "endpoint_logits", "task_id", "future")
            )
        }
        if forbidden:
            raise ValueError(
                f"forbidden candidate-generator inputs: {sorted(forbidden)}"
            )
        normalized.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "action_score": float(row["action_score"]),
                "dense_score": float(row["dense_score"]),
            }
        )
    if len(normalized) < int(count):
        raise ValueError("candidate seed pool is smaller than selected pool")
    if not all(
        math.isfinite(row["action_score"])
        and math.isfinite(row["dense_score"])
        for row in normalized
    ):
        raise ValueError("candidate-generator scores must be finite")

    proposals: List[Tuple[Tuple[float, float, str, str], List[Dict[str, Any]]]] = []
    for dense_min in normalized:
        for action_min in normalized:
            if dense_min["candidate_id"] == action_min["candidate_id"]:
                continue
            if not (
                dense_min["dense_score"] < action_min["dense_score"]
                and action_min["action_score"] < dense_min["action_score"]
            ):
                continue
            admissible = [
                row
                for row in normalized
                if row["dense_score"] >= dense_min["dense_score"]
                and row["action_score"] >= action_min["action_score"]
            ]
            if len(admissible) < int(count):
                continue
            fixed = [dense_min, action_min]
            remaining = [
                row
                for row in admissible
                if row["candidate_id"]
                not in {
                    dense_min["candidate_id"],
                    action_min["candidate_id"],
                }
            ]
            # Fill near the dense minimum so the disagreement is not diluted
            # by an arbitrary extreme-risk token.  All tie breaks are stable.
            remaining.sort(
                key=lambda row: (
                    row["dense_score"],
                    row["action_score"],
                    row["candidate_id"],
                )
            )
            selected = fixed + remaining[: int(count) - 2]
            dense_values = [row["dense_score"] for row in selected]
            span = max(dense_values) - min(dense_values)
            regret = (
                action_min["dense_score"] - dense_min["dense_score"]
            ) / max(span, 1.0e-30)
            key = (
                -float(regret),
                -float(
                    action_min["dense_score"] - dense_min["dense_score"]
                ),
                dense_min["candidate_id"],
                action_min["candidate_id"],
            )
            proposals.append((key, selected))
    if not proposals:
        raise RuntimeError(
            "seed pool contains no eight-candidate action/dense argmin disagreement"
        )
    key, selected = sorted(proposals, key=lambda item: item[0])[0]
    selected_ids = [row["candidate_id"] for row in selected]
    action_choice = min(
        selected, key=lambda row: (row["action_score"], row["candidate_id"])
    )
    dense_choice = min(
        selected, key=lambda row: (row["dense_score"], row["candidate_id"])
    )
    if action_choice["candidate_id"] == dense_choice["candidate_id"]:
        raise RuntimeError("mechanism-disagreement selector failed its identity")
    return selected_ids, {
        "candidate_seed_count": len(normalized),
        "selected_count": len(selected),
        "action_argmin_candidate_id": action_choice["candidate_id"],
        "dense_argmin_candidate_id": dense_choice["candidate_id"],
        "predicted_normalized_regret": -float(key[0]),
        "exact_physical_kl_used": False,
        "candidate_endpoint_logits_used": False,
        "task_id_used": False,
    }


def vector_metrics(predicted: Any, truth: Any) -> Dict[str, float]:
    left = np.asarray(predicted, dtype=np.float64).reshape(-1)
    right = np.asarray(truth, dtype=np.float64).reshape(-1)
    difference = left - right
    denominator = max(
        float(np.linalg.norm(left) * np.linalg.norm(right)), 1.0e-30
    )
    return {
        "cosine": float(np.dot(left, right) / denominator),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(right), 1.0e-30)
        ),
        "norm_ratio": float(
            np.linalg.norm(left) / max(np.linalg.norm(right), 1.0e-30)
        ),
        "maximum_absolute_error": float(
            np.max(np.abs(difference), initial=0.0)
        ),
    }


def deterministic_projection(
    input_dimension: int, output_dimension: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    return generator.normal(
        0.0,
        1.0 / math.sqrt(float(output_dimension)),
        size=(int(output_dimension), int(input_dimension)),
    ).astype(np.float64)


def sequence_first_metrics(
    rows: pd.DataFrame,
    score_column: str,
    target_column: str = "exact_physical_kl",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    records = []
    unit_keys = ["sample_id", "task", "target_anchor"]
    for key, group in rows.groupby(unit_keys, sort=True):
        metrics = ranking_metrics(
            group[target_column].to_numpy(),
            group[score_column].to_numpy(),
        )
        records.append(
            {
                **dict(zip(unit_keys, key)),
                "score": score_column,
                **metrics,
            }
        )
    frame = pd.DataFrame(records)
    task_spearman = {
        str(task): float(group["spearman"].mean())
        for task, group in frame.groupby("task", sort=True)
    }
    summary = {
        "score": score_column,
        "sequence_count": len(frame),
        "overall_spearman": float(frame["spearman"].mean()),
        "task_spearman": task_spearman,
        "pairwise_accuracy": float(frame["pairwise_accuracy"].mean()),
        "top1_accuracy": float(frame["top1_correct"].mean()),
        "normalized_regret": float(frame["normalized_regret"].mean()),
        "positive_sequence_fraction": float(
            (frame["spearman"] > 0).mean()
        ),
    }
    return frame, summary
