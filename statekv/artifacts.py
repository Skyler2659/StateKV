"""Atomic fragments, NPZ references, and consolidated discovery tables."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from statekv.backend import ReferenceTrajectory
from statekv.config import DiscoveryConfig
from statekv.storage import (
    atomic_frame,
    atomic_gzip_text,
    atomic_json,
    atomic_npz,
    atomic_text,
)


TABLES = ("reference", "candidate", "step", "horizon", "temporal")
PARQUET_NAMES = {
    "reference": "reference_trajectories.parquet",
    "candidate": "candidate_sets.parquet",
    "step": "step_losses.parquet",
    "horizon": "horizon_losses.parquet",
    "temporal": "temporal_signals.parquet",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: %s" % type(value).__name__)


def json_text(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def _git(command: List[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            command, cwd=str(cwd), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception as exc:
        print(
            "[artifacts] WARNING: could not capture git provenance (%s) "
            "with %r: %s: %s"
            % (cwd, command, type(exc).__name__, exc),
            file=sys.stderr,
        )
        return None


class ArtifactStore:
    def __init__(self, cfg: DiscoveryConfig, repository_root: Path):
        self.cfg = cfg
        self.repository_root = repository_root.resolve()
        run_id = cfg.runtime.run_id or (
            "%s_seed%d" % (cfg.config_hash[:12], int(cfg.runtime.seed))
        )
        root = Path(cfg.runtime.output_root)
        if not root.is_absolute():
            root = self.repository_root / root
        self.run_dir = (root / run_id).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            (self.run_dir / "fragments" / table).mkdir(parents=True, exist_ok=True)
        (self.run_dir / "references").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "figures").mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.status: Dict[str, Any] = self._load_status()
        resolved_path = self.run_dir / "resolved_config.yaml"
        resolved_payload = cfg.to_dict()
        if resolved_path.exists():
            with open(resolved_path, "r", encoding="utf-8") as handle:
                prior_payload = yaml.safe_load(handle) or {}
            if prior_payload != resolved_payload:
                raise RuntimeError(
                    "run directory contains a different resolved configuration: %s"
                    % self.run_dir
                )
        else:
            atomic_text(
                resolved_path,
                yaml.safe_dump(
                    resolved_payload, sort_keys=False, allow_unicode=True
                ),
            )

    def _load_status(self) -> Dict[str, Any]:
        if not self.status_path.exists():
            return {"state": "created", "completed": {}, "failed": {}}
        with open(self.status_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_status(self) -> None:
        atomic_json(
            self.status_path,
            self.status,
            default=_json_default,
        )

    def is_complete(self, key: str) -> bool:
        return bool(self.status.get("completed", {}).get(key))

    def mark_complete(self, key: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.status.setdefault("completed", {})[key] = {
            "completed_at": time.time(),
            **(metadata or {}),
        }
        self.status.setdefault("failed", {}).pop(key, None)
        self.save_status()

    def mark_failed(self, key: str, error: str) -> None:
        self.status.setdefault("failed", {})[key] = {
            "failed_at": time.time(),
            "error": str(error),
        }
        self.save_status()

    def write_metadata(
        self,
        model_info: Dict[str, Any],
        task_events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        commit = _git(["git", "rev-parse", "HEAD"], self.repository_root)
        status = _git(["git", "status", "--short"], self.repository_root)
        metadata = {
            "run_id": self.run_dir.name,
            "experiment_name": self.cfg.experiment_name,
            "config_hash": self.cfg.config_hash,
            "seed": int(self.cfg.runtime.seed),
            "git_commit": commit,
            "git_dirty": bool(status),
            "git_status": status.splitlines() if status else [],
            "model": model_info,
            "task_load_events": task_events,
            "created_at_unix": time.time(),
            "protocol": {
                "name": "teacher_forced_frozen_core",
                "anchor_query_replay": "rewind_one_token",
                "core_refresh_during_horizon": False,
                "recent_policy": "fifo",
            },
        }
        atomic_json(
            self.run_dir / "metadata.json",
            metadata,
            default=_json_default,
        )
        return metadata

    @staticmethod
    def safe_slug(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)

    def write_fragment(
        self,
        table: str,
        key: str,
        rows: Iterable[Dict[str, Any]],
    ) -> Path:
        if table not in TABLES:
            raise ValueError("unknown fragment table: %s" % table)
        path = (
            self.run_dir
            / "fragments"
            / table
            / (self.safe_slug(key) + ".json.gz")
        )
        payload = list(rows)
        atomic_gzip_text(
            path,
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=_json_default,
            ),
        )
        return path

    def append_error(self, record: Dict[str, Any]) -> None:
        path = self.run_dir / "errors.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")

    def save_reference_npz(self, reference: ReferenceTrajectory) -> Path:
        slug = self.safe_slug(reference.sample_id)
        path = self.run_dir / "references" / (slug + ".npz")
        max_keys = max(
            (
                int(value.shape[-1])
                for record in reference.query_records
                for value in record.oracle_attention_by_layer.values()
            ),
            default=0,
        )
        num_records = len(reference.query_records)
        num_layers = max(
            (len(record.oracle_attention_by_layer) for record in reference.query_records),
            default=0,
        )
        num_kv_heads = (
            int(
                next(iter(reference.query_records[0].oracle_attention_by_layer.values())).shape[
                    0
                ]
            )
            if reference.query_records
            else 0
        )
        oracle = np.zeros(
            (num_records, num_layers, num_kv_heads, max_keys), dtype=np.float16
        )
        oracle_lengths = np.zeros((num_records,), dtype=np.int32)
        for step, record in enumerate(reference.query_records):
            lengths = []
            for layer, value in record.oracle_attention_by_layer.items():
                array = value.detach().cpu().numpy().astype(np.float16, copy=False)
                oracle[step, int(layer), :, : array.shape[-1]] = array
                lengths.append(array.shape[-1])
            oracle_lengths[step] = max(lengths, default=0)
        arrays: Dict[str, Any] = {
            "prompt_token_ids": np.asarray(reference.prompt_token_ids, dtype=np.int32),
            "generated_token_ids": np.asarray(
                reference.generated_token_ids, dtype=np.int32
            ),
            "reference_log_probabilities": np.asarray(
                reference.reference_log_probabilities, dtype=np.float32
            ),
            "top_ids": reference.top_ids.numpy().astype(np.int32),
            "top_probabilities": reference.top_probabilities.numpy().astype(
                np.float32
            ),
            "oracle_attention": oracle,
            "oracle_attention_lengths": oracle_lengths,
            "query_positions": np.asarray(
                [record.query_position for record in reference.query_records],
                dtype=np.int32,
            ),
        }
        for layer in reference.selected_layers:
            for head in reference.selected_heads[layer]:
                key = "%d:%d" % (layer, head)
                arrays["query_%s" % key.replace(":", "_")] = np.stack(
                    [
                        record.queries[key].numpy().astype(np.float32)
                        for record in reference.query_records
                    ]
                )
                arrays["attention_output_%s" % key.replace(":", "_")] = np.stack(
                    [
                        record.attention_outputs[key].numpy().astype(np.float32)
                        for record in reference.query_records
                    ]
                )
                attention = np.zeros(
                    (num_records, max_keys), dtype=np.float16
                )
                attention_lengths = np.zeros(
                    (num_records,), dtype=np.int32
                )
                for step, record in enumerate(reference.query_records):
                    distribution = (
                        record.attention_distributions[key]
                        .numpy()
                        .astype(np.float16, copy=False)
                    )
                    attention[step, : distribution.shape[-1]] = distribution
                    attention_lengths[step] = distribution.shape[-1]
                suffix = key.replace(":", "_")
                arrays["attention_distribution_%s" % suffix] = attention
                arrays["attention_distribution_lengths_%s" % suffix] = (
                    attention_lengths
                )
        atomic_npz(path, arrays)
        return path

    def consolidate(self) -> Dict[str, Path]:
        outputs: Dict[str, Path] = {}
        for table in TABLES:
            rows: List[Dict[str, Any]] = []
            directory = self.run_dir / "fragments" / table
            fragments = sorted(directory.glob("*.json")) + sorted(
                directory.glob("*.json.gz")
            )
            for fragment in fragments:
                opener = gzip.open if fragment.suffix == ".gz" else open
                with opener(fragment, "rt", encoding="utf-8") as handle:
                    value = json.load(handle)
                if not isinstance(value, list):
                    raise RuntimeError("fragment is not a row list: %s" % fragment)
                rows.extend(value)
            frame = pd.DataFrame(rows)
            path = self.run_dir / PARQUET_NAMES[table]
            atomic_frame(frame, path)
            outputs[table] = path
        return outputs
