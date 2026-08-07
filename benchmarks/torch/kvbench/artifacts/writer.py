"""Atomic run writer with deterministic run identity and sample resume."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from kvbench.config import ExperimentConfig
from kvbench.types import SampleResult, ScoreBundle, SelectionDecision


def _slug(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in value)
    return "_".join(part for part in normalized.split("_") if part)[:96]


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _git(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


class RunWriter:
    def __init__(self, cfg: ExperimentConfig, command: Optional[List[str]] = None):
        self.cfg = cfg
        self.command = list(command or sys.argv)
        identity = cfg.to_dict()
        # Storage location and resume/error-handling controls do not change the
        # scientific experiment identity.
        identity["output"] = {"experiment_name": cfg.output.experiment_name}
        identity["runtime"] = {
            key: value
            for key, value in identity["runtime"].items()
            if key not in {"resume", "fail_on_error"}
        }
        config_payload = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=_json_default
        )
        self.config_hash = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()
        run_id = cfg.output.run_id or "%s_%s" % (
            _slug(cfg.output.experiment_name), self.config_hash[:12]
        )
        protocol = "%s_%s_%s" % (
            cfg.protocol.visibility,
            cfg.protocol.cache_mode,
            cfg.protocol.update_policy,
        )
        self.run_dir = (
            Path(os.path.expandvars(cfg.output.root))
            / _slug(cfg.benchmark.name)
            / _slug(Path(cfg.model.name).name)
            / _slug(protocol)
            / _slug(cfg.method.name)
            / _slug(run_id)
        )
        if self.run_dir.exists() and cfg.output.overwrite:
            raise RuntimeError(
                "overwrite=true is intentionally unsupported for paper artifacts; choose a new run_id"
            )
        for directory in ("samples", "decisions", "scores", "logs"):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)
        self.status_path = self.run_dir / "status.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self.status = self._load_json(self.status_path, {"samples": {}, "state": "running"})
        self._write_initial_files()

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_initial_files(self) -> None:
        config_path = self.run_dir / "resolved_config.yaml"
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.cfg.to_dict(), handle, sort_keys=False, allow_unicode=True
            )
        existing = self._load_json(self.metadata_path, {})
        source_tree_sha256 = self._source_tree_hash()
        if existing:
            if existing.get("config_hash") not in {None, self.config_hash}:
                raise RuntimeError(
                    "existing run directory has a different resolved config; choose a new run_id"
                )
            previous_source = existing.get("source_tree_sha256")
            if previous_source is not None and previous_source != source_tree_sha256:
                raise RuntimeError(
                    "source code changed since this run started; choose a new output.run_id "
                    "instead of mixing samples from different implementations"
                )
        metadata = {
            **existing,
            "schema_version": 1,
            "config_hash": self.config_hash,
            "command": self.command,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(_git(["status", "--porcelain=v1"])),
            "git_diff_sha256": self._git_diff_hash(),
            "source_tree_sha256": source_tree_sha256,
            "start_time": existing.get("start_time") or datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "state": "running",
        }
        _atomic_json(self.metadata_path, metadata)
        _atomic_json(self.status_path, self.status)

    @staticmethod
    def _source_tree_hash() -> str:
        project_root = Path(__file__).resolve().parents[2]
        digest = hashlib.sha256()
        paths = list((project_root / "kvbench").rglob("*.py"))
        paths.extend(
            path
            for path in (project_root / "scripts").rglob("*.sh")
            if path.is_file()
        )
        paths.extend(
            path
            for path in (project_root / "configs").rglob("*.yaml")
            if path.is_file()
        )
        paths.extend(
            path
            for path in (
                project_root / "pyproject.toml",
                project_root / "requirements.txt",
            )
            if path.is_file()
        )
        for path in sorted(paths):
            digest.update(str(path.relative_to(project_root)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _git_diff_hash() -> Optional[str]:
        try:
            diff = subprocess.check_output(
                ["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL
            )
            status = subprocess.check_output(
                ["git", "status", "--porcelain=v1"], stderr=subprocess.DEVNULL
            )
            payload = diff + b"\n--status--\n" + status
            return hashlib.sha256(payload).hexdigest()
        except Exception:
            return None

    def save_environment(self, model_info: Dict[str, Any]) -> None:
        versions: Dict[str, Any] = {}
        for name in ("torch", "transformers", "accelerate", "datasets", "numpy", "pandas"):
            try:
                module = __import__(name)
                versions[name] = getattr(module, "__version__", "unknown")
            except Exception as exc:
                versions[name] = "unavailable: %s" % exc
        try:
            import torch

            versions["torch_cuda"] = torch.version.cuda
            versions["cudnn"] = torch.backends.cudnn.version()
            versions["gpu"] = (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            )
        except Exception:
            pass
        _atomic_json(
            self.run_dir / "environment.json",
            {"versions": versions, "model": model_info},
        )

    def write_sample_manifest(self, rows: Iterable[Dict[str, Any]]) -> None:
        payload = list(rows)
        _atomic_json(self.run_dir / "sample_manifest.json", payload)

    def is_complete(self, sample_id: str) -> bool:
        entry = self.status.get("samples", {}).get(str(sample_id), {})
        return entry.get("state") == "complete" and (
            self.run_dir / "samples" / (self.sample_filename(sample_id) + ".json")
        ).exists()

    @staticmethod
    def sample_filename(sample_id: str) -> str:
        digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:12]
        return "%s_%s" % (_slug(str(sample_id))[:48], digest)

    def mark_running(self, sample_id: str) -> None:
        self.status.setdefault("samples", {})[str(sample_id)] = {
            "state": "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.status_path, self.status)

    def save_sample(
        self,
        result: SampleResult,
        decisions: List[SelectionDecision],
        score_bundles: List[ScoreBundle],
    ) -> None:
        name = self.sample_filename(result.sample_id)
        _atomic_json(self.run_dir / "samples" / (name + ".json"), result.to_dict())
        _atomic_json(
            self.run_dir / "decisions" / (name + ".json"),
            (
                [decision.to_dict() for decision in decisions]
                if self.cfg.diagnostics.save_selections
                else []
            ),
        )
        score_payload = []
        if self.cfg.diagnostics.save_scores:
            for bundle in score_bundles:
                score_payload.append(
                    {
                        "aggregate": bundle.aggregate,
                        "by_head": bundle.by_head,
                        "components": bundle.components,
                        "components_by_head": bundle.components_by_head,
                        "diagnostics": bundle.diagnostics,
                    }
                )
        _atomic_json(self.run_dir / "scores" / (name + ".json"), score_payload)
        self.status.setdefault("samples", {})[str(result.sample_id)] = {
            "state": "complete",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sample_file": "samples/%s.json" % name,
        }
        _atomic_json(self.status_path, self.status)

    def save_failure(self, sample_id: str, error: str, traceback_text: str) -> None:
        name = self.sample_filename(sample_id)
        _atomic_json(
            self.run_dir / "samples" / (name + ".error.json"),
            {"sample_id": sample_id, "error": error, "traceback": traceback_text},
        )
        self.status.setdefault("samples", {})[str(sample_id)] = {
            "state": "failed",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
        _atomic_json(self.status_path, self.status)

    def finalize(self, expected_samples: int) -> None:
        results = []
        for path in sorted((self.run_dir / "samples").glob("*.json")):
            if path.name.endswith(".error.json"):
                continue
            with open(path, "r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        predictions_path = self.run_dir / "predictions.jsonl"
        with open(predictions_path, "w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        scores = [float(row["score"]) for row in results if row.get("score") is not None]
        metrics = {
            "expected_samples": int(expected_samples),
            "completed_samples": len(results),
            "failed_samples": sum(
                1
                for value in self.status.get("samples", {}).values()
                if value.get("state") == "failed"
            ),
            "mean_score": sum(scores) / len(scores) if scores else None,
            "effective_n": len(scores),
        }
        _atomic_json(self.run_dir / "metrics.json", metrics)
        complete = len(results) == int(expected_samples) and metrics["failed_samples"] == 0
        self.status["state"] = "complete" if complete else "incomplete"
        _atomic_json(self.status_path, self.status)
        metadata = self._load_json(self.metadata_path, {})
        metadata["end_time"] = datetime.now(timezone.utc).isoformat()
        metadata["state"] = self.status["state"]
        _atomic_json(self.metadata_path, metadata)
