"""Validation helpers for frozen-experiment and future run registries."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from statekv.io.schemas import RUN_MANIFEST_VERSION, require_schema_version


RUN_STATUSES = frozenset(
    {
        "complete",
        "negative-result",
        "failed-run",
        "interrupted-run",
        "obsolete-protocol",
        "smoke",
        "in-progress",
        "not-run",
    }
)

REQUIRED_RUN_FIELDS = (
    "run_manifest_version",
    "config_schema_version",
    "artifact_schema_version",
    "run_id",
    "phase",
    "status",
    "protocol_version",
    "config_hash",
    "git_commit",
    "dirty_diff_hash",
    "backend",
    "model",
    "model_revision",
    "tokenizer_revision",
    "dataset",
    "dataset_revision",
    "artifact_path",
    "paper_usage",
)


class FrozenExperimentError(RuntimeError):
    """Raised when a caller attempts ordinary mutation of frozen evidence."""


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a repository-relative path: {value!r}")
    return path


def validate_run_record(record: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_RUN_FIELDS if name not in record]
    if missing:
        raise ValueError(f"run record missing required fields: {missing}")
    require_schema_version(
        record, "run_manifest_version", RUN_MANIFEST_VERSION
    )
    require_schema_version(record, "config_schema_version", 1)
    require_schema_version(record, "artifact_schema_version", 1)
    if str(record["status"]) not in RUN_STATUSES:
        raise ValueError(f"unknown run status: {record['status']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["config_hash"])):
        raise ValueError("config_hash must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{40}", str(record["git_commit"])):
        raise ValueError("git_commit must be a lowercase 40-character commit")
    dirty_hash = record["dirty_diff_hash"]
    if dirty_hash is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(dirty_hash)
    ):
        raise ValueError("dirty_diff_hash must be null or a SHA-256 digest")
    _relative_path(record["artifact_path"], "artifact_path")


def validate_frozen_registry(
    registry: Mapping[str, Any],
    *,
    repository_root: Optional[Path] = None,
) -> None:
    if int(registry.get("registry_version", -1)) != 1:
        raise ValueError("unsupported frozen registry version")
    experiments = registry.get("experiments")
    if not isinstance(experiments, Mapping) or not experiments:
        raise ValueError("frozen registry has no experiments")
    root = Path(repository_root).resolve() if repository_root else None
    for experiment_id, payload in experiments.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"invalid frozen entry: {experiment_id}")
        for field in (
            "path",
            "status",
            "scientific_role",
            "mutable",
            "manifest",
            "ledger",
            "config_paths",
            "entry_points",
            "canonical_claims",
            "notes",
        ):
            if field not in payload:
                raise ValueError(f"{experiment_id} missing {field}")
        if payload["mutable"] is not False:
            raise ValueError(f"frozen experiment is mutable: {experiment_id}")
        paths = [payload["path"], payload["manifest"]]
        paths.extend(payload["ledger"] or [])
        paths.extend(payload["config_paths"] or [])
        paths.extend(payload["entry_points"] or [])
        for value in paths:
            relative = _relative_path(value, str(experiment_id))
            if root is not None and not (root / relative).exists():
                raise ValueError(
                    f"{experiment_id} references missing path: {relative}"
                )


def assert_experiment_mutable(
    registry: Mapping[str, Any], experiment_id: str
) -> None:
    experiments = registry.get("experiments", {})
    if experiment_id not in experiments:
        raise KeyError(f"experiment is not registered: {experiment_id}")
    if experiments[experiment_id].get("mutable") is not True:
        raise FrozenExperimentError(
            f"experiment {experiment_id!r} is frozen and cannot be modified "
            "by ordinary migration tooling"
        )
