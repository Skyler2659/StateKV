"""Canonical repository paths for evaluation-time provenance names.

Frozen manifests intentionally keep the logical filenames used when an
experiment ran.  The current checkout does not materialize those historical
names at the repository root; callers must resolve them through this module.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import yaml


HISTORICAL_ROOT_PATHS: Mapping[str, str] = MappingProxyType(
    {
        "CURRENT_THEORY_MODEL_AND_EXPERIMENTAL_VALIDATION_ZH_REVISED.md": "docs/archive/theory_iterations/current_theory_model_revised.md",
        "P0_V2_CODE_AUDIT.md": "experiments/p0_v2_fixed_boundary/docs/code_audit.md",
        "P0_V2_EXPERIMENT_PLAN.md": "experiments/p0_v2_fixed_boundary/docs/experiment_plan.md",
        "P0_V2_RESULTS.md": "experiments/p0_v2_fixed_boundary/docs/results.md",
        "P1_CALIBRATION_REPORT.md": "experiments/p1_state_conditioned/docs/calibration.md",
        "P1_CODE_AUDIT.md": "experiments/p1_state_conditioned/docs/code_audit.md",
        "P1_STATE_CONDITIONED_EXPERIMENT_PLAN.md": "experiments/p1_state_conditioned/docs/experiment_plan.md",
        "P1_STATE_CONDITIONED_FAILURE_ANALYSIS.md": "experiments/p1_state_conditioned/docs/failure_analysis.md",
        "P1_STATE_CONDITIONED_RESULTS.md": "experiments/p1_state_conditioned/docs/results.md",
        "P2_CALIBRATION_REPORT.md": "experiments/p2_state_local_risk/docs/calibration.md",
        "P2_CODE_AUDIT.md": "experiments/p2_state_local_risk/docs/code_audit.md",
        "P2_STATE_LOCAL_EXPERIMENT_PLAN.md": "experiments/p2_state_local_risk/docs/experiment_plan.md",
        "P2_STATE_LOCAL_FAILURE_ANALYSIS.md": "experiments/p2_state_local_risk/docs/failure_analysis.md",
        "P2_STATE_LOCAL_RESULTS.md": "experiments/p2_state_local_risk/docs/results.md",
        "R0_P2_FAILURE_MAP.md": "experiments/p2_recovery/docs/r0_failure_map.md",
        "P2_RECOVERY_MASTER_LOG.md": "experiments/p2_recovery/docs/master_log.md",
        "P2_RECOVERY_DECISION_TREE.md": "experiments/p2_recovery/docs/decision_tree.md",
        "P2_RECOVERY_CUMULATIVE_RESULTS.md": "experiments/p2_recovery/docs/cumulative_results.md",
        "P2_RECOVERY_FINAL_RECOMMENDATION.md": "experiments/p2_recovery/docs/final_recommendation.md",
        "P2_RECOVERY_EXPERIMENT_RESULTS_SUMMARY_ZH.md": "experiments/p2_recovery/docs/results_summary.md",
        "P3PR_CROSS_MODEL_TASK_GENERALIZATION_REPORT_ZH.md": "docs/statekv/generalization.md",
        "P3_PHYSICAL_RECOVERY_DETAILED_EXPERIMENT_REPORT_ZH.md": "experiments/p3_physical_recovery/docs/detailed_report.md",
        "P3_PHYSICAL_RECOVERY_FINAL_SUMMARY_ZH.md": "docs/statekv/physical_same_step.md",
        "p0_v2_config.yaml": "configs/frozen/p0_v2_config.yaml",
        "p1_state_conditioned_config.yaml": "configs/frozen/p1_state_conditioned_config.yaml",
        "p2_state_local_config.yaml": "configs/frozen/p2_state_local_config.yaml",
        "P2_RECOVERY_DATA_LEDGER.yaml": "configs/frozen/P2_RECOVERY_DATA_LEDGER.yaml",
        "torch_project": "benchmarks/torch",
    }
)

LAYOUT_MIGRATION_LEDGER = Path("experiments/layout_migrations.yaml")
RETIRED_DOCUMENT_LEDGER = Path("experiments/retired_documents.yaml")


def resolve_repository_path(
    repository_root: Path, recorded_path: Union[str, Path]
) -> Path:
    """Resolve a current or evaluation-time repository-relative path."""

    path = Path(recorded_path)
    if path.is_absolute():
        return path
    relative = path.as_posix()
    canonical = HISTORICAL_ROOT_PATHS.get(relative, relative)
    return repository_root / canonical


def repository_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a repository file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _layout_migrations(repository_root: Path) -> Mapping[str, Any]:
    ledger = repository_root / LAYOUT_MIGRATION_LEDGER
    if not ledger.is_file():
        return {}
    payload = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
    migrations = payload.get("source_migrations", {})
    return migrations if isinstance(migrations, Mapping) else {}


@lru_cache(maxsize=4)
def _retired_documents(repository_root: Path) -> Mapping[str, str]:
    ledger = repository_root / RETIRED_DOCUMENT_LEDGER
    if not ledger.is_file():
        return {}
    payload = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
    documents = payload.get("documents", {})
    return documents if isinstance(documents, Mapping) else {}


def retired_document_sha256(
    repository_root: Path, recorded_path: Union[str, Path]
) -> Optional[str]:
    """Return the archived digest for an intentionally retired document."""

    root = repository_root.resolve()
    target = resolve_repository_path(root, recorded_path)
    try:
        canonical = target.relative_to(root).as_posix()
    except ValueError:
        return None
    digest = _retired_documents(root).get(canonical)
    return str(digest) if digest is not None else None


def verify_repository_checksum(
    repository_root: Path,
    recorded_path: Union[str, Path],
    evaluation_sha256: str,
) -> bool:
    """Verify an evaluation checksum or an explicitly recorded layout edit.

    Frozen manifests remain evaluation-time records. A differing current hash
    is accepted only when the migration ledger binds that exact old digest to
    the exact current digest for the canonical path.
    """

    root = repository_root.resolve()
    target = resolve_repository_path(root, recorded_path)
    try:
        canonical = target.relative_to(root).as_posix()
    except ValueError:
        return False
    if target.is_file():
        current_sha256 = repository_sha256(target)
    else:
        current_sha256 = retired_document_sha256(root, canonical)
        if current_sha256 is None:
            return False
    if current_sha256 == str(evaluation_sha256):
        return True
    migration = _layout_migrations(root).get(canonical, {})
    return bool(
        isinstance(migration, Mapping)
        and migration.get("evaluation_sha256") == str(evaluation_sha256)
        and migration.get("current_sha256") == current_sha256
        and migration.get("change_class") == "repository-path-only"
    )
