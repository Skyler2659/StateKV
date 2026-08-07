import hashlib
from pathlib import Path

import pytest
import yaml

from statekv.io.registry import (
    FrozenExperimentError,
    assert_experiment_mutable,
    load_yaml_mapping,
    validate_frozen_registry,
)
from statekv.repository_layout import (
    retired_document_sha256,
    repository_sha256,
    resolve_repository_path,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "experiments" / "frozen_registry.yaml"
EXPECTED_EXPERIMENTS = {
    "predictive_closure",
    "local_truncated_jacobian",
    "p0_v2_fixed_boundary",
    "p1_state_conditioned",
    "p2_state_local_risk",
    "p2_recovery",
    "p3_decision_validity",
    "p3_physical_recovery",
    "p3pr_generalization",
}


def test_frozen_registry_is_complete_and_resolves_paths() -> None:
    registry = load_yaml_mapping(REGISTRY_PATH)
    validate_frozen_registry(registry, repository_root=ROOT)
    assert set(registry["experiments"]) == EXPECTED_EXPERIMENTS


def test_registered_manifests_expose_checksum_provenance() -> None:
    registry = load_yaml_mapping(REGISTRY_PATH)
    for experiment_id, entry in registry["experiments"].items():
        manifest = ROOT / entry["manifest"]
        text = manifest.read_text(encoding="utf-8").lower()
        assert "sha256" in text or "checksum" in text, experiment_id


def test_ordinary_migration_is_blocked_for_every_frozen_phase() -> None:
    registry = load_yaml_mapping(REGISTRY_PATH)
    for experiment_id in EXPECTED_EXPERIMENTS:
        with pytest.raises(FrozenExperimentError):
            assert_experiment_mutable(registry, experiment_id)


def test_checksum_bound_mlx_compatibility_copy_is_unchanged() -> None:
    frozen = ROOT / "benchmarks/torch/kvbench/temporal/backend_mlx.py"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == (
        "07f961347f99c16d7bdf187f76ecc7be6c1bd8a894f01ad67d55c857f031d0a7"
    )


def test_layout_migration_ledger_binds_old_and_current_hashes() -> None:
    registry = load_yaml_mapping(REGISTRY_PATH)
    ledger_path = ROOT / registry["policy"]["layout_migration_ledger"]
    ledger = load_yaml_mapping(ledger_path)
    migrations = ledger["source_migrations"]
    manifest_bindings = set()
    for entry in registry["experiments"].values():
        manifest = yaml.safe_load((ROOT / entry["manifest"]).read_text())
        for recorded_path, digest in manifest.get("checksums", {}).items():
            canonical = resolve_repository_path(
                ROOT, recorded_path
            ).relative_to(ROOT).as_posix()
            manifest_bindings.add((canonical, str(digest)))

    assert migrations
    for canonical, migration in migrations.items():
        target = ROOT / canonical
        assert migration["change_class"] == "repository-path-only"
        if target.is_file():
            current_sha256 = repository_sha256(target)
        else:
            current_sha256 = retired_document_sha256(ROOT, canonical)
        assert current_sha256 == migration["current_sha256"]
        assert (
            canonical,
            migration["evaluation_sha256"],
        ) in manifest_bindings


def test_retired_document_ledger_is_complete_and_absent() -> None:
    registry = load_yaml_mapping(REGISTRY_PATH)
    ledger_path = ROOT / registry["policy"]["retired_document_ledger"]
    ledger = load_yaml_mapping(ledger_path)
    documents = ledger["documents"]
    assert ledger["document_count"] == len(documents)
    assert documents
    assert "README.md" not in documents
    for relative, digest in documents.items():
        assert relative.endswith(".md")
        assert len(digest) == 64
        assert not (ROOT / relative).exists(), relative
