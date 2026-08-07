from pathlib import Path

from statekv.repository_layout import (
    HISTORICAL_ROOT_PATHS,
    retired_document_sha256,
    resolve_repository_path,
    verify_repository_checksum,
)


ROOT = Path(__file__).resolve().parents[2]


def test_historical_root_paths_resolve_without_root_aliases() -> None:
    for historical, canonical in HISTORICAL_ROOT_PATHS.items():
        assert not (ROOT / historical).exists()
        assert resolve_repository_path(ROOT, historical) == ROOT / canonical
        if canonical.endswith(".md"):
            assert retired_document_sha256(ROOT, historical) is not None
        else:
            assert (ROOT / canonical).exists()


def test_canonical_and_absolute_paths_are_preserved() -> None:
    canonical = Path("configs/frozen/p0_v2_config.yaml")
    assert resolve_repository_path(ROOT, canonical) == ROOT / canonical
    absolute = ROOT / canonical
    assert resolve_repository_path(ROOT, absolute) == absolute


def test_checksum_verification_rejects_unrecorded_digest() -> None:
    assert not verify_repository_checksum(
        ROOT, "README.md", "0" * 64
    )


def test_retired_document_checksum_is_still_verifiable() -> None:
    digest = retired_document_sha256(ROOT, "docs/statekv/status.md")
    assert digest is not None
    assert verify_repository_checksum(ROOT, "docs/statekv/status.md", digest)
