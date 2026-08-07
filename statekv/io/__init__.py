"""Versioned, framework-neutral I/O and provenance interfaces.

These helpers are additive. Existing experiment writers continue to use their
frozen implementations until parity fixtures justify migration.
"""

from statekv.io.atomic import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
)
from statekv.io.provenance import (
    collect_git_provenance,
    collect_runtime_provenance,
    sha256_file,
)
from statekv.io.schemas import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    RUN_MANIFEST_VERSION,
    UnsupportedSchemaVersion,
    require_schema_version,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "RUN_MANIFEST_VERSION",
    "UnsupportedSchemaVersion",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "atomic_write_yaml",
    "collect_git_provenance",
    "collect_runtime_provenance",
    "require_schema_version",
    "sha256_file",
]
