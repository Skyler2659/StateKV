"""Schema-version constants and fail-closed version checks."""
from __future__ import annotations

from typing import Any, Mapping


CONFIG_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
RUN_MANIFEST_VERSION = 1


class UnsupportedSchemaVersion(ValueError):
    """Raised when an artifact is missing or uses an unknown schema version."""


def require_schema_version(
    payload: Mapping[str, Any],
    field: str,
    expected: int,
) -> int:
    """Return a supported version or raise instead of silently guessing."""
    if field not in payload:
        raise UnsupportedSchemaVersion(
            f"missing required schema version field {field!r}"
        )
    value = payload[field]
    if isinstance(value, bool):
        raise UnsupportedSchemaVersion(
            f"invalid {field!r}: expected integer {expected}, got boolean"
        )
    try:
        actual = int(value)
    except (TypeError, ValueError) as error:
        raise UnsupportedSchemaVersion(
            f"invalid {field!r}: expected integer {expected}, got {value!r}"
        ) from error
    if actual != int(expected):
        raise UnsupportedSchemaVersion(
            f"unsupported {field}={actual}; supported version is {int(expected)}"
        )
    return actual
