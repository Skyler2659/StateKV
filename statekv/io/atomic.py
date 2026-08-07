"""Atomic file writers for future StateKV artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import yaml


def _prepare_path(path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    """Write bytes in the destination directory, fsync, then replace."""
    target = _prepare_path(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(Path(path), str(text).encode(encoding))


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    default: Optional[Callable[[Any], Any]] = None,
) -> Path:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=default,
    )
    return atomic_write_text(Path(path), payload + "\n")


def atomic_write_yaml(path: Path, value: Any) -> Path:
    payload = yaml.safe_dump(
        value,
        sort_keys=False,
        allow_unicode=True,
    )
    return atomic_write_text(Path(path), payload)
