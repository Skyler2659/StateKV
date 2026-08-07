"""Atomic DataFrame output for new, non-frozen artifact writers."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_frame(path: Path, frame: Any, *, index: bool = False) -> Path:
    """Atomically write a pandas-compatible frame as CSV or Parquet.

    The function is intentionally not wired into existing experiment writers;
    their byte-level behavior remains frozen until parity fixtures exist.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError(f"unsupported table suffix: {target.suffix!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.stem + ".", suffix=suffix, dir=str(target.parent)
    )
    os.close(descriptor)
    try:
        if suffix == ".csv":
            frame.to_csv(temporary, index=index)
        else:
            frame.to_parquet(temporary, index=index)
        read_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(read_descriptor)
        finally:
            os.close(read_descriptor)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
