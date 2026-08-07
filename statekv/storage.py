"""Small, reliable writers for mutable run artifacts.

This module deliberately handles write safety only.  Experiment identity and
scientific evidence remain in resolved configurations and structured results.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd


def _temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=path.suffix, dir=str(path.parent)
    )
    os.close(descriptor)
    return Path(temporary)


def atomic_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(
    path: Path,
    payload: Any,
    *,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    atomic_text(
        path,
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=default,
        ),
    )


def atomic_gzip_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = _temporary_path(path)
    try:
        if path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        elif path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError("unsupported table output: %s" % path)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    temporary = _temporary_path(path)
    try:
        np.savez_compressed(temporary, **dict(arrays))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
