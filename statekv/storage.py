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


def safe_path_component(value: Any) -> str:
    """Return the stable filesystem spelling used for logical artifact IDs.

    Artifact IDs such as ``gov_report:192`` are meaningful to analysis code,
    but ``:`` and ``/`` are unsuitable in a portable path component.  Keep
    the historical encoding in one place so writers and readers cannot drift.
    """

    return str(value).replace(":", "__").replace("/", "_")


def atomic_npz(
    path: Path,
    arrays: Optional[Mapping[str, Any]] = None,
    *,
    compressed: bool = True,
    **named_arrays: Any,
) -> None:
    """Atomically publish a compressed NumPy archive.

    ``arrays`` accepts the mapping form used by data collectors. Keyword
    arrays are supported for concise single-artifact writes. Set
    ``compressed=False`` only where an existing artifact contract requires an
    uncompressed archive. The two payload forms are deliberately exclusive to
    make accidental partial payload assembly obvious at the call site.
    """

    if arrays is not None and named_arrays:
        raise ValueError("pass NPZ arrays as either a mapping or keyword arrays, not both")
    payload = dict(named_arrays if arrays is None else arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        # Writing through the descriptor prevents NumPy from appending its
        # own ``.npz`` suffix.  The temporary file therefore never matches a
        # collector's ``*.npz`` publication glob.
        with os.fdopen(descriptor, "wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
