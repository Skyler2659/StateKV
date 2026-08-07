import gzip
import json

import numpy as np
import pandas as pd

from statekv.storage import (
    atomic_frame,
    atomic_gzip_text,
    atomic_json,
    atomic_npz,
    atomic_text,
)


def test_atomic_writers_commit_complete_artifacts(tmp_path) -> None:
    text_path = tmp_path / "value.txt"
    json_path = tmp_path / "value.json"
    gzip_path = tmp_path / "value.json.gz"
    parquet_path = tmp_path / "value.parquet"
    npz_path = tmp_path / "value.npz"

    atomic_text(text_path, "statekv")
    atomic_json(json_path, {"status": "ok"})
    atomic_gzip_text(gzip_path, "compressed")
    atomic_frame(pd.DataFrame({"value": [1, 2]}), parquet_path)
    atomic_npz(npz_path, {"value": np.asarray([3, 4])})

    assert text_path.read_text() == "statekv"
    assert json.loads(json_path.read_text()) == {"status": "ok"}
    with gzip.open(gzip_path, "rt", encoding="utf-8") as handle:
        assert handle.read() == "compressed"
    assert pd.read_parquet(parquet_path)["value"].tolist() == [1, 2]
    with np.load(npz_path) as arrays:
        assert arrays["value"].tolist() == [3, 4]
    assert {path.name for path in tmp_path.iterdir()} == {
        "value.txt",
        "value.json",
        "value.json.gz",
        "value.parquet",
        "value.npz",
    }
