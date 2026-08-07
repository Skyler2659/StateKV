from pathlib import Path

import pandas as pd
import pytest
import yaml

from statekv.io.atomic import atomic_write_json, atomic_write_yaml
from statekv.io.provenance import collect_runtime_provenance, sha256_file
from statekv.io.schemas import UnsupportedSchemaVersion, require_schema_version
from statekv.io.tables import atomic_write_frame


def test_atomic_structured_writers_replace_complete_files(tmp_path: Path) -> None:
    json_path = tmp_path / "record.json"
    yaml_path = tmp_path / "record.yaml"
    atomic_write_json(json_path, {"version": 1, "value": "first"})
    atomic_write_json(json_path, {"version": 1, "value": "second"})
    atomic_write_yaml(yaml_path, {"version": 1, "value": "second"})
    assert '"value": "second"' in json_path.read_text(encoding="utf-8")
    assert yaml.safe_load(yaml_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "value": "second",
    }
    assert len(sha256_file(json_path)) == 64


def test_atomic_csv_writer_preserves_frame(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    frame = pd.DataFrame({"candidate": ["a", "b"], "risk": [1.0, 2.0]})
    atomic_write_frame(path, frame)
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(loaded, frame)


def test_runtime_provenance_is_framework_neutral() -> None:
    value = collect_runtime_provenance(
        command=["statekv", "example"], packages=()
    )
    assert value["command"] == ["statekv", "example"]
    assert value["python_version"]
    assert value["hardware"]["machine"]


def test_schema_version_checks_fail_closed() -> None:
    assert require_schema_version({"version": 1}, "version", 1) == 1
    with pytest.raises(UnsupportedSchemaVersion):
        require_schema_version({}, "version", 1)
    with pytest.raises(UnsupportedSchemaVersion):
        require_schema_version({"version": 2}, "version", 1)
