from pathlib import Path

import pytest
import yaml

from statekv.io.registry import REQUIRED_RUN_FIELDS, validate_run_record
from statekv.io.schemas import UnsupportedSchemaVersion


ROOT = Path(__file__).resolve().parents[2]


def test_run_registry_schema_covers_required_record_fields() -> None:
    schema = yaml.safe_load(
        (ROOT / "artifacts/registry.schema.yaml").read_text(encoding="utf-8")
    )
    assert set(REQUIRED_RUN_FIELDS) <= set(schema["required"])
    assert set(REQUIRED_RUN_FIELDS) <= set(schema["properties"])


def test_documentation_example_is_a_valid_v1_record() -> None:
    example = yaml.safe_load(
        (ROOT / "artifacts/example-run.yaml").read_text(encoding="utf-8")
    )
    validate_run_record(example)
    assert example["example"] is True
    assert example["paper_usage"] == "none"


def test_unknown_run_manifest_version_fails_closed() -> None:
    example = yaml.safe_load(
        (ROOT / "artifacts/example-run.yaml").read_text(encoding="utf-8")
    )
    example["run_manifest_version"] = 999
    with pytest.raises(UnsupportedSchemaVersion):
        validate_run_record(example)
