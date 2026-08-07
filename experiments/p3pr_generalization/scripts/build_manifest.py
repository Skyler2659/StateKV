#!/usr/bin/env python3
"""Build a checksum manifest for the completed generalization experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3pr_generalization"
REPORT = ROOT / "docs/statekv/generalization.md"
MANIFEST = EXPERIMENT / "P3PR_GENERALIZATION_MANIFEST.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [
        EXPERIMENT / "p3pr_generalization_config.yaml",
        *sorted((EXPERIMENT / "scripts").glob("*.py")),
        *sorted((EXPERIMENT / "results").rglob("*.json")),
        *sorted((EXPERIMENT / "results").rglob("*.parquet")),
        *sorted((EXPERIMENT / "results").rglob("*.csv")),
        REPORT,
        ROOT / "benchmarks/torch/kvbench/temporal/backend_mlx.py",
        ROOT / "tests/test_p3pr_generalization.py",
    ]
    paths = sorted({path.resolve() for path in paths if path.exists()})
    payload = {
        "schema_version": 1,
        "program": "p3pr_cross_model_task_generalization",
        "report": str(REPORT.relative_to(ROOT)),
        "entry_count": len(paths),
        "checksums": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in paths
        },
    }
    MANIFEST.write_text(
        yaml.safe_dump(
            payload,
            allow_unicode=True,
            sort_keys=True,
            width=1000,
        ),
        encoding="utf-8",
    )
    verification = {
        relative: sha256_file(ROOT / relative) == expected
        for relative, expected in payload["checksums"].items()
    }
    if not all(verification.values()):
        raise RuntimeError("manifest verification failed")
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "entry_count": len(paths),
                "verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
