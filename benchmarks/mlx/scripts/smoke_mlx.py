#!/usr/bin/env python3
"""End-to-end MLX smoke with hard scientific-validity assertions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/experiments/dev/qwen25_05b_mlx_method_sanity.yaml"
OUTPUT_ROOT = Path("/tmp/l1_robust_kv_cache_smoke/mlx_qwen25_05b_inst_4bit/niah")


def run(command):
    print("$", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.artifacts import ScoreArtifact, SelectionArtifact, load_artifact
    from src.config import ExperimentConfig
    from src.eviction.registry import get_method_spec

    cfg = ExperimentConfig.from_yaml(CONFIG)
    py = sys.executable
    run(
        [
            py,
            "scripts/run_benchmark.py",
            "--config",
            str(CONFIG.relative_to(ROOT)),
            "--num_samples",
            "1",
            "--skip_analysis",
        ]
    )
    run_dir = sorted(path for path in OUTPUT_ROOT.glob("20*") if path.is_dir())[-1]
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    expected_methods = list(cfg.methods or [])
    assert [row["method"] for row in results] == expected_methods
    assert not [row for row in results if row.get("error")]
    assert not [row for row in results if row.get("skipped")]
    assert not [row for row in results if row.get("sanity_check_failed")]
    assert not [row for row in results if int(row.get("attention_hook_errors") or 0)]
    assert not [row for row in results if row.get("estimator_failures")]
    assert not [row for row in results if int(row.get("estimator_fallback_count") or 0)]

    snapshots = set()
    for row in results:
        assert row.get("artifact_schema_version") == 2
        selection_path = Path(row["selection_artifact_path"])
        assert selection_path.exists()
        selection = load_artifact(selection_path)
        assert isinstance(selection, SelectionArtifact)
        snapshots.add(selection.snapshot.snapshot_id)
        spec = get_method_spec(row["method"])
        if spec.requires_scores:
            score_path = Path(row["score_artifact_path"])
            assert score_path.exists()
            assert isinstance(load_artifact(score_path), ScoreArtifact)
            assert row["mechanism_artifacts_alignment_safe"] is True
    assert len(snapshots) == 1

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("manifest_schema_version") == 1
    for key in (
        "git_commit",
        "tracked_patch_sha256",
        "repository_state_id",
        "config_hash",
        "sample_manifest_hash",
    ):
        assert manifest.get(key)
    assert manifest.get("untracked_file_sha256")
    assert all(manifest["untracked_file_sha256"].values())

    run(
        [
            py,
            "scripts/run_analysis.py",
            "--input",
            str(run_dir),
            "--config",
            "configs/analysis/basic.yaml",
        ]
    )
    overlap = json.loads(
        (run_dir / "analysis/overlap_summary.json").read_text(encoding="utf-8")
    )
    rank = json.loads(
        (run_dir / "analysis/rank_correlation_summary.json").read_text(encoding="utf-8")
    )
    assert overlap["pairs"]
    assert rank["pairs"]
    assert overlap["confidence_intervals"]
    assert rank["confidence_intervals"]
    assert (run_dir / "analysis/overlap_per_sample_layer_head.csv").exists()
    assert (run_dir / "analysis/rank_per_sample_layer_head.csv").exists()
    assert any(
        "budget scopes do not match" in item["reason"]
        for item in overlap["mismatched_pairs"]
    )
    print(f"MLX smoke passed: {run_dir}")


if __name__ == "__main__":
    main()
