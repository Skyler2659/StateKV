#!/usr/bin/env python3
"""Compute the gated Stage L3 same-step output-ranking diagnostic."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOCAL_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LOCAL_SCRIPTS))

from local_core import atomic_frame, atomic_json, sha256_file
from run_l1_formal import _spearman


PREREGISTRATION_SHA256 = (
    "d21532527849ad9cf644458bbb622ba6afe9088443556d289401ece6e4c0b28e"
)
L1_SUMMARY = (
    ROOT
    / "experiments/local_truncated_jacobian/raw"
    / "l1_local_linearization/formal/l1_summary.json"
)
L1_SUMMARY_SHA256 = (
    "f127bfb9a0706719d04695013e3b943ebd8f92b66b17dde34778aa3efd0790c6"
)
L2_SUMMARY = (
    ROOT
    / "experiments/local_truncated_jacobian/raw"
    / "l2_depth_ablation/formal/l2_summary.json"
)
L2_SUMMARY_SHA256 = (
    "9e0132419c9cab4987a14be010b2aecc0347cf989527a876ac84a583b0d44ca8"
)
PHYSICAL_ROWS = (
    ROOT
    / "experiments/local_truncated_jacobian/raw"
    / "l2_depth_ablation/formal/physical_output_rows.parquet"
)
PHYSICAL_ROWS_SHA256 = (
    "a8136ca599707eb1d0f2d36668dbd58316e4fd196b55dc6b54477c63bdbcec96"
)


def group_row(group: pd.DataFrame) -> Dict[str, Any]:
    if len(group) != 8 or group["mask_hash"].nunique() != 8:
        raise RuntimeError("L3 group is not eight-distinct")
    sj1 = group["sj1_predicted_energy"].to_numpy(dtype=np.float64)
    true1 = group["strue1_physical_energy"].to_numpy(dtype=np.float64)
    s0 = group["s0_direct_energy"].to_numpy(dtype=np.float64)
    exact_kl = group["exact_kl_full_to_physical"].to_numpy(
        dtype=np.float64
    )
    rho_j_true = _spearman(sj1, true1)
    rho_true_kl = _spearman(true1, exact_kl)
    rho_j_kl = _spearman(sj1, exact_kl)
    rho_s0_kl = _spearman(s0, exact_kl)
    first = group.iloc[0]
    return {
        "sample_id": first["sample_id"],
        "task": first["task"],
        "split": first["split"],
        "anchor": int(first["anchor"]),
        "layer": int(first["layer"]),
        "candidate_count": 8,
        "rho_sj1_strue1": rho_j_true,
        "rho_strue1_exact_kl": rho_true_kl,
        "rho_sj1_exact_kl": rho_j_kl,
        "rho_s0_exact_kl": rho_s0_kl,
        "delta_rho_j1_minus_s0_to_kl": float(
            rho_j_kl - rho_s0_kl
        ),
        "finite": bool(
            np.isfinite(sj1).all()
            and np.isfinite(true1).all()
            and np.isfinite(s0).all()
            and np.isfinite(exact_kl).all()
            and np.isfinite(
                [
                    rho_j_true,
                    rho_true_kl,
                    rho_j_kl,
                    rho_s0_kl,
                ]
            ).all()
        ),
    }


def aggregate(frame: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    metrics = [
        "rho_sj1_strue1",
        "rho_strue1_exact_kl",
        "rho_sj1_exact_kl",
        "rho_s0_exact_kl",
        "delta_rho_j1_minus_s0_to_kl",
    ]
    return frame.groupby(by)[metrics].median().reset_index()


def run(output_dir: Path) -> Dict[str, Any]:
    preregistration = (
        ROOT / "experiments/local_truncated_jacobian/PREREGISTRATION.md"
    )
    checksums = {
        "preregistration": (
            sha256_file(preregistration) == PREREGISTRATION_SHA256
        ),
        "l1_summary": sha256_file(L1_SUMMARY) == L1_SUMMARY_SHA256,
        "l2_summary": sha256_file(L2_SUMMARY) == L2_SUMMARY_SHA256,
        "physical_rows": (
            sha256_file(PHYSICAL_ROWS) == PHYSICAL_ROWS_SHA256
        ),
    }
    if not all(checksums.values()):
        raise RuntimeError(f"L3 source checksum failure: {checksums}")
    l1 = json.loads(L1_SUMMARY.read_text(encoding="utf-8"))
    if not bool(l1["formal_l1_local_passed"]):
        raise RuntimeError("formal L1 gate does not authorize L3")
    candidates = pd.read_parquet(PHYSICAL_ROWS)
    groups = pd.DataFrame(
        [
            group_row(group)
            for _key, group in candidates.groupby(
                ["sample_id", "task", "anchor", "layer"],
                sort=False,
            )
        ]
    )
    sequence = aggregate(groups, ["sample_id", "task"])
    task = aggregate(sequence, ["task"])
    metric_columns = [
        "rho_sj1_strue1",
        "rho_strue1_exact_kl",
        "rho_sj1_exact_kl",
        "rho_s0_exact_kl",
        "delta_rho_j1_minus_s0_to_kl",
    ]
    pooled = {
        column: float(sequence[column].median())
        for column in metric_columns
    }
    integrity_checks = {
        **checksums,
        "candidate_rows": len(candidates) == 640,
        "group_rows": len(groups) == 80,
        "sequence_rows": len(sequence) == 4,
        "task_rows": len(task) == 2,
        "candidate_groups": bool(
            candidates.groupby(
                ["sample_id", "anchor", "layer"]
            ).size().eq(8).all()
        ),
        "candidate_finite": bool(candidates["finite"].all()),
        "group_finite": bool(groups["finite"].all()),
        "exact_kl_nonnegative": bool(
            candidates["exact_kl_full_to_physical"].ge(0.0).all()
        ),
        "calibration_or_test_not_loaded": True,
    }
    summary = {
        "stage": "formal_l3_secondary_diagnostic",
        "status": "complete",
        "is_primary_gate": False,
        "changes_l0_l1_l2_outcome": False,
        "split": "train",
        "calibration_or_test_loaded": False,
        "integrity": {
            "passed": bool(all(integrity_checks.values())),
            "checks": integrity_checks,
        },
        "sequence_independent_pooled_medians": pooled,
        "task_medians": task.to_dict("records"),
        "exact_kl_distribution": {
            "minimum": float(
                candidates["exact_kl_full_to_physical"].min()
            ),
            "median": float(
                candidates["exact_kl_full_to_physical"].median()
            ),
            "maximum": float(
                candidates["exact_kl_full_to_physical"].max()
            ),
        },
        "interpretation_scope": (
            "same-step output-ranking diagnostic only; no Fisher, "
            "policy, held-out, or causal utility claim"
        ),
        "row_counts": {
            "candidate": int(len(candidates)),
            "group": int(len(groups)),
            "sequence": int(len(sequence)),
            "task": int(len(task)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_frame(output_dir / "candidate_rows.parquet", candidates)
    atomic_frame(output_dir / "group_ranking_rows.parquet", groups)
    atomic_frame(output_dir / "sequence_aggregate.parquet", sequence)
    atomic_frame(output_dir / "task_aggregate.parquet", task)
    atomic_json(output_dir / "l3_summary.json", summary)
    pd.DataFrame([pooled]).to_csv(
        output_dir / "l3_summary.csv", index=False
    )
    atomic_json(
        output_dir / "status.json",
        {
            "stage": "formal_l3_secondary_diagnostic",
            "state": "complete",
            "is_primary_gate": False,
            "integrity_passed": summary["integrity"]["passed"],
            "calibration_or_test_loaded": False,
            "errors": [],
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "experiments/local_truncated_jacobian/raw"
        / "l3_output_diagnostic/formal",
    )
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
