#!/usr/bin/env python
"""Validate numerical and protocol invariants of final output artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.trajectory_analysis import atomic_json


REQUIRED = [
    "OUTPUT_SENSITIVITY_READOUT_DESIGN_ZH.md",
    "configs/stages/output_sensitivity_config.yaml",
    "output_bridge_rows.parquet",
    "output_bridge_coverage_summary.json",
    "output_bridge_tightness_summary.json",
    "output_bridge_ranking_summary.json",
    "pairwise_action_rows.parquet",
    "pairwise_action_calibration_summary.json",
    "selection_policy_rows.parquet",
    "selection_policy_summary.json",
    "refresh_lcb_policy_rows.parquet",
    "refresh_lcb_policy_summary.json",
    "free_generation_results.json",
    "OUTPUT_SENSITIVITY_ANALYTICAL_DERIVATION_ZH.md",
    "OUTPUT_SENSITIVITY_READOUT_RESULTS_ZH.md",
    "THEORY_MODEL_UPDATE_AFTER_OUTPUT_READOUT_ZH.md",
]


def _close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(
        math.isclose(
            float(left), float(right), rel_tol=tolerance, abs_tol=tolerance
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    repo = Path(args.repo_root).resolve()
    checks = {}
    for name in REQUIRED:
        path = repo / name if name.endswith((".md", ".yaml")) else run_dir / name
        checks["artifact:%s" % name] = path.exists()
    raw = pd.read_parquet(run_dir / "output_candidate_rows.parquet")
    inventory = pd.read_parquet(
        run_dir / "output_candidate_inventory.parquet"
    )
    probes = pd.read_parquet(
        run_dir / "output_jacobian_probe_rows.parquet"
    )
    output = pd.read_parquet(run_dir / "output_bridge_rows.parquet")
    pair = pd.read_parquet(run_dir / "pairwise_action_rows.parquet")
    selection = pd.read_parquet(run_dir / "selection_policy_rows.parquet")
    metadata = json.load((run_dir / "metadata.json").open())
    coverage = json.load(
        (run_dir / "output_bridge_coverage_summary.json").open()
    )
    folds = json.load(
        (run_dir / "output_bridge_fold_models.json").open()
    )
    checks.update(
        {
            "24_independent_sequences": raw["sample_id"].nunique() == 24,
            "12_sequences_per_task": (
                raw.groupby("task")["sample_id"].nunique().sort_values().tolist()
                == [12, 12]
            ),
            "official_govreport": any(
                event["source"] == "official_longbench_gov_report"
                and event["dataset_official"]
                and event["count"] == 12
                for event in metadata["task_load_events"]
            ),
            "24_candidates_per_anchor": (
                inventory.groupby(["sample_id", "anchor"])[
                    "candidate_id"
                ].nunique().eq(24).all()
            ),
            "24_distinct_physical_masks_per_anchor": (
                inventory.groupby(["sample_id", "anchor"])[
                    "mask_hash"
                ].nunique().eq(24).all()
            ),
            "candidate_budgets_equal": (
                inventory.groupby(["sample_id", "anchor"])[
                    "total_budget"
                ].nunique().eq(1).all()
            ),
            "physical_shared_masks": bool(
                inventory["physical_layer_shared_mask"].all()
                and inventory["gqa_shared"].all()
            ),
            "no_future_truth": bool(
                not raw["uses_future_compressed_truth"].any()
                and not inventory["uses_future_compressed_truth"].any()
            ),
            "task_not_a_feature": bool(
                not inventory["uses_task_feature"].any()
                and not output["task_feature_used"].any()
            ),
            "future_label_not_a_feature": bool(
                not output["future_label_used"].any()
            ),
            "token_alignment": bool(raw["token_position_aligned"].all()),
            "exact_budget_128": bool(
                raw["active_cache_tokens"].eq(128).all()
            ),
            "three_fd_radii": probes.groupby(
                ["sample_id", "anchor", "layer"]
            )["relative_radius"].nunique().eq(3).all(),
            "eight_fd_directions": probes.groupby(
                ["sample_id", "anchor", "layer"]
            )["direction_index"].nunique().eq(8).all(),
            "fd_not_claimed_operator_norm": bool(
                not probes["claimed_operator_norm"].any()
            ),
            "softmax_kl_each_row": bool(
                np.all(
                    raw["exact_kl"].to_numpy()
                    <= 0.25 * raw["logit_l2_sq"].to_numpy() + 1e-7
                )
            ),
            "logit_and_kl_coverage_separate": bool(
                {"logit_covered", "kl_covered"}.issubset(output.columns)
            ),
            "pair_interval_ordered": bool(
                (pair["interval_lower"] <= pair["interval_upper"]).all()
            ),
            "pair_interval_center": bool(
                np.allclose(
                    pair["predicted_delta"],
                    0.5 * (
                        pair["interval_lower"] + pair["interval_upper"]
                    ),
                )
            ),
            "pair_bootstrap_sequence": bool(
                pair["bootstrap_unit"].eq("sequence").all()
            ),
            "selection_candidate_count": bool(
                selection["candidate_count"].eq(24).all()
            ),
            "layer27_reported": any(
                int(row["layer"]) == 27 for row in coverage["layer_27"]
            ),
            "task_split_reported": set(
                row["task_bucket"] for row in coverage["output_bridge"]
            )
            == {"NIAH", "GovReport"},
        }
    )
    checks["nested_split_and_nonnegative_models"] = all(
        len(payload["fit_sequences"]) == 10
        and len(payload["state_margin_sequences"]) == 5
        and len(payload["output_calibration_sequences"]) == 8
        and not (
            set(payload["fit_sequences"])
            & set(payload["state_margin_sequences"])
        )
        and not (
            set(payload["fit_sequences"])
            & set(payload["output_calibration_sequences"])
        )
        and payload["e2"]["nonnegative"]
        and all(
            bridge.get("nonnegative", True)
            for bridge in payload["bridges"].values()
        )
        and payload["pairwise_monotone_correction"]["nonnegative"]
        and not payload["pairwise_monotone_correction"][
            "task_feature_used"
        ]
        for payload in folds.values()
    )
    actual_coverage = (
        output.groupby(["bridge_family", "task_bucket"])[
            ["logit_covered", "kl_covered"]
        ]
        .mean()
    )
    checks["coverage_json_matches_parquet"] = all(
        _close(
            actual_coverage.loc[
                (row["bridge_family"], row["task_bucket"]),
                "logit_covered",
            ],
            row["logit_coverage"],
        )
        and _close(
            actual_coverage.loc[
                (row["bridge_family"], row["task_bucket"]),
                "kl_covered",
            ],
            row["induced_kl_coverage"],
        )
        for row in coverage["output_bridge"]
    )
    refresh = pd.read_parquet(run_dir / "refresh_lcb_policy_rows.parquet")
    if len(refresh):
        checks.update(
            {
                "refresh_maximum_respected": bool(
                    (
                        refresh["actual_refresh_count"]
                        <= refresh["maximum_refresh_count"]
                    ).all()
                ),
                "refresh_state_not_reset": bool(
                    not refresh["refresh_reset_state_error"].any()
                ),
                "refresh_no_deleted_kv_recall": bool(
                    not refresh["refresh_recalled_deleted_kv"].any()
                ),
                "matched_refresh_count_columns": bool(
                    {
                        "maximum_refresh_count",
                        "actual_refresh_count",
                    }.issubset(refresh.columns)
                ),
            }
        )
    free = json.load((run_dir / "free_generation_results.json").open())
    checks["teacher_forced_freegen_separated"] = bool(
        free.get("teacher_forced_and_free_generation_separated", False)
        or free.get("protocol", "").startswith("free generation")
    )
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "passed": bool(all(checks.values())),
        "passed_count": int(sum(bool(value) for value in checks.values())),
        "check_count": int(len(checks)),
        "checks": checks,
    }
    atomic_json(run_dir / "output_sensitivity_artifact_validation.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
