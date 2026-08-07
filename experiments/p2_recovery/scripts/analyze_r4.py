#!/usr/bin/env python3
"""Analyze R4 scalar decision-risk calibration, formal, and replication."""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
P2_DIR = ROOT / "experiments/p2_state_local_risk/scripts"
for value in (SCRIPT_DIR, P2_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from analyze_r3 import (  # noqa: E402
    UNIT,
    decision_gate,
    make_rankings,
    response_breakdown,
    score_rows,
    sequence_rankings,
)
from p2_core import atomic_frame, atomic_json, sha256_file  # noqa: E402
from recovery_core import state_local_quadratic_risk  # noqa: E402


EXPERIMENT = (
    ROOT / "experiments/p2_recovery/r4_scalar_decision_risk"
)


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(
        (EXPERIMENT / "r4_config.yaml").read_text(
            encoding="utf-8"
        )
    )


def load_stage(stage: str) -> Dict[str, Any]:
    directory = EXPERIMENT / "results" / stage
    metadata = json.loads(
        (directory / "stage_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if not metadata["completed"]:
        raise RuntimeError(f"R4 {stage} is incomplete")
    return {
        "directory": directory,
        "metadata": metadata,
        "response": pd.read_parquet(
            directory / "path_response_rows.parquet"
        ),
        "directions": pd.read_parquet(
            directory / "direction_rows.parquet"
        ),
        "vectors": pd.read_parquet(
            directory / "sequence_vector_metrics.parquet"
        ),
    }


def integrity(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    response = data["response"]
    numeric = response.select_dtypes(include=[np.number])
    checks = {
        "all_numeric_finite": bool(
            np.isfinite(numeric.to_numpy()).all()
        ),
        "all_row_finite": bool(response["finite"].all()),
        "only_frozen_method": set(response["method"])
        == {"midpoint_k2"},
        "only_frozen_backend": set(response["numeric_backend"])
        == {config["numeric_backend"]["name"]},
        "no_exact_kl_predictor_parameter": (
            "exact_kl"
            not in inspect.signature(
                state_local_quadratic_risk
            ).parameters
        ),
        "no_truth_predictor_parameter": (
            "truth"
            not in inspect.signature(
                state_local_quadratic_risk
            ).parameters
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def analyze_formal(
    stage: str, data: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    scores = score_rows(data["response"], data["directions"])
    rankings = make_rankings(scores)
    sequence = sequence_rankings(rankings)
    directory = data["directory"]
    atomic_frame(directory / "score_rows.parquet", scores)
    atomic_frame(directory / "ranking_rows.parquet", rankings)
    atomic_frame(
        directory / "sequence_first_rankings.parquet", sequence
    )
    sequence.to_csv(
        directory / "sequence_first_rankings.csv", index=False
    )
    decision = decision_gate(
        sequence,
        "midpoint_k2",
        config["formal_gates"]["decision"],
    )
    integrity_gate = integrity(data, config)
    if stage == "evaluation":
        inherited = json.loads(
            (
                ROOT
                / "experiments/p2_recovery/"
                "r3_path_integrated_readout/results/evaluation/"
                "analysis_summary.json"
            ).read_text(encoding="utf-8")
        )["reduced"]
        reduced_checks = {
            "directional_derivative_cost": 2
            <= int(
                config["formal_gates"]["reduced"][
                    "maximum_directional_derivatives"
                ]
            ),
            "forward_probe_cost": 4
            <= int(
                config["formal_gates"]["reduced"][
                    "maximum_forward_probes"
                ]
            ),
            "inherited_r3_oracle_gate": bool(inherited["passed"]),
            "inherited_gain_retention": float(
                inherited["metrics"]["gain_retention"]
            )
            >= float(
                config["formal_gates"]["reduced"][
                    "inherited_oracle_gain_retention_min"
                ]
            ),
            "inherited_oracle_gap": float(
                inherited["metrics"]["oracle_spearman_gap"]
            )
            <= float(
                config["formal_gates"]["reduced"][
                    "inherited_oracle_spearman_gap_max"
                ]
            ),
        }
        reduced = {
            "passed": all(reduced_checks.values()),
            "checks": reduced_checks,
            "inherited_r3_metrics": inherited["metrics"],
            "directional_derivative_cost": 2,
            "forward_probe_cost": 4,
        }
    else:
        reduced = {
            "passed": True,
            "not_recomputed": (
                "Frozen R3 oracle evidence and R4 formal cost "
                "evidence remain applicable."
            ),
        }
    scalar_mechanism = {
        "passed": bool(integrity_gate["passed"] and decision["passed"]),
        "claim": "scalar_decision_risk_only",
        "full_vector_closure_claimed": False,
        "integrity": integrity_gate,
    }
    return {
        "stage": stage,
        "predictor": "midpoint_k2_state_local_scalar_risk",
        "scalar_mechanism": scalar_mechanism,
        "decision": decision,
        "reduced": reduced,
        "passed": bool(
            scalar_mechanism["passed"]
            and decision["passed"]
            and reduced["passed"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["calibration", "evaluation", "replication"],
    )
    args = parser.parse_args()
    config = load_config()
    data = load_stage(args.stage)
    if args.stage == "calibration":
        result = {
            "stage": "calibration",
            "predictor_preselected": True,
            "no_method_or_threshold_tuning": True,
            "integrity": integrity(data, config),
        }
        result["passed"] = result["integrity"]["passed"]
    else:
        result = analyze_formal(args.stage, data, config)
    direction_norms = data["directions"][
        UNIT + ["candidate_id", "action_r_norm"]
    ]
    enriched = data["response"].merge(
        direction_norms,
        on=UNIT + ["candidate_id"],
        validate="many_to_one",
    )
    breakdown = response_breakdown(enriched)
    atomic_frame(
        data["directory"] / "response_breakdown.parquet",
        breakdown,
    )
    breakdown.to_csv(
        data["directory"] / "response_breakdown.csv", index=False
    )
    result["stage_metadata_sha256"] = sha256_file(
        data["directory"] / "stage_metadata.json"
    )
    result["row_counts"] = {
        "directions": len(data["directions"]),
        "response": len(data["response"]),
        "vectors": len(data["vectors"]),
    }
    atomic_json(data["directory"] / "analysis_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
