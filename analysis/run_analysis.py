#!/usr/bin/env python3
"""Run the complete offline analysis in the required stage order."""
from __future__ import annotations

import argparse
from pathlib import Path

from build_analysis_table import build as build_tables
from generate_report import build as generate_report
from geometry_metrics import build as build_geometry
from inspect_outputs import audit
from make_figures import build as make_figures
from refresh_benefit import build as build_refresh
from stability_metrics import build as build_stability
from statistical_analysis import build as build_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    audit(input_dir, analysis_dir)
    build_tables(input_dir, analysis_dir)
    build_stability(input_dir, analysis_dir)
    build_geometry(input_dir, analysis_dir)
    build_refresh(analysis_dir)
    build_statistics(input_dir, analysis_dir, args.bootstrap_draws)
    make_figures(analysis_dir)
    generate_report(analysis_dir, input_dir)
    print(analysis_dir / "final_analysis_report.md")
    print(analysis_dir / "david_update.md")
    print(analysis_dir / "manifest.json")


if __name__ == "__main__":
    main()
