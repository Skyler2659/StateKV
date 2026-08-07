#!/usr/bin/env python3
"""Render every P3PR Markdown with Pandoc MathML and audit leftovers."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3_physical_recovery"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3pr_core import atomic_json  # noqa: E402


def main() -> None:
    documents = sorted(EXPERIMENT.glob("*.md")) + [
        ROOT / "docs/statekv/physical_same_step.md",
        ROOT / "experiments/p3_physical_recovery/docs/detailed_report.md",
    ]
    rows = []
    for path in documents:
        source = path.read_text(encoding="utf-8")
        command = [
            "pandoc",
            str(path),
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash",
            "--to",
            "html5",
            "--mathml",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        html = completed.stdout
        warnings = [
            line for line in completed.stderr.splitlines() if line.strip()
        ]
        raw_patterns = {
            "double_dollar": len(re.findall(r"\$\$", html)),
            "single_backslash_open": len(re.findall(r"\\\(", html)),
            "single_backslash_close": len(re.findall(r"\\\)", html)),
            "display_backslash_open": len(re.findall(r"\\\[", html)),
            "display_backslash_close": len(re.findall(r"\\\]", html)),
        }
        source_math_delimiters = (
            len(re.findall(r"\$\$", source))
            + len(re.findall(r"\\\(", source))
            + len(re.findall(r"\\\[", source))
        )
        mathml_nodes = len(re.findall(r"<math(?:\s|>)", html))
        raw_count = sum(raw_patterns.values())
        row = {
            "path": str(path.relative_to(ROOT)),
            "return_code": completed.returncode,
            "warning_count": len(warnings),
            "warnings": warnings,
            "mathml_node_count": mathml_nodes,
            "source_math_delimiter_count": source_math_delimiters,
            "raw_math_leftover_count": raw_count,
            "raw_math_leftovers": raw_patterns,
            "passed": bool(
                completed.returncode == 0
                and not warnings
                and raw_count == 0
                and (
                    source_math_delimiters == 0
                    or mathml_nodes > 0
                )
            ),
        }
        rows.append(row)
    result = {
        "schema_version": 1,
        "renderer": "pandoc --to html5 --mathml",
        "pandoc_version": subprocess.run(
            ["pandoc", "--version"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()[0],
        "document_count": len(rows),
        "mathml_node_count": sum(
            row["mathml_node_count"] for row in rows
        ),
        "warning_count": sum(row["warning_count"] for row in rows),
        "raw_math_leftover_count": sum(
            row["raw_math_leftover_count"] for row in rows
        ),
        "rows": rows,
    }
    result["passed"] = bool(rows) and all(row["passed"] for row in rows)
    atomic_json(EXPERIMENT / "results/formula_render_audit.json", result)
    if not result["passed"]:
        raise SystemExit(
            "P3PR formula rendering audit failed: "
            + str(
                [
                    row["path"]
                    for row in rows
                    if not row["passed"]
                ]
            )
        )
    print(
        {
            key: result[key]
            for key in (
                "passed",
                "document_count",
                "mathml_node_count",
                "warning_count",
                "raw_math_leftover_count",
            )
        }
    )


if __name__ == "__main__":
    main()
