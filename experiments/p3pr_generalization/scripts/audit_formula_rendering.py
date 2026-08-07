#!/usr/bin/env python3
"""Render the sole final Markdown report to MathML and audit leftovers."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p3pr_generalization"
REPORT = ROOT / "docs/statekv/generalization.md"
OUTPUT = EXPERIMENT / "results/formula_render_audit.json"


def main() -> None:
    source = REPORT.read_text(encoding="utf-8")
    completed = subprocess.run(
        [
            "pandoc",
            str(REPORT),
            "--from",
            "markdown+tex_math_dollars+tex_math_single_backslash",
            "--to",
            "html5",
            "--mathml",
        ],
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
        "inline_dollar": len(
            re.findall(r"(?<![A-Za-z0-9])\\$[^\\n$]+\\$", html)
        ),
        "inline_backslash_open": len(re.findall(r"\\\(", html)),
        "inline_backslash_close": len(re.findall(r"\\\)", html)),
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
    result = {
        "schema_version": 1,
        "report": str(REPORT.relative_to(ROOT)),
        "renderer": "pandoc --to html5 --mathml",
        "pandoc_version": subprocess.run(
            ["pandoc", "--version"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()[0],
        "return_code": completed.returncode,
        "warning_count": len(warnings),
        "warnings": warnings,
        "source_math_delimiter_count": source_math_delimiters,
        "mathml_node_count": mathml_nodes,
        "raw_math_leftover_count": raw_count,
        "raw_math_leftovers": raw_patterns,
    }
    result["passed"] = bool(
        completed.returncode == 0
        and not warnings
        and raw_count == 0
        and source_math_delimiters > 0
        and mathml_nodes > 0
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
