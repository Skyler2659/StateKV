#!/usr/bin/env python3
"""Render every P3 Markdown formula through Pandoc MathML."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p3_core import atomic_json, sha256_file  # noqa: E402


EXPERIMENT = ROOT / "experiments/p3_decision_validity"
OUTPUT = EXPERIMENT / "results/formula_render_audit.json"


def main() -> None:
    renderer = subprocess.run(
        ["pandoc", "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    documents = sorted(EXPERIMENT.rglob("*.md"))
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="p3-formula-audit-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        for index, source in enumerate(documents):
            rendered = temporary / f"{index:03d}.html"
            process = subprocess.run(
                [
                    "pandoc",
                    "--from=gfm+tex_math_dollars",
                    "--to=html5",
                    "--mathml",
                    "--standalone",
                    str(source),
                    "-o",
                    str(rendered),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            html = (
                rendered.read_text(encoding="utf-8")
                if rendered.exists()
                else ""
            )
            without_annotations = re.sub(
                r"<annotation\b.*?</annotation>",
                "",
                html,
                flags=re.DOTALL,
            )
            raw = re.findall(
                r"\$\$|(?<!\\)\$(?!\$)|"
                r"\\(?:widehat|Delta|delta|tau|epsilon|alpha|frac|"
                r"operatorname|mathbb|mathbf|mathcal|lVert|rVert|"
                r"begin|middle|sum|int|geq|leq|rightarrow|text)\b",
                without_annotations,
            )
            source_text = source.read_text(encoding="utf-8")
            has_math = bool(
                re.search(r"\$\$|(?<!\\)\$(?!\$)", source_text)
            )
            mathml = html.count("<math")
            warnings = len(
                re.findall(
                    r"warning", process.stderr, flags=re.IGNORECASE
                )
            )
            rows.append(
                {
                    "path": str(source.relative_to(ROOT)),
                    "sha256": sha256_file(source),
                    "has_math_input": has_math,
                    "pandoc_exit_code": process.returncode,
                    "pandoc_stderr": process.stderr,
                    "warning_count": warnings,
                    "mathml_node_count": mathml,
                    "raw_math_leftover_count": len(raw),
                    "passed": bool(
                        process.returncode == 0
                        and process.stderr == ""
                        and (not has_math or mathml > 0)
                        and not raw
                    ),
                }
            )
    result = {
        "renderer": renderer,
        "input_format": "gfm+tex_math_dollars",
        "output_math": "MathML",
        "document_count": len(rows),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "documents": rows,
        "total_warning_count": sum(
            row["warning_count"] for row in rows
        ),
        "total_raw_math_leftover_count": sum(
            row["raw_math_leftover_count"] for row in rows
        ),
        "total_mathml_node_count": sum(
            row["mathml_node_count"] for row in rows
        ),
    }
    atomic_json(OUTPUT, result)
    if not result["passed"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
