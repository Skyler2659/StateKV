#!/usr/bin/env python3
"""Render every P2 Markdown formula to MathML and reject leftovers."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments/p2_state_local_risk/results"
DOCUMENTS = [
    ROOT / "experiments/p2_state_local_risk/docs/code_audit.md",
    ROOT / "experiments/p2_state_local_risk/docs/experiment_plan.md",
    ROOT / "experiments/p2_state_local_risk/docs/calibration.md",
    ROOT / "experiments/p2_state_local_risk/docs/results.md",
    ROOT / "experiments/p2_state_local_risk/docs/failure_analysis.md",
    ROOT / "experiments/p2_state_local_risk/README.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    pandoc = subprocess.run(
        ["pandoc", "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="p2-formula-render-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        for source in DOCUMENTS:
            rendered = temporary / f"{source.stem}.html"
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
            raw_patterns = re.findall(
                r"\$\$|(?<!\\)\$(?!\$)|"
                r"\\(?:widehat|delta|varepsilon|frac|operatorname|"
                r"mathcal|lVert|rVert|begin|middle|top|qquad)\b",
                without_annotations,
            )
            relative = str(source.relative_to(ROOT))
            row = {
                "path": relative,
                "sha256": sha256(source),
                "pandoc_exit_code": process.returncode,
                "pandoc_stderr": process.stderr,
                "warning_count": len(
                    re.findall(
                        r"warning",
                        process.stderr,
                        flags=re.IGNORECASE,
                    )
                ),
                "mathml_node_count": html.count("<math"),
                "raw_math_leftover_count": len(raw_patterns),
                "passed": bool(
                    process.returncode == 0
                    and process.stderr == ""
                    and html.count("<math") > 0
                    and not raw_patterns
                ),
            }
            rows.append(row)
    result = {
        "renderer": pandoc,
        "input_format": "gfm+tex_math_dollars",
        "output_math": "MathML",
        "passed": all(row["passed"] for row in rows),
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "formula_render_audit.json"
    destination.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
