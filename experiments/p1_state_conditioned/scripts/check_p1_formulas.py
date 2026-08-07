#!/usr/bin/env python3
"""Render every P1 Markdown formula to MathML and reject raw leftovers."""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments/p1_state_conditioned/results"
DOCUMENTS = [
    "experiments/p1_state_conditioned/docs/code_audit.md",
    "experiments/p1_state_conditioned/docs/experiment_plan.md",
    "experiments/p1_state_conditioned/docs/calibration.md",
    "experiments/p1_state_conditioned/docs/results.md",
    "experiments/p1_state_conditioned/docs/failure_analysis.md",
]


def sha256(path: Path) -> str:
    import hashlib

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
    with tempfile.TemporaryDirectory(prefix="p1-formula-render-") as temp:
        temporary = Path(temp)
        for relative in DOCUMENTS:
            source = ROOT / relative
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
            html = rendered.read_text(encoding="utf-8")
            without_annotations = re.sub(
                r"<annotation\b.*?</annotation>",
                "",
                html,
                flags=re.DOTALL,
            )
            raw_patterns = re.findall(
                r"\$\$|\\(?:widehat|delta|frac|operatorname|"
                r"mathcal|lVert|begin)\b",
                without_annotations,
            )
            row = {
                "path": relative,
                "sha256": sha256(source),
                "pandoc_exit_code": process.returncode,
                "pandoc_stderr": process.stderr,
                "mathml_node_count": html.count("<math"),
                "raw_math_leftover_count": len(raw_patterns),
                "passed": (
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
        "total_mathml_node_count": sum(
            row["mathml_node_count"] for row in rows
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "formula_render_audit.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
