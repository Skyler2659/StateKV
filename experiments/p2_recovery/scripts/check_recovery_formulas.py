#!/usr/bin/env python3
"""Render every P2-Recovery Markdown formula to MathML."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments/p2_recovery"
OUTPUT = EXPERIMENT / "results/formula_render_audit.json"
ROOT_DOCUMENT_NAMES = {
    "experiments/p2_recovery/docs/r0_failure_map.md",
    "experiments/p2_recovery/docs/master_log.md",
    "experiments/p2_recovery/docs/decision_tree.md",
    "experiments/p2_recovery/docs/cumulative_results.md",
    "experiments/p2_recovery/docs/final_recommendation.md",
    "experiments/p2_recovery/docs/results_summary.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def documents() -> list[Path]:
    paths = [
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.name in ROOT_DOCUMENT_NAMES
    ]
    paths.extend(
        path
        for path in EXPERIMENT.rglob("*.md")
        if "__pycache__" not in path.parts
    )
    return sorted(set(paths))


def main() -> None:
    renderer = subprocess.run(
        ["pandoc", "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    rows = []
    with tempfile.TemporaryDirectory(
        prefix="p2-recovery-formulas-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        for index, source in enumerate(documents()):
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
            raw_patterns = re.findall(
                r"\$\$|(?<!\\)\$(?!\$)|"
                r"\\(?:widehat|Delta|delta|alpha|gamma|frac|"
                r"operatorname|mathcal|lVert|rVert|begin|middle|"
                r"sum|int|geq|leq|rightarrow|longrightarrow|text)\b",
                without_annotations,
            )
            source_text = source.read_text(encoding="utf-8")
            has_math_input = bool(
                re.search(r"\$\$|(?<!\\)\$(?!\$)", source_text)
            )
            mathml_count = html.count("<math")
            warning_count = len(
                re.findall(
                    r"warning",
                    process.stderr,
                    flags=re.IGNORECASE,
                )
            )
            row = {
                "path": str(source.relative_to(ROOT)),
                "sha256": sha256(source),
                "has_math_input": has_math_input,
                "pandoc_exit_code": process.returncode,
                "pandoc_stderr": process.stderr,
                "warning_count": warning_count,
                "mathml_node_count": mathml_count,
                "raw_math_leftover_count": len(raw_patterns),
                "passed": bool(
                    process.returncode == 0
                    and process.stderr == ""
                    and (not has_math_input or mathml_count > 0)
                    and not raw_patterns
                ),
            }
            rows.append(row)
    result = {
        "renderer": renderer,
        "input_format": "gfm+tex_math_dollars",
        "output_math": "MathML",
        "document_count": len(rows),
        "passed": bool(rows and all(row["passed"] for row in rows)),
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            result, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    for iteration in (
        "r0_failure_map",
        "r1_amplitude_trust_region",
        "r3_path_integrated_readout",
        "r4_scalar_decision_risk",
    ):
        prefix = f"experiments/p2_recovery/{iteration}/"
        iteration_rows = [
            row
            for row in rows
            if row["path"].startswith(prefix)
        ]
        iteration_result = {
            "renderer": renderer,
            "input_format": "gfm+tex_math_dollars",
            "output_math": "MathML",
            "iteration": iteration,
            "document_count": len(iteration_rows),
            "passed": bool(
                iteration_rows
                and all(row["passed"] for row in iteration_rows)
            ),
            "documents": iteration_rows,
            "total_warning_count": sum(
                row["warning_count"] for row in iteration_rows
            ),
            "total_raw_math_leftover_count": sum(
                row["raw_math_leftover_count"]
                for row in iteration_rows
            ),
            "total_mathml_node_count": sum(
                row["mathml_node_count"] for row in iteration_rows
            ),
        }
        destination = (
            EXPERIMENT
            / iteration
            / "results/formula_render_audit.json"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                iteration_result,
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
