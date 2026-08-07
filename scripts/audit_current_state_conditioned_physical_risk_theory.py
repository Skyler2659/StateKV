#!/usr/bin/env python3
"""Audit Markdown math and emit the required reproducible JSON report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "statekv" / "theory.md"
DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "statekv"
    / "audits"
    / "theory_formula_audit.json"
)
DEFAULT_HTML = Path("/tmp/current_theory_audit.html")

REQUIRED_MAJOR_SYMBOLS = {
    "t": "$t$",
    "ell": "$\\ell$",
    "b": "$b$",
    "C": "$C$",
    "candidate_collection": "$\\mathcal C_t$",
    "full_boundary_state": "$x_{t,b}^{\\mathrm{full}}$",
    "historical_boundary_state": "$x_{t,b}^{\\mathrm{hist}}$",
    "historical_state": "$s_{t,b}$",
    "boundary_implicit_state": "$s_t$",
    "state_delta": "$\\delta x_{t,b}$",
    "action_injection": "$u_{t,\\ell}(C)$",
    "boundary_response": "$r_{t,b}(C)$",
    "reference_logits": "$z_0$",
    "state_logits": "$z_s$",
    "reference_probability": "$p_0",
    "state_probability": "$p_s$",
    "path_jacobian": "$J_{s+\\alpha r}$",
    "two_midpoint_summary": "$\\widehat{\\Delta z}_2(C)$",
    "state_gradient": "$g_s$",
    "state_fisher": "$F_s$",
    "scalar_risk": "$\\widehat R_s(C)$",
    "selected_candidate": "$C_t^\\star$",
    "physical_state": "$S_t^{\\mathrm{physical}}$",
    "physical_target": "$K_t^{\\mathrm{physical}}(C)$",
}

RAW_TEX_COMMANDS = (
    r"\operatorname",
    r"\operatorname{softmax}",
    r"\operatorname{Diag}",
    r"\operatorname{KL}",
    r"\arg",
    r"\frac",
    r"\left",
    r"\right",
    r"\middle",
    r"\top",
    r"\mathcal",
    r"\widehat",
    r"\begin",
    r"\end",
)

COMPLEX_TABLE_COMMANDS = (
    r"\begin",
    r"\end",
    r"\frac",
    r"\sum",
    r"\int",
    r"\operatorname",
    r"\arg",
)

ALLOWED_ENVIRONMENTS = {
    "aligned",
    "aligned*",
    "matrix",
    "pmatrix",
    "bmatrix",
    "cases",
    "split",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_code(markdown: str) -> str:
    without_fences = re.sub(
        r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$",
        "",
        markdown,
    )
    return re.sub(r"`[^`\n]*`", "", without_fences)


def unescaped_dollar_positions(text: str) -> list[int]:
    positions: list[int] = []
    for index, character in enumerate(text):
        if character != "$":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            positions.append(index)
    return positions


def extract_math(text: str) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    blocks = re.findall(r"(?s)\$\$(.+?)\$\$", text)
    block_delimiter_count = len(re.findall(r"(?<!\\)\$\$", text))
    if block_delimiter_count % 2:
        errors.append("odd number of block-math delimiters")
    without_blocks = re.sub(r"(?s)\$\$(.+?)\$\$", "", text)
    dollar_positions = unescaped_dollar_positions(without_blocks)
    if len(dollar_positions) % 2:
        errors.append("odd number of inline-math delimiters")
    inline = re.findall(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$", without_blocks)
    return blocks, inline, errors


def braces_balanced(expression: str) -> bool:
    depth = 0
    escaped = False
    for character in expression:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def source_sections(markdown: str) -> list[dict[str, Any]]:
    matches = list(
        re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$", markdown)
    )
    sections: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        title = re.sub(
            r"\s+",
            " ",
            re.sub(r"[`*_]", "", match.group(2)),
        ).strip()
        body = markdown[match.end() : end]
        cleaned = strip_code(body)
        has_formula = bool(re.search(r"(?<!\\)\$", cleaned))
        sections.append(
            {
                "level": len(match.group(1)),
                "title": title,
                "has_formula": has_formula,
            }
        )
    return sections


def rendered_sections(rendered: str) -> dict[str, int]:
    matches = list(
        re.finditer(
            r"(?is)<h([1-6])(?:\s+[^>]*)?>(.*?)</h\1>",
            rendered,
        )
    )
    output: dict[str, int] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rendered)
        title = re.sub(
            r"\s+",
            " ",
            html.unescape(re.sub(r"(?s)<[^>]+>", "", match.group(2))),
        ).strip()
        body = rendered[match.end() : end]
        output[title] = len(re.findall(r"(?i)<math(?:\s|>)", body))
    return output


def raw_residues(rendered: str) -> tuple[int, int]:
    without_math = re.sub(
        r"(?is)<math(?:\s+[^>]*)?>.*?</math>",
        "",
        rendered,
    )
    plain = html.unescape(re.sub(r"(?s)<[^>]+>", " ", without_math))
    block_residue = len(re.findall(r"(?<!\\)\$\$", plain))
    inline_residue = len(
        re.findall(r"(?<!\\)\$(?!\$)[^$\n]+?(?<!\\)\$", plain)
    )
    tex_residue = sum(plain.count(command) for command in RAW_TEX_COMMANDS)
    return block_residue + inline_residue, tex_residue


def warning_lines(stderr: str) -> list[str]:
    return [
        line
        for line in stderr.splitlines()
        if re.search(r"(?i)\bwarning\b", line)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    rendered_path = args.html.resolve()
    markdown = source.read_text(encoding="utf-8")
    cleaned = strip_code(markdown)
    blocks, inline, delimiter_errors = extract_math(cleaned)
    expressions = blocks + inline

    fullwidth_delimiters = {
        character: markdown.count(character)
        for character in ("＄", "﹩")
        if character in markdown
    }
    environments = re.findall(r"\\(?:begin|end)\{([^}]+)\}", cleaned)
    illegal_environments = sorted(
        {name for name in environments if name not in ALLOWED_ENVIRONMENTS}
    )
    unbalanced_expressions = [
        index
        for index, expression in enumerate(expressions)
        if not braces_balanced(expression)
    ]
    left_count = sum(expression.count(r"\left") for expression in expressions)
    right_count = sum(expression.count(r"\right") for expression in expressions)
    paired_left_right = left_count == right_count

    missing_symbols = [
        name
        for name, literal in REQUIRED_MAJOR_SYMBOLS.items()
        if literal not in markdown
    ]
    complex_table_lines = [
        number
        for number, line in enumerate(markdown.splitlines(), start=1)
        if line.lstrip().startswith("|")
        and (
            "$$" in line
            or any(command in line for command in COMPLEX_TABLE_COMMANDS)
        )
    ]

    rendered_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pandoc",
        "--from=gfm+tex_math_dollars",
        "--to=html5",
        "--mathml",
        str(source),
        "-o",
        str(rendered_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    rendered = (
        rendered_path.read_text(encoding="utf-8")
        if rendered_path.exists()
        else ""
    )
    warnings = warning_lines(process.stderr)
    mathml_count = len(re.findall(r"(?i)<math(?:\s|>)", rendered))
    raw_delimiter_count, raw_tex_count = raw_residues(rendered)

    rendered_by_heading = rendered_sections(rendered)
    formula_sections = [
        section for section in source_sections(markdown) if section["has_formula"]
    ]
    section_mathml = {
        section["title"]: rendered_by_heading.get(section["title"], 0)
        for section in formula_sections
    }
    sections_without_mathml = [
        title for title, count in section_mathml.items() if count <= 0
    ]

    version_process = subprocess.run(
        ["pandoc", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    pandoc_version = (
        version_process.stdout.splitlines()[0]
        if version_process.stdout
        else "unknown"
    )

    checks = {
        "pandoc_return_code_zero": process.returncode == 0,
        "pandoc_warning_count_zero": len(warnings) == 0,
        "mathml_node_count_nonzero": mathml_count > 0,
        "raw_delimiter_residue_zero": raw_delimiter_count == 0,
        "raw_tex_residue_zero": raw_tex_count == 0,
        "dollar_delimiters_paired": not delimiter_errors,
        "fullwidth_delimiters_absent": not fullwidth_delimiters,
        "illegal_latex_environments_absent": not illegal_environments,
        "formula_braces_balanced": not unbalanced_expressions,
        "left_right_delimiters_paired": paired_left_right,
        "major_symbol_definitions_complete": not missing_symbols,
        "complex_formulas_absent_from_tables": not complex_table_lines,
        "formula_sections_have_mathml": not sections_without_mathml,
    }
    passed = all(checks.values())
    payload = {
        "schema_version": 1,
        "audited_file": str(source.relative_to(ROOT)),
        "audited_file_sha256": sha256_file(source),
        "pandoc_version": pandoc_version,
        "pandoc_command": command,
        "return_code": process.returncode,
        "warning_count": len(warnings),
        "warnings": warnings,
        "mathml_node_count": mathml_count,
        "raw_delimiter_residue_count": raw_delimiter_count,
        "raw_tex_residue_count": raw_tex_count,
        "block_formula_count": len(blocks),
        "inline_formula_count": len(inline),
        "formula_section_count": len(formula_sections),
        "formula_section_mathml_counts": section_mathml,
        "delimiter_errors": delimiter_errors,
        "fullwidth_delimiters": fullwidth_delimiters,
        "illegal_latex_environments": illegal_environments,
        "unbalanced_formula_indices": unbalanced_expressions,
        "left_delimiter_count": left_count,
        "right_delimiter_count": right_count,
        "missing_major_symbol_definitions": missing_symbols,
        "complex_formula_table_lines": complex_table_lines,
        "sections_without_mathml": sections_without_mathml,
        "checks": checks,
        "passed": passed,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
