"""Deterministic parser for RULER variable-tracking dependency paths."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from src.benchmarks.niah import _token_span_from_char_span


ASSIGNMENT_RE = re.compile(
    r"\b(?:(?:VAR\s+(?P<lhs>[A-Z][A-Z0-9_]*))|(?P<legacy_lhs>VAR_\d+))"
    r"\s*=\s*(?:(?:VAR\s+)?(?P<var>[A-Z][A-Z0-9_]*)|"
    r"(?P<literal>-?\d+(?:\.\d+)?|[a-z][a-z0-9_-]*))\b"
)
TARGET_RE = re.compile(
    r"(?:assigned\s+the\s+value|value(?:\s+of)?)[^\d-]*(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_vt_prompt(
    prompt: str,
    tokenizer: Any,
    *,
    target_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Parse assignments and return target-chain/distractor token spans.

    The official RULER format uses ``VAR A = VAR B`` and ``VAR B = 123``.
    A dependency edge is the complete assignment statement. A path is complete
    only when following every referenced variable terminates at the queried
    literal without a missing edge or cycle.
    """
    assignments: List[Dict[str, Any]] = []
    by_lhs: Dict[str, Dict[str, Any]] = {}
    for edge_id, match in enumerate(ASSIGNMENT_RE.finditer(prompt)):
        rhs_var = match.group("var")
        literal = match.group("literal")
        start, end, token_positions = _token_span_from_char_span(
            tokenizer, prompt, match.start(), match.end()
        )
        lhs = match.group("lhs") or match.group("legacy_lhs")
        lhs_group = "lhs" if match.group("lhs") else "legacy_lhs"
        lhs_char_start = match.start(lhs_group)
        lhs_char_end = match.end(lhs_group)
        _, _, node_positions = _token_span_from_char_span(
            tokenizer, prompt, lhs_char_start, lhs_char_end
        )
        edge = {
            "edge_id": edge_id,
            "lhs": lhs,
            "rhs": rhs_var or literal,
            "rhs_is_variable": rhs_var is not None,
            "char_start": match.start(),
            "char_end": match.end(),
            "token_start": start,
            "token_end": end,
            "token_positions": token_positions,
            "node_positions": node_positions,
        }
        assignments.append(edge)
        by_lhs[edge["lhs"]] = edge

    if target_value is None:
        matches = list(TARGET_RE.finditer(prompt))
        target_value = matches[-1].group(1) if matches else None
    target_value = None if target_value is None else str(target_value).strip()

    memo: Dict[str, Optional[str]] = {}

    def resolve(name: str, visiting: Optional[Set[str]] = None) -> Optional[str]:
        if name in memo:
            return memo[name]
        visiting = set(visiting or ())
        if name in visiting:
            memo[name] = None
            return None
        visiting.add(name)
        edge = by_lhs.get(name)
        if edge is None:
            memo[name] = None
        elif not edge["rhs_is_variable"]:
            memo[name] = str(edge["rhs"])
        else:
            memo[name] = resolve(str(edge["rhs"]), visiting)
        return memo[name]

    target_variables = sorted(
        name for name in by_lhs if target_value is not None and resolve(name) == target_value
    )
    target_edges: Set[int] = set()
    path_by_variable: Dict[str, List[int]] = {}
    complete_by_variable: Dict[str, bool] = {}
    for name in target_variables:
        path: List[int] = []
        current = name
        seen: Set[str] = set()
        complete = False
        while current not in seen:
            seen.add(current)
            edge = by_lhs.get(current)
            if edge is None:
                break
            path.append(int(edge["edge_id"]))
            target_edges.add(int(edge["edge_id"]))
            if not edge["rhs_is_variable"]:
                complete = str(edge["rhs"]) == target_value
                break
            current = str(edge["rhs"])
        path_by_variable[name] = path
        complete_by_variable[name] = complete

    evidence_positions = sorted(
        {
            int(position)
            for edge in assignments
            if int(edge["edge_id"]) in target_edges
            for position in edge["token_positions"]
        }
    )
    node_positions = sorted(
        {
            int(position)
            for edge in assignments
            if int(edge["edge_id"]) in target_edges
            for position in edge["node_positions"]
        }
    )
    distractor_positions = sorted(
        {
            int(position)
            for edge in assignments
            if int(edge["edge_id"]) not in target_edges
            for position in edge["token_positions"]
        }
    )
    return {
        "parser_version": "ruler_vt_v1",
        "target_value": target_value,
        "assignments": assignments,
        "target_variables": target_variables,
        "target_edge_ids": sorted(target_edges),
        "path_edge_ids_by_variable": path_by_variable,
        "path_complete_by_variable": complete_by_variable,
        "parser_complete": bool(target_variables) and all(complete_by_variable.values()),
        "evidence_positions": evidence_positions,
        "assignment_node_positions": node_positions,
        "distractor_positions": distractor_positions,
    }
