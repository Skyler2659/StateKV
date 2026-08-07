"""Fail-fast fairness and completeness validation for paper runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from kvbench.methods.policy import get_method_spec


def _hash_manifest(path: Path) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _comparison_identity(cfg: Dict[str, Any]) -> str:
    runtime = {
        key: value
        for key, value in cfg["runtime"].items()
        if key not in {"resume", "fail_on_error"}
    }
    return _canonical_hash(
        {
            "runtime": runtime,
            "model": cfg["model"],
            "benchmark": cfg["benchmark"],
            "protocol": cfg["protocol"],
            "budget": cfg["budget"],
            "generation": cfg["generation"],
        }
    )


def _expected_mandatory(decision: Dict[str, Any], cfg: Dict[str, Any]) -> set:
    universe = [int(value) for value in decision["universe_positions"]]
    sink = universe[: int(cfg["budget"]["sink_size"])]
    recent_size = int(cfg["budget"]["recent_size"])
    recent = universe[-recent_size:] if recent_size else []
    current = universe[-1:] if cfg["budget"]["protect_current"] else []
    return set(sink + recent + current)


def validate_results(root: Path) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    runs = []
    comparison_manifests: Dict[Tuple[Any, ...], Dict[str, str]] = defaultdict(dict)
    comparison_tokenizers: Dict[str, Dict[str, str]] = defaultdict(dict)
    comparison_sources: Dict[str, Dict[str, str]] = defaultdict(dict)
    for config_path in sorted(root.rglob("resolved_config.yaml")):
        run_dir = config_path.parent
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        status_path = run_dir / "status.json"
        metadata_path = run_dir / "metadata.json"
        manifest_path = run_dir / "sample_manifest.json"
        predictions_path = run_dir / "predictions.jsonl"
        environment_path = run_dir / "environment.json"
        for required in (
            status_path, metadata_path, manifest_path, predictions_path, environment_path
        ):
            if not required.exists():
                errors.append("%s: missing %s" % (run_dir, required.name))
        if any(not path.exists() for path in (status_path, manifest_path, predictions_path)):
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") != "complete":
            errors.append("%s: run state is %s" % (run_dir, status.get("state")))
        failed = [key for key, value in status.get("samples", {}).items() if value.get("state") == "failed"]
        if failed:
            errors.append("%s: failed samples %s" % (run_dir, failed[:10]))

        configured_method = cfg["method"]["name"]
        spec = get_method_spec(configured_method)
        method = spec.name
        if spec.fidelity not in {"control", "core", "project_definition"}:
            warnings.append(
                "%s: method=%s fidelity=%s (%s)"
                % (run_dir, configured_method, spec.fidelity, spec.notes)
            )
        if cfg["protocol"]["visibility"] == "query_agnostic" and spec.requires_visible_query:
            errors.append("%s: future-query signal leakage for method=%s" % (run_dir, method))
        if cfg["budget"]["scope"] == "total_kv":
            budget = int(cfg["budget"]["cache_budget"])
            for decision_path in sorted((run_dir / "decisions").glob("*.json")):
                decisions = json.loads(decision_path.read_text(encoding="utf-8"))
                for decision in decisions:
                    universe_size = len(decision["universe_positions"])
                    expected = universe_size if method == "full" else min(budget, universe_size)
                    if int(decision["effective_budget"]) != expected:
                        errors.append(
                            "%s: retained=%s expected=%d"
                            % (decision_path, decision["effective_budget"], expected)
                        )
                    if int(decision["effective_budget"]) != len(decision["selected_positions"]):
                        errors.append("%s: effective budget does not match selection" % decision_path)
                    mandatory = set(decision["mandatory_positions"])
                    if not mandatory.issubset(set(decision["selected_positions"])):
                        errors.append("%s: mandatory tokens were dropped" % decision_path)
                    if method != "full" and mandatory != _expected_mandatory(decision, cfg):
                        errors.append("%s: mandatory policy does not match config" % decision_path)

        predictions = []
        with open(predictions_path, "r", encoding="utf-8") as handle:
            predictions = [json.loads(line) for line in handle if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if len(predictions) != len(manifest):
            errors.append(
                "%s: prediction count=%d manifest count=%d"
                % (run_dir, len(predictions), len(manifest))
            )
        if {row.get("sample_id") for row in predictions} != {
            row.get("sample_id") for row in manifest
        }:
            errors.append("%s: prediction IDs do not equal manifest IDs" % run_dir)
        for row in predictions:
            row_meta = row.get("metadata", {})
            cache = row.get("cache", {})
            if row_meta.get("target_used_for_generation") is not False:
                errors.append("%s: target/generation separation is not asserted" % run_dir)
            trace = [int(value) for value in cache.get("occupancy_trace", [])]
            if (
                method != "full"
                and cfg["protocol"]["cache_mode"] == "live_bounded"
                and trace
                and max(trace) > int(cfg["budget"]["cache_budget"])
            ):
                errors.append("%s: live-bounded cache exceeded the budget" % run_dir)
            if method == "full" and row_meta.get("truncation", {}).get("truncated"):
                warnings.append(
                    "%s sample=%s: Full Cache used explicit %s truncation"
                    % (run_dir, row.get("sample_id"), cfg["benchmark"]["truncation"])
                )
            if cfg["benchmark"].get("require_official"):
                if row_meta.get("dataset_official") is not True:
                    errors.append("%s: official run contains non-official data" % run_dir)
                implementation = row_meta.get("metric_implementation")
                expected_metric = {
                    "longbench": "longbench_official_python_port",
                    "ruler": "ruler_public_string_match",
                    "scbench": "scbench_official_python_port",
                }.get(cfg["benchmark"]["name"])
                if expected_metric and implementation != expected_metric:
                    errors.append(
                        "%s: metric=%s expected=%s"
                        % (run_dir, implementation, expected_metric)
                    )
                if cfg["benchmark"]["name"] == "scbench":
                    if row_meta.get("scbench_schema") != "official_context_multi_turns":
                        errors.append("%s: SCBench did not use the official raw schema" % run_dir)
                    if row_meta.get("scbench_prompt_implementation") != (
                        "minference_create_scdq_prompt_python_port"
                    ):
                        errors.append("%s: SCBench did not use the official SCDQ prompt" % run_dir)
            if (
                cfg["benchmark"]["name"] == "scbench"
                and cfg["protocol"]["reuse_mode"] == "multi_query"
                and row_meta.get("query_count_executed") != row_meta.get("query_count")
            ):
                errors.append("%s: not every SCBench future query was executed" % run_dir)

        comparison_key = (_comparison_identity(cfg),)
        method_variant = _canonical_hash(cfg["method"])[:12]
        method_key = "%s:%s" % (method, method_variant)
        comparison_manifests[comparison_key][method_key] = _hash_manifest(manifest_path)
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        model_info = environment.get("model", {})
        tokenizer_identity = _canonical_hash(
            {
                "model_name": model_info.get("model_name"),
                "revision": model_info.get("revision"),
                "checkpoint_commit_hash": model_info.get("checkpoint_commit_hash"),
                "tokenizer_name_or_path": model_info.get("tokenizer_name_or_path"),
                "tokenizer_class": model_info.get("tokenizer_class"),
                "tokenizer_vocab_size": model_info.get("tokenizer_vocab_size"),
            }
        )
        comparison_tokenizers[comparison_key[0]][method_key] = tokenizer_identity
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_identity = metadata.get("source_tree_sha256")
        if source_identity:
            comparison_sources[comparison_key[0]][method_key] = source_identity
        runs.append(str(run_dir))

    for key, by_method in comparison_manifests.items():
        hashes = set(by_method.values())
        if len(hashes) > 1:
            errors.append("comparison group uses different sample manifests: %s %s" % (key, by_method))
    for key, by_method in comparison_tokenizers.items():
        if len(set(by_method.values())) > 1:
            errors.append(
                "comparison group uses different model/tokenizer identities: %s %s"
                % (key, by_method)
            )
    for key, by_method in comparison_sources.items():
        if len(set(by_method.values())) > 1:
            errors.append(
                "comparison group uses different source trees: %s %s"
                % (key, by_method)
            )
    if not runs:
        errors.append("no run directories found under %s" % root)
    return {
        "valid": not errors,
        "run_count": len(runs),
        "errors": errors,
        "warnings": warnings,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = validate_results(Path(args.results_root))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
