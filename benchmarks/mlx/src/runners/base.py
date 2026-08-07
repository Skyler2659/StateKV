"""Shared runner utilities for experiment backends."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import ExperimentConfig
from src.utils.io import save_jsonl, save_results


def text_hash(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


class BaseRunner:
    """Common result-directory and metadata helpers."""

    backend_name = "base"

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.run_provenance: Dict[str, Any] = {}

    def make_run_dir(self) -> Path:
        model_slug = self.model_slug()
        bench_slug = self.cfg.benchmark.name.lower()
        base = Path(self.cfg.output_dir) / model_slug / bench_slug
        run_id = self.cfg.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base / run_id
        if out_dir.exists() and not self.cfg.overwrite:
            out_dir = base / f"{run_id}_{datetime.now().strftime('%f')}"
        for child in (
            "samples",
            "selected_tokens",
            "scores",
            "artifacts",
            "analysis",
            "figures",
            "logs",
        ):
            (out_dir / child).mkdir(parents=True, exist_ok=True)
        self.cfg.run_id = out_dir.name
        return out_dir

    def model_slug(self) -> str:
        name = self.cfg.model.name.split("/")[-1].lower()
        name = (
            name.replace(".", "")
            .replace("-", "_")
            .replace("instruct", "inst")
        )
        quant = (
            f"{self.cfg.model.quant_bits}bit"
            if self.cfg.model.quant_bits
            else str(self.cfg.model.dtype).lower()
        )
        if quant and quant in name:
            return f"{self.backend_name}_{name}"
        return f"{self.backend_name}_{name}_{quant}"

    def save_run_metadata(
        self,
        out_dir: Path,
        model_info: Dict[str, Any],
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.cfg.to_yaml(out_dir / "config.yaml")
        save_results(model_info, out_dir / "model_info.json")
        env = self.environment_info()
        save_results(env, out_dir / "env.json")
        config_payload = json.dumps(
            self.cfg.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        )
        sample_manifest = self.sample_manifest(samples or [])
        provenance = {
            "manifest_schema_version": 1,
            **self.repository_provenance(),
            "config_hash": hashlib.sha256(config_payload.encode("utf-8")).hexdigest(),
            "sample_manifest_hash": hashlib.sha256(
                json.dumps(sample_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "sample_count": len(sample_manifest),
        }
        self.run_provenance = provenance
        save_results(sample_manifest, out_dir / "sample_manifest.json")
        save_results(provenance, out_dir / "run_manifest.json")

    def environment_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "git_commit": self.git_commit(),
            "backend": self.backend_name,
        }
        for mod_name in ("mlx", "mlx_lm", "torch", "transformers", "datasets", "numpy"):
            try:
                mod = __import__(mod_name)
                info[f"{mod_name}_version"] = getattr(mod, "__version__", "unknown")
            except Exception as exc:
                info[f"{mod_name}_version"] = f"unavailable: {exc}"
        return info

    @staticmethod
    def git_commit() -> Optional[str]:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    @staticmethod
    def _git_output(args: List[str], *, binary: bool = False):
        try:
            return subprocess.check_output(
                ["git", *args],
                stderr=subprocess.DEVNULL,
                text=not binary,
            )
        except Exception:
            return b"" if binary else ""

    def repository_provenance(self) -> Dict[str, Any]:
        status = str(self._git_output(["status", "--porcelain=v1", "--untracked-files=all"])).splitlines()
        tracked_patch = self._git_output(["diff", "--binary", "HEAD"], binary=True)
        repo_root_text = str(self._git_output(["rev-parse", "--show-toplevel"])).strip()
        repo_root = Path(repo_root_text) if repo_root_text else Path.cwd()
        untracked_raw = self._git_output(
            ["ls-files", "-z", "--others", "--exclude-standard"], binary=True
        )
        untracked = [
            os.fsdecode(value)
            for value in bytes(untracked_raw).split(b"\0")
            if value
        ]
        untracked_hashes: Dict[str, Optional[str]] = {}
        for relative in sorted(untracked):
            path = repo_root / relative
            try:
                if path.is_file():
                    digest = hashlib.sha256()
                    with open(path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    untracked_hashes[relative] = digest.hexdigest()
                else:
                    untracked_hashes[relative] = None
            except OSError:
                untracked_hashes[relative] = None
        patch_hash = hashlib.sha256(tracked_patch).hexdigest()
        state_payload = json.dumps(
            {
                "commit": self.git_commit(),
                "patch_hash": patch_hash,
                "untracked_hashes": untracked_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "git_commit": self.git_commit(),
            "git_dirty": bool(status),
            "git_status": status,
            "tracked_patch_sha256": patch_hash,
            "untracked_file_sha256": untracked_hashes,
            "repository_state_id": hashlib.sha256(state_payload.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def sample_manifest(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        manifest = []
        for index, sample in enumerate(samples):
            metadata = sample.get("metadata", {}) or {}
            prompt = sample.get("prompt") or sample.get("full_text") or ""
            manifest.append(
                {
                    "sample_id": index,
                    "official_dataset_index": metadata.get("official_dataset_index"),
                    "task": metadata.get("task"),
                    "prompt_hash": text_hash(str(prompt)),
                    "ground_truth_hash": text_hash(str(sample.get("ground_truth") or "")),
                }
            )
        return manifest

    def save_result_bundle(self, results: List[Dict[str, Any]], out_dir: Path) -> None:
        save_results(results, out_dir / "results.json")
        save_jsonl(results, out_dir / "results.jsonl")
        save_jsonl(results, out_dir / "samples.jsonl")
        self.save_metrics_csv(results, out_dir / "metrics.csv")
        save_results(self.summary(results), out_dir / "summary.json")
        save_results(self.profiling_summary(results), out_dir / "profiling_summary.json")
        self.write_run_log(results, out_dir / "logs" / "run.log")

    def save_metrics_csv(self, results: List[Dict[str, Any]], path: Path) -> None:
        metric_keys = [
            "experiment_name",
            "run_id",
            "sample_id",
            "official_dataset_index",
            "code_commit_hash",
            "repository_state_id",
            "config_hash",
            "model_name",
            "model_family",
            "backend",
            "quant_bits",
            "method",
            "method_family",
            "budget",
            "benchmark",
            "context_length",
            "needle_depth",
            "ppl",
            "mean_nll",
            "n_eval_tokens",
            "contains_ground_truth",
            "exact_match",
            "answer_f1",
            "correct",
            "official_score",
            "official_correct",
            "official_metric_name",
            "official_metric_implementation",
            "dataset_official",
            "primary_metric",
            "primary_score",
            "evidence_recall",
            "evidence_precision",
            "max_kv_len",
            "final_kv_len",
            "avg_kv_len",
            "peak_token_head_slots",
            "final_token_head_slots",
            "avg_token_head_slots",
            "slot_time_integral",
            "peak_memory_bytes",
            "active_memory_bytes",
            "total_time_s",
            "generation_time_s",
            "prefill_time_s",
            "decode_time_s",
            "eviction_time_s",
            "decode_loop_time_s",
            "score_time_s",
            "topk_time_s",
            "cache_rebuild_time_s",
            "generated_token_count",
            "tokens_per_second",
            "end_to_end_decode_tokens_per_second",
            "eviction_overhead_fraction",
            "avg_ms_per_token",
            "ppl_time_s",
            "ppl_prefill_time_s",
            "ppl_decode_time_s",
            "ppl_eviction_time_s",
            "ppl_score_time_s",
            "ppl_end_to_end_decode_tokens_per_second",
            "ppl_eviction_overhead_fraction",
            "ppl_score_refit_count",
            "update_policy",
            "update_interval",
            "score_update_count",
            "eviction_count",
            "skipped",
            "skipped_reason",
            "unsupported_reason",
            "oracle",
            "artifact_schema_version",
            "mechanism_artifacts_alignment_safe",
            "sanity_check_failed",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_keys)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in metric_keys})

    def summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [r for r in results if "error" not in r and not r.get("skipped")]
        errors = [r for r in results if "error" in r]
        skipped = [r for r in results if r.get("skipped")]
        by_method: Dict[str, Dict[str, Any]] = {}
        for method in sorted({r.get("method") for r in ok}):
            rows = [r for r in ok if r.get("method") == method]
            by_method[str(method)] = {
                "n": len(rows),
                "avg_ppl": safe_mean(r.get("ppl") for r in rows),
                "avg_official_score": safe_mean(r.get("official_score") for r in rows),
                "avg_primary_score": safe_mean(r.get("primary_score") for r in rows),
                "official_accuracy": safe_mean(
                    1.0 if r.get("official_correct") else 0.0
                    for r in rows
                    if r.get("official_correct") is not None
                ),
                "accuracy": safe_mean(1.0 if r.get("correct") else 0.0 for r in rows),
                "contains_ground_truth_rate": safe_mean(
                    1.0 if r.get("contains_ground_truth") else 0.0 for r in rows
                ),
                "avg_evidence_recall": safe_mean(
                    r.get("evidence_recall") for r in rows
                ),
                "avg_tokens_per_second": safe_mean(
                    r.get("tokens_per_second") for r in rows
                ),
                "avg_end_to_end_decode_tokens_per_second": safe_mean(
                    r.get("end_to_end_decode_tokens_per_second") for r in rows
                ),
                "avg_eviction_overhead_fraction": safe_mean(
                    r.get("eviction_overhead_fraction") for r in rows
                ),
                "avg_final_kv_len": safe_mean(r.get("final_kv_len") for r in rows),
            }
        return {
            "num_results": len(results),
            "num_errors": len(errors),
            "num_skipped": len(skipped),
            "backend": self.backend_name,
            "model": self.cfg.model.name,
            "benchmark": self.cfg.benchmark.name,
            "by_method": by_method,
        }

    def profiling_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ok = [r for r in results if "error" not in r and not r.get("skipped")]
        keys = [
            "total_time_s",
            "generation_time_s",
            "prefill_time_s",
            "decode_time_s",
            "eviction_time_s",
            "decode_loop_time_s",
            "score_time_s",
            "topk_time_s",
            "cache_rebuild_time_s",
            "tokens_per_second",
            "end_to_end_decode_tokens_per_second",
            "eviction_overhead_fraction",
            "avg_ms_per_token",
            "ppl_time_s",
            "ppl_prefill_time_s",
            "ppl_decode_time_s",
            "ppl_eviction_time_s",
            "ppl_score_time_s",
            "ppl_end_to_end_decode_tokens_per_second",
            "ppl_eviction_overhead_fraction",
            "score_update_count",
            "eviction_count",
        ]
        by_method: Dict[str, Dict[str, float]] = {}
        for method in sorted({r.get("method") for r in ok}):
            rows = [r for r in ok if r.get("method") == method]
            by_method[str(method)] = {k: safe_mean(r.get(k) for r in rows) for k in keys}
        return {"backend": self.backend_name, "by_method": by_method}

    def write_run_log(self, results: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.summary(results)
        lines = [
            f"backend={self.backend_name}",
            f"model={self.cfg.model.name}",
            f"benchmark={self.cfg.benchmark.name}",
            f"run_id={self.cfg.run_id}",
            f"num_results={summary['num_results']}",
            f"num_errors={summary['num_errors']}",
        ]
        for method, values in summary.get("by_method", {}).items():
            lines.append(f"{method}: {values}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def write_json(data: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
