#!/usr/bin/env python3
"""Main benchmark runner — evaluates eviction methods on long-context tasks.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark/niah.yaml
    python scripts/run_benchmark.py --config configs/benchmark/niah.yaml --method l1_mixed
    python scripts/run_benchmark.py --model Qwen/Qwen2.5-1.5B --benchmark niah --budget 128
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

import torch
from torch.nn import CrossEntropyLoss

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig, ModelConfig, EvictionConfig, BenchmarkConfig
from src.eviction.registry import create_eviction, PAPER_BASELINES, list_methods
from src.eviction.kv_utils import get_kv_seq_len, to_legacy_cache
from src.models import load_model_and_tokenizer
from src.profiling.throughput import ThroughputTracker
from src.profiling.memory import MemoryTracker
from src.utils.seed import set_global_seed
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.io import save_jsonl, save_results, save_scores, save_selected_tokens


logger = get_logger("run_benchmark")


def eviction_kwargs_from_config(cfg: EvictionConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    for key in ("method", "methods", "cache_size", "cache_budget_ratio", "seed"):
        data.pop(key, None)
    return data


def make_run_dir(cfg: ExperimentConfig) -> Path:
    base = Path(cfg.output_dir) / cfg.experiment_name
    run_id = cfg.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / run_id
    if out_dir.exists() and not cfg.overwrite:
        suffix = datetime.now().strftime("%f")
        out_dir = base / f"{run_id}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_id = out_dir.name
    return out_dir


def text_hash(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Core decode loop ────────────────────────────────────────────────────

@torch.no_grad()
def run_decode_eval(
    model,
    input_ids: torch.Tensor,
    eviction,
    label: str,
    k_seq_dim: int,
    max_steps: int,
    eval_target_positions=None,
    progress_every: int = 100,
    tracker: ThroughputTracker | None = None,
    memory_tracker: MemoryTracker | None = None,
    output_attentions: bool = False,
) -> dict:
    """Token-by-token decode with eviction, returning PPL + throughput metrics."""
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    nlls: list = []
    step_times: list = []
    kv_lens: list = []

    total_steps = min(input_ids.size(1) - 1, max_steps)
    eval_set = set(eval_target_positions) if eval_target_positions else None
    wall_start = time.perf_counter()

    if eviction is not None:
        eviction.reset()

    if tracker:
        tracker.reset()

    for idx in range(total_steps):
        token = input_ids[:, idx : idx + 1]
        target = input_ids[:, idx + 1 : idx + 2].to(token.device).view(-1)

        # Pre-eviction: make room for incoming token
        if eviction is not None:
            t_evict_start = time.perf_counter()
            past_key_values = eviction.evict_for_space(past_key_values, num_coming=1)
            if tracker:
                tracker.record_phase("eviction", time.perf_counter() - t_evict_start)

        # Forward pass
        if tracker:
            tracker.begin_step()
        outputs = model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
        )
        if tracker:
            step_elapsed = tracker.end_step()
        else:
            step_elapsed = 0.0

        if output_attentions and eviction is not None:
            attentions = getattr(outputs, "attentions", None)
            if attentions is not None:
                for layer_idx, attention_weights in enumerate(attentions):
                    eviction.update_attention(layer_idx, attention_weights)

        # Compute loss
        nll = loss_fn(
            outputs.logits[:, -1, :].view(-1, model.config.vocab_size), target
        )
        if eval_set is None or (idx + 1) in eval_set:
            nlls.append(nll)

        # Post-forward eviction
        past_key_values = outputs.past_key_values
        if eviction is not None:
            t_evict_start = time.perf_counter()
            past_key_values = eviction(past_key_values)
            if tracker:
                tracker.record_phase("eviction", time.perf_counter() - t_evict_start)

        # Track KV length
        pkv_legacy, _ = to_legacy_cache(past_key_values)
        if pkv_legacy is not None:
            kl = get_kv_seq_len(pkv_legacy[0][0], k_seq_dim)
            kv_lens.append(kl)
            if tracker:
                tracker.record_phase("kv_len", kl)

        # Progress
        step_id = idx + 1
        if progress_every > 0 and (step_id % progress_every == 0 or step_id == total_steps):
            elapsed = time.perf_counter() - wall_start
            tok_s = step_id / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{label}] step={step_id}/{total_steps} "
                f"kv={kv_lens[-1] if kv_lens else 0} "
                f"tok/s={tok_s:.2f} elapsed={elapsed:.1f}s"
            )

    # Memory snapshot
    peak_mb = 0.0
    if memory_tracker:
        peak_mb = memory_tracker.record_peak()
        if tracker:
            tracker.record_memory(peak_mb)

    if not nlls:
        raise ValueError(
            f"No target tokens evaluated. eval_target_positions={eval_target_positions}, "
            f"max_steps={max_steps}"
        )

    mean_nll = torch.stack(nlls).mean().item()
    total_s = time.perf_counter() - wall_start
    stats = tracker.get_stats() if tracker else None

    result = {
        "label": label,
        "steps": total_steps,
        "ppl": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tok_per_s": total_steps / total_s if total_s > 0 else float("inf"),
        "avg_ms_per_tok": (total_s / total_steps) * 1000.0,
        "max_kv_len": max(kv_lens) if kv_lens else 0,
        "final_kv_len": kv_lens[-1] if kv_lens else 0,
        "peak_memory_mb": peak_mb,
        "total_time_s": total_s,
    }

    if stats:
        result["throughput"] = stats.to_dict()

    # Collect per-layer diagnostics
    if eviction is not None and hasattr(eviction, "last_scores"):
        result["has_scores"] = len(eviction.last_scores) > 0
    if eviction is not None and hasattr(eviction, "last_selected"):
        result["has_selected"] = len(eviction.last_selected) > 0

    return result


def method_needs_attentions(method_name: str) -> bool:
    key = method_name.lower()
    return key in {
        "attention",
        "h2o",
        "snapkv",
        "pyramidkv",
        "hybrid",
        "attn_l1",
        "attn_l2",
        "attn_recency",
        "attention+l1",
    }


def summarize_score_stats(scores: Dict[int, torch.Tensor], topk: int = 5) -> Dict[str, Any]:
    if not scores:
        return {}
    flat = torch.cat([s.flatten().float() for s in scores.values() if s.numel() > 0])
    if flat.numel() == 0:
        return {}
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {"all_non_finite": True}
    k = min(topk, finite.numel())
    return {
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        "top_values": [float(x) for x in torch.topk(finite, k).values.tolist()],
    }


# ── Benchmark loading ───────────────────────────────────────────────────

def _construct_from_signature(cls: Type, values: Dict[str, Any]):
    """Construct *cls* using only keyword arguments accepted by its __init__."""
    sig = inspect.signature(cls.__init__)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    kwargs = {k: v for k, v in values.items() if k in allowed and v is not None}
    ignored = sorted(k for k, v in values.items() if k not in allowed and v is not None)
    if ignored:
        logger.debug("%s ignored benchmark fields: %s", cls.__name__, ignored)
    return cls(**kwargs)


def instantiate_benchmark(cfg: ExperimentConfig):
    """Instantiate the configured benchmark without loading samples."""
    bench_cfg = cfg.benchmark
    bench_name = bench_cfg.name.lower()
    seed = cfg.seed
    max_samples = bench_cfg.num_samples

    if bench_name == "niah":
        from src.benchmarks.niah import NIAHBenchmark

        values = {
            "depths": bench_cfg.depths or [bench_cfg.needle_depth],
            "max_words": bench_cfg.max_words,
            "needles_per_depth": bench_cfg.needles_per_depth,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
            "context_length": bench_cfg.context_length,
        }
        return _construct_from_signature(NIAHBenchmark, values)

    if bench_name in ("multi_needle", "multi_niah"):
        from src.benchmarks.niah import MultiNeedleNIAH

        values = {
            "n_needles": bench_cfg.num_needles,
            "max_words": bench_cfg.max_words,
            "n_samples": bench_cfg.n_samples,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
        }
        return _construct_from_signature(MultiNeedleNIAH, values)

    if bench_name == "variable_depth":
        from src.benchmarks.niah import VariableDepthNIAH

        values = {
            "n_depths": bench_cfg.n_depths,
            "max_words": bench_cfg.max_words,
            "seed": seed,
            "max_samples": max_samples,
            "use_synthetic_haystack": bench_cfg.use_synthetic_haystack,
            "haystack_repeats": bench_cfg.haystack_repeats,
        }
        return _construct_from_signature(VariableDepthNIAH, values)

    if bench_name == "ruler":
        from src.benchmarks.ruler import RULERBenchmark

        values = {
            "tasks": bench_cfg.tasks or [bench_cfg.ruler_task],
            "n_samples_per_task": bench_cfg.n_samples_per_task,
            "seq_words": bench_cfg.seq_words,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(RULERBenchmark, values)

    if bench_name == "longbench":
        from src.benchmarks.longbench import LongBenchWrapper

        values = {
            "tasks": bench_cfg.tasks or [bench_cfg.longbench_task],
            "max_words": bench_cfg.max_words,
            "n_samples_per_task": bench_cfg.n_samples_per_task,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(LongBenchWrapper, values)

    if bench_name in ("hotpotqa", "multihop"):
        from src.benchmarks.multihop import MultiHopQA

        values = {
            "dataset": bench_cfg.dataset if bench_name == "multihop" else "hotpotqa",
            "split": bench_cfg.split,
            "max_words": bench_cfg.max_words,
            "n_samples": bench_cfg.n_samples,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(MultiHopQA, values)

    if bench_name == "reasoning":
        from src.benchmarks.reasoning import ReasoningWithDistractors

        values = {
            "n_samples": bench_cfg.n_samples,
            "n_distractors": bench_cfg.n_distractors,
            "seed": seed,
            "max_samples": max_samples,
        }
        return _construct_from_signature(ReasoningWithDistractors, values)

    raise ValueError(f"Unknown benchmark: {bench_name}")


def load_benchmark(cfg: ExperimentConfig, tokenizer):
    """Load benchmark samples based on config."""
    bench = instantiate_benchmark(cfg)
    samples = bench.load_samples(tokenizer, cfg.benchmark.num_samples)
    return bench, samples


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="L1 KV Cache Benchmark Runner")
    p.add_argument("--config", type=str, default=None, help="YAML config file")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)
    p.add_argument("--benchmark", type=str, default=None,
                    choices=["niah", "multi_needle", "variable_depth", "ruler",
                             "longbench", "hotpotqa", "reasoning"])
    p.add_argument("--method", type=str, default=None, nargs="+",
                    help="Eviction method(s). Default: paper baselines")
    p.add_argument("--budget", type=int, default=None, nargs="+",
                    help="Cache budget(s)")
    p.add_argument("--budget_ratio", type=float, default=None, nargs="+",
                    help="Cache budget ratio(s) of context length")
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--sink_size", type=int, default=None)
    p.add_argument("--recent_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--save_scores", action="store_true")
    p.add_argument("--save_selected", action="store_true")
    p.add_argument("--skip_analysis", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()

    # Load config
    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
    else:
        cfg = ExperimentConfig()

    # Apply CLI overrides
    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.model.device = args.device
    if args.dtype:
        cfg.model.dtype = args.dtype
    if args.benchmark:
        cfg.benchmark.name = args.benchmark
    if args.context_length:
        cfg.benchmark.context_length = args.context_length
    if args.max_steps:
        cfg.benchmark.max_steps = args.max_steps
    if args.num_samples:
        cfg.benchmark.num_samples = args.num_samples
    if args.sink_size is not None:
        cfg.eviction.sink_size = args.sink_size
    if args.recent_size is not None:
        cfg.eviction.recent_size = args.recent_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.output_dir = args.output_dir

    # Set seed
    set_global_seed(cfg.seed)

    # Determine methods to run
    methods = args.method or cfg.methods or cfg.eviction.methods or [cfg.eviction.method]
    if methods == ["paper"]:
        methods = PAPER_BASELINES

    # Determine budgets
    budgets = args.budget or cfg.cache_budgets or [cfg.eviction.cache_size]
    budget_ratios = args.budget_ratio or cfg.cache_budget_ratios or []

    # Load model
    logger.info(f"Loading model: {cfg.model.name}")
    model, tokenizer, model_info = load_model_and_tokenizer(cfg.model)
    cfg.model.device = model_info["device"]
    k_seq_dim = model_info["k_seq_dim"]
    v_seq_dim = model_info["v_seq_dim"]
    logger.info(f"Model info: {model_info}")

    # Load benchmark
    logger.info(f"Loading benchmark: {cfg.benchmark.name}")
    bench, samples = load_benchmark(cfg, tokenizer)
    logger.info(f"Loaded {len(samples)} samples")

    # Prepare output directory
    out_dir = make_run_dir(cfg)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Save config
    cfg.to_yaml(out_dir / "config.yaml")

    # Run experiments
    all_results: list = []
    memory_tracker = MemoryTracker(cfg.model.device)

    for budget in budgets:
        for method_name in methods:
            for sample_idx, sample in enumerate(samples):
                input_ids = sample["input_ids"].to(cfg.model.device)
                answer_positions = sample.get("answer_positions")
                eval_positions = answer_positions if cfg.benchmark.eval_target_only else None

                # Compute actual cache size from ratio if specified
                actual_budget = budget
                for ratio in budget_ratios:
                    actual_budget = max(1, int(input_ids.size(1) * ratio))

                logger.info(
                    f"Running: method={method_name} budget={actual_budget} "
                    f"sample={sample_idx}/{len(samples)}"
                )

                # Create eviction method
                eviction = create_eviction(
                    method=method_name,
                    cache_size=actual_budget,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                    seed=cfg.seed,
                    **eviction_kwargs_from_config(cfg.eviction),
                ) if method_name != "full" else None

                # Tracker
                tracker = ThroughputTracker(cfg.model.device)
                memory_tracker.reset_peak()

                # Run
                try:
                    result = run_decode_eval(
                        model=model,
                        input_ids=input_ids,
                        eviction=eviction,
                        label=f"{method_name}_b{actual_budget}_s{sample_idx}",
                        k_seq_dim=k_seq_dim,
                        max_steps=cfg.benchmark.max_steps,
                        eval_target_positions=eval_positions,
                        progress_every=args.progress_every,
                        tracker=tracker,
                        memory_tracker=memory_tracker,
                        output_attentions=method_needs_attentions(method_name),
                    )
                except Exception as exc:
                    logger.error(f"Failed: {method_name} budget={actual_budget}: {exc}")
                    result = {
                        "label": f"{method_name}_b{actual_budget}_s{sample_idx}",
                        "error": str(exc),
                    }

                # Add metadata
                prompt_text = sample.get("prompt")
                result.update({
                    "sample_id": sample_idx,
                    "method": method_name,
                    "budget": actual_budget,
                    "sample_idx": sample_idx,
                    "model": cfg.model.name,
                    "benchmark": cfg.benchmark.name,
                    "context_length": input_ids.size(1),
                    "prompt": prompt_text,
                    "prompt_hash": text_hash(prompt_text),
                    "prediction": None,
                    "correct": None,
                    "metric": None if "error" in result else {"ppl": result.get("ppl")},
                    "loss": result.get("mean_nll"),
                    "latency": result.get("total_time_s"),
                    "tokens_per_second": result.get("tok_per_s"),
                    "peak_memory": result.get("peak_memory_mb"),
                    "evidence_positions": sample.get("evidence_positions"),
                    "ground_truth": sample.get("ground_truth"),
                    "metadata": sample.get("metadata", {}),
                    "score_update_count": getattr(eviction, "score_update_count", None) if eviction else None,
                })

                selected_path = None
                scores_path = None

                save_selected_flag = args.save_selected or cfg.save_selected_tokens
                save_scores_flag = args.save_scores or cfg.save_scores

                if save_scores_flag and eviction and hasattr(eviction, "last_scores"):
                    scores_path = out_dir / "scores" / f"{method_name}_b{actual_budget}_s{sample_idx}.pt"
                    save_scores(eviction.last_scores, scores_path)
                    result["scores_path"] = str(scores_path)

                if save_selected_flag and eviction and hasattr(eviction, "last_selected"):
                    selected_path = out_dir / "selected" / f"{method_name}_b{actual_budget}_s{sample_idx}.pt"
                    save_selected_tokens(eviction.last_selected, selected_path)
                    result["selected_tokens_path"] = str(selected_path)
                    result["selected_tokens"] = {
                        str(k): v.detach().cpu().tolist()
                        for k, v in eviction.last_selected.items()
                    }

                if eviction is not None and hasattr(eviction, "last_scores"):
                    stats = summarize_score_stats(eviction.last_scores)
                    if stats:
                        result["score_stats"] = stats
                        if method_needs_attentions(method_name):
                            logger.info(
                                "Score stats for %s sample=%s: min=%.4g max=%.4g "
                                "mean=%.4g std=%.4g top=%s",
                                method_name,
                                sample_idx,
                                stats.get("min", float("nan")),
                                stats.get("max", float("nan")),
                                stats.get("mean", float("nan")),
                                stats.get("std", float("nan")),
                                stats.get("top_values", []),
                            )

                all_results.append(result)
                save_results(
                    result,
                    out_dir / "samples" / f"{method_name}_b{actual_budget}_s{sample_idx}.json",
                )

    # Save all results
    save_results(all_results, out_dir / "results.json")
    save_jsonl(all_results, out_dir / "results.jsonl")

    # Print summary table
    _print_summary_table(all_results)

    # Run analysis if enabled
    if not args.skip_analysis and cfg.analysis.overlap:
        logger.info("Running post-hoc analysis...")
        try:
            from scripts.run_analysis import run_analysis
            run_analysis(all_results, cfg, out_dir)
        except Exception as exc:
            logger.warning(f"Analysis failed: {exc}")

    logger.info(f"Results saved to {out_dir}")


def _print_summary_table(results: list):
    """Print a summary table of results grouped by method."""
    print("\n" + "=" * 100)
    print(f"{'method':<20} {'budget':>8} {'ppl':>10} {'tok/s':>10} {'avg_ms':>10} {'max_kv':>8} {'mem_mb':>8}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['method']:<20} {r.get('budget', '?'):>8} {'ERROR':>10} {r['error'][:40]}")
        else:
            print(
                f"{r['method']:<20} {r['budget']:>8} {r['ppl']:>10.4f} "
                f"{r['tok_per_s']:>10.2f} {r['avg_ms_per_tok']:>10.3f} "
                f"{r['max_kv_len']:>8} {r.get('peak_memory_mb', 0):>8.1f}"
            )
    print("=" * 100)


if __name__ == "__main__":
    main()
