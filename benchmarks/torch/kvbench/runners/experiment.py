"""The only generation/evaluation loop used by the Torch paper path."""
from __future__ import annotations

import hashlib
import math
import os
import random
import resource
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from kvbench.analysis.diagnostics import (
    compute_decision_diagnostics,
    evidence_token_positions,
)
from kvbench.artifacts.writer import RunWriter
from kvbench.backends.huggingface import HFCacheState, HuggingFaceBackend
from kvbench.benchmarks.factory import load_benchmark
from kvbench.config import ExperimentConfig
from kvbench.evaluation.metrics import evaluate_prediction
from kvbench.methods.policy import EvictionPolicy
from kvbench.protocols.base import Protocol
from kvbench.types import (
    BenchmarkSample,
    GenerationResult,
    SampleResult,
    ScoreBundle,
    SelectionDecision,
)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _peak_cpu_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig, command: Optional[List[str]] = None):
        self.cfg = cfg
        self.protocol = Protocol(cfg.protocol)
        self.policy = EvictionPolicy(cfg.method, cfg.budget, cfg.runtime.seed)
        self.protocol.validate_method(
            self.policy.spec.requires_visible_query, self.policy.name
        )
        if cfg.protocol.visibility == "query_agnostic" and cfg.model.prompt_format != "plain":
            raise ValueError(
                "query_agnostic shared-prefix runs require model.prompt_format=plain so "
                "the future query is not inserted into a prebuilt generation template"
            )
        if cfg.protocol.reuse_mode == "multi_query" and cfg.benchmark.name != "scbench":
            raise ValueError(
                "multi_query reuse currently requires the explicit SCBench shared-prefix schema"
            )
        self.backend = HuggingFaceBackend(cfg)
        self.writer = RunWriter(cfg, command=command)

    def run(self) -> str:
        set_reproducibility(self.cfg.runtime.seed, self.cfg.runtime.deterministic)
        model_load_started = time.perf_counter()
        model_info = self.backend.load()
        model_info["model_load_s"] = time.perf_counter() - model_load_started
        model_info["warmup_steps"] = int(self.cfg.runtime.warmup_steps)
        model_info["warmup_s"] = self.backend.warmup(self.cfg.runtime.warmup_steps)
        self.writer.save_environment(model_info)
        samples = load_benchmark(self.cfg)
        manifest = [
            {
                "sample_id": sample.sample_id,
                "task": sample.task,
                "prompt_sha256": _hash_text(sample.prompt),
                "reference_sha256": _hash_text("\n".join(sample.references)),
                "shared_prefix_sha256": (
                    _hash_text(sample.shared_prefix)
                    if sample.shared_prefix is not None
                    else None
                ),
                "query_sha256": [_hash_text(query) for query in sample.queries],
                "official_dataset_index": sample.metadata.get("official_dataset_index"),
                "dataset_official": sample.metadata.get("dataset_official"),
            }
            for sample in samples
        ]
        self.writer.write_sample_manifest(manifest)

        for sample in samples:
            if self.cfg.runtime.resume and self.writer.is_complete(sample.sample_id):
                continue
            self.writer.mark_running(sample.sample_id)
            try:
                result, generation = self.run_sample(sample)
                self.writer.save_sample(
                    result,
                    generation.decisions,
                    generation.score_bundles,
                )
            except Exception as exc:
                trace = traceback.format_exc()
                self.writer.save_failure(sample.sample_id, str(exc), trace)
                if self.cfg.runtime.fail_on_error:
                    self.writer.finalize(len(samples))
                    raise
        self.writer.finalize(len(samples))
        return str(self.writer.run_dir)

    def _truncate(self, token_ids: List[int]) -> Tuple[List[int], Dict[str, Any]]:
        maximum = min(
            int(self.cfg.benchmark.context_length),
            int(self.cfg.model.max_context),
        )
        if len(token_ids) <= maximum:
            return token_ids, {"truncated": False, "original_length": len(token_ids)}
        if self.cfg.benchmark.truncation == "error":
            raise RuntimeError(
                "tokenized prompt length %d exceeds configured context length %d"
                % (len(token_ids), maximum)
            )
        left = maximum // 2
        right = maximum - left
        return token_ids[:left] + token_ids[-right:], {
            "truncated": True,
            "original_length": len(token_ids),
            "strategy": "head_tail",
        }

    def run_sample(self, sample: BenchmarkSample) -> Tuple[SampleResult, GenerationResult]:
        sample_started = time.perf_counter()
        sample_seed = int(self.cfg.runtime.seed) + int(
            hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()[:8], 16
        )
        # Policies hold random/sketch/update state.  Resetting per sample makes a
        # resumed run identical to an uninterrupted run and removes order effects.
        self.policy = EvictionPolicy(self.cfg.method, self.cfg.budget, sample_seed)
        self.backend.reset_peak_memory()
        if self.protocol.query_visible:
            prompt_ids = self.backend.encode_prompt(
                sample.prompt,
                use_chat_template=(
                    self.cfg.model.prompt_format == "chat_template"
                    and not bool(sample.metadata.get("disable_chat_template", False))
                ),
            )
            prompt_ids, truncation = self._truncate(prompt_ids)
            generation, diagnostic_events = self._generate_query_visible(
                sample, prompt_ids
            )
            evidence_positions = evidence_token_positions(
                self.backend,
                prompt_ids,
                sample.metadata.get("evidence_texts", []),
            )
        else:
            generation, diagnostic_events, prompt_ids, truncation = self._generate_query_agnostic(
                sample
            )
            evidence_positions = evidence_token_positions(
                self.backend,
                prompt_ids,
                sample.metadata.get("evidence_texts", []),
            )

        if self.cfg.generation.compute_teacher_forced_ppl and self.protocol.query_visible:
            generation_policy = self.policy
            try:
                # PPL is a separate teacher-forced replay.  It must not inherit
                # selection state from free generation.
                self.policy = EvictionPolicy(
                    self.cfg.method, self.cfg.budget, sample_seed
                )
                generation.teacher_forced_ppl = self._teacher_forced_ppl(
                    sample, prompt_ids
                )
            finally:
                self.policy = generation_policy

        predictions = generation.query_texts or [generation.text]
        if sample.shared_prefix is not None:
            if len(predictions) > len(sample.references):
                raise RuntimeError("more SCBench predictions than reference answers")
            query_evaluations = []
            query_labels = sample.metadata.get("query_labels")
            query_tasks = sample.metadata.get("query_tasks")
            for index, prediction in enumerate(predictions):
                evaluation_metadata = dict(sample.metadata)
                if query_labels is not None:
                    if index >= len(query_labels):
                        raise RuntimeError("SCBench query_labels length mismatch")
                    evaluation_metadata["query_label"] = query_labels[index]
                if query_tasks is not None:
                    if index >= len(query_tasks):
                        raise RuntimeError("SCBench query_tasks length mismatch")
                    evaluation_metadata["query_task"] = query_tasks[index]
                query_evaluation = evaluate_prediction(
                    self.cfg.benchmark.name,
                    sample.task,
                    prediction,
                    [sample.references[index]],
                    evaluation_metadata,
                )
                query_evaluations.append(
                    {
                        **query_evaluation,
                        "query_index": index,
                        "query_task": evaluation_metadata.get(
                            "query_task", sample.task
                        ),
                    }
                )
            scores = [
                float(item["score"])
                for item in query_evaluations
                if item.get("score") is not None
            ]
            correctness = [item.get("correct") for item in query_evaluations]
            evaluation = {
                "score": sum(scores) / len(scores) if scores else None,
                "metric_name": query_evaluations[0]["metric_name"],
                "correct": (
                    None
                    if any(value is None for value in correctness)
                    else all(bool(value) for value in correctness)
                ),
                "metric_implementation": query_evaluations[0].get(
                    "metric_implementation"
                ),
            }
        else:
            query_evaluations = []
            evaluation = evaluate_prediction(
                self.cfg.benchmark.name,
                sample.task,
                generation.text,
                sample.references,
                sample.metadata,
            )
        max_cache = max(generation.cache_lengths) if generation.cache_lengths else 0
        final_cache = generation.cache_lengths[-1] if generation.cache_lengths else 0
        generated_token_count = (
            sum(len(values) for values in generation.query_token_ids)
            if generation.query_token_ids
            else len(generation.token_ids)
        )
        cache_bytes_per_token = self.backend.cache_bytes_per_token()
        end_to_end_s = time.perf_counter() - sample_started
        compression_ratio = (
            float(self.cfg.budget.cache_budget) / max(1, len(prompt_ids))
            if self.policy.name != "full"
            else 1.0
        )
        diagnostics = {
            "events": diagnostic_events,
            "evidence_positions": evidence_positions,
            "teacher_forced_ppl": generation.teacher_forced_ppl,
            "attention_hook_errors": 0,
            "query_evaluations": query_evaluations,
        }
        result = SampleResult(
            sample_id=sample.sample_id,
            task=sample.task,
            prediction=generation.text,
            references=list(sample.references),
            score=evaluation["score"],
            metric_name=evaluation["metric_name"],
            correct=evaluation["correct"],
            status="complete",
            error=None,
            metadata={
                **sample.metadata,
                "benchmark": self.cfg.benchmark.name,
                "method": self.policy.name,
                "method_variant": self.policy.variant,
                "method_fidelity": self.policy.spec.fidelity,
                "protocol_visibility": self.cfg.protocol.visibility,
                "cache_mode": self.cfg.protocol.cache_mode,
                "update_policy": self.cfg.protocol.update_policy,
                "reuse_mode": self.cfg.protocol.reuse_mode,
                "query_count_executed": len(predictions),
                "context_length": int(self.cfg.benchmark.context_length),
                "actual_input_length": len(prompt_ids),
                "cache_budget": int(self.cfg.budget.cache_budget),
                "compression_ratio": compression_ratio,
                "seed": int(self.cfg.runtime.seed),
                "truncation": truncation,
                "metric_implementation": evaluation.get("metric_implementation"),
                "max_new_tokens_used": self._max_new_tokens(sample),
                "generated_token_count": generated_token_count,
                "target_used_for_generation": False,
            },
            timing={
                "prefill_s": generation.prefill_time_s,
                "scoring_s": generation.score_time_s,
                "compression_s": generation.compression_time_s,
                "decode_s": generation.decode_time_s,
                "total_model_s": generation.prefill_time_s + generation.decode_time_s,
                "end_to_end_s": end_to_end_s,
                "prefill_tokens_per_s": (
                    len(prompt_ids) / generation.prefill_time_s
                    if generation.prefill_time_s > 0
                    else None
                ),
                "decode_tokens_per_s": (
                    generated_token_count / generation.decode_time_s
                    if generation.decode_time_s > 0
                    else None
                ),
                "selection_event_count": len(generation.score_bundles),
                "score_computation_count": sum(
                    bundle.diagnostics.get("score_refresh") != "cached"
                    for bundle in generation.score_bundles
                ),
                "decode_recomputation_count": max(
                    0,
                    sum(
                        bundle.diagnostics.get("score_refresh") != "cached"
                        for bundle in generation.score_bundles
                    )
                    - 1,
                ),
                "cached_priority_selection_count": sum(
                    bundle.diagnostics.get("score_refresh") == "cached"
                    for bundle in generation.score_bundles
                ),
            },
            cache={
                "requested_budget": int(self.cfg.budget.cache_budget),
                "max_physical_length": max_cache,
                "final_physical_length": final_cache,
                "occupancy_trace": generation.cache_lengths,
                "budget_scope": self.cfg.budget.scope,
                "peak_gpu_memory_bytes": generation.peak_gpu_memory_bytes,
                "peak_cpu_rss_bytes": _peak_cpu_rss_bytes(),
                "bytes_per_token_position": cache_bytes_per_token,
                "requested_budget_bytes": (
                    int(self.cfg.budget.cache_budget) * cache_bytes_per_token
                ),
                "max_physical_cache_bytes": max_cache * cache_bytes_per_token,
                "final_physical_cache_bytes": final_cache * cache_bytes_per_token,
            },
            diagnostics=diagnostics,
            predictions=predictions,
        )
        return result, generation

    def _prefill_and_compress(
        self,
        sample: BenchmarkSample,
        prefill_ids: List[int],
        evidence_positions: List[int],
    ) -> Tuple[
        HFCacheState,
        torch.Tensor,
        float,
        float,
        float,
        List[SelectionDecision],
        List[ScoreBundle],
        List[Dict[str, Any]],
    ]:
        capture = self.policy.spec.requires_attention
        state, prefill_logits, prefill_time = self.backend.prefill(prefill_ids, capture)
        snapshot = self.backend.snapshot(state, sample.sample_id, "pre_answer", 0)
        self.backend.synchronize()
        score_started = time.perf_counter()
        decisions, bundle = self.policy.decide(snapshot, recompute=True)
        self.backend.synchronize()
        score_time = time.perf_counter() - score_started
        diagnostic = compute_decision_diagnostics(
            snapshot,
            decisions,
            bundle,
            evidence_positions,
            self.cfg.diagnostics,
        )
        compression_time = self.backend.apply_decisions(state, decisions)
        return (
            state,
            prefill_logits,
            prefill_time,
            score_time,
            compression_time,
            list(decisions),
            [bundle],
            [diagnostic],
        )

    def _generate_query_visible(
        self, sample: BenchmarkSample, prompt_ids: List[int]
    ) -> Tuple[GenerationResult, List[Dict[str, Any]]]:
        if not prompt_ids:
            raise RuntimeError("query-visible prompt must contain at least one token")
        evidence_positions = evidence_token_positions(
            self.backend,
            prompt_ids,
            sample.metadata.get("evidence_texts", []),
        )
        (
            state,
            prefill_logits,
            prefill_time,
            score_time,
            compression_time,
            decisions,
            bundles,
            diagnostic_events,
        ) = self._prefill_and_compress(sample, prompt_ids, evidence_positions)
        # The final prompt token was processed before compression, so its
        # attention is available to query-visible methods.  Its logits produce
        # the first answer token; later steps consume the compressed cache.
        generated: List[int] = [int(torch.argmax(prefill_logits, dim=-1).item())]
        cache_lengths = [self.backend.cache_length(state)]
        decode_time = 0.0
        current = generated[0]
        capture_decode = self.policy.spec.requires_attention and self.protocol.live_bounded
        max_new_tokens = self._max_new_tokens(sample)
        decode_steps = (
            range(0)
            if (
                self.cfg.generation.stop_on_eos
                and generated[0] == int(self.backend.tokenizer.eos_token_id)
            )
            else range(1, max_new_tokens)
        )
        for step in decode_steps:
            logits, elapsed = self.backend.step(state, current, capture_decode)
            decode_time += elapsed
            token = int(torch.argmax(logits, dim=-1).item())
            generated.append(token)
            if (
                self.protocol.live_bounded
                and self.policy.name != "full"
                and self.backend.cache_length(state) > int(self.cfg.budget.cache_budget)
            ):
                snapshot = self.backend.snapshot(
                    state, sample.sample_id, "decode", step
                )
                recompute = self.protocol.should_recompute(step)
                self.backend.synchronize()
                score_started = time.perf_counter()
                step_decisions, bundle = self.policy.decide(
                    snapshot, recompute=recompute
                )
                self.backend.synchronize()
                score_time += time.perf_counter() - score_started
                diagnostic_events.append(
                    compute_decision_diagnostics(
                        snapshot,
                        step_decisions,
                        bundle,
                        evidence_positions,
                        self.cfg.diagnostics,
                    )
                )
                compression_time += self.backend.apply_decisions(
                    state, step_decisions
                )
                decisions.extend(step_decisions)
                bundles.append(bundle)
            cache_lengths.append(self.backend.cache_length(state))
            current = token
            if (
                self.cfg.generation.stop_on_eos
                and token == int(self.backend.tokenizer.eos_token_id)
            ):
                break
        return GenerationResult(
            token_ids=generated,
            text=self.backend.decode(generated),
            prefill_time_s=prefill_time,
            score_time_s=score_time,
            compression_time_s=compression_time,
            decode_time_s=decode_time,
            peak_gpu_memory_bytes=self.backend.peak_memory_bytes(),
            cache_lengths=cache_lengths,
            decisions=decisions,
            score_bundles=bundles,
        ), diagnostic_events

    def _generate_query_agnostic(
        self, sample: BenchmarkSample
    ) -> Tuple[GenerationResult, List[Dict[str, Any]], List[int], Dict[str, Any]]:
        if not sample.shared_prefix or not sample.queries:
            raise RuntimeError(
                "query_agnostic samples require shared_prefix and at least one future query"
            )
        prefix_ids = self.backend.encode_text(sample.shared_prefix, add_special_tokens=True)
        prefix_ids, truncation = self._truncate(prefix_ids)
        evidence_positions = evidence_token_positions(
            self.backend,
            prefix_ids,
            sample.metadata.get("evidence_texts", []),
        )
        (
            state,
            _,
            prefill_time,
            score_time,
            compression_time,
            decisions,
            bundles,
            diagnostic_events,
        ) = self._prefill_and_compress(sample, prefix_ids, evidence_positions)
        # The base cache is compressed exactly once before any future query is
        # observed.  Every future query forks that same immutable prefix cache.
        queries = (
            sample.queries
            if self.cfg.protocol.reuse_mode == "multi_query"
            else sample.queries[:1]
        )
        query_texts: List[str] = []
        query_token_ids: List[List[int]] = []
        cache_lengths: List[int] = [self.backend.cache_length(state)]
        decode_time = 0.0
        for query_index, query in enumerate(queries):
            request_state = self.backend.fork_state(state)
            (
                generated,
                elapsed,
                request_score_time,
                request_compression_time,
                request_trace,
                request_decisions,
                request_bundles,
                request_diagnostics,
            ) = self._run_future_query(
                sample,
                query_index,
                query,
                request_state,
                evidence_positions,
            )
            decode_time += elapsed
            score_time += request_score_time
            compression_time += request_compression_time
            cache_lengths.extend(request_trace)
            decisions.extend(request_decisions)
            bundles.extend(request_bundles)
            diagnostic_events.extend(request_diagnostics)
            query_token_ids.append(generated)
            query_texts.append(self.backend.decode(generated))
        generated = query_token_ids[0] if query_token_ids else []
        generation = GenerationResult(
            token_ids=generated,
            text=query_texts[0] if query_texts else "",
            prefill_time_s=prefill_time,
            score_time_s=score_time,
            compression_time_s=compression_time,
            decode_time_s=decode_time,
            peak_gpu_memory_bytes=self.backend.peak_memory_bytes(),
            cache_lengths=cache_lengths,
            decisions=decisions,
            score_bundles=bundles,
            query_texts=query_texts,
            query_token_ids=query_token_ids,
        )
        return generation, diagnostic_events, prefix_ids, truncation

    def _run_future_query(
        self,
        sample: BenchmarkSample,
        query_index: int,
        query: str,
        state: HFCacheState,
        evidence_positions: List[int],
    ) -> Tuple[
        List[int],
        float,
        float,
        float,
        List[int],
        List[SelectionDecision],
        List[ScoreBundle],
        List[Dict[str, Any]],
    ]:
        query_ids = self.backend.encode_text(
            "\n\n" + query, add_special_tokens=False
        )
        if not query_ids:
            raise RuntimeError("future query tokenized to an empty sequence")
        elapsed_total = 0.0
        score_time = 0.0
        compression_time = 0.0
        logits = None
        decisions: List[SelectionDecision] = []
        bundles: List[ScoreBundle] = []
        diagnostics: List[Dict[str, Any]] = []
        trace: List[int] = []
        request_id = "%s:q%d" % (sample.sample_id, query_index)
        event_step = 0
        for query_token in query_ids:
            logits, elapsed = self.backend.step(state, query_token, False)
            elapsed_total += elapsed
            score_elapsed, compression_elapsed = self._compress_live_event(
                state,
                request_id,
                event_step,
                evidence_positions,
                decisions,
                bundles,
                diagnostics,
            )
            score_time += score_elapsed
            compression_time += compression_elapsed
            trace.append(self.backend.cache_length(state))
            event_step += 1
        if logits is None:
            raise RuntimeError("query ingestion produced no logits")
        generated: List[int] = [int(torch.argmax(logits, dim=-1).item())]
        for _ in range(1, self._max_new_tokens(sample)):
            logits, elapsed = self.backend.step(state, generated[-1], False)
            elapsed_total += elapsed
            score_elapsed, compression_elapsed = self._compress_live_event(
                state,
                request_id,
                event_step,
                evidence_positions,
                decisions,
                bundles,
                diagnostics,
            )
            score_time += score_elapsed
            compression_time += compression_elapsed
            trace.append(self.backend.cache_length(state))
            event_step += 1
            generated.append(int(torch.argmax(logits, dim=-1).item()))
            if (
                self.cfg.generation.stop_on_eos
                and generated[-1] == int(self.backend.tokenizer.eos_token_id)
            ):
                break
        return (
            generated,
            elapsed_total,
            score_time,
            compression_time,
            trace,
            decisions,
            bundles,
            diagnostics,
        )

    def _max_new_tokens(self, sample: BenchmarkSample) -> int:
        official = sample.metadata.get("official_max_new_tokens")
        if self.cfg.benchmark.use_official_generation_length and official is not None:
            return int(official)
        return int(self.cfg.generation.max_new_tokens)

    def _compress_live_event(
        self,
        state: HFCacheState,
        sample_id: str,
        event_step: int,
        evidence_positions: List[int],
        decisions: List[SelectionDecision],
        bundles: List[ScoreBundle],
        diagnostics: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        if not self.protocol.live_bounded:
            return 0.0, 0.0
        if self.policy.name == "full":
            return 0.0, 0.0
        if self.backend.cache_length(state) <= int(self.cfg.budget.cache_budget):
            return 0.0, 0.0
        snapshot = self.backend.snapshot(state, sample_id, "request", event_step)
        self.backend.synchronize()
        score_started = time.perf_counter()
        step_decisions, bundle = self.policy.decide(
            snapshot, recompute=self.protocol.should_recompute(event_step)
        )
        self.backend.synchronize()
        score_elapsed = time.perf_counter() - score_started
        diagnostics.append(
            compute_decision_diagnostics(
                snapshot,
                step_decisions,
                bundle,
                evidence_positions,
                self.cfg.diagnostics,
            )
        )
        compression_elapsed = self.backend.apply_decisions(state, step_decisions)
        decisions.extend(step_decisions)
        bundles.append(bundle)
        return score_elapsed, compression_elapsed

    def _teacher_forced_ppl(
        self, sample: BenchmarkSample, prompt_ids: List[int]
    ) -> Optional[float]:
        if not sample.answer_text or len(prompt_ids) < 2:
            return None
        targets = self.backend.encode_text(
            " " + sample.answer_text.lstrip(), add_special_tokens=False
        )
        if not targets:
            return None
        state, logits, _, _, _, _, _, _ = self._prefill_and_compress(
            sample, prompt_ids, []
        )
        nlls: List[float] = []
        capture_decode = self.policy.spec.requires_attention and self.protocol.live_bounded
        for step, target in enumerate(targets):
            log_probability = torch.log_softmax(logits.float(), dim=-1)[0, int(target)]
            nlls.append(-float(log_probability.item()))
            if step + 1 >= len(targets):
                break
            logits, _ = self.backend.step(state, int(target), capture_decode)
            if (
                self.protocol.live_bounded
                and self.policy.name != "full"
                and self.backend.cache_length(state) > int(self.cfg.budget.cache_budget)
            ):
                snapshot = self.backend.snapshot(state, sample.sample_id, "decode", step)
                decisions, _ = self.policy.decide(
                    snapshot, recompute=self.protocol.should_recompute(step)
                )
                self.backend.apply_decisions(state, decisions)
        return math.exp(sum(nlls) / len(nlls)) if nlls else None
