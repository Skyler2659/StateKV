"""Resumable CUDA experiment jobs with per-arm atomic results."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from kvbench.benchmarks.longbench import LongBenchBenchmark, MAX_NEW_TOKENS
from kvbench.config import BenchmarkConfig
from statekv.cuda_runtime import CudaRuntime
from statekv.storage import atomic_json
from statekv.tasks import (_extend_retrieval_prompt, _synthetic_niah_multikey,
                           _synthetic_niah_multiquery, _synthetic_variable_tracking)
from src.evaluation.official_metrics import longbench_score, ruler_score


def synthetic_sample(runtime: CudaRuntime, job: dict[str, Any], index: int):
    task = job["task"]
    target = job["context_length"]
    hint = target
    factory = _synthetic_niah_multikey if task.startswith("multikey") else (
        _synthetic_niah_multiquery if task == "multiquery" else _synthetic_variable_tracking)
    seed = job["seed"] + index * (9173 if task == "variable_tracking" else 1009)
    parameter = int(task.rsplit("_", 1)[1]) if task.startswith("multikey") else (4 if task == "multiquery" else 8)
    for _ in range(8):
        sample = _extend_retrieval_prompt(factory(seed, 1, hint, parameter)[0])
        sample.sample_id = f"{task}:{index}"
        ids = runtime.backend.encode_prompt(sample.prompt)
        if target * .95 <= len(ids) <= target:
            sample.metadata.update(requested_context_length=target, generator_context_hint=hint)
            return sample, ids, False
        hint = max(256, int(hint * (target - 128) / max(1, len(ids) - 128)))
    raise RuntimeError(f"could not calibrate synthetic context: {task} target={target} actual={len(ids)}")


def run_job(root: Path, job: dict[str, Any], output: Path) -> None:
    key = job["model"].split("/")[-1].lower().replace(".", "p")
    checkpoint = json.loads((root / "cache/ready" / (key + ".json")).read_text())
    config = dict(job)
    config["model"] = {"name": checkpoint["path"], "dtype": job["dtype"],
                       "chat_template_kwargs": {"enable_thinking": False}}
    config["local_files_only"] = True
    config["stop_on_eos"] = job["benchmark"] == "longbench"
    torch.manual_seed(job["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = CudaRuntime(config)
    revision_path = root / "code/source_revision.txt"
    source_revision = revision_path.read_text().strip() if revision_path.exists() else "working-tree"
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    atomic_json(output / "job.json", dict(config=job, checkpoint=checkpoint,
                model_info=runtime.info, source_revision=source_revision, job_identity=identity))
    samples = {}
    if job["benchmark"] == "longbench":
        data = json.loads((root / "cache/longbench_manifest.json").read_text())[job["task"]]
        indices = [data["indices"][index] for index in job["sample_indices"]]
        cfg = BenchmarkConfig(task=job["task"], data_path=data["path"],
              num_samples=len(indices), sample_indices=indices, max_words=0,
              require_official=True, dataset_revision=data["revision"])
        samples = dict(zip(job["sample_indices"], LongBenchBenchmark(cfg, job["seed"]).load()))
    try:
        for index in job["sample_indices"]:
            arms = [(policy, 0 if policy == "FULL" else budget)
                    for policy in job["policies"]
                    for budget in ([0] if policy == "FULL" else job["budgets"])]
            paths = {(policy, budget): output / "results" / f"{index}_{policy}_{budget}.json"
                     for policy, budget in arms}
            pending = [(policy, budget) for policy, budget in arms if not paths[policy, budget].exists()]
            if not pending:
                continue
            if job["benchmark"] == "synthetic":
                sample, ids, truncated = synthetic_sample(runtime, job, index)
                cycles = job["control_cycles"]
            else:
                sample = samples[index]
                ids, truncated = runtime.encode(sample, job["context_length"])
                cycles = MAX_NEW_TOKENS[job["task"]]
            anchor, _, prefill_s = runtime.prefill(ids[:-1])
            for policy, budget in pending:
                started = time.time()
                atomic_json(output / "progress.json", dict(status="running", sample_index=index,
                    policy=policy, budget=budget, started_at=started))
                with (output / "events.jsonl").open("a", buffering=1) as events:
                    def event(row):
                        events.write(json.dumps(dict(sample_index=index, policy=policy,
                                      budget=budget, timestamp=time.time(), **row)) + "\n")
                    result = runtime.run_arm(anchor, ids, policy, budget, cycles, prefill_s, event)
                score = (ruler_score(sample.task, result["generation_text"], sample.references)
                         if job["benchmark"] == "synthetic" else
                         longbench_score(sample.task, result["generation_text"], sample.references,
                                         official=True, all_classes=sample.metadata.get("all_classes")))
                if score is None:
                    raise RuntimeError("task has no implemented metric")
                result.update(sample_index=index, sample_id=sample.sample_id, task=job["task"],
                    benchmark=job["benchmark"], model=job["model"], checkpoint_revision=checkpoint["revision"],
                    source_revision=source_revision, seed=job["seed"], official_score=score,
                    references=sample.references, actual_input_tokens=len(ids),
                    requested_context_length=job["context_length"], prompt_truncated=truncated,
                    sample_metadata=sample.metadata, job_identity=identity,
                    rollout_horizon=job["rollout_horizon"], completed_at=time.time())
                atomic_json(paths[policy, budget], result)
                print(f"{sample.sample_id} {policy} budget={budget} score={score} seconds={time.time()-started:.1f}", flush=True)
            del anchor
            torch.cuda.empty_cache()
        atomic_json(output / "progress.json", {"status": "complete", "completed_at": time.time()})
        atomic_json(output / "complete.json", {"completed_at": time.time(), "job_identity": identity})
    finally:
        runtime.close()
