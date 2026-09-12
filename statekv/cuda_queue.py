"""Four independent GPU workers over a resumable file-backed experiment queue."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from statekv.storage import atomic_json


def build_queue(plan_path: Path, root: Path) -> Path:
    plan = yaml.safe_load(plan_path.read_text())
    run = root / "runs" / plan["run_name"]
    jobs = []

    def add(model, task, benchmark, length, seed, policies=None, budgets=None, horizon=None):
        for offset in range(0, plan["num_samples"], plan["shard_size"]):
            start = plan["sample_offset"] if benchmark == "synthetic" else 0
            job = {key: plan[key] for key in ("dtype", "control_cycles", "rollout_horizon",
                "sink_size", "recent_size", "snapkv_window", "snapkv_pooling_kernel", "prefill_chunk_size")}
            job.update(model=model, task=task, benchmark=benchmark, context_length=length,
                seed=seed, policies=policies or plan["policies"], budgets=budgets or plan["budgets"],
                sample_indices=list(range(start + offset,
                    start + min(plan["num_samples"], offset + plan["shard_size"]))))
            if horizon is not None:
                job["rollout_horizon"] = horizon
            if model.endswith("Qwen3-8B"):
                job["score_layers"] = [0, 7, 14, 15, 21, 27]
            identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
            jobs.append(dict(id=identity, config=job))

    for model in plan["models"]:
        for length in plan["context_lengths"]:
            for task in plan["synthetic_tasks"]:
                add(model, task, "synthetic", length, plan["seed"])
    for task in plan["longbench_tasks"]:
        add(plan["longbench_model"], task, "longbench", plan["longbench_max_context"], plan["seed"])
    for seed in plan.get("robustness_seeds", []):
        for task in ("multikey_4", "multikey_8"):
            add(plan["longbench_model"], task, "synthetic", 8192, seed)
    for horizon in plan.get("horizon_ablations", []):
        add(plan["longbench_model"], "multikey_4", "synthetic", 8192, plan["seed"],
            policies=["FULL", "CHEAP_R2"], budgets=[256], horizon=horizon)
    path = run / "queue.json"
    if path.exists() and json.loads(path.read_text())["jobs"] != jobs:
        raise ValueError("existing queue differs from plan; use another run_name")
    atomic_json(path, {"plan": plan, "jobs": jobs})
    return run


def gpu_available(gpu: int) -> tuple[bool, str]:
    if gpu not in (3, 4, 5, 6):
        raise ValueError("only physical GPU 3,4,5,6 are allowed")
    row = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid = row[0].strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader,nounits"], text=True)
    return int(row[1]) < 256 and int(row[2]) <= 5 and uuid not in apps, uuid


def status(run: Path, jobs: list[dict]) -> dict:
    complete, failed, running = 0, 0, 0
    for job in jobs:
        output = run / "jobs" / job["id"]
        if (output / "complete.json").exists():
            complete += 1
        elif (output / "failure.json").exists():
            failed += 1
        else:
            active = output / "active.json"
            if active.exists():
                item = json.loads(active.read_text())
                running += int(Path(f"/proc/{item['pid']}").exists())
    value = dict(total_jobs=len(jobs), complete_jobs=complete, failed_jobs=failed,
                 running_jobs=running, updated_at=time.time(),
                 status="complete" if complete == len(jobs) else "running")
    atomic_json(run / "status.json", value)
    return value


def worker(root: Path, run: Path, gpu: int) -> None:
    jobs = json.loads((run / "queue.json").read_text())["jobs"]
    while True:
        if status(run, jobs)["status"] == "complete":
            return
        available, uuid = gpu_available(gpu)
        if not available:
            atomic_json(run / f"worker_gpu{gpu}.json", {"status": "waiting_for_gpu", "updated_at": time.time()})
            time.sleep(15)
            continue
        ran = False
        for job in jobs:
            output = run / "jobs" / job["id"]
            if (output / "complete.json").exists():
                continue
            key = job["config"]["model"].split("/")[-1].lower().replace(".", "p")
            if not (root / "cache/ready" / (key + ".json")).exists():
                continue
            if job["config"]["benchmark"] == "longbench" and not (root / "cache/longbench_manifest.json").exists():
                continue
            output.mkdir(parents=True, exist_ok=True)
            with (output / "claim.lock").open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (output / "complete.json").exists():
                    continue
                active = output / "active.json"
                if active.exists() and Path(f"/proc/{json.loads(active.read_text())['pid']}").exists():
                    continue
                failure = output / "failure.json"
                attempts = json.loads(failure.read_text())["attempts"] if failure.exists() else 0
                if attempts >= 3:
                    continue
                atomic_json(output / "config.json", job["config"])
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
                command = [sys.executable, str(root / "code/scripts/run_cuda_experiments.py"),
                           "job", "--root", str(root), "--output", str(output)]
                with (output / "run.log").open("a", buffering=1) as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(active, {"pid": process.pid, "gpu": gpu, "uuid": uuid, "started_at": time.time()})
                    atomic_json(run / f"worker_gpu{gpu}.json", {"status": "running", "job_id": job["id"], "pid": process.pid})
                    code = process.wait()
                if code != 0:
                    atomic_json(failure, {"attempts": attempts + 1, "exit_code": code, "updated_at": time.time()})
                elif failure.exists():
                    failure.unlink()
                active.unlink(missing_ok=True)
                status(run, jobs)
                ran = True
                break
        if not ran:
            time.sleep(15)
