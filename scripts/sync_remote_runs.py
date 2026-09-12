#!/usr/bin/env python3
"""Continuously pull immutable results and growing logs over an SSH connection."""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def aggregate(local: Path, seen: dict) -> None:
    columns = ("model", "benchmark", "task", "requested_context_length", "seed",
               "rollout_horizon", "policy", "budget")
    values = ("official_score", "mean_trajectory_exact_kl", "wall_time_s", "physical_cache_peak_bytes")
    for path in local.glob("jobs/*/results/*.json"):
        if str(path) not in seen:
            result = json.loads(path.read_text())
            seen[str(path)] = {key: result[key] for key in columns + values}
    groups = defaultdict(list)
    for result in seen.values():
        groups[tuple(result[key] for key in columns)].append(result)
    rows = []
    for identity, results in groups.items():
        row = dict(zip(columns, identity), n=len(results))
        for key in values:
            row["mean_" + key] = statistics.mean(result[key] for result in results)
        row["median_wall_time_s"] = statistics.median(result["wall_time_s"] for result in results)
        rows.append(row)
    if rows:
        temporary = local / "metrics.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(local / "metrics.csv")
    temporary = local / "summary.json.tmp"
    temporary.write_text(json.dumps(dict(completed_arms=len(seen), updated_at=time.time(), groups=rows), indent=2))
    temporary.replace(local / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="120")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    args.local.mkdir(parents=True, exist_ok=True)
    seen = {}
    ssh = shlex.join(["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"])
    while True:
        started = time.time()
        command = ["rsync", "-az", "--partial", "--delay-updates", "--timeout=30",
                   "--exclude=claim.lock", "-e", ssh,
                   f"{args.host}:{args.remote.rstrip('/')}/", str(args.local) + "/"]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            status = dict(updated_at=time.time(), exit_code=result.returncode,
                          error=result.stderr[-2000:])
        except subprocess.TimeoutExpired:
            status = dict(updated_at=time.time(), exit_code=-1, error="rsync timed out")
        temporary = args.local / ".sync_status.tmp"
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(args.local / "sync_status.json")
        print(json.dumps(status), flush=True)
        if status["exit_code"] == 0:
            aggregate(args.local, seen)
            progress = args.local / "status.json"
            if progress.exists() and json.loads(progress.read_text()).get("status") == "complete":
                # One final pull includes the last worker's completion files.
                subprocess.run(command, check=True, timeout=120)
                return
        time.sleep(max(1, args.interval - (time.time() - started)))


if __name__ == "__main__":
    main()
