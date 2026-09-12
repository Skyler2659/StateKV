#!/usr/bin/env python3
"""Download pinned HF checkpoints and the official LongBench archive."""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmarks/torch"), str(ROOT / "benchmarks/mlx")]

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from statekv.storage import atomic_json


def model_key(name: str) -> str:
    return name.split("/")[-1].lower().replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/cuda/validation.yaml")
    args = parser.parse_args()
    root = args.root.resolve()
    plan = yaml.safe_load(args.plan.read_text())
    api = HfApi()
    manifest_path = root / "cache/model_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in plan["models"]:
        if name not in manifest:
            manifest[name] = {"revision": api.model_info(name).sha}
    atomic_json(manifest_path, manifest)
    for name, item in manifest.items():
        ready = root / "cache/ready" / (model_key(name) + ".json")
        if ready.exists() and json.loads(ready.read_text())["revision"] == item["revision"]:
            continue
        path = snapshot_download(name, revision=item["revision"],
            cache_dir=str(root / "cache/huggingface"), max_workers=4,
            allow_patterns=["config.json", "generation_config.json", "tokenizer*",
                            "*.jinja", "*.safetensors", "*.safetensors.index.json"])
        item.update(path=path, name=name)
        atomic_json(ready, item)
        print(f"model ready: {name} {item['revision']}", flush=True)
    data_manifest = root / "cache/longbench_manifest.json"
    if data_manifest.exists():
        print("all inputs already ready", flush=True)
        return
    revision = api.dataset_info("THUDM/LongBench").sha
    archive = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset",
        revision=revision, cache_dir=str(root / "cache/huggingface"))
    target = root / "cache/longbench"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
    data = {}
    for task in plan["longbench_tasks"]:
        path = next(target.rglob(task + ".jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        buckets = [[], [], []]
        for index, row in enumerate(rows):
            length = int(row.get("length", 0))
            buckets[0 if length < 4000 else (1 if length < 8000 else 2)].append(index)
        rng = random.Random(plan["seed"])
        for bucket in buckets:
            rng.shuffle(bucket)
        selected = [bucket[i] for i in range(max(map(len, buckets)))
                    for bucket in buckets if i < len(bucket)][:plan["num_samples"]]
        data[task] = {"path": str(path), "indices": sorted(selected), "revision": revision}
    atomic_json(root / "cache/longbench_manifest.json", data)
    atomic_json(root / "cache/inputs_complete.json", {"models": manifest, "longbench_revision": revision})
    print("all inputs ready", flush=True)


if __name__ == "__main__":
    main()
