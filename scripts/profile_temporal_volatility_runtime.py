#!/usr/bin/env python
"""Profile CPU arithmetic and score-state size for temporal volatility."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from statekv.direct_policy_runtime import RollingTemporalVolatilityPolicy
from statekv.storage import atomic_frame, atomic_json, atomic_text


def _normalized_attention(
    rng: np.random.Generator, heads: int, tokens: int
) -> np.ndarray:
    values = rng.random((heads, tokens), dtype=np.float32)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output_dir = Path(config["output_run"])
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = tuple(range(int(config["diagnostic_layers"])))
    window = int(config["window"])
    heads = int(config["query_heads"])
    core_budget = int(config["core_budget"])
    repeats = int(config["timing_repeats"])
    rng = np.random.default_rng(int(config["seed"]))
    records = []
    for tokens in (int(value) for value in config["token_counts"]):
        bank = {
            layer: np.stack(
                [
                    _normalized_attention(rng, heads, tokens)
                    for _ in range(window)
                ],
                axis=0,
            )
            for layer in layers
        }
        eligible = np.arange(tokens, dtype=np.int64)
        update_ms = []
        completed = None
        for _ in range(repeats):
            rolling = RollingTemporalVolatilityPolicy(layers, window)
            started = time.perf_counter_ns()
            for query in range(window):
                for layer in layers:
                    rolling.update_layer(layer, bank[layer][query])
            update_ms.append(
                (time.perf_counter_ns() - started) / 1.0e6 / window
            )
            completed = rolling
        if completed is None:
            raise RuntimeError("profile did not build a rolling state")
        refresh_ms = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            completed.select(eligible, core_budget)
            refresh_ms.append((time.perf_counter_ns() - started) / 1.0e6)
        records.append(
            {
                "token_count": tokens,
                "rolling_update_per_decode_step_median_ms": float(
                    np.median(update_ms)
                ),
                "rolling_update_per_decode_step_p95_ms": float(
                    np.quantile(update_ms, 0.95)
                ),
                "refresh_median_ms": float(np.median(refresh_ms)),
                "refresh_p95_ms": float(np.quantile(refresh_ms, 0.95)),
                "working_set_bytes": completed.working_set_bytes,
                "bytes_per_context_token": float(
                    completed.working_set_bytes / tokens
                ),
            }
        )
    frame = pd.DataFrame(records)
    largest = frame.sort_values("token_count").iloc[-1]
    summary = {
        "experiment": str(config["experiment_name"]),
        "status": "cpu_arithmetic_microbenchmark",
        "signal": "attention_temporal_volatility_w4_shared",
        "candidate_algorithms_run_per_decision": 0,
        "diagnostic_layers": len(layers),
        "window": window,
        "query_heads": heads,
        "largest_token_count": int(largest["token_count"]),
        "largest_refresh_median_ms": float(largest["refresh_median_ms"]),
        "largest_update_per_decode_step_median_ms": float(
            largest["rolling_update_per_decode_step_median_ms"]
        ),
        "bytes_per_context_token": float(largest["bytes_per_context_token"]),
        "scope": (
            "CPU score arithmetic and persistent score state only; attention "
            "capture and end-to-end decode latency are excluded."
        ),
    }
    atomic_frame(frame, output_dir / "runtime_metrics.csv")
    atomic_json(output_dir / "summary.json", summary)
    atomic_text(
        output_dir / "config.yaml",
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
