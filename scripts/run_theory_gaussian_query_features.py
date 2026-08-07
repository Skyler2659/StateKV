#!/usr/bin/env python3
"""Regenerate only anchor-time shrinkage Gaussian-query features."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPORT_ROOTS = (
    REPOSITORY_ROOT,
    REPOSITORY_ROOT / "benchmarks" / "torch",
    REPOSITORY_ROOT / "benchmarks" / "mlx",
)
for import_root in IMPORT_ROOTS:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from statekv.config import load_discovery_config
from statekv.functional_probe import _condition_cache
from statekv.selectors import CoreSelector
from statekv.tasks import load_discovery_tasks
from statekv.theory_closing import TheoryClosingRunner, _atomic_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_discovery_config(args.config)
    runner = TheoryClosingRunner(cfg, REPOSITORY_ROOT)
    # This pass needs the 32 historical/current query records and anchors 0/32
    # only; it does not repeat subset enumeration or stateful replay.
    feature_theory = replace(cfg.theory_closing, horizons=[1])
    feature_generation = replace(cfg.generation, max_new_tokens=33)
    feature_cfg = replace(
        cfg,
        theory_closing=feature_theory,
        generation=feature_generation,
    )
    runner.model.cfg = feature_cfg
    samples, _ = load_discovery_tasks(cfg)
    runner.model.load()
    runner._projection_bases = runner._build_projection_bases()
    fragment_dir = (
        runner.store.run_dir
        / "fragments"
        / "theory_closing"
        / "gaussian_query_anchor_features"
    )
    fragment_dir.mkdir(parents=True, exist_ok=True)
    try:
        for sample in samples:
            path = fragment_dir / (
                runner.store.safe_slug(sample.sample_id) + ".parquet"
            )
            if cfg.runtime.resume and path.exists():
                continue
            reference = runner.model.generate_reference(
                sample.sample_id, sample.task, sample.prompt
            )
            rows = []
            try:
                for recent in cfg.theory_closing.protected_recent_sizes:
                    cache_cfg = _condition_cache(
                        cfg,
                        cfg.theory_closing.total_budget,
                        int(recent),
                    )
                    selector = CoreSelector(replace(cfg, cache=cache_cfg))
                    old = selector.select(
                        reference.anchors[
                            cfg.theory_closing.horizon_anchor_step
                        ].snapshot(reference.sample_id),
                        cfg.theory_closing.selector,
                    )
                    fresh = selector.select(
                        reference.anchors[
                            cfg.theory_closing.horizon_start_step
                        ].snapshot(reference.sample_id),
                        cfg.theory_closing.selector,
                    )
                    for layer in runner.model.selected_layers:
                        for head in runner.model.selected_heads[int(layer)]:
                            features = runner._anchor_history_features(
                                reference,
                                int(layer),
                                int(head),
                                old.by_layer[int(layer)].selected_positions,
                                fresh.by_layer[
                                    int(layer)
                                ].selected_positions,
                                int(recent),
                            )
                            rows.append(
                                {
                                    "sample_id": sample.sample_id,
                                    "task": sample.task,
                                    "protected_recent_size": int(recent),
                                    "layer": int(layer),
                                    "head": int(head),
                                    **{
                                        key: value
                                        for key, value in features.items()
                                        if key.startswith(
                                            "anchor_gaussian_q_"
                                        )
                                    },
                                }
                            )
                _atomic_frame(pd.DataFrame(rows), path)
            finally:
                runner.model.release(reference)
        fragments = sorted(fragment_dir.glob("*.parquet"))
        output = pd.concat(
            [pd.read_parquet(path) for path in fragments],
            ignore_index=True,
        )
        _atomic_frame(
            output,
            runner.store.run_dir
            / "gaussian_query_anchor_features.parquet",
        )
        _atomic_frame(
            output,
            runner.store.run_dir / "gaussian_query_anchor_features.csv",
        )
    finally:
        runner.model.close()
    print(runner.store.run_dir)


if __name__ == "__main__":
    main()
