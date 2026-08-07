"""Single benchmark factory for every protocol and backend."""
from __future__ import annotations

from typing import List

from kvbench.benchmarks.longbench import LongBenchBenchmark
from kvbench.benchmarks.ruler import RULERBenchmark
from kvbench.benchmarks.scbench import SCBenchBenchmark
from kvbench.config import ExperimentConfig
from kvbench.types import BenchmarkSample


def load_benchmark(cfg: ExperimentConfig) -> List[BenchmarkSample]:
    name = cfg.benchmark.name.lower()
    if name == "ruler":
        adapter = RULERBenchmark(cfg.benchmark, cfg.runtime.seed)
    elif name == "longbench":
        adapter = LongBenchBenchmark(cfg.benchmark, cfg.runtime.seed)
    elif name == "scbench":
        adapter = SCBenchBenchmark(cfg.benchmark, cfg.runtime.seed)
    else:
        raise ValueError("unsupported benchmark: %s" % cfg.benchmark.name)
    return adapter.load()

