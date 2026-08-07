"""Benchmark adapter interface and deterministic sample selection."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from kvbench.config import BenchmarkConfig
from kvbench.types import BenchmarkSample


class BenchmarkAdapter(ABC):
    def __init__(self, cfg: BenchmarkConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)

    @abstractmethod
    def load(self) -> List[BenchmarkSample]:
        raise NotImplementedError

    def select(self, samples: List[BenchmarkSample]) -> List[BenchmarkSample]:
        if self.cfg.sample_indices is not None:
            invalid = [
                index
                for index in self.cfg.sample_indices
                if index < 0 or index >= len(samples)
            ]
            if invalid:
                raise IndexError("sample indices outside dataset: %s" % invalid[:10])
            return [samples[index] for index in self.cfg.sample_indices[: self.cfg.num_samples]]
        limit = min(len(samples), int(self.cfg.num_samples))
        if self.cfg.sample_strategy == "first":
            return samples[:limit]
        if self.cfg.sample_strategy == "random":
            indices = sorted(random.Random(self.seed).sample(range(len(samples)), limit))
            return [samples[index] for index in indices]
        raise ValueError("unsupported sample_strategy: %s" % self.cfg.sample_strategy)


def references_from_value(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]

