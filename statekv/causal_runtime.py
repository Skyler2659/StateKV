"""Minimal model runtime shared by the active causal and free-generation runs.

The causal/R2 programs only require a configured temporal model.  Earlier
versions inherited that runtime from the Fisher pullback experiment stack,
which pulled several completed experiment families into every active run.
Keeping this adapter small makes the runtime contract explicit and prevents
closed analysis code from becoming an accidental production dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from statekv.backend import TemporalModel
from statekv.config import DiscoveryConfig


class CausalRuntimeRunner:
    """Own a temporal model without an experiment-specific artifact pipeline."""

    def __init__(self, cfg: DiscoveryConfig, repository_root: Path) -> None:
        self.cfg = cfg
        self.repository_root = Path(repository_root).resolve()
        if cfg.model.backend == "mlx":
            from statekv.backend_mlx import MLXTemporalModel

            self.model: Any = MLXTemporalModel(cfg)
        else:
            self.model = TemporalModel(cfg)
