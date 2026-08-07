"""Backend interface used by the experiment runner."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch

from kvbench.types import CacheSnapshot, SelectionDecision


class BackendAdapter(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def encode_prompt(self, prompt: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def prefill(self, token_ids: List[int], capture_attention: bool) -> Tuple[Any, torch.Tensor, float]:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: Any, token_id: int, capture_attention: bool) -> Tuple[torch.Tensor, float]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, state: Any, sample_id: str, phase: str, decode_step: int) -> CacheSnapshot:
        raise NotImplementedError

    @abstractmethod
    def apply_decisions(self, state: Any, decisions: List[SelectionDecision]) -> float:
        raise NotImplementedError

    @abstractmethod
    def cache_length(self, state: Any) -> int:
        raise NotImplementedError

