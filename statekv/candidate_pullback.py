"""Compatibility name for the retired candidate-pullback runner.

Candidate-pullback collection belonged to the closed Fisher investigation.
Active causal experiments only used its inherited model runtime, which now
lives in :mod:`statekv.causal_runtime`.
"""
from statekv.causal_runtime import CausalRuntimeRunner


CandidatePullbackRunner = CausalRuntimeRunner

__all__ = ["CandidatePullbackRunner"]
