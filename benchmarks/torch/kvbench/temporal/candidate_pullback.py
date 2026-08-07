"""Compatibility alias for :mod:`statekv.candidate_pullback`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("statekv.candidate_pullback")
