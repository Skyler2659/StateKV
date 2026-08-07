"""Compatibility alias for :mod:`statekv.output_sensitivity_freegen`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("statekv.output_sensitivity_freegen")
