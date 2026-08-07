"""Compatibility alias for :mod:`statekv.gauge_geometry_analysis`."""

from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("statekv.gauge_geometry_analysis")
