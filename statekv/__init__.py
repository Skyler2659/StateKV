"""State-conditioned KV-cache selection and refresh research package.

The package contains the reusable StateKV mechanism, evaluator, analysis, and
experiment primitives.  Protocol-aware eviction benchmarking remains in the
independent :mod:`kvbench` package.
"""

from statekv.config import DiscoveryConfig, load_discovery_config

__all__ = ["DiscoveryConfig", "load_discovery_config"]
