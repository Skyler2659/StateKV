"""Compatibility namespace for the former ``kvbench.temporal`` package.

New code should import from :mod:`statekv`.  The per-module shims in this
package preserve historical scripts without pulling StateKV into kvbench's
benchmark core.
"""

from statekv import DiscoveryConfig, load_discovery_config

__all__ = ["DiscoveryConfig", "load_discovery_config"]
