"""Performance optimization modules for Datarax.

This package provides XLA compilation strategies, goodput tracking and
host/device synchronization helpers. Roofline analysis and compilation
profiling live in calibrax (``calibrax.profiling``).
"""

from datarax.performance.goodput import GoodputMetrics, GoodputTracker
from datarax.performance.synchronization import (
    block_until_ready_tree,
    copy_to_host_async_tree,
)
from datarax.performance.xla_optimization import (
    apply_xla_flags,
    get_xla_flags,
    MemoryEfficientCompilation,
    SmartCompilation,
    XLAOptimizer,
)


__all__ = [
    "XLAOptimizer",
    "SmartCompilation",
    "MemoryEfficientCompilation",
    "GoodputTracker",
    "GoodputMetrics",
    "get_xla_flags",
    "apply_xla_flags",
    "block_until_ready_tree",
    "copy_to_host_async_tree",
]
