"""Performance optimization modules for Datarax.

This package provides hardware-aware performance optimization tools based on
roofline analysis, XLA compilation strategies, and JAX transformation
optimization patterns.
"""

from datarax.performance.goodput import GoodputMetrics, GoodputTracker
from datarax.performance.roofline import HardwareSpecs, RooflineAnalyzer
from datarax.performance.synchronization import (
    block_until_ready_tree,
    copy_to_host_async_tree,
)
from datarax.performance.xla_optimization import (
    apply_xla_flags,
    CompilationProfiler,
    get_xla_flags,
    MemoryEfficientCompilation,
    SmartCompilation,
    XLAOptimizer,
)


__all__ = [
    "RooflineAnalyzer",
    "HardwareSpecs",
    "XLAOptimizer",
    "SmartCompilation",
    "MemoryEfficientCompilation",
    "CompilationProfiler",
    "GoodputTracker",
    "GoodputMetrics",
    "get_xla_flags",
    "apply_xla_flags",
    "block_until_ready_tree",
    "copy_to_host_async_tree",
]
