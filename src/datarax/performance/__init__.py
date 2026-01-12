"""Performance optimization modules for Datarax.

This package provides hardware-aware performance optimization tools based on
roofline analysis, XLA compilation strategies, and JAX transformation
optimization patterns.
"""

from datarax.performance.roofline import HardwareSpecs, RooflineAnalyzer
from datarax.performance.xla_optimization import (
    CompilationProfiler,
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
]
