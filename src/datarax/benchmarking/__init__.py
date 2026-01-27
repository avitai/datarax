"""Advanced benchmarking and profiling system for Datarax.

This package provides benchmarking capabilities including:

- GPU memory profiling and optimization suggestions
- Performance regression detection
- Memory leak detection for long-running pipelines
- Comparative benchmarking tools
- Production monitoring integration

The system builds upon the existing datarax.monitoring and datarax.utils.benchmark
infrastructure to provide advanced performance insights.
"""

from datarax.benchmarking.comparative import (
    BenchmarkComparison,
    ComparativeBenchmark,
    LibraryComparison,
)
from datarax.benchmarking.monitor import (
    AdvancedMonitor,
    AlertManager,
    ProductionMonitor,
)
from datarax.benchmarking.profiler import (
    AdvancedProfiler,
    GPUMemoryProfiler,
    MemoryOptimizer,
    ProfileResult,
)
from datarax.benchmarking.regression import (
    PerformanceRegression,
    RegressionDetector,
    RegressionReport,
)


__all__ = [
    # Profiling
    "AdvancedProfiler",
    "GPUMemoryProfiler",
    "MemoryOptimizer",
    "ProfileResult",
    # Regression Detection
    "PerformanceRegression",
    "RegressionDetector",
    "RegressionReport",
    # Comparative Benchmarking
    "ComparativeBenchmark",
    "BenchmarkComparison",
    "LibraryComparison",
    # Advanced Monitoring
    "AdvancedMonitor",
    "ProductionMonitor",
    "AlertManager",
]
