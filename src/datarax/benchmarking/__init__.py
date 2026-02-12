"""Advanced benchmarking and profiling system for Datarax.

This package provides benchmarking capabilities including:

- Framework-agnostic timing and resource monitoring
- Statistical analysis with bootstrap CI and significance testing
- Performance regression detection
- Comparative benchmarking tools
- GPU memory profiling and optimization suggestions
- Production monitoring and alerting

Engine layer modules (installable with the library):
- timing.py: TimingCollector for framework-agnostic iteration timing
- statistics.py: StatisticalAnalyzer for measurement analysis
- resource_monitor.py: ResourceMonitor for background resource sampling
- results.py: BenchmarkResult for serializable result containers
- profiler.py: GPUMemoryProfiler, MemoryOptimizer, AdaptiveOperation
- regression.py: RegressionDetector for baseline comparison
- comparative.py: BenchmarkComparison for cross-config comparison
- monitor.py: AdvancedMonitor, ProductionMonitor, AlertManager
"""

from datarax.benchmarking.comparative import BenchmarkComparison
from datarax.benchmarking.monitor import (
    AdvancedMonitor,
    AlertManager,
    ProductionMonitor,
)
from datarax.benchmarking.profiler import (
    AdaptiveOperation,
    GPUMemoryProfiler,
    MemoryOptimizer,
)
from datarax.benchmarking.regression import (
    PerformanceRegression,
    RegressionDetector,
    RegressionReport,
)
from datarax.benchmarking.resource_monitor import (
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
)
from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.statistics import StatisticalAnalyzer, StatisticalResult
from datarax.benchmarking.timing import TimingCollector, TimingSample


__all__ = [
    # Timing
    "TimingCollector",
    "TimingSample",
    # Statistics
    "StatisticalAnalyzer",
    "StatisticalResult",
    # Resource Monitoring
    "ResourceMonitor",
    "ResourceSample",
    "ResourceSummary",
    # Results
    "BenchmarkResult",
    # Hardware Profiling
    "AdaptiveOperation",
    "GPUMemoryProfiler",
    "MemoryOptimizer",
    # Regression Detection
    "PerformanceRegression",
    "RegressionDetector",
    "RegressionReport",
    # Comparative Benchmarking
    "BenchmarkComparison",
    # Advanced Monitoring
    "AdvancedMonitor",
    "ProductionMonitor",
    "AlertManager",
]
