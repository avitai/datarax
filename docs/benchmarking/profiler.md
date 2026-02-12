# Profiler

GPU memory profiling, hardware-adaptive optimization, and memory analysis for Datarax pipelines.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Performance Tools](../performance/index.md) - Optimization
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

This module provides three components:

- **GPUMemoryProfiler** — Detects GPU availability and reports memory usage (used/total/utilization). Also analyzes memory patterns across multiple measurements to detect leaks and high utilization.
- **MemoryOptimizer** — Analyzes a pipeline function's memory footprint by measuring baseline, peak, and post-GC memory. Returns optimization suggestions.
- **AdaptiveOperation** — Auto-detects hardware (CPU/GPU/TPU) and configures optimal tile sizes, precision, and batch sizes. Also provides Grain auto-optimization.

## Quick Start

### Check GPU memory

```python
from datarax.benchmarking import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
usage = profiler.get_memory_usage()
print(f"GPU memory: {usage['gpu_memory_used_mb']:.1f} / {usage['gpu_memory_total_mb']:.1f} MB")
print(f"Utilization: {usage.get('gpu_memory_utilization', 0):.1%}")
```

### Analyze pipeline memory

```python
from datarax.benchmarking import MemoryOptimizer

optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
print(f"Peak usage: {analysis['peak_usage_mb']:.1f} MB")
print(f"Memory efficiency: {analysis['memory_efficiency']:.1%}")
for suggestion in analysis["suggestions"]:
    print(f"  - {suggestion}")
```

---

::: datarax.benchmarking.profiler
