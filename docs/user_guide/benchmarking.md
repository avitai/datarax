# Benchmarking Guide

Datarax provides a benchmarking suite to measure and analyze the performance of your data pipelines.

## Overview

The benchmarking module (`datarax.benchmarking`) allows you to:

1.  **Time pipeline throughput** with `TimingCollector` (GPU-sync aware).
2.  **Profile GPU memory** with `GPUMemoryProfiler` and optimize with `MemoryOptimizer`.
3.  **Compare configurations** side-by-side with `BenchmarkComparison`.
4.  **Detect regressions** against historical baselines with `RegressionDetector`.
5.  **Monitor in real time** with `AdvancedMonitor` and `ProductionMonitor`.

## Quick Start

### Measuring Throughput

Use `TimingCollector` to measure samples/sec with optional GPU synchronization:

```python
from datarax.benchmarking import TimingCollector

# CPU timing (pass sync_fn for GPU — see docstring)
timer = TimingCollector()
result = timer.measure_iteration(
    iter(pipeline),
    num_batches=100,
    count_fn=lambda batch: batch["image"].shape[0],
)
throughput = result.num_elements / result.wall_clock_sec
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"First batch: {result.first_batch_time:.4f}s (includes JIT)")
```

### Memory Profiling

Use `GPUMemoryProfiler` to check GPU memory usage:

```python
from datarax.benchmarking import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
usage = profiler.get_memory_usage()
print(f"GPU memory used: {usage['gpu_memory_used_mb']:.1f} MB")
```

Use `MemoryOptimizer` to analyze a pipeline's memory footprint:

```python
from datarax.benchmarking import MemoryOptimizer

optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
print(f"Peak usage: {analysis['peak_usage_mb']:.1f} MB")
for suggestion in analysis["suggestions"]:
    print(f"  - {suggestion}")
```

### Comparative Benchmarking

Use `BenchmarkComparison` to compare results from different configurations:

```python
from datarax.benchmarking import BenchmarkComparison, BenchmarkResult

comparison = BenchmarkComparison()
comparison.add_result("baseline", baseline_result)
comparison.add_result("optimized", optimized_result)

ratios = comparison.get_performance_ratio()
print(f"Best config: {comparison.best_config}")
for name, ratio in ratios.items():
    print(f"  {name}: {ratio:.2%} of best")

comparison.save("comparison_results.json")
```

## Advanced Usage

For more detailed information on the classes and functions available, please refer to the [Benchmarking API Reference](../benchmarking/index.md).
