# Benchmarking Guide

Datarax provides a benchmarking suite to measure and analyze the performance of your data pipelines.

## Overview

The benchmarking module (`calibrax`) allows you to:

1.  **Time pipeline throughput** with `TimingCollector` (GPU-sync aware).
2.  **Profile GPU memory** with `GPUMemoryProfiler` and optimize with `MemoryOptimizer`.
3.  **Compare frameworks** with `rank_table()` and `compare_configurations()`.
4.  **Detect regressions** against historical baselines with `detect_regressions()`.
5.  **Monitor in real time** with `AdvancedMonitor` and `ProductionMonitor`.

## Quick Start

### Measuring Throughput

Use `TimingCollector` to measure samples/sec with optional GPU synchronization:

```python
from calibrax.profiling import TimingCollector

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
from calibrax.profiling import GPUMemoryProfiler

profiler = GPUMemoryProfiler()
usage = profiler.get_memory_usage()
print(f"GPU memory used: {usage['gpu_memory_used_mb']:.1f} MB")
```

Use `MemoryOptimizer` to analyze a pipeline's memory footprint:

```python
from calibrax.profiling import MemoryOptimizer

optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
print(f"Peak usage: {analysis['peak_usage_mb']:.1f} MB")
for suggestion in analysis["suggestions"]:
    print(f"  - {suggestion}")
```

### Comparative Benchmarking

Use `rank_table()` to rank frameworks by any metric:

```python
from calibrax.analysis import rank_table
from calibrax.core import Metric, MetricDef, MetricDirection, Point, Run

run = Run(
    points=(
        Point(
            name="CV-1/baseline",
            scenario="CV-1",
            tags={"framework": "baseline"},
            metrics={"throughput": Metric(value=15000.0)},
        ),
        Point(
            name="CV-1/optimized",
            scenario="CV-1",
            tags={"framework": "optimized"},
            metrics={"throughput": Metric(value=20000.0)},
        ),
    ),
    metric_defs={
        "throughput": MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
        ),
    },
)

rankings = rank_table(run, "throughput")
for row in rankings:
    print(f"  {row.rank}. {row.label}: {row.value:.0f} elem/s")
```

## Advanced Usage

For more detailed information on the classes and functions available, please refer to the [Benchmarking API Reference](../benchmarking/index.md).
