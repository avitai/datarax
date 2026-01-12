# Benchmarking Guide

Datarax provides a benchmarking suite to measure and analyze the performance of your data pipelines.

## Overview

The benchmarking module (`datarax.benchmarking`) allows you to:

1.  **Comparatively Benchmark** different configurations.
2.  **Profile** execution time and memory usage.
3.  **Analyze Throughput** to identify bottlenecks.
4.  **Perform Regression Testing** to ensure performance stability.

## Quick Start

### Comparative Benchmarking

Use `ComparativeBenchmark` to compare different pipeline settings:

```python
from datarax.benchmarking import ComparativeBenchmark

def pipeline_fn(config):
    # build your pipeline based on config
    pass

benchmark = ComparativeBenchmark(
    name="my_benchmark",
    pipeline_fn=pipeline_fn,
    configs={
        "baseline": {"batch_size": 32},
        "optimized": {"batch_size": 128},
    }
)
results = benchmark.run()
print(results)
```

### Profiling

Use the `Profiler` to get detailed execution metrics:

```python
from datarax.benchmarking import Profiler

profiler = Profiler()
with profiler.profile():
    # run your pipeline code
    pass

profiler.print_stats()
```

## Advanced Usage

For more detailed information on the classes and functions available, please refer to the [Benchmarking API Reference](../benchmarking/index.md).
