# Benchmarking

Performance measurement and analysis tools for data pipelines. Use these tools to measure throughput, identify bottlenecks, and track performance regressions.

## Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| **TimingCollector** | Measure samples/sec with GPU sync | Throughput metrics |
| **GPUMemoryProfiler** | GPU memory profiling | Memory usage stats |
| **MemoryOptimizer** | Pipeline memory analysis | Optimization suggestions |
| **BenchmarkComparison** | A/B comparison | Relative performance ratios |
| **RegressionDetector** | Track over time | Regression reports |
| **AdvancedMonitor** | Real-time monitoring | Live metrics + alerts |

!!! tip "Benchmarking best practices"
    - Always warm up pipelines before benchmarking (JIT compilation)
    - Use `block_until_ready()` for accurate JAX timing
    - Comparative benchmarks control for variance automatically
    - Profile first, optimize second

## Quick Start

```python
from datarax.benchmarking import TimingCollector

# Measure throughput (CPU — pass sync_fn for GPU)
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

## Modules

- [profiler](profiler.md) - GPU memory profiling and hardware-adaptive optimization
- [comparative](comparative.md) - Compare configurations side-by-side
- [regression](regression.md) - Detect performance regressions over time
- [monitor](monitor.md) - Real-time performance monitoring and alerting
- [timing](timing.md) - Framework-agnostic timing with GPU sync
- [statistics](statistics.md) - Statistical analysis with bootstrap CI
- [resource_monitor](resource_monitor.md) - Background resource sampling
- [results](results.md) - Serializable benchmark result containers

## GPU Memory Profiling

```python
from datarax.benchmarking import GPUMemoryProfiler, MemoryOptimizer

# Check GPU memory usage
profiler = GPUMemoryProfiler()
usage = profiler.get_memory_usage()
print(f"GPU memory: {usage['gpu_memory_used_mb']:.1f} MB used")

# Analyze pipeline memory patterns
optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
for suggestion in analysis["suggestions"]:
    print(f"  Suggestion: {suggestion}")
```

## See Also

- [Benchmarking User Guide](../user_guide/benchmarking.md)
- [Performance](../performance/index.md) - Optimization tools
- [Monitoring](../monitoring/index.md) - Runtime metrics
