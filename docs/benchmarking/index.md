# Benchmarking

!!! info "External package"
    Benchmarking is provided by [calibrax](https://github.com/avitai/calibrax), which
    datarax depends on. Its API reference lives at
    [calibrax.readthedocs.io](https://calibrax.readthedocs.io); this page shows how the
    tools apply to a datarax pipeline.

Performance measurement and analysis tools for data pipelines. Use these tools to measure throughput, identify bottlenecks, and track performance regressions.

## Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| **TimingCollector** | Measure samples/sec with GPU sync | Throughput metrics |
| **GPUMemoryProfiler** | GPU memory profiling | Memory usage stats |
| **MemoryOptimizer** | Pipeline memory analysis | Optimization suggestions |
| **detect_regressions** | Track over time | Regression alerts |
| **rank_table** | Compare frameworks | Ranked performance tables |
| **AdvancedMonitor** | Real-time monitoring | Live metrics + alerts |

!!! tip "Benchmarking best practices"
    - Always warm up pipelines before benchmarking (JIT compilation)
    - Use `block_until_ready()` for accurate JAX timing
    - Attach confidence bounds via `Metric(lower=, upper=, samples=)` and use `calibrax.statistics` for significance testing
    - Profile first, optimize second

## Quick Start

```python
from calibrax.profiling import TimingCollector

# Measure throughput; each batch is awaited with jax.block_until_ready by default
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

## Reference

Each tool is documented in calibrax's API reference:

- [profiling](https://calibrax.readthedocs.io/en/latest/api-reference/profiling/) - timing, GPU memory profiling, hardware-adaptive optimization, background resource sampling
- [analysis](https://calibrax.readthedocs.io/en/latest/api-reference/analysis/) - side-by-side comparison and regression detection
- [monitoring](https://calibrax.readthedocs.io/en/latest/api-reference/monitoring/) - real-time monitoring and alerting
- [statistics](https://calibrax.readthedocs.io/en/latest/api-reference/statistics/) - bootstrap confidence intervals and significance tests
- [core](https://calibrax.readthedocs.io/en/latest/api-reference/core/) - serializable result containers

## GPU Memory Profiling

```python
from calibrax.profiling import GPUMemoryProfiler, MemoryOptimizer

# Check GPU memory usage
profiler = GPUMemoryProfiler()
usage = profiler.memory()
if usage is not None:
    print(f"GPU memory: {usage.used_mb:.1f} MB of {usage.total_mb:.1f} MB")

# Analyze pipeline memory patterns
optimizer = MemoryOptimizer()
analysis = optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
if analysis is not None:
    for suggestion in analysis.suggestions:
        print(f"  Suggestion: {suggestion}")
```

## See Also

- [Benchmarking User Guide](../user_guide/benchmarking.md)
- [Performance](../performance/index.md) - Optimization tools
- [Monitoring](../monitoring/index.md) - Runtime metrics
