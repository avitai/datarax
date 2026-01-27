# Benchmarking

Performance measurement and analysis tools for data pipelines. Use these tools to measure throughput, identify bottlenecks, and track performance regressions.

## Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| **Pipeline Throughput** | Measure samples/sec | Throughput metrics |
| **Profiler** | Detailed timing | Per-operation breakdown |
| **Comparative** | A/B comparison | Relative performance |
| **Regression** | Track over time | Performance trends |
| **Monitor** | Real-time display | Live metrics |

`★ Insight ─────────────────────────────────────`

- Always warm up pipelines before benchmarking (JIT compilation)
- Use `block_until_ready()` for accurate JAX timing
- Comparative benchmarks control for variance automatically
- Profile first, optimize second

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.benchmarking import measure_pipeline_throughput

# Measure throughput
result = measure_pipeline_throughput(
    pipeline,
    num_batches=100,
    warmup_batches=10,
)
print(f"Throughput: {result.samples_per_second:.2f} samples/sec")
```

## Modules

- [pipeline_throughput](pipeline_throughput.md) - Pipeline throughput measurement
- [profiler](profiler.md) - Detailed profiling with per-operation timing
- [comparative](comparative.md) - Compare implementations side-by-side
- [regression](regression.md) - Detect performance regressions over time
- [monitor](monitor.md) - Real-time performance monitoring display

## Profiling Example

```python
from datarax.benchmarking import Profiler

with Profiler() as profiler:
    for batch in pipeline:
        with profiler.section("forward"):
            output = model(batch)
        with profiler.section("loss"):
            loss = compute_loss(output)

# Print timing breakdown
profiler.summary()
```

## See Also

- [Benchmarking User Guide](../user_guide/benchmarking.md)
- [Performance](../performance/index.md) - Optimization tools
- [Monitoring](../monitoring/index.md) - Runtime metrics
