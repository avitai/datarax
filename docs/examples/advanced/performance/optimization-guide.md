# Performance Optimization Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~60 min |
| **Prerequisites** | Pipeline Tutorial, Monitoring Quick Reference |
| **Format** | Python + Jupyter |
| **Memory** | ~2 GB RAM |

## Overview

Master data pipeline performance optimization for Datarax. This guide covers
profiling techniques, batch size tuning, operator optimization, and
thorough benchmarking methodology.

## What You'll Learn

1. Profile pipeline performance to identify bottlenecks
2. Optimize batch size for your hardware
3. Measure and improve operator throughput
4. Compare different pipeline configurations
5. Generate performance benchmarks and visualizations

## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `num_workers` in DataLoader | Single-threaded (JAX handles parallelism) |
| `pin_memory=True` | JAX device placement |
| `torch.utils.benchmark` | Custom timing with `time.time()` |
| `prefetch_factor` | JAX async dispatch |

**Key difference:** Datarax relies on JAX's XLA compilation for performance rather than Python multiprocessing.

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `dataset.prefetch(AUTOTUNE)` | JAX async execution |
| `dataset.interleave()` | Explicit interleaving |
| `tf.profiler` | Custom profiling |
| `dataset.cache()` | Manual caching strategies |

## Files

- **Python Script**: [`examples/advanced/performance/01_optimization_guide.py`](https://github.com/avitai/datarax/blob/main/examples/advanced/performance/01_optimization_guide.py)
- **Jupyter Notebook**: [`examples/advanced/performance/01_optimization_guide.ipynb`](https://github.com/avitai/datarax/blob/main/examples/advanced/performance/01_optimization_guide.ipynb)

## Quick Start

```bash
python examples/advanced/performance/01_optimization_guide.py
```

## Performance Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Throughput** | Samples/second | Maximize |
| **Latency** | Time per batch | Minimize |
| **Memory** | Peak RAM usage | Within limits |
| **Utilization** | CPU/GPU usage | High |

## Part 1: Baseline Measurement

Use the `PipelineBenchmark` utility to measure throughput and latency
percentiles for any pipeline configuration.

```python
import time
import numpy as np
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

class PipelineBenchmark:
    """Utility for benchmarking pipeline configurations."""

    def __init__(self, warmup_batches: int = 5, measure_batches: int = 50):
        """Initialize PipelineBenchmark."""
        self.warmup_batches = warmup_batches
        self.measure_batches = measure_batches
        self.results = []

    def benchmark(self, pipeline, name: str = "Pipeline") -> dict:
        """Benchmark a pipeline and return metrics."""
        # Warmup
        warmup_count = 0
        for batch in pipeline:
            _ = batch["image"].block_until_ready()
            warmup_count += 1
            if warmup_count >= self.warmup_batches:
                break

        # Measurement
        latencies = []
        samples = 0
        batch_count = 0

        start_total = time.time()
        for batch in pipeline:
            start_batch = time.time()
            _ = batch["image"].block_until_ready()
            latencies.append(time.time() - start_batch)

            samples += batch["image"].shape[0]
            batch_count += 1

            if batch_count >= self.measure_batches + self.warmup_batches:
                break

        total_time = time.time() - start_total

        measured_latencies = latencies[self.warmup_batches :]
        result = {
            "name": name,
            "total_samples": samples,
            "total_time": total_time,
            "throughput": samples / total_time if total_time > 0 else 0,
            "avg_latency_ms": np.mean(measured_latencies) * 1000 if measured_latencies else 0,
            "p50_latency_ms": (
                np.percentile(measured_latencies, 50) * 1000 if measured_latencies else 0
            ),
            "p95_latency_ms": (
                np.percentile(measured_latencies, 95) * 1000 if measured_latencies else 0
            ),
            "p99_latency_ms": (
                np.percentile(measured_latencies, 99) * 1000 if measured_latencies else 0
            ),
        }

        self.results.append(result)
        return result


benchmark = PipelineBenchmark(warmup_batches=3, measure_batches=30)

# Baseline measurement
result = benchmark.benchmark(pipeline, name="Baseline")
print(f"Throughput: {result['throughput']:,.0f} samples/s")
print(f"Avg latency: {result['avg_latency_ms']:.2f} ms (p95: {result['p95_latency_ms']:.2f} ms)")
```

Throughput and latency depend heavily on your hardware, so run the benchmark
locally rather than relying on fixed numbers.

## Part 2: Batch Size Optimization

```python
def preprocess(element, key=None):
    """Simple normalization."""
    del key
    image = element.data["image"] / 255.0
    return element.update_data({"image": image})


def create_memory_pipeline(data, batch_size):
    """Create pipeline from memory data."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(ElementOperatorConfig(stochastic=False), fn=preprocess, rngs=nnx.Rngs(0))
    return Pipeline(source=source, stages=[prep], batch_size=batch_size, rngs=nnx.Rngs(0))


# Benchmark different batch sizes with the Datarax DAG pipeline
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
batch_results = []

print("Batch Size Sweep (Datarax Pipeline):")
for bs in batch_sizes:
    # Run multiple trials for stable measurements
    throughputs = []
    for trial in range(3):
        pipeline = create_memory_pipeline(test_data, bs)

        samples = 0
        start = time.time()
        for i, batch in enumerate(pipeline):
            if i >= 50:
                break
            _ = batch["image"].block_until_ready()
            samples += batch["image"].shape[0]
        elapsed = time.time() - start
        throughputs.append(samples / elapsed)

    avg_tp = np.mean(throughputs)
    batch_results.append({"batch_size": bs, "throughput": avg_tp, "std": np.std(throughputs)})
    print(f"  Batch {bs:4d}: {avg_tp:,.0f} samples/s (±{np.std(throughputs):.0f})")

# Find optimal
optimal = max(batch_results, key=lambda x: x["throughput"])
print(f"\nOptimal batch size: {optimal['batch_size']}")
```

Each batch size runs 3 trials of 50 batches so the reported throughput carries
a standard deviation. The optimal size is hardware dependent.

## Part 3: Operator Profiling

```python
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
    RotationOperator,
    RotationOperatorConfig,
)


def create_operator_pipeline(data, operator, batch_size=64):
    """Create pipeline with a specific operator."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(ElementOperatorConfig(stochastic=False), fn=preprocess, rngs=nnx.Rngs(0))

    stages = [prep]
    if operator is not None:
        stages.append(operator)

    return Pipeline(source=source, stages=stages, batch_size=batch_size, rngs=nnx.Rngs(0))


def benchmark_operator(name, operator, data, num_batches=30):
    """Benchmark a single operator, reporting per-batch latency in ms."""
    # Warmup
    pipeline = create_operator_pipeline(data, operator)
    for i, batch in enumerate(pipeline):
        if i >= 5:
            break
        _ = batch["image"].block_until_ready()

    # Measure
    pipeline = create_operator_pipeline(data, operator)
    latencies = []
    for i, batch in enumerate(pipeline):
        if i >= num_batches + 5:
            break
        start = time.time()
        _ = batch["image"].block_until_ready()
        latencies.append(time.time() - start)

    measured = latencies[5:]
    return {
        "name": name,
        "avg_ms": np.mean(measured) * 1000,
        "p50_ms": np.percentile(measured, 50) * 1000,
        "p95_ms": np.percentile(measured, 95) * 1000,
    }


operators = {
    "Baseline": None,
    "Brightness": BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image", brightness_range=(-0.2, 0.2), stochastic=True, stream_name="b"
        ),
        rngs=nnx.Rngs(b=1),
    ),
    "Contrast": ContrastOperator(
        ContrastOperatorConfig(
            field_key="image", contrast_range=(0.8, 1.2), stochastic=True, stream_name="c"
        ),
        rngs=nnx.Rngs(c=2),
    ),
    "Rotation": RotationOperator(
        RotationOperatorConfig(field_key="image", angle_range=(-15, 15)),
        rngs=nnx.Rngs(0),
    ),
    "Noise": NoiseOperator(
        NoiseOperatorConfig(
            field_key="image", mode="gaussian", noise_std=0.1, stochastic=True, stream_name="n"
        ),
        rngs=nnx.Rngs(n=3),
    ),
}

op_results = []
print("Operator Benchmarks:")
for name, op in operators.items():
    result = benchmark_operator(name, op, test_data)
    op_results.append(result)
    print(f"  {name:12s}: {result['avg_ms']:6.2f} ms (p95: {result['p95_ms']:.2f} ms)")
```

Each operator is measured against the `Baseline` (normalization only), so you
can read off the marginal latency each augmentation adds per batch.

## Part 4: Pipeline Optimization Strategies

### Strategy 1: Minimize Operators

The `normalize`, `scale`, and `shift` names below are illustrative pseudocode
standing in for three separate `ElementOperator` stages; the point is that
fusing them into one operator removes per-stage call overhead.

```python
# Inefficient: Many small operators (illustrative)
pipeline_slow = (
    Pipeline(source=source, stages=[normalize, scale, shift], batch_size=64, rngs=nnx.Rngs(0))
)

# Efficient: Combined operator
def combined_transform(element, key=None):
    image = element.data["image"]
    image = (image / 255.0 - 0.5) * 2.0  # Combined
    return element.update_data({"image": image})

combined_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=combined_transform,
    rngs=nnx.Rngs(0),
)
pipeline_fast = Pipeline(
    source=source,
    stages=[combined_op],
    batch_size=64,
    rngs=nnx.Rngs(0),
)
```

### Strategy 2: JIT Compilation

```python
@jax.jit
def jitted_transform(image):
    """JIT-compiled transformation."""
    return (image / 255.0 - 0.5) * 2.0

def jit_operator(element, key=None):
    image = jitted_transform(element.data["image"])
    return element.update_data({"image": image})
```

### Strategy 3: Memory-Efficient Operations

```python
# Avoid: Creates temporary arrays
def inefficient(element, key=None):
    image = element.data["image"]
    temp1 = image / 255.0
    temp2 = temp1 - 0.5
    temp3 = temp2 * 2.0
    return element.update_data({"image": temp3})

# Better: In-place style (JAX creates efficient fusion)
def efficient(element, key=None):
    image = element.data["image"]
    result = (image / 255.0 - 0.5) * 2.0
    return element.update_data({"image": result})
```

## Part 5: Visualization

The batch-size sweep is rendered as a 2-panel figure: a line plot of
throughput versus batch size (log-scaled, with the optimal size annotated) and
a bar chart of the same data.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bs_list = [r["batch_size"] for r in batch_results]
tp_list = [r["throughput"] for r in batch_results]
std_list = [r["std"] for r in batch_results]

# Panel 1: Throughput vs batch size (log-scaled x, optimal annotated)
ax1 = axes[0]
ax1.errorbar(bs_list, tp_list, yerr=std_list, fmt="o-", capsize=5, linewidth=2, markersize=8)
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Throughput (samples/second)")
ax1.set_title("Throughput vs Batch Size")
ax1.set_xscale("log", base=2)
ax1.grid(True, alpha=0.3)

optimal_idx = np.argmax(tp_list)
ax1.axvline(x=bs_list[optimal_idx], color="red", linestyle="--", alpha=0.5)
ax1.annotate(
    f"Optimal: {bs_list[optimal_idx]}",
    xy=(bs_list[optimal_idx], tp_list[optimal_idx]),
    xytext=(bs_list[optimal_idx] * 1.5, tp_list[optimal_idx] * 0.95),
    arrowprops=dict(arrowstyle="->"),
)

# Panel 2: Throughput bar chart with the optimal size highlighted
ax2 = axes[1]
bars = ax2.bar([str(bs) for bs in bs_list], tp_list, color="steelblue")
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Throughput (samples/second)")
ax2.set_title("Throughput by Batch Size")

optimal_tp = max(tp_list)
for bar, tp in zip(bars, tp_list):
    color = "darkgreen" if tp == optimal_tp else "black"
    x_pos = bar.get_x() + bar.get_width() / 2
    ax2.text(x_pos, tp + 100, f"{tp:,.0f}", ha="center", fontsize=8, color=color)

ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("docs/assets/images/examples/perf-batch-size-sweep.png", dpi=150)
```

### Further Analysis

The full script continues with several more analyses, each producing its own
figure:

- **Operator comparison** (`perf-throughput-comparison.png`): average and P95
  per-operator latency as horizontal and grouped bar charts.
- **Latency distribution** (`perf-latency-distribution.png`): a histogram per
  operator over 100 batches, with mean and P95 lines marked.
- **Memory profiling** (`perf-memory-profile.png`): estimated batch memory
  versus batch size, plus a throughput-per-MB efficiency chart.
- **Optimization report**: a printed summary of the optimal batch size, each
  operator's overhead relative to baseline, the most memory-efficient batch
  size, and general tuning recommendations.

## Results Summary

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| Optimal batch size | 1.5-2x | Hardware dependent |
| Combined operators | 1.3x | Reduce function call overhead |
| JIT compilation | 2-5x | One-time compilation cost |
| Memory efficiency | 1.2x | Reduce allocations |

**Performance Targets:**

| Hardware | Expected Throughput |
|----------|---------------------|
| CPU (8 cores) | 5,000-15,000 samples/sec |
| Single GPU | 50,000-100,000 samples/sec |
| Multi-GPU (4x) | 150,000-300,000 samples/sec |

## Best Practices

1. **Measure first**: Always profile before optimizing
2. **Warmup**: Skip first 10-20 batches for accurate timing
3. **Batch size**: Start at 64-128, sweep to find optimal
4. **Combine operators**: Fewer operators = less overhead
5. **JIT everything**: Use `@jax.jit` for custom transforms

## Next Steps

- [Distributed Training](../distributed/sharding-guide.md) - Scale across devices
- [End-to-End Training](../training/e2e-cifar10-guide.md) - Apply optimizations
- [API Reference: Benchmarking](../../../benchmarking/index.md) - Built-in tools
