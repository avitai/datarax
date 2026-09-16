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

Every measurement in this guide goes through one timing helper. It draws the batch inside
the timed span, from the request to the batch being ready on the device, after warm-up
batches that absorb the session's compilation. Timing only `block_until_ready()` on a batch
the iterator has already produced measures nothing. The `PipelineBenchmark` utility turns
those per-batch times into throughput and latency percentiles for any pipeline configuration.

```python
import time
import numpy as np
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


def time_batches(pipeline, warmup_batches: int, num_batches: int) -> list[float]:
    """Seconds from requesting each batch to that batch being ready on the device."""
    batches = iter(pipeline)
    for _ in range(warmup_batches):
        next(batches)["image"].block_until_ready()

    latencies = []
    for _ in range(num_batches):
        start = time.perf_counter()
        next(batches)["image"].block_until_ready()
        latencies.append(time.perf_counter() - start)
    return latencies


class PipelineBenchmark:
    """Utility for benchmarking pipeline configurations."""

    def __init__(self, warmup_batches: int = 5, measure_batches: int = 50):
        """Initialize PipelineBenchmark."""
        self.warmup_batches = warmup_batches
        self.measure_batches = measure_batches
        self.results = []

    def benchmark(self, pipeline, name: str = "Pipeline") -> dict:
        """Time ``measure_batches`` batches of a pipeline and return throughput and latencies."""
        latencies = time_batches(pipeline, self.warmup_batches, self.measure_batches)
        samples = self.measure_batches * pipeline.batch_size
        total_time = sum(latencies)

        result = {
            "name": name,
            "total_samples": samples,
            "total_time": total_time,
            "throughput": samples / total_time,
            "avg_latency_ms": np.mean(latencies) * 1000,
            "p50_latency_ms": np.percentile(latencies, 50) * 1000,
            "p95_latency_ms": np.percentile(latencies, 95) * 1000,
            "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        }

        self.results.append(result)
        return result


benchmark = PipelineBenchmark(warmup_batches=3, measure_batches=30)
```

Throughput and latency depend heavily on your hardware. The numbers below come from one
run on an NVIDIA L40S; measure on the hardware you deploy on.

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


# Baseline measurement
result = benchmark.benchmark(create_memory_pipeline(test_data, 64), name="Baseline")
print(f"Baseline (batch 64): {result['throughput']:,.0f} samples/s")
print(f"Avg latency: {result['avg_latency_ms']:.2f} ms (p95: {result['p95_latency_ms']:.2f} ms)")

# Benchmark different batch sizes with the Datarax DAG pipeline. Each trial times up to 50
# batches after 3 warm-up batches; an epoch of NUM_SAMPLES bounds the count at the large sizes.
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
batch_results = []

print("\nBatch Size Sweep (Datarax Pipeline):")
for bs in batch_sizes:
    # Run multiple trials
    throughputs = []
    num_batches = min(50, NUM_SAMPLES // bs - 3)
    for trial in range(3):
        pipeline = create_memory_pipeline(test_data, bs)
        latencies = time_batches(pipeline, warmup_batches=3, num_batches=num_batches)
        throughputs.append(num_batches * bs / sum(latencies))

    avg_tp = np.mean(throughputs)
    batch_results.append({"batch_size": bs, "throughput": avg_tp, "std": np.std(throughputs)})
    print(f"  Batch {bs:4d}: {avg_tp:,.0f} samples/s (±{np.std(throughputs):.0f})")
```

**Terminal Output:**
```
Baseline (batch 64): 270,217 samples/s
Avg latency: 0.24 ms (p95: 0.31 ms)

Batch Size Sweep (Datarax Pipeline):
  Batch    8: 36,204 samples/s (±827)
  Batch   16: 73,563 samples/s (±3534)
  Batch   32: 149,587 samples/s (±2100)
  Batch   64: 288,573 samples/s (±13653)
  Batch  128: 636,699 samples/s (±30710)
  Batch  256: 1,086,639 samples/s (±48406)
  Batch  512: 2,000,521 samples/s (±182913)
```

Each batch size runs 3 trials, so the reported throughput carries a standard deviation. On the
L40S the per-batch time barely moves with the batch size, so throughput grows almost linearly
with it up to 512. The optimal size is hardware dependent.

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
    """Per-batch latency of a pipeline with one operator, after 5 warm-up batches."""
    pipeline = create_operator_pipeline(data, operator)
    measured = time_batches(pipeline, warmup_batches=5, num_batches=num_batches)
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
        RotationOperatorConfig(
            field_key="image", angle_range=(-15, 15), stochastic=True, stream_name="r"
        ),
        rngs=nnx.Rngs(r=4),
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

**Terminal Output:**
```
Operator Benchmarks:
  Baseline    :   0.23 ms (p95: 0.32 ms)
  Brightness  :   0.25 ms (p95: 0.30 ms)
  Contrast    :   0.24 ms (p95: 0.32 ms)
  Rotation    :   0.32 ms (p95: 0.50 ms)
  Noise       :   0.26 ms (p95: 0.35 ms)
```

Each operator is measured against the `Baseline` (normalization only), so you
can read off the marginal latency each augmentation adds per batch: on the L40S the
pixel-wise operators add a few hundredths of a millisecond and rotation, which resamples
the image, about a tenth.

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
from substrax.artifacts import resolve_output_dir

output_dir = resolve_output_dir("examples").path

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
plt.savefig(output_dir / "perf-batch-size-sweep.png", dpi=150)
```

### Further Analysis

The full script continues with several more analyses, each producing its own
figure:

- **Operator comparison** (`perf-throughput-comparison.png`): average and P95
  per-operator latency as horizontal and grouped bar charts.
- **Latency distribution** (`perf-latency-distribution.png`): a histogram per
  operator over 60 batches after 10 warm-up batches, with mean and P95 lines marked.
- **Memory profiling** (`perf-memory-profile.png`): estimated batch memory
  versus batch size, plus a throughput-per-MB efficiency chart.
- **Optimization report**: a printed summary of the optimal batch size, each
  operator's overhead relative to baseline, the most memory-efficient batch
  size, and general tuning recommendations.

**Terminal Output:**
```
============================================================
OPTIMIZATION REPORT
============================================================
1. BATCH SIZE OPTIMIZATION
   Optimal batch size: 512
   Peak throughput: 2,000,521 samples/s
   Recommendation: Use batch sizes between 256 and 512
2. OPERATOR OVERHEAD
   Baseline latency: 0.23 ms
   Brightness: +0.01 ms (+6%)
   Contrast: +0.01 ms (+2%)
   Rotation: +0.09 ms (+39%)
   Noise: +0.03 ms (+13%)
3. MEMORY EFFICIENCY
   Most efficient batch size: 128
   Throughput/MB: 337226 samples/s/MB
4. GENERAL RECOMMENDATIONS
   - Use JIT compilation for custom operators
   - Minimize Python overhead in operator functions
   - Prefer vectorized operations over loops
   - Consider operator order (cheap before expensive)
============================================================
```

## What to Expect

These techniques have no fixed speedup: the gain from batch size, operator fusion, JIT
compilation and fewer allocations depends on the hardware, the data and the pipeline shape.
Measure each change with the profiling steps above, on the hardware you deploy on.

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
