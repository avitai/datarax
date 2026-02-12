# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
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

## Learning Goals

By the end of this guide, you will be able to:

1. Profile pipeline performance to identify bottlenecks
2. Optimize batch size for your hardware
3. Measure and improve operator throughput
4. Compare different pipeline configurations
5. Generate performance benchmarks and visualizations
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[tfds]" matplotlib
```
"""

# %%
# GPU Memory Configuration
import os

os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Core imports
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# Datarax imports
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
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
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Part 1: Understanding Pipeline Performance

### Performance Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Throughput** | Samples/second | Maximize |
| **Latency** | Time per batch | Minimize |
| **Memory** | Peak RAM usage | Within limits |
| **Utilization** | CPU/GPU usage | High |

### Common Bottlenecks

| Bottleneck | Symptom | Solution |
|------------|---------|----------|
| I/O bound | Low CPU usage | Increase prefetch |
| CPU bound | High CPU, low throughput | JIT compile, vectorize |
| Memory bound | OOM errors | Reduce batch size, stream |
| GPU idle | Low GPU util | Increase batch size |
"""

# %% [markdown]
"""
## Part 2: Benchmarking Utilities
"""


# %%
class PipelineBenchmark:
    """Utility for benchmarking pipeline configurations."""

    def __init__(self, warmup_batches: int = 5, measure_batches: int = 50):
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

        # Need fresh pipeline after warmup
        # (For real benchmarks, create new pipeline)

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

        # Compute metrics
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
print("PipelineBenchmark utility created")

# %% [markdown]
"""
## Part 3: Batch Size Optimization

Finding the optimal batch size balances throughput and memory.
"""

# %%
# Create test data
NUM_SAMPLES = 5000
IMAGE_SHAPE = (32, 32, 3)

np.random.seed(42)
test_data = {
    "image": np.random.rand(NUM_SAMPLES, *IMAGE_SHAPE).astype(np.float32),
    "label": np.random.randint(0, 10, (NUM_SAMPLES,)).astype(np.int32),
}

print(f"Test data: {NUM_SAMPLES} samples, shape={IMAGE_SHAPE}")


# %%
def preprocess(element, key=None):  # noqa: ARG001
    """Simple normalization."""
    del key
    image = element.data["image"] / 255.0
    return element.update_data({"image": image})


def create_memory_pipeline(data, batch_size):
    """Create pipeline from memory data."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(ElementOperatorConfig(stochastic=False), fn=preprocess, rngs=nnx.Rngs(0))
    return from_source(source, batch_size=batch_size).add(OperatorNode(prep))


# %%
# Benchmark different batch sizes with Datarax DAG pipeline
batch_sizes = [8, 16, 32, 64, 128, 256, 512]
batch_results = []

print("\nBatch Size Sweep (Datarax Pipeline):")
for bs in batch_sizes:
    # Run multiple trials
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

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot batch size sweep
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bs_list = [r["batch_size"] for r in batch_results]
tp_list = [r["throughput"] for r in batch_results]
std_list = [r["std"] for r in batch_results]

# Throughput vs batch size
ax1 = axes[0]
ax1.errorbar(bs_list, tp_list, yerr=std_list, fmt="o-", capsize=5, linewidth=2, markersize=8)
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Throughput (samples/second)")
ax1.set_title("Throughput vs Batch Size")
ax1.set_xscale("log", base=2)
ax1.grid(True, alpha=0.3)

# Mark optimal
optimal_idx = np.argmax(tp_list)
ax1.axvline(x=bs_list[optimal_idx], color="red", linestyle="--", alpha=0.5)
ax1.annotate(
    f"Optimal: {bs_list[optimal_idx]}",
    xy=(bs_list[optimal_idx], tp_list[optimal_idx]),
    xytext=(bs_list[optimal_idx] * 1.5, tp_list[optimal_idx] * 0.95),
    arrowprops=dict(arrowstyle="->"),
)

# Throughput as bar chart with values labeled
ax2 = axes[1]
bars = ax2.bar([str(bs) for bs in bs_list], tp_list, color="steelblue")
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Throughput (samples/second)")
ax2.set_title("Throughput by Batch Size")

# Highlight optimal
optimal_tp = max(tp_list)
for bar, tp in zip(bars, tp_list):
    color = "darkgreen" if tp == optimal_tp else "black"
    x_pos = bar.get_x() + bar.get_width() / 2
    ax2.text(x_pos, tp + 100, f"{tp:,.0f}", ha="center", fontsize=8, color=color)

ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    output_dir / "perf-batch-size-sweep.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'perf-batch-size-sweep.png'}")

# %% [markdown]
"""
## Part 4: Operator Performance Comparison

Compare throughput of different augmentation operators.
"""


# %%
def create_operator_pipeline(data, operator, batch_size=64):
    """Create pipeline with specific operator."""
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    prep = ElementOperator(ElementOperatorConfig(stochastic=False), fn=preprocess, rngs=nnx.Rngs(0))

    pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(prep))

    if operator is not None:
        pipeline = pipeline.add(OperatorNode(operator))

    return pipeline


def benchmark_operator(name, operator, data, num_batches=30):
    """Benchmark a single operator."""
    pipeline = create_operator_pipeline(data, operator)

    # Warmup
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


# %%
# Benchmark operators
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
print("\nOperator Benchmarks:")
for name, op in operators.items():
    result = benchmark_operator(name, op, test_data)
    op_results.append(result)
    print(f"  {name:12s}: {result['avg_ms']:6.2f} ms (p95: {result['p95_ms']:.2f} ms)")

# %%
# Plot operator comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = [r["name"] for r in op_results]
avg_times = [r["avg_ms"] for r in op_results]
p95_times = [r["p95_ms"] for r in op_results]

# Average latency
ax1 = axes[0]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
bars = ax1.barh(names, avg_times, color=colors)
ax1.set_xlabel("Latency (ms)")
ax1.set_title("Average Operator Latency per Batch")
for bar, val in zip(bars, avg_times):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}", va="center")

# P95 vs Average
ax2 = axes[1]
x = np.arange(len(names))
width = 0.35
ax2.bar(x - width / 2, avg_times, width, label="Average", color="steelblue")
ax2.bar(x + width / 2, p95_times, width, label="P95", color="coral")
ax2.set_xticks(x)
ax2.set_xticklabels(names, rotation=45, ha="right")
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Average vs P95 Latency")
ax2.legend()

plt.tight_layout()
plt.savefig(
    output_dir / "perf-throughput-comparison.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'perf-throughput-comparison.png'}")

# %% [markdown]
"""
## Part 5: Latency Distribution Analysis
"""

# %%
# Collect latency distributions
latency_distributions = {}

for name, op in operators.items():
    pipeline = create_operator_pipeline(test_data, op)
    latencies = []

    for i, batch in enumerate(pipeline):
        if i >= 100:
            break
        start = time.time()
        _ = batch["image"].block_until_ready()
        latencies.append((time.time() - start) * 1000)

    latency_distributions[name] = latencies[10:]  # Skip warmup

# %%
# Plot latency distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (name, latencies) in zip(axes, latency_distributions.items()):
    ax.hist(latencies, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    ax.axvline(x=mean_lat, color="red", linestyle="--", label=f"Mean: {mean_lat:.2f}")
    ax.axvline(x=p95_lat, color="orange", linestyle="--", label=f"P95: {p95_lat:.2f}")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} Latency Distribution")
    ax.legend(fontsize=8)

# Hide extra subplot
if len(axes) > len(latency_distributions):
    axes[-1].axis("off")

plt.tight_layout()
plt.savefig(
    output_dir / "perf-latency-distribution.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'perf-latency-distribution.png'}")

# %% [markdown]
"""
## Part 6: Memory Profiling
"""

# %%
# Estimate memory usage


def estimate_batch_memory(batch_size, image_shape, dtype_bytes=4):
    """Estimate memory for a batch."""
    image_mem = batch_size * np.prod(image_shape) * dtype_bytes
    label_mem = batch_size * 4  # int32
    overhead = 1.2  # JAX/NumPy overhead factor
    return (image_mem + label_mem) * overhead


# Memory vs batch size
memory_estimates = []
for bs in batch_sizes:
    mem = estimate_batch_memory(bs, IMAGE_SHAPE)
    memory_estimates.append({"batch_size": bs, "memory_mb": mem / 1e6})
    print(f"Batch {bs}: ~{mem / 1e6:.1f} MB")

# %%
# Plot memory profile
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Memory vs batch size
ax1 = axes[0]
mem_values = [m["memory_mb"] for m in memory_estimates]
ax1.plot(batch_sizes, mem_values, "o-", linewidth=2, markersize=8)
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Estimated Memory (MB)")
ax1.set_title("Memory Usage vs Batch Size")
ax1.set_xscale("log", base=2)
ax1.grid(True, alpha=0.3)

# Throughput per memory
ax2 = axes[1]
tp_per_mem = [tp / mem for tp, mem in zip(tp_list, mem_values)]
ax2.bar([str(bs) for bs in batch_sizes], tp_per_mem, color="steelblue")
ax2.set_xlabel("Batch Size")
ax2.set_ylabel("Throughput per MB (samples/s/MB)")
ax2.set_title("Memory Efficiency")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(output_dir / "perf-memory-profile.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'perf-memory-profile.png'}")

# %% [markdown]
"""
## Part 7: Optimization Recommendations
"""

# %%
# Generate optimization report
print()
print("=" * 60)
print("OPTIMIZATION REPORT")
print("=" * 60)

# Best batch size
best_bs_idx = np.argmax(tp_list)
best_bs = batch_sizes[best_bs_idx]
best_tp = tp_list[best_bs_idx]

print("\n1. BATCH SIZE OPTIMIZATION")
print(f"   Optimal batch size: {best_bs}")
print(f"   Peak throughput: {best_tp:,.0f} samples/s")
lower_bs = batch_sizes[max(0, best_bs_idx - 1)]
upper_bs = batch_sizes[min(len(batch_sizes) - 1, best_bs_idx + 1)]
print(f"   Recommendation: Use batch sizes between {lower_bs} and {upper_bs}")

# Operator overhead
baseline_time = next(r["avg_ms"] for r in op_results if r["name"] == "Baseline")
print("\n2. OPERATOR OVERHEAD")
print(f"   Baseline latency: {baseline_time:.2f} ms")
for r in op_results:
    if r["name"] != "Baseline":
        overhead = r["avg_ms"] - baseline_time
        overhead_pct = (overhead / baseline_time) * 100
        print(f"   {r['name']}: +{overhead:.2f} ms (+{overhead_pct:.0f}%)")

# Memory efficiency
best_mem_eff_idx = np.argmax(tp_per_mem)
print("\n3. MEMORY EFFICIENCY")
print(f"   Most efficient batch size: {batch_sizes[best_mem_eff_idx]}")
print(f"   Throughput/MB: {tp_per_mem[best_mem_eff_idx]:.0f} samples/s/MB")

print("\n4. GENERAL RECOMMENDATIONS")
print("   - Use JIT compilation for custom operators")
print("   - Minimize Python overhead in operator functions")
print("   - Prefer vectorized operations over loops")
print("   - Consider operator order (cheap before expensive)")
print("=" * 60)

# %% [markdown]
"""
## Results Summary

### Performance Tuning Checklist

| Area | Action | Impact |
|------|--------|--------|
| Batch size | Find optimal via sweep | High |
| Operators | Minimize custom logic | Medium |
| I/O | Use streaming/prefetch | Medium |
| JIT | Enable for operators | High |
| Memory | Monitor and limit | Medium |

### Key Takeaways

1. **Batch size**: Optimal is typically 64-256 for most hardware
2. **Latency**: P95 matters more than average for consistency
3. **Memory**: Linear with batch size, monitor for OOM
4. **Operators**: Rotation is typically most expensive
5. **Baseline**: Always compare against no-augmentation baseline
"""

# %% [markdown]
"""
## Next Steps

- **Distributed**: [Sharding guide](../distributed/02_sharding_guide.ipynb)
- **Checkpointing**: [Resumable training](../checkpointing/02_resumable_training_guide.ipynb)
- **Full training**: [End-to-end CIFAR-10](../training/01_e2e_cifar10_guide.ipynb)
"""


# %%
def main():
    """Run the performance optimization guide."""
    print("Performance Optimization Guide")
    print("=" * 50)

    # Quick benchmark
    np.random.seed(42)
    data = {
        "image": np.random.rand(1000, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (1000,)).astype(np.int32),
    }

    # Test different batch sizes
    for bs in [32, 64, 128]:
        pipeline = create_memory_pipeline(data, bs)
        samples = 0
        start = time.time()
        for i, batch in enumerate(pipeline):
            if i >= 20:
                break
            samples += batch["image"].shape[0]
        elapsed = time.time() - start
        print(f"Batch {bs}: {samples / elapsed:.0f} samples/s")

    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
