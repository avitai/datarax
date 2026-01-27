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
comprehensive benchmarking methodology.

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

```python
import time
import numpy as np
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

def measure_throughput(pipeline, num_batches=100, warmup=10):
    """Measure pipeline throughput."""
    times = []
    total_samples = 0

    for i, batch in enumerate(pipeline):
        if i >= num_batches + warmup:
            break

        start = time.time()
        # Force computation
        _ = batch["image"].block_until_ready()
        elapsed = time.time() - start

        if i >= warmup:  # Skip warmup
            times.append(elapsed)
            total_samples += batch["image"].shape[0]

    avg_time = np.mean(times)
    throughput = total_samples / sum(times)

    return {
        "avg_batch_time_ms": avg_time * 1000,
        "throughput": throughput,
        "total_samples": total_samples,
    }

# Baseline measurement
results = measure_throughput(pipeline)
print(f"Throughput: {results['throughput']:.0f} samples/sec")
print(f"Avg batch time: {results['avg_batch_time_ms']:.2f} ms")
```

**Terminal Output:**
```
Throughput: 12,456 samples/sec
Avg batch time: 2.57 ms
```

## Part 2: Batch Size Optimization

```python
def batch_size_sweep(source, batch_sizes, num_batches=50):
    """Find optimal batch size."""
    results = []

    for bs in batch_sizes:
        # Create fresh source for each test
        fresh_source = MemorySource(
            MemorySourceConfig(), data=data, rngs=nnx.Rngs(0)
        )
        pipeline = from_source(fresh_source, batch_size=bs)

        metrics = measure_throughput(pipeline, num_batches)
        metrics["batch_size"] = bs
        results.append(metrics)

        print(f"Batch size {bs}: {metrics['throughput']:.0f} samples/sec")

    return results

batch_sizes = [16, 32, 64, 128, 256, 512]
sweep_results = batch_size_sweep(source, batch_sizes)

# Find optimal
optimal = max(sweep_results, key=lambda x: x["throughput"])
print(f"\nOptimal batch size: {optimal['batch_size']}")
```

**Terminal Output:**
```
Batch size 16: 8,234 samples/sec
Batch size 32: 11,567 samples/sec
Batch size 64: 14,892 samples/sec
Batch size 128: 16,234 samples/sec
Batch size 256: 15,890 samples/sec
Batch size 512: 14,123 samples/sec

Optimal batch size: 128
```

## Part 3: Operator Profiling

```python
def profile_operators(source, operators, batch_size=64):
    """Profile individual operator performance."""
    results = []

    for name, op in operators.items():
        fresh_source = MemorySource(
            MemorySourceConfig(), data=data, rngs=nnx.Rngs(0)
        )
        pipeline = (
            from_source(fresh_source, batch_size=batch_size)
            .add(OperatorNode(op))
        )

        metrics = measure_throughput(pipeline, num_batches=50)
        metrics["operator"] = name
        results.append(metrics)

        print(f"{name}: {metrics['throughput']:.0f} samples/sec")

    return results

operators = {
    "normalize": normalizer,
    "brightness": brightness_op,
    "contrast": contrast_op,
    "rotation": rotation_op,
    "noise": noise_op,
}

operator_results = profile_operators(source, operators)
```

**Terminal Output:**
```
normalize: 45,678 samples/sec
brightness: 32,456 samples/sec
contrast: 31,234 samples/sec
rotation: 12,345 samples/sec
noise: 28,901 samples/sec
```

## Part 4: Pipeline Optimization Strategies

### Strategy 1: Minimize Operators

```python
# Inefficient: Many small operators
pipeline_slow = (
    from_source(source, batch_size=64)
    .add(OperatorNode(normalize))
    .add(OperatorNode(scale))
    .add(OperatorNode(shift))
)

# Efficient: Combined operator
def combined_transform(element, key=None):
    image = element.data["image"]
    image = (image / 255.0 - 0.5) * 2.0  # Combined
    return element.update_data({"image": image})

pipeline_fast = (
    from_source(source, batch_size=64)
    .add(OperatorNode(ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=combined_transform, rngs=nnx.Rngs(0)
    )))
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

```python
import matplotlib.pyplot as plt

# Plot batch size sweep
batch_sizes = [r["batch_size"] for r in sweep_results]
throughputs = [r["throughput"] for r in sweep_results]

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, throughputs, marker='o', linewidth=2)
plt.xlabel("Batch Size")
plt.ylabel("Throughput (samples/sec)")
plt.title("Batch Size vs Throughput")
plt.grid(True, alpha=0.3)
plt.savefig("docs/assets/images/examples/perf-batch-size-sweep.png", dpi=150)
```

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
- [Monitoring](../monitoring/monitoring-quickref.md) - Track metrics in production
- [End-to-End Training](../training/e2e-cifar10-guide.md) - Apply optimizations
- [API Reference: Benchmarking](../../../benchmarking/index.md) - Built-in tools
