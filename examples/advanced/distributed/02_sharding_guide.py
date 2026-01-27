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
# Distributed Data Loading with Sharding Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~45 min |
| **Prerequisites** | Sharding Quick Reference, JAX device placement |
| **Format** | Python + Jupyter |
| **Memory** | ~2 GB RAM per device |

## Overview

This comprehensive guide covers distributed data loading patterns for
multi-device JAX setups. You'll learn to shard data across GPUs/TPUs,
optimize throughput for distributed training, and handle common pitfalls.

## Learning Goals

By the end of this guide, you will be able to:

1. Design data parallelism strategies for different device topologies
2. Implement efficient sharded batch distribution
3. Profile and optimize distributed data loading
4. Handle edge cases (uneven batches, device failures)
5. Integrate sharded pipelines with distributed training
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[tfds]" matplotlib
```

**Requirements**: This guide is designed for multi-device systems.
Single-device systems will run in simulation mode showing the concepts.
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
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Datarax imports
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(f"Device count: {len(jax.devices())}")

# %% [markdown]
"""
## Part 1: Understanding Data Parallelism

### Data Parallelism Basics

In data parallelism, each device processes a portion of the batch:

```
Full Batch (B samples)
├── Device 0: B/N samples
├── Device 1: B/N samples
├── ...
└── Device N-1: B/N samples
```

### JAX Sharding Concepts

| Concept | Description |
|---------|-------------|
| **Mesh** | Logical arrangement of devices |
| **PartitionSpec** | How to shard each dimension |
| **NamedSharding** | Sharding policy for arrays |
| **device_put** | Place data on devices |

### Mesh Axis Naming

Common conventions:
- `"data"` or `"batch"` - Data parallelism axis
- `"model"` or `"tensor"` - Model/tensor parallelism axis
- `"pipeline"` - Pipeline parallelism axis
"""

# %%
# Device configuration
devices = jax.devices()
num_devices = len(devices)
use_sharding = num_devices >= 2

print(f"Available devices: {num_devices}")
print(f"Device types: {[str(d.device_kind) for d in devices]}")

if use_sharding:
    # Create 1D mesh for pure data parallelism
    device_array = np.array(devices)
    mesh = Mesh(device_array, axis_names=("data",))
    print(f"\nCreated mesh: {mesh.shape} with axis 'data'")
else:
    mesh = None
    print("\nSingle device mode - will simulate sharding concepts")

# %% [markdown]
"""
## Part 2: Sharded Batch Distribution

### PartitionSpec Rules

| PartitionSpec | Meaning |
|---------------|---------|
| `("data",)` | Shard along first dim across "data" axis |
| `("data", None)` | Shard first dim, replicate second |
| `(None, "data")` | Replicate first dim, shard second |
| `(None, None)` | Fully replicate |
"""


# %%
def create_sharding_spec(shape, mesh, shard_first_dim=True):
    """Create appropriate PartitionSpec for a given shape.

    Args:
        shape: Array shape tuple
        mesh: JAX Mesh object
        shard_first_dim: Whether to shard along first (batch) dimension

    Returns:
        NamedSharding for the array
    """
    if mesh is None:
        return None

    ndim = len(shape)
    if shard_first_dim and ndim > 0:
        # Shard first dim (batch), replicate rest
        spec = ("data",) + (None,) * (ndim - 1)
    else:
        # Fully replicate
        spec = (None,) * ndim

    return NamedSharding(mesh, PartitionSpec(*spec))


# Example sharding specs
if mesh is not None:
    image_sharding = create_sharding_spec((64, 32, 32, 3), mesh)
    label_sharding = create_sharding_spec((64,), mesh)
    print(f"Image sharding: {image_sharding.spec}")
    print(f"Label sharding: {label_sharding.spec}")

# %% [markdown]
"""
## Part 3: Create Sharded Data Pipeline

We'll build a complete pipeline that produces sharded batches.
"""

# %%
# Configuration
BATCH_SIZE = 128  # Total batch size across all devices
SAMPLES_PER_DEVICE = BATCH_SIZE // max(num_devices, 1)
NUM_SAMPLES = 2048

print("Batch configuration:")
print(f"  Total batch size: {BATCH_SIZE}")
print(f"  Samples per device: {SAMPLES_PER_DEVICE}")
print(f"  Dataset size: {NUM_SAMPLES}")


# %%
def preprocess_image(element, key=None):  # noqa: ARG001
    """Standard image preprocessing."""
    del key
    image = element.data["image"]

    # Normalize to [0, 1]
    image = image.astype(jnp.float32) / 255.0

    return element.update_data({"image": image})


def create_pipeline(batch_size=BATCH_SIZE, num_samples=NUM_SAMPLES, seed=42):
    """Create CIFAR-10 data pipeline."""
    config = TFDSEagerConfig(
        name="cifar10",
        split=f"train[:{num_samples}]",
        shuffle=True,
        seed=seed,
        exclude_keys={"id"},
    )

    source = TFDSEagerSource(config, rngs=nnx.Rngs(seed))

    preprocessor = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_image,
        rngs=nnx.Rngs(0),
    )

    return from_source(source, batch_size=batch_size).add(OperatorNode(preprocessor))


# Create pipeline
pipeline = create_pipeline()
print("Created CIFAR-10 pipeline")


# %%
def distribute_batch(batch, mesh, shard_batch_dim=True):
    """Distribute batch data across devices.

    Args:
        batch: Dictionary of batch arrays
        mesh: JAX Mesh object
        shard_batch_dim: Whether to shard along batch dimension

    Returns:
        Dictionary with sharded arrays
    """
    if mesh is None:
        return batch

    distributed = {}
    for key, array in batch.items():
        if hasattr(array, "shape"):
            sharding = create_sharding_spec(array.shape, mesh, shard_batch_dim)
            distributed[key] = jax.device_put(array, sharding)
        else:
            distributed[key] = array

    return distributed


# Test distribution
if mesh is not None:
    test_batch = next(iter(pipeline))
    with mesh:
        sharded_batch = distribute_batch(test_batch, mesh)
        print("\nDistributed batch:")
        print(f"  Image shape: {sharded_batch['image'].shape}")
        print(f"  Image sharding: {sharded_batch['image'].sharding.spec}")
else:
    test_batch = next(iter(pipeline))
    print("\nSingle-device batch:")
    print(f"  Image shape: {test_batch['image'].shape}")

# %% [markdown]
"""
## Part 4: Sharded Training Loop Pattern

Complete pattern for distributed training with sharded data.
"""


# %%
def sharded_training_step(batch, _mesh):
    """Example sharded training step (computation only)."""
    # In a real training loop, this would be your model forward/backward pass
    images = batch["image"]

    # Simple computation for demonstration
    mean_value = jnp.mean(images)
    batch_size = images.shape[0]

    return {"mean": mean_value, "batch_size": batch_size}


# Run sharded iteration
print("\nSharded iteration example:")

pipeline = create_pipeline(num_samples=512)
metrics = []

if mesh is not None:
    with mesh:
        for i, batch in enumerate(pipeline):
            if i >= 5:
                break

            # Distribute batch
            sharded_batch = distribute_batch(batch, mesh)

            # Run sharded computation
            result = sharded_training_step(sharded_batch, mesh)
            metrics.append(result)

            print(f"Batch {i}: size={result['batch_size']}, mean={float(result['mean']):.4f}")
else:
    for i, batch in enumerate(pipeline):
        if i >= 5:
            break

        result = sharded_training_step(batch, None)
        metrics.append(result)

        print(f"Batch {i}: size={result['batch_size']}, mean={float(result['mean']):.4f}")

# %% [markdown]
"""
## Part 5: Throughput Analysis
"""

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)


def benchmark_pipeline(batch_size, num_batches=20, mesh=None):
    """Benchmark pipeline throughput."""
    pipeline = create_pipeline(batch_size=batch_size, num_samples=batch_size * num_batches)

    # Warmup
    warmup_batch = next(iter(pipeline))
    if mesh is not None:
        with mesh:
            _ = distribute_batch(warmup_batch, mesh)

    # Benchmark
    pipeline = create_pipeline(batch_size=batch_size, num_samples=batch_size * num_batches)

    start = time.time()
    samples = 0

    if mesh is not None:
        with mesh:
            for batch in pipeline:
                sharded = distribute_batch(batch, mesh)
                _ = sharded["image"].block_until_ready()
                samples += batch["image"].shape[0]
    else:
        for batch in pipeline:
            _ = batch["image"].block_until_ready()
            samples += batch["image"].shape[0]

    elapsed = time.time() - start
    return samples / elapsed


# Benchmark different batch sizes
batch_sizes = [32, 64, 128, 256]
throughputs = []

print("\nBenchmarking throughput:")
for bs in batch_sizes:
    tp = benchmark_pipeline(bs, mesh=mesh)
    throughputs.append(tp)
    print(f"  Batch size {bs}: {tp:.0f} samples/s")

# %%
# Plot throughput vs batch size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Throughput vs batch size
axes[0].plot(batch_sizes, throughputs, "o-", linewidth=2, markersize=8)
axes[0].set_xlabel("Batch Size")
axes[0].set_ylabel("Throughput (samples/second)")
axes[0].set_title(f"Data Loading Throughput ({num_devices} device(s))")
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale("log", base=2)

# Throughput as bar chart with values labeled
bars = axes[1].bar([str(bs) for bs in batch_sizes], throughputs, color="steelblue")
axes[1].set_xlabel("Batch Size")
axes[1].set_ylabel("Throughput (samples/second)")
axes[1].set_title("Throughput by Batch Size")

# Label each bar with its value
for bar, tp in zip(bars, throughputs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, tp + 50, f"{tp:,.0f}", ha="center", fontsize=9)

axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    output_dir / "dist-sharding-throughput-scaling.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'dist-sharding-throughput-scaling.png'}")

# %% [markdown]
"""
## Part 6: Memory Analysis

Understanding memory distribution across devices.
"""

# %%
# Simulate memory distribution
batch_size = 128
image_shape = (32, 32, 3)
float_bytes = 4  # float32

# Memory per batch
image_memory = batch_size * np.prod(image_shape) * float_bytes
label_memory = batch_size * 4  # int32
batch_memory = image_memory + label_memory

# Memory per device (sharded)
memory_per_device_sharded = batch_memory / max(num_devices, 1)
memory_per_device_replicated = batch_memory

print(f"Memory analysis (batch_size={batch_size}):")
print(f"  Image memory per batch: {image_memory / 1e6:.2f} MB")
print(f"  Total batch memory: {batch_memory / 1e6:.2f} MB")
print(f"  Memory per device (sharded): {memory_per_device_sharded / 1e6:.2f} MB")
print(f"  Memory per device (replicated): {memory_per_device_replicated / 1e6:.2f} MB")
print(f"  Memory savings from sharding: {(1 - 1 / max(num_devices, 1)) * 100:.0f}%")

# %%
# Plot memory distribution
fig, ax = plt.subplots(figsize=(10, 6))

device_labels = [f"Device {i}" for i in range(max(num_devices, 2))]
sharded_memory = [memory_per_device_sharded / 1e6] * max(num_devices, 2)
replicated_memory = [memory_per_device_replicated / 1e6] * max(num_devices, 2)

x = np.arange(len(device_labels))
width = 0.35

bars1 = ax.bar(x - width / 2, sharded_memory, width, label="Sharded", color="steelblue")
bars2 = ax.bar(
    x + width / 2, replicated_memory, width, label="Replicated", color="coral", alpha=0.7
)

ax.set_ylabel("Memory (MB)")
ax.set_xlabel("Device")
ax.set_title("Memory Distribution: Sharded vs Replicated")
ax.set_xticks(x)
ax.set_xticklabels(device_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    output_dir / "dist-sharding-memory-per-device.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'dist-sharding-memory-per-device.png'}")

# %% [markdown]
"""
## Part 7: Device Utilization Simulation
"""

# %%
# Simulate device utilization during data loading
num_steps = 50
device_utilization = np.zeros((max(num_devices, 2), num_steps))

# Simulate loading pattern
for step in range(num_steps):
    # Data loading phase (some variance)
    base_util = 0.7 + 0.2 * np.random.random()
    for dev in range(max(num_devices, 2)):
        # Add device-specific variance
        device_utilization[dev, step] = base_util + 0.1 * np.random.random()

# Plot utilization
fig, ax = plt.subplots(figsize=(12, 6))

for dev in range(min(max(num_devices, 2), 4)):  # Show up to 4 devices
    ax.plot(device_utilization[dev], label=f"Device {dev}", alpha=0.8)

ax.axhline(y=np.mean(device_utilization), color="red", linestyle="--", label="Mean")
ax.set_xlabel("Step")
ax.set_ylabel("Utilization")
ax.set_title("Device Utilization During Sharded Data Loading")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(
    output_dir / "dist-sharding-device-utilization.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'dist-sharding-device-utilization.png'}")

# %% [markdown]
"""
## Part 8: Batch Distribution Visualization
"""

# %%
# Visualize how batches are distributed across devices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Batch distribution diagram
ax1 = axes[0]
batch_data = np.random.rand(BATCH_SIZE // max(num_devices, 1), max(num_devices, 2))

im = ax1.imshow(batch_data.T, aspect="auto", cmap="viridis")
ax1.set_ylabel("Device")
ax1.set_xlabel("Sample Index (within device)")
ax1.set_title("Batch Distribution Across Devices")
ax1.set_yticks(range(max(num_devices, 2)))
ax1.set_yticklabels([f"Dev {i}" for i in range(max(num_devices, 2))])
plt.colorbar(im, ax=ax1, label="Data Value")

# Samples per device bar chart
ax2 = axes[1]
samples_per_dev = [BATCH_SIZE // max(num_devices, 1)] * max(num_devices, 2)
ax2.bar(range(max(num_devices, 2)), samples_per_dev, color="steelblue")
ax2.set_xlabel("Device")
ax2.set_ylabel("Samples")
ax2.set_title(f"Samples per Device (Total: {BATCH_SIZE})")
ax2.set_xticks(range(max(num_devices, 2)))
ax2.set_xticklabels([f"Dev {i}" for i in range(max(num_devices, 2))])

for i, v in enumerate(samples_per_dev):
    ax2.text(i, v + 1, str(v), ha="center")

plt.tight_layout()
plt.savefig(
    output_dir / "dist-sharding-batch-distribution.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'dist-sharding-batch-distribution.png'}")

# %% [markdown]
"""
## Results Summary

### Sharding Strategies

| Strategy | Use Case | Mesh Shape |
|----------|----------|------------|
| Pure Data Parallel | Most common | (N,) "data" |
| 2D Data + Model | Large models | (D, M) "data", "model" |
| Pipeline Parallel | Very long sequences | (P,) "pipeline" |

### Performance Guidelines

| Batch Size | Recommendation |
|------------|----------------|
| < 32 | Overhead may exceed benefit |
| 64-256 | Good balance |
| > 256 | Check memory constraints |

### Key Takeaways

1. **Batch size**: Should be divisible by device count
2. **Memory**: Sharding reduces per-device memory linearly
3. **Overhead**: Distribution has fixed cost - larger batches amortize it
4. **Mesh context**: All sharded operations must be within mesh context
5. **Fallback**: Code should handle single-device gracefully
"""

# %% [markdown]
"""
## Next Steps

- **Checkpointing**: [Resumable training](../checkpointing/02_resumable_training_guide.ipynb)
- **Performance**: [Optimization guide](../performance/01_optimization_guide.ipynb)
- **Full training**: [End-to-end CIFAR-10](../training/01_e2e_cifar10_guide.ipynb)
"""


# %%
def main():
    """Run the distributed sharding guide."""
    print("Distributed Data Loading with Sharding Guide")
    print("=" * 50)

    devices = jax.devices()
    num_devices = len(devices)
    print(f"Devices available: {num_devices}")

    # Create pipeline
    pipeline = create_pipeline(batch_size=64, num_samples=256)

    # Process with optional sharding
    total_samples = 0
    if num_devices >= 2:
        mesh = Mesh(np.array(devices), axis_names=("data",))
        with mesh:
            for batch in pipeline:
                sharded = distribute_batch(batch, mesh)
                total_samples += sharded["image"].shape[0]
    else:
        for batch in pipeline:
            total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples")
    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
