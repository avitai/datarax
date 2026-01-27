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
# Sharded Pipeline Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline, JAX sharding concepts |
| **Format** | Python + Jupyter |

## Overview

Distribute data processing across multiple JAX devices using Datarax sharding.
This enables efficient utilization of multi-GPU setups for large-scale data
pipelines, essential for training on large datasets.

## Learning Goals

By the end of this example, you will be able to:

1. Create a JAX device mesh for multi-device execution
2. Configure Datarax pipelines for sharded data distribution
3. Verify data is properly distributed across devices
4. Handle single-device fallback gracefully
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax
uv pip install datarax
```

**Note**: Multi-GPU sharding requires at least 2 JAX devices.
Single-device systems will run in fallback mode.
"""

# %%
# Imports
import jax
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {len(jax.devices())}")

# %% [markdown]
"""
## Step 1: Check Device Availability

Sharding requires multiple JAX devices. The example gracefully
handles single-device environments.
"""

# %%
# Check device availability
devices = jax.devices()
use_sharding = len(devices) >= 2

if use_sharding:
    print(f"Multi-device mode: {len(devices)} devices available")
    print(f"Devices: {[str(d) for d in devices]}")
else:
    print(f"Single-device mode: Only {len(devices)} device(s) found")
    print("Sharding demo will show concepts without actual distribution")

# %% [markdown]
"""
## Step 2: Create Data and Pipeline

Standard pipeline setup - the sharding is applied at the mesh level,
not changing how you define sources or operators.
"""

# %%
# Create sample data
num_samples = 1024
data = {
    "image": np.random.rand(num_samples, 32, 32, 3).astype(np.float32),
    "feature": np.random.rand(num_samples, 128).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

# Create source
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

print(f"Data samples: {num_samples}")
print(f"Image shape per sample: {data['image'].shape[1:]}")


# %%
# Define normalization operator
def normalize(element, key=None):
    """Normalize image to [0, 1] range."""
    return element.update_data({"image": element.data["image"] / 255.0})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
)

# Build pipeline
pipeline = from_source(source, batch_size=128).add(OperatorNode(normalizer))

print("Pipeline created with batch_size=128")

# %% [markdown]
"""
## Step 3: Device Mesh Setup

A JAX `Mesh` defines how devices are organized. Common patterns:

- `("data",)` - Data parallelism across all devices
- `("data", "model")` - 2D mesh for data + model parallelism
"""

# %%
# Create device mesh
if use_sharding:
    # Reshape devices into a mesh
    # For data parallelism: all devices along "data" axis
    device_mesh = np.array(devices).reshape(-1)
    mesh = Mesh(device_mesh, axis_names=("data",))
    print(f"Created mesh with {len(device_mesh)} devices along 'data' axis")

    # Define partition spec for batched data
    # batch dimension sharded across "data" axis, others replicated
    data_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))
    label_sharding = NamedSharding(mesh, PartitionSpec("data"))
else:
    mesh = None
    print("Skipping mesh creation (single device)")

# %% [markdown]
"""
## Step 4: Process with Sharding

When running inside a mesh context, JAX operations automatically
use the sharded execution.
"""

# %%
# Process batches
print("\nProcessing batches:")

if use_sharding and mesh is not None:
    with mesh:
        for i, batch in enumerate(pipeline):
            if i >= 2:
                break

            # Apply sharding to batch data
            image_batch = jax.device_put(batch["image"], data_sharding)
            label_batch = jax.device_put(batch["label"], label_sharding)

            print(f"Batch {i}:")
            print(f"  Image shape: {image_batch.shape}")
            print(f"  Image sharding: {image_batch.sharding}")
            print(f"  Label shape: {label_batch.shape}")
else:
    # Single device fallback
    for i, batch in enumerate(pipeline):
        if i >= 2:
            break

        print(f"Batch {i}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Label shape: {batch['label'].shape}")
        print("  (Running on single device)")

# Expected output (multi-GPU):
# Batch 0:
#   Image shape: (128, 32, 32, 3)
#   Image sharding: NamedSharding(mesh=..., spec=PartitionSpec('data',))
#   Label shape: (128,)

# %% [markdown]
"""
## Results Summary

| Feature | Value |
|---------|-------|
| Device Count | Depends on system |
| Mesh Shape | (N,) for N devices |
| Data Parallelism | Batch dimension sharded |
| Fallback | Single-device execution |

Sharding benefits:

- **Memory efficiency**: Data distributed across device memories
- **Throughput**: Parallel preprocessing on multiple devices
- **Scalability**: Easily scales with more devices
"""

# %% [markdown]
"""
## Next Steps

- **Advanced sharding**: Explore model parallelism for large models
- **TPU sharding**: Configure meshes for TPU pod slices
- **Pipeline parallelism**: Overlap data loading and computation
- **Checkpointing**: [Checkpointing](../checkpointing/01_checkpoint_quickref.ipynb)
"""


# %%
def main():
    """Run the sharded pipeline example."""
    print("Sharded Pipeline Example")
    print("=" * 50)

    # Check devices
    devices = jax.devices()
    use_sharding = len(devices) >= 2
    print(f"Devices: {len(devices)}, Sharding: {use_sharding}")

    # Create data and pipeline
    num_samples = 1024
    data = {
        "image": np.random.rand(num_samples, 32, 32, 3).astype(np.float32),
        "feature": np.random.rand(num_samples, 128).astype(np.float32),
        "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
    }

    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )
    pipeline = from_source(source, batch_size=128).add(OperatorNode(normalizer))

    # Process batches
    total_samples = 0
    for i, batch in enumerate(pipeline):
        if i >= 5:
            break
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
