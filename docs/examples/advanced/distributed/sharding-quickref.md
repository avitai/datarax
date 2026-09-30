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

## What You'll Learn

1. Create a JAX device mesh for multi-device execution
2. Configure Datarax pipelines for sharded data distribution
3. Verify data is properly distributed across devices
4. Handle single-device fallback gracefully

## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `DistributedSampler(dataset)` | JAX `Mesh` with `PartitionSpec` |
| `DataParallel(model)` | Data sharded along batch dimension |
| `torch.distributed.init_process_group()` | `DeviceMeshManager.create_data_parallel_mesh()` |
| `sampler.set_epoch(epoch)` | RNG-based shuffling per device |

**Key difference:** Datarax uses JAX's built-in GSPMD for transparent sharding without explicit communication.

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tf.distribute.MirroredStrategy` | `Mesh` with data axis |
| `strategy.experimental_distribute_dataset` | `jax.device_put(batch, sharding)` |
| `tf.distribute.Strategy.scope()` | `jax.set_mesh(mesh)` |
| `strategy.reduce()` | JAX handles via GSPMD |

## Files

- **Python Script**: [`examples/advanced/distributed/01_sharding_quickref.py`](https://github.com/avitai/datarax/blob/main/examples/advanced/distributed/01_sharding_quickref.py)
- **Jupyter Notebook**: [`examples/advanced/distributed/01_sharding_quickref.ipynb`](https://github.com/avitai/datarax/blob/main/examples/advanced/distributed/01_sharding_quickref.ipynb)

## Quick Start

```bash
python examples/advanced/distributed/01_sharding_quickref.py
```

## Architecture

```mermaid
flowchart TB
    subgraph Source["Data Source"]
        D[MemorySource<br/>1024 samples]
    end

    subgraph Pipeline["Pipeline"]
        P[Pipeline<br/>batch_size=128]
    end

    subgraph Mesh["Device Mesh"]
        direction LR
        G0[GPU 0<br/>batch[0:64]]
        G1[GPU 1<br/>batch[64:128]]
    end

    D --> P
    P --> G0
    P --> G1
```

## Key Concepts

The terminal output below is from a run on two GPUs (2x NVIDIA L40S), so each step takes its
multi-device branch. With a single device the same cells skip the mesh and run unsharded.

### Step 1: Check Device Availability

```python
# Check device availability
devices = jax.devices()
use_sharding = len(devices) >= 2

if use_sharding:
    print(f"Multi-device mode: {len(devices)} devices available")
    print(f"Devices: {[str(d) for d in devices]}")
else:
    print(f"Single-device mode: Only {len(devices)} device(s) found")
    print("Sharding demo will show concepts without actual distribution")
```

**Terminal Output:**
```
Multi-device mode: 2 devices available
Devices: ['cuda:0', 'cuda:1']
```

### Step 2: Create Data and Pipeline

Standard pipeline setup - sharding is applied at the mesh level, not by
changing how you define sources or operators:

```python
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
```

**Terminal Output:**
```
Data samples: 1024
Image shape per sample: (32, 32, 3)
```

```python
# Define normalization operator
def normalize(element, key=None):
    """Normalize image to [0, 1] range."""
    return element.update_data({"image": element.data["image"] / 255.0})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
)

# Build pipeline
pipeline = Pipeline(source=source, stages=[normalizer], batch_size=128, rngs=nnx.Rngs(0))

print("Pipeline created with batch_size=128")
```

**Terminal Output:**
```
Pipeline created with batch_size=128
```

### Step 3: Create Device Mesh

```python
# Create device mesh
if use_sharding:
    # Data parallelism: every device along the "data" axis
    mesh = DeviceMeshManager.create_data_parallel_mesh()
    print(f"Created mesh with {mesh.devices.size} devices along 'data' axis")
else:
    mesh = None
    print("Skipping mesh creation (single device)")
```

**Terminal Output:**
```
Created mesh with 2 devices along 'data' axis
```

### Step 4: Process with Sharding

```python
# Process batches
print("\nProcessing batches:")

if use_sharding and mesh is not None:
    # The batch dimension of every array is split across the "data" axis;
    # the remaining dimensions are replicated.
    batch_sharding = create_data_parallel_sharding(mesh)

    with jax.set_mesh(mesh):
        for i, batch in enumerate(pipeline):
            if i >= 2:
                break

            # Place every array of the batch on the mesh
            sharded_batch = place_batch_on_shards(batch, batch_sharding)

            print(f"Batch {i}:")
            print(f"  Image shape: {sharded_batch['image'].shape}")
            print(f"  Image sharding: {sharded_batch['image'].sharding}")
            print(f"  Label shape: {sharded_batch['label'].shape}")
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
#   Image sharding: NamedSharding(mesh=Mesh('data': 2, axis_types=(Auto,)), spec=P('data',), memory_kind=device)  # noqa: E501
#   Label shape: (128,)
```

**Terminal Output:**
```
Processing batches:
Batch 0:
  Image shape: (128, 32, 32, 3)
  Image sharding: NamedSharding(mesh=Mesh('data': 2, axis_types=(Auto,)), spec=P('data',), memory_kind=device)
  Label shape: (128,)
Batch 1:
  Image shape: (128, 32, 32, 3)
  Image sharding: NamedSharding(mesh=Mesh('data': 2, axis_types=(Auto,)), spec=P('data',), memory_kind=device)
  Label shape: (128,)
```

## Mesh Configurations

| Pattern | Mesh Shape | Use Case |
|---------|------------|----------|
| Data Parallel | `("data",)` | Replicate model, shard data |
| Model Parallel | `("model",)` | Shard model, replicate data |
| Hybrid | `("data", "model")` | Large models + large batches |

## Results Summary

| Feature | Value |
|---------|-------|
| Device Count | Depends on system |
| Mesh Shape | (N,) for N devices |
| Data Parallelism | Batch dimension sharded |
| Fallback | Single-device execution |

**Sharding benefits:**

- **Memory efficiency**: Data distributed across device memories
- **Throughput**: Parallel preprocessing on multiple devices
- **Scalability**: Easily scales with more devices

## Next Steps

- [Sharding Guide](sharding-guide.md) - Advanced sharding patterns
- [Checkpointing](../checkpointing/checkpoint-quickref.md) - Save distributed state
- [Performance Guide](../performance/optimization-guide.md) - Optimize throughput
- [API Reference: Sharding](../../../sharding/index.md) - Complete API
