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
| `torch.distributed.init_process_group()` | `Mesh(devices, axis_names)` |
| `sampler.set_epoch(epoch)` | RNG-based shuffling per device |

**Key difference:** Datarax uses JAX's built-in GSPMD for transparent sharding without explicit communication.

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tf.distribute.MirroredStrategy` | `Mesh` with data axis |
| `strategy.experimental_distribute_dataset` | `jax.device_put(batch, sharding)` |
| `tf.distribute.Strategy.scope()` | `with mesh:` context |
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
        P[from_source<br/>batch_size=128]
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

### Step 1: Check Device Availability

```python
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

devices = jax.devices()
use_sharding = len(devices) >= 2

print(f"JAX devices: {devices}")
print(f"Device count: {len(devices)}")
```

**Terminal Output:**
```
JAX devices: [cuda:0, cuda:1]
Device count: 2
```

### Step 2: Create Pipeline

Standard pipeline setup - sharding is applied at the mesh level:

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

data = {
    "image": np.random.rand(1024, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, (1024,)).astype(np.int32),
}

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = from_source(source, batch_size=128)

print(f"Pipeline: {len(source)} samples, batch_size=128")
```

**Terminal Output:**
```
Pipeline: 1024 samples, batch_size=128
```

### Step 3: Create Device Mesh

```python
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Create mesh for data parallelism
device_mesh = np.array(devices).reshape(-1)
mesh = Mesh(device_mesh, axis_names=("data",))

# Define sharding specs
# Batch dimension sharded, others replicated
data_sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))
label_sharding = NamedSharding(mesh, PartitionSpec("data"))

print(f"Mesh: {len(device_mesh)} devices along 'data' axis")
```

**Terminal Output:**
```
Mesh: 2 devices along 'data' axis
```

### Step 4: Process with Sharding

```python
with mesh:
    for i, batch in enumerate(pipeline):
        if i >= 2:
            break

        # Apply sharding to batch
        image_batch = jax.device_put(batch["image"], data_sharding)
        label_batch = jax.device_put(batch["label"], label_sharding)

        print(f"Batch {i}:")
        print(f"  Image sharding: {image_batch.sharding}")
```

**Terminal Output:**
```
Batch 0:
  Image sharding: NamedSharding(mesh=Mesh('data': 2), spec=PartitionSpec('data',))
Batch 1:
  Image sharding: NamedSharding(mesh=Mesh('data': 2), spec=PartitionSpec('data',))
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
