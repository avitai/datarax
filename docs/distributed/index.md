# Distributed

Distributed training and multi-device data processing support. This module
provides tools for scaling Datarax pipelines across multiple GPUs, TPUs, and
hosts.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Device Placement** | Hardware detection | Auto-select best devices |
| **Device Mesh** | Mesh configuration | Multi-device layouts |
| **Data Parallel** | Data parallelism | Replicate across devices |
| **Metrics** | Distributed metrics | Aggregate across hosts |
| **Sharding** | Named-axis rules | Partition specs for a mesh |

!!! note "Key points"

    - JAX handles device placement automatically in most cases
    - Use `jax.devices()` to see available hardware
    - Device mesh enables advanced sharding patterns
    - Data parallelism is the simplest multi-device strategy

## Quick Start

```python
import jax
from datarax.distributed import get_batch_size_recommendation

# Check available devices
print(f"Devices: {jax.devices()}")

# Get a batch-size recommendation for the detected hardware
recommendation = get_batch_size_recommendation()
print(f"Recommended batch size: {recommendation.optimal_batch_size}")
```

## Modules

- [device_placement](device_placement.md) - Device detection and placement strategies
- [device_mesh](device_mesh.md) - Device mesh configuration for sharding
- [data_parallel](data_parallel.md) - Data parallelism patterns
- [metrics](metrics.md) - Distributed metrics collection and aggregation
- [sharding](sharding.md) - Named-axis sharding rules and partition specs

## Device Mesh Example

```python
from datarax.distributed import DeviceMeshManager

# Create a 2D mesh for data + model parallelism
mesh = DeviceMeshManager.create_device_mesh({"data": 2, "model": 4})
```

## Multi-Host Training

For multi-host setups, `JaxProcessSharderModule` derives the shard topology
from Grain's `ShardByJaxProcess`, so each process automatically slices its
local shard:

```python
# Each host runs this code
from datarax.sharding import JaxProcessSharderModule

sharder = JaxProcessSharderModule()

# Shard data across hosts (process index/count are auto-derived)
local_batch = sharder.shard_data(global_batch)
```

## See Also

- [Device Placement Guide](device_placement.md) - Detailed placement docs
- [Distributed Training Guide](../user_guide/distributed_training.md) - User guide
- [Sharding](../sharding/index.md) - Data sharding utilities
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
