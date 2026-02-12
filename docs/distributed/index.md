# Distributed

Distributed training and multi-device data processing support. This module provides tools for scaling Datarax pipelines across multiple GPUs, TPUs, and hosts.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Device Placement** | Hardware detection | Auto-select best devices |
| **Device Mesh** | Mesh configuration | Multi-device layouts |
| **Data Parallel** | Data parallelism | Replicate across devices |
| **Metrics** | Distributed metrics | Aggregate across hosts |

`★ Insight ─────────────────────────────────────`

- JAX handles device placement automatically in most cases
- Use `jax.devices()` to see available hardware
- Device mesh enables advanced sharding patterns
- Data parallelism is the simplest multi-device strategy

`─────────────────────────────────────────────────`

## Quick Start

```python
import jax
from datarax.distributed import get_device_placement_recommendation

# Check available devices
print(f"Devices: {jax.devices()}")

# Get placement recommendation
recommendation = get_device_placement_recommendation()
print(f"Recommended: {recommendation}")
```

## Modules

- [device_placement](device_placement.md) - Device detection and placement strategies
- [device_mesh](device_mesh.md) - Device mesh configuration for sharding
- [data_parallel](data_parallel.md) - Data parallelism patterns
- [metrics](metrics.md) - Distributed metrics collection and aggregation

## Device Mesh Example

```python
from datarax.distributed import create_device_mesh

# Create 2D mesh for data + model parallelism
mesh = create_device_mesh(
    devices=jax.devices(),
    mesh_shape=(2, 4),  # 2 data parallel, 4 model parallel
    axis_names=("data", "model"),
)
```

## Multi-Host Training

For multi-host setups:

```python
# Each host runs this code
from datarax.sharding import JaxProcessSharder

sharder = JaxProcessSharder(
    num_processes=jax.process_count(),
    process_index=jax.process_index(),
)

# Shard data across hosts
local_batch = sharder.shard(global_batch)
```

## See Also

- [Device Placement Guide](device_placement.md) - Detailed placement docs
- [Distributed Training Guide](../user_guide/distributed_training.md) - User guide
- [Sharding](../sharding/index.md) - Data sharding utilities
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
