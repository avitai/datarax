# Distributed

Device detection and placement, device meshes, SPMD data parallelism and cross-device metric
reduction come from [substrax](https://github.com/avitai/substrax), which datarax depends on:
[`substrax.devices`](https://github.com/avitai/substrax/blob/main/docs/api/devices.md),
[`substrax.mesh`](https://github.com/avitai/substrax/blob/main/docs/api/mesh.md) and
[`substrax.spmd`](https://github.com/avitai/substrax/blob/main/docs/api/spmd.md).
`prefetch_to_device` is datarax's own: it lives in `datarax.control.prefetcher` and is exported
from the package root.

## Quick Start

```python
import jax
from substrax.devices import get_batch_size_recommendation
from substrax.mesh import DeviceMeshManager

# Check available devices
print(f"Devices: {jax.devices()}")

# Get a batch-size recommendation for the detected hardware
recommendation = get_batch_size_recommendation()
print(f"Recommended batch size: {recommendation.optimal_batch_size}")

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

- [Distributed Training Guide](../user_guide/distributed_training.md) - User guide
- [Prefetcher](../control/prefetcher.md) - Host-to-device prefetching
- [Sharding](../sharding/index.md) - Data sharding utilities
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
