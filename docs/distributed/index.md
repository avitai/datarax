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

With one process per GPU (`jax.distributed`), every process builds the same pipeline with a
`sharding` over a mesh of all the processes' devices. `batch_size` is the global batch: each
process reads only the rows its own devices hold of every global batch, and the placement
assembles them, so every process's batches are the single-process batches row for row. The
source is an `INDEXED` source of all its records; [Sharding](../sharding/index.md) lists what is
refused across processes.

## See Also

- [Distributed Training Guide](../user_guide/distributed_training.md) - User guide
- [Prefetcher](../control/prefetcher.md) - Host-to-device prefetching
- [Sharding](../sharding/index.md) - Data sharding utilities
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
