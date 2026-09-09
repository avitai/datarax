# Distributed

!!! warning "Moved to substrax"
    Device detection and placement, device meshes, SPMD data parallelism and
    cross-device metric reduction live in
    [substrax](https://github.com/avitai/substrax), which datarax depends on.
    The `datarax.distributed` package is gone as of 0.1.6; import from
    `substrax.devices`, `substrax.mesh` and `substrax.spmd` instead.

    `prefetch_to_device` is datarax's own. It lives in `datarax.control.prefetcher`
    and is exported from the package root.

## Where each name went

| datarax 0.1.5 | datarax 0.1.6 |
|---------------|---------------|
| `DevicePlacement`, `HardwareType`, `BatchSizeRecommendation`, `place_on_device`, `distribute_batch`, `get_batch_size_recommendation` | [`substrax.devices`](https://github.com/avitai/substrax/blob/main/docs/api/devices.md) |
| `DeviceMeshManager`, `MeshRules`, `data_parallel_rules`, `fsdp_rules`, `create_named_sharding`, `partition_spec_for_names` | [`substrax.mesh`](https://github.com/avitai/substrax/blob/main/docs/api/mesh.md) |
| `create_data_parallel_sharding`, `place_batch_on_shards`, `place_nnx_state_on_shards`, `spmd_train_step`, `reduce_gradient_tree`, the `reduce_*` and `*_collective` functions, `all_gather`, `collect_from_devices` | [`substrax.spmd`](https://github.com/avitai/substrax/blob/main/docs/api/spmd.md) |
| `prefetch_to_device` | `datarax.control.prefetcher` (also `from datarax import prefetch_to_device`) |
| `DevicePlacement.prefetch_to_device` | The `prefetch_to_device` function |
| `data_parallel_train_step`, `place_model_state_on_shards`, `reduce_gradients_across_devices` | Removed. Use `spmd_train_step`, `place_nnx_state_on_shards` and `reduce_gradient_tree`. |

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
