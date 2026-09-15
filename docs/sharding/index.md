# Sharding

Datarax slices data for the current JAX process. Placing batches on a device mesh, naming mesh
axes and partitioning parameters come from substrax and Flax NNX.

| Need | API |
|------|-----|
| Slice records for the current JAX process | `datarax.sharding.JaxProcessSharderModule` |
| A data-parallel mesh and its batch sharding | `substrax.mesh.DeviceMeshManager`, `substrax.spmd.create_data_parallel_sharding` |
| Place a batch on the mesh | `substrax.spmd.place_batch_on_shards` |
| Map logical axis names to mesh axes | `substrax.mesh.MeshRules`, `substrax.mesh.partition_spec_for_names` |
| Apply a function to each shard | `flax.nnx.shard_map` |
| Create a parameter with a partitioning annotation | `flax.nnx.with_partitioning` |

## Place Pipeline Batches on a Mesh

`place_batch_on_shards` hands every array leaf of a batch to one
`jax.make_array_from_process_local_data` call. On one process that is a single batched
transfer; on several, each process passes the batch it loaded and jax assembles the global
batch. The batch size must be a multiple of the number of devices on the `data` axis.

```python
import jax
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards

mesh = DeviceMeshManager.create_data_parallel_mesh()
sharding = create_data_parallel_sharding(mesh)

with jax.set_mesh(mesh):
    for batch in pipeline:
        batch = place_batch_on_shards(batch, sharding)
        loss = train_step(model, optimizer, batch)
```

## Slice Data per Process

```python
from datarax.sharding import JaxProcessSharderModule

# The module reads the process index and count from Grain's ShardByJaxProcess.
sharder = JaxProcessSharderModule()

# ``shard_data`` slices arrays, lists and tuples to this process's portion.
local_images = sharder.shard_data(global_images)
local_labels = sharder.shard_data(global_labels)
```

## Modules

- [jax_process_sharder](jax_process_sharder.md) - Process-level data slicing

## See Also

- [Distributed](../distributed/index.md) - Distributed training
- [Distributed Training Guide](../user_guide/distributed_training.md)
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
