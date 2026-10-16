# Sharding

Datarax slices data for the current JAX process, and a pipeline given a `sharding` places its
batches on that data mesh. Building meshes, naming mesh axes, placing other arrays and
partitioning parameters come from substrax and Flax NNX.

| Need | API |
|------|-----|
| Slice records for the current JAX process | `datarax.sharding.JaxProcessSharderModule` |
| A data-parallel mesh and its batch sharding | `substrax.mesh.DeviceMeshManager`, `substrax.spmd.create_data_parallel_sharding` |
| Place a pipeline's batches on the mesh | `Pipeline(..., sharding=create_data_parallel_sharding(mesh))` |
| Place another batch or array on the mesh | `substrax.spmd.place_batch_on_shards` |
| Map logical axis names to mesh axes | `substrax.mesh.MeshRules`, `substrax.mesh.partition_spec_for_names` |
| Apply a function to each shard | `flax.nnx.shard_map` |
| Create a parameter with a partitioning annotation | `flax.nnx.with_partitioning` |

## Place Pipeline Batches on a Mesh

Give the pipeline the sharding of a batch's rows. Iteration (`for batch in pipeline` and
`pipeline.raw_batches()`) then places every batch on the mesh as it serves it:

```python
import jax
from flax import nnx
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding

from datarax import Pipeline

mesh = DeviceMeshManager.create_data_parallel_mesh()
pipeline = Pipeline(
    source=source,
    stages=stages,
    batch_size=256,
    rngs=nnx.Rngs(0),
    shuffle=True,
    sharding=create_data_parallel_sharding(mesh),
)

with jax.set_mesh(mesh):
    for batch in pipeline:
        loss = train_step(model, optimizer, batch)
```

- A full batch has its rows on the sharding and its batch-level state (MixUp's `lam`)
  replicated. A `raw_batches(chunk=K)` chunk `(K, B, ...)` has its rows on the axes after the
  chunk axis, `P(None, "data")`.
- A run bounded by `num_epochs` with `drop_last=False` ends on a short batch, which the data
  axis cannot split. The pipeline places it replicated, so an evaluation covers every record
  once with no padding row. Its step compiles once more, once per process, never per
  evaluation.
- Replicated, that final batch puts all of its rows, up to `batch_size - 1`, on every device,
  where a full batch puts `batch_size / N` on each of `N` devices. Its step can need up to `N`
  times the per-device memory of a full batch's step: for a convolutional evaluation step on 8
  emulated CPU devices with `batch_size=256`, a 255-row final batch took 7.97 times a full
  batch's per-device bytes in XLA's memory analysis of the compiled step. Size an evaluation so
  that `batch_size` rows fit one device's step, or choose a `batch_size` that divides the run's
  record count (`num_epochs` times the source's length), so the run ends on a full batch.
- The batch size must split over the sharding's row axes; one that does not is refused when the
  pipeline is built.
- The mesh's axes must be `AxisType.Auto`. `DeviceMeshManager` builds Auto axes; `jax.make_mesh`
  on jax 0.11 builds Explicit ones unless given `axis_types`. On Explicit axes an operator that
  reads another record (MixUp, CutMix) must name the sharding of what it gathers, which
  datarax's operators do not, so such a mesh is refused.
- The mesh is this process's devices. A mesh spanning several processes is refused: each
  process would place a whole batch of the same records as its slice.
- `step()`, `scan` and `session()` serve batches through the compiled step, which places nothing
  on a mesh, so they refuse a pipeline holding a sharding.

`place_batch_on_shards` places any other batch or array: every array leaf goes to one
`jax.make_array_from_process_local_data` call, on one process a single batched transfer.

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
