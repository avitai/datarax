# Sharding

A pipeline given a `sharding` places its batches on that data mesh, which may span processes.
Building meshes, naming mesh axes, placing other arrays and partitioning parameters come from
substrax and Flax NNX.

| Need | API |
|------|-----|
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

- A full batch has its rows on the sharding, its mix (`partners`, `lams`) included, and its
  batch-level state replicated. A `raw_batches(chunk=K)` chunk `(K, B, ...)` has its rows on the
  axes after the chunk axis, `P(None, "data")`.
- A run bounded by `num_epochs` with `drop_last=False` ends on a short batch, which the data
  axis cannot split. The pipeline places it replicated, so an evaluation covers every record
  once with no padding row. Its step compiles once more, once per process, never per
  evaluation.
- Replicated, that final batch puts all of its rows, up to `batch_size - 1`, on every device,
  where a full batch puts `batch_size / N` on each of `N` devices. Its step can need up to `N`
  times the per-device memory of a full batch's step: for a convolutional evaluation step on 8
  emulated CPU devices with `batch_size=256`, a 255-row final batch took 7.97 times a full
  batch's per-device bytes in XLA's memory analysis of the compiled step, and on two A100 GPUs
  with `batch_size=1024` a 1023-row final batch took 2.0 times. Size an evaluation so
  that `batch_size` rows fit one device's step, or choose a `batch_size` that divides the run's
  record count (`num_epochs` times the source's length), so the run ends on a full batch.
- The batch size must split over the sharding's row axes; one that does not is refused when the
  pipeline is built.
- Iterate in the same `jax.set_mesh(mesh)` context you warmed up in. The context mesh is part of
  every jitted function's cache key, so a pipeline first iterated outside `jax.set_mesh` compiles
  its DAG again on its first batch inside it, and a timing that starts there measures that
  compile.
- The mesh's axes must be `AxisType.Auto`. `DeviceMeshManager` builds Auto axes; `jax.make_mesh`
  on jax 0.11 builds Explicit ones unless given `axis_types`. On Explicit axes an operator that
  reads another record (MixUp, CutMix) must name the sharding of what it gathers, which
  datarax's operators do not, so such a mesh is refused.
- The mesh may span processes (`jax.distributed`, one process per GPU). `batch_size` is then
  the global batch: every process builds the same pipeline and reads only the rows its own
  devices hold of each global batch, so the batches, their per-record draws and MixUp's batch
  draws are the single-process batches row for row. The short final batch is read whole by
  every process and placed replicated. A chunk holds each process's rows of its `K` batches,
  and a unit's provenance names the records this process read.
- Across processes the source is an `INDEXED` source of all its records. A stream (read
  forward, with no position to read a process's rows by) and a source serving a worker's shard
  (`shard_id`, `num_workers`) are refused when the pipeline is built, and a run whose
  configuration, cursor or mesh differs between processes is refused in every process when it
  opens.
- Across processes, worker processes (`host_resources` for a GIL-bound read) need each
  process's share of `/dev/shm`, `HostResources(shm_budget_bytes=...)`: every process on a
  machine sees the same free bytes, and each would plan against all of them.
- `HostResources` is one process's budget: with one process per GPU, each process passes its
  share of the machine's RAM and CPUs, not the machine's totals.
- A source that holds its records in memory (`MemorySource`, an eager TFDS source) holds them
  in every process, so N processes hold N copies. A decoded copy reads its columns through
  memory maps, which the processes share in the machine's page cache.
- Save and restore a pipeline's state on every process at the same step: the checkpoint
  store's save and restore are collective across the processes.
- `step()`, `scan` and `session()` serve batches through the compiled step, which places nothing
  on a mesh, so they refuse a pipeline holding a sharding.

`place_batch_on_shards` places any other batch or array: every array leaf goes to one
`jax.make_array_from_process_local_data` call, on one process a single batched transfer.

## See Also

- [Distributed](../distributed/index.md) - Distributed training
- [Distributed Training Guide](../user_guide/distributed_training.md)
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
