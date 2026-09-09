# Distributed Training

This guide shows how to train on several devices or hosts with a datarax pipeline.

## Overview

The device, mesh and SPMD utilities live in
[substrax](https://github.com/avitai/substrax), the infrastructure package datarax
depends on. They allow for:

- Data-parallel training across multiple devices
- Model-parallel training for large models
- Hybrid parallelism combining both approaches
- Distributed metrics collection and aggregation

## Distributed Components

Three groups of APIs cover distributed training: a mesh manager
(`substrax.mesh`), data-parallel functions and metrics functions
(`substrax.spmd`).

### DeviceMeshManager

`DeviceMeshManager` handles JAX device mesh creation and management through
static methods (no instance is required):

```python
from substrax.mesh import DeviceMeshManager

# Create a data-parallel mesh
mesh = DeviceMeshManager.create_data_parallel_mesh()

# Or create a model-parallel mesh
model_mesh = DeviceMeshManager.create_model_parallel_mesh(num_devices=4)

# Or create a hybrid mesh
hybrid_mesh = DeviceMeshManager.create_hybrid_mesh(
    data_parallel_size=2,
    model_parallel_size=4,
)

# Get information about the mesh
mesh_info = DeviceMeshManager.get_mesh_info(mesh)
print(f"Mesh info: {mesh_info}")
```

### Data-parallel functions

The data-parallel functions build sharding specifications and place data and
model state across devices:

```python
import flax.nnx as nnx
from substrax.mesh import DeviceMeshManager
from substrax.spmd import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    place_nnx_state_on_shards,
    reduce_gradient_tree,
)

# Create a data-parallel mesh
mesh = DeviceMeshManager.create_data_parallel_mesh()

# Create sharding specification for data parallelism
sharding = create_data_parallel_sharding(mesh)

# Shard a batch across devices
sharded_batch = place_batch_on_shards(batch, sharding)

# Place the model's NNX state on the mesh (replicated by default)
sharded_state = place_nnx_state_on_shards(nnx.state(model), mesh)

# Reduce a gradient tree across devices (inside nnx.jit with an active mesh)
reduced_grads = reduce_gradient_tree(gradients, reduce_type="mean")
```

### Metrics functions

The metrics functions aggregate values across devices. Two variants are
provided:

- **SPMD functions** (`reduce_mean`, `reduce_sum`, `reduce_custom`, ...) operate
  on global arrays and work inside `nnx.jit` with an active mesh. They take no
  `axis_name`.
- **Collective functions** (`reduce_mean_collective`, `reduce_sum_collective`,
  `all_gather`) use `lax.p*` collectives and are only valid inside a `pmap` or
  `shard_map` context. They accept an `axis_name`.

```python
from substrax.spmd import (
    collect_from_devices,
    reduce_custom,
    reduce_mean,
    reduce_sum,
)

# Compute mean of metrics across devices (SPMD)
reduced_metrics = reduce_mean(metrics)

# Compute sum of metrics across devices (SPMD)
sum_metrics = reduce_sum(metrics)

# Apply custom per-metric reduction operations
custom_metrics = reduce_custom(
    metrics,
    reduce_fn={
        "loss": "mean",
        "accuracy": "mean",
        "step": "max",
    },
)

# Split stacked per-device metrics into per-device lists
device_metrics = collect_from_devices(metrics)
```

## Example: Data-Parallel Training

Here's a simple example of data-parallel training using the SPMD path
(`nnx.jit` with an active mesh). Parameters are replicated across devices and
the batch is sharded along the data axis; the XLA compiler handles gradient
all-reduce automatically.

```python
import flax.nnx as nnx
import jax
import optax
from substrax.mesh import DeviceMeshManager
from substrax.spmd import create_data_parallel_sharding, place_batch_on_shards

# Create the device mesh and data-parallel sharding
mesh = DeviceMeshManager.create_data_parallel_mesh()
sharding = create_data_parallel_sharding(mesh)

# Define model and optimizer
model = MyNNXModel()
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3), wrt=nnx.Param)


@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        # Call the model directly on the batch
        logits = model(batch["inputs"])
        return compute_loss(logits, batch["targets"])

    # Compute loss and gradients; the compiler all-reduces sharded grads
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update parameters in place
    optimizer.update(model, grads)
    return loss


# Train for multiple steps under the mesh context
with jax.set_mesh(mesh):
    for step in range(num_steps):
        batch = load_data_batch()
        sharded_batch = place_batch_on_shards(batch, sharding)
        loss = train_step(model, optimizer, sharded_batch)
```

The training-step body above is also available as the `spmd_train_step`
convenience function in `substrax.spmd`, which wraps `nnx.value_and_grad` and
`optimizer.update`.

## Using with pmap and collectives

For the explicit `pmap` path, gradients and metrics must be reduced with the
collective functions inside the mapped function. Build the `pmap` with the
assignment form so the `axis_name` matches the collective reductions:

```python
import flax.nnx as nnx
import jax
from substrax.mesh import DeviceMeshManager
from substrax.spmd import (
    create_data_parallel_sharding,
    place_batch_on_shards,
    reduce_mean_collective,
)


def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch["inputs"])
        return compute_loss(logits, batch["targets"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Average gradients across devices with a collective reduction
    grads = reduce_mean_collective(grads, axis_name="batch")

    optimizer.update(model, grads)
    return loss


# Build the pmapped step with the assignment form (pmap already compiles)
train_step = jax.pmap(train_step, axis_name="batch")

# Shard the batch and run across devices
mesh = DeviceMeshManager.create_data_parallel_mesh()
sharding = create_data_parallel_sharding(mesh)
sharded_batch = place_batch_on_shards(load_data_batch(), sharding)
loss = train_step(model, optimizer, sharded_batch)
```

## Feeding the devices

`prefetch_to_device` overlaps host-to-device transfer with compute and is
datarax's own (`datarax.control.prefetcher`, also exported from the package
root):

```python
from datarax import prefetch_to_device

for batch in prefetch_to_device(pipeline, size=2):
    loss = train_step(model, optimizer, batch)
```

## Recommended Practices

1. **Scale batch size with device count** to maintain the effective batch size:
   ```python
   batch_size = per_device_batch_size * jax.device_count()
   ```

2. **Do not wrap `pmap` in `jit`** — `pmap` already compiles its function:
   ```python
   # pmap compiles on its own; no jax.jit wrapper needed
   train_step = jax.pmap(train_step_fn, axis_name="batch")
   ```

3. **Be consistent with axis names** when using `pmap` and collective reductions:
   ```python
   # Use the same axis_name in pmap and in the collective reduction
   train_step = jax.pmap(fn, axis_name="batch")
   reduced = reduce_mean_collective(values, axis_name="batch")
   ```

4. **Shard data correctly** to match the device arrangement:
   ```python
   mesh = DeviceMeshManager.create_data_parallel_mesh()
   sharding = create_data_parallel_sharding(mesh)
   sharded_batch = place_batch_on_shards(batch, sharding)
   ```

5. **Use SPMD metric reductions for accuracy** when reporting metrics under
   `nnx.jit`:
   ```python
   metrics = {"loss": loss, "accuracy": accuracy}
   reduced_metrics = reduce_mean(metrics)
   ```

## Next Steps

For complete examples, see the [examples section](../examples/overview.md):

- [Sharding Quick Reference](../examples/advanced/distributed/sharding-quickref.md) - JAX sharding basics

## See Also

- [Distributed](../distributed/index.md) - Where each name lives now
- [Sharding](../sharding/index.md) - Data sharding utilities
- [Performance Tools](../performance/index.md) - Optimization utilities
- [NNX Best Practices](nnx_best_practices.md) - JAX/Flax optimization tips
