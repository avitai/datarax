# Distributed Training with NNX Modules

This guide shows how to use Datarax's NNX-based distributed training components for multi-device and multi-host training.

## Overview

Datarax provides NNX-based modules for distributed training that leverage JAX's powerful distributed computing capabilities. These modules allow for:

- Data-parallel training across multiple devices
- Model-parallel training for large models
- Hybrid parallelism combining both approaches
- Distributed metrics collection and aggregation

## NNX-based Distributed Components

Datarax provides three main NNX modules for distributed training:

### DeviceMeshModule

The `DeviceMeshModule` handles JAX device mesh creation and management:

```python
from datarax.distributed import DeviceMeshModule

# Create the mesh module
mesh_module = DeviceMeshModule()

# Create a data-parallel mesh
mesh = mesh_module.create_data_parallel_mesh()

# Or create a model-parallel mesh
model_mesh = mesh_module.create_model_parallel_mesh(num_devices=4)

# Or create a hybrid mesh
hybrid_mesh = mesh_module.create_hybrid_mesh(
    data_parallel_size=2,
    model_parallel_size=4
)

# Get information about the mesh
mesh_info = mesh_module.get_mesh_info(mesh)
print(f"Mesh info: {mesh_info}")
```

### DataParallelModule

The `DataParallelModule` provides utilities for data-parallel training:

```python
from datarax.distributed import DataParallelModule, DeviceMeshModule

# Create the modules
mesh_module = DeviceMeshModule()
dp_module = DataParallelModule()

# Create a data-parallel mesh
mesh = mesh_module.create_data_parallel_mesh()

# Create sharding specification for data parallelism
sharding = dp_module.create_data_parallel_sharding(mesh)

# Shard a batch across devices
sharded_batch = dp_module.shard_batch(batch, sharding)

# Shard model state across devices
sharded_state = dp_module.shard_model_state(state, mesh)

# Reduce gradients across devices
reduced_grads = dp_module.all_reduce_gradients(gradients, reduce_type="mean")
```

### DistributedMetricsModule

The `DistributedMetricsModule` handles metrics collection and aggregation:

```python
from datarax.distributed import DistributedMetricsModule

# Create the metrics module
metrics_module = DistributedMetricsModule()

# Compute mean of metrics across devices
reduced_metrics = metrics_module.reduce_mean(metrics)

# Compute sum of metrics across devices
sum_metrics = metrics_module.reduce_sum(metrics)

# Apply custom reduction operations
custom_metrics = metrics_module.reduce_custom(
    metrics,
    reduce_fn={
        "loss": "mean",
        "accuracy": "mean",
        "step": "max",
    }
)

# Collect metrics from all devices
device_metrics = metrics_module.collect_from_devices(metrics)
```

## Example: Data-Parallel Training

Here's a simple example of data-parallel training with Datarax's NNX-based distributed components:

```python
import flax.nnx as nnx
import jax
import optax

from datarax.distributed import (
    DataParallelModule,
    DeviceMeshModule,
    DistributedMetricsModule,
)

# Initialize modules
mesh_module = DeviceMeshModule()
dp_module = DataParallelModule()
metrics_module = DistributedMetricsModule()

# Create device mesh
mesh = mesh_module.create_data_parallel_mesh()
sharding = dp_module.create_data_parallel_sharding(mesh)

# Define model and optimizer
model = MyNNXModel()
optimizer = optax.adam(learning_rate=1e-3)

# Create training state
state = TrainingState(model=model, optimizer=optimizer)

# Load data and shard it
batch = load_data_batch()
sharded_batch = dp_module.shard_batch(batch, sharding)

# Define a pmapped training step
@jax.pmap(axis_name="batch")
def train_step(state, batch):
    def loss_fn(params):
        # Forward pass
        outputs = state.model.apply(params, batch["inputs"])
        loss = compute_loss(outputs, batch["targets"])
        return loss

    # Compute gradients
    grads = jax.grad(loss_fn)(state.params)

    # Average gradients across devices
    grads = metrics_module.reduce_mean(grads, axis_name="batch")

    # Update parameters
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)

    # Update state
    new_state = state.replace(params=new_params, opt_state=new_opt_state)

    return new_state

# Train for multiple steps
for step in range(num_steps):
    state = train_step(state, sharded_batch)
```

## Using with JAX Transformations

Datarax's NNX-based distributed modules work seamlessly with JAX transformations:

```python
# Define a model
model = MyNNXModel()

# Apply vmap to process multiple examples in parallel
batch_size = 32
vmapped_model = jax.vmap(model, in_axes=0, out_axes=0)

# Create a pmap function to run across devices
pmapped_forward = jax.pmap(vmapped_model, axis_name="batch")

# Combine with distributed modules
mesh_module = DeviceMeshModule()
dp_module = DataParallelModule()

mesh = mesh_module.create_data_parallel_mesh()
sharding = dp_module.create_data_parallel_sharding(mesh)

batch = load_data_batch()
sharded_batch = dp_module.shard_batch(batch, sharding)

# Run forward pass across devices
outputs = pmapped_forward(sharded_batch["inputs"])
```

## Recommended Practices

When using Datarax's distributed training components:

1. **Scale batch size with device count** to maintain the effective batch size:
   ```python
   batch_size = per_device_batch_size * jax.device_count()
   ```

2. **Use XLA compilation for performance**:
   ```python
   train_step = jax.jit(jax.pmap(train_step_fn, axis_name="batch"))
   ```

3. **Be consistent with axis names** when using pmap and pmean/psum:
   ```python
   # Use the same axis_name in pmap and reduction operations
   pmap_fn = jax.pmap(fn, axis_name="batch")
   reduced = lax.pmean(values, axis_name="batch")
   ```

4. **Shard data correctly** to match the device arrangement:
   ```python
   mesh = mesh_module.create_data_parallel_mesh()
   sharding = dp_module.create_data_parallel_sharding(mesh)
   sharded_batch = dp_module.shard_batch(batch, sharding)
   ```

5. **Use DistributedMetricsModule for accuracy** when reporting metrics:
   ```python
   metrics = {"loss": loss, "accuracy": accuracy}
   reduced_metrics = metrics_module.reduce_mean(metrics)
   ```

## Next Steps

For more advanced distributed training scenarios, check out:

- Datarax Multi-Host Training Guide (Coming Soon)
- Model Parallelism with Datarax (Coming Soon)
- Performance Tuning for Distributed Training (Coming Soon)

For complete examples, see the [examples section](examples/overview.md):
- [Sharding Quick Reference](examples/advanced/distributed/01_sharding_quickref.ipynb) - JAX sharding basics
