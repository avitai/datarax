# Sharding

Data sharding utilities for distributed processing. Shard data across devices and hosts for parallel training.

## Sharders

| Sharder | Scope | Use Case |
|---------|-------|----------|
| **ArraySharder** | Single host | Multi-GPU on one machine |
| **JaxProcessSharder** | Multi-host | TPU pods, multi-node |

`★ Insight ─────────────────────────────────────`

- Sharding splits data across devices for parallel processing
- Use `ArraySharder` for single-host multi-GPU
- Use `JaxProcessSharder` for multi-host (TPU pods)
- JAX handles the communication automatically

`─────────────────────────────────────────────────`

## Quick Start

```python
import jax
from flax import nnx

from datarax.sharding import ArraySharder

# Build a single-axis device mesh and the corresponding NamedSharding.
mesh = jax.make_mesh((len(jax.devices()),), ("data",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

sharder = ArraySharder(rngs=nnx.Rngs(0))
sharded_batch = sharder.shard(batch, sharding)
# Each device now holds ``batch_size / num_devices`` samples.
```

## Modules

- [array_sharder](array_sharder.md) - Shard arrays across devices on single host
- [jax_process_sharder](jax_process_sharder.md) - Process-level sharding for multi-host

## Multi-Host Example

```python
import jax
from flax import nnx

from datarax.sharding import JaxProcessSharderModule

# Each host instantiates its own sharder; the module reads
# ``jax.process_count()`` and ``jax.process_index()`` on construction.
sharder = JaxProcessSharderModule(rngs=nnx.Rngs(0))

# ``shard_data`` slices arrays / lists / tuples down to the
# current host's portion (``Grain``-style bounds).
local_images = sharder.shard_data(global_images)
local_labels = sharder.shard_data(global_labels)
```

## With Pipelines

Wrap the sharder in an `nnx.Module` and place it in `stages=[...]`:

```python
import jax
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sharding import ArraySharder


class _Shard(nnx.Module):
    """Wrap ``ArraySharder.shard`` as a Pipeline-compatible stage."""

    def __init__(self, sharder: ArraySharder, sharding: jax.sharding.Sharding) -> None:
        self.sharder = sharder
        self.sharding = sharding

    def __call__(self, batch):
        return self.sharder.shard(batch, self.sharding)


mesh = jax.make_mesh((len(jax.devices()),), ("data",))
sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
shard_stage = _Shard(ArraySharder(rngs=nnx.Rngs(0)), sharding)

pipeline = Pipeline(
    source=source,
    stages=[shard_stage, transform],
    batch_size=256,
    rngs=nnx.Rngs(0),
)
```

## See Also

- [Distributed](../distributed/index.md) - Distributed training
- [Distributed Training Guide](../user_guide/distributed_training.md)
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
