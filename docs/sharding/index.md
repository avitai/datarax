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
from datarax.sharding import ArraySharder
import jax

# Shard across available devices
sharder = ArraySharder(devices=jax.devices())

# Shard a batch
sharded_batch = sharder.shard(batch)
# Each device gets batch_size / num_devices samples
```

## Modules

- [array_sharder](array_sharder.md) - Shard arrays across devices on single host
- [jax_process_sharder](jax_process_sharder.md) - Process-level sharding for multi-host

## Multi-Host Example

```python
from datarax.sharding import JaxProcessSharder
import jax

# Each host creates its sharder
sharder = JaxProcessSharder(
    num_processes=jax.process_count(),
    process_index=jax.process_index(),
)

# Each host gets its portion of global batch
local_batch = sharder.shard(global_batch)
```

## With Pipelines

```python
from datarax.dag import from_source
from datarax.dag.nodes import SharderNode

pipeline = (
    from_source(source, batch_size=256)
    >> SharderNode(ArraySharder(jax.devices()))
    >> transform
)
```

## See Also

- [Distributed](../distributed/index.md) - Distributed training
- [Distributed Training Guide](../user_guide/distributed_training.md)
- [Sharding Tutorial](../examples/advanced/distributed/sharding-quickref.md)
