# Checkpointing Guide

Datarax provides checkpointing capabilities through Orbax integration. This guide covers how to effectively checkpoint and restore pipeline state.

## Overview

Datarax's checkpointing system is built on:

1. **Orbax Integration**: Uses Orbax's `StandardCheckpointer` for PyTree serialization
2. **DataraxModule State**: All Datarax modules inherit from NNX Module with state management
3. **Checkpointable Iterators**: Iterator modules that can save and restore position

## Core Handler

The main checkpointing handler is `OrbaxCheckpointHandler`:

```python
from datarax.checkpoint import OrbaxCheckpointHandler

# Create handler
handler = OrbaxCheckpointHandler()

# Use as context manager for automatic cleanup
with OrbaxCheckpointHandler() as handler:
    handler.save("./checkpoints", state, step=1)
    restored = handler.restore("./checkpoints")
```

## Basic Checkpointing

### Saving Checkpoints

```python
from datarax.checkpoint import OrbaxCheckpointHandler
import jax.numpy as jnp

# Create handler
handler = OrbaxCheckpointHandler()

# State to checkpoint (dict or Checkpointable object)
state = {
    "model_weights": jnp.ones((10, 10)),
    "position": 42,
    "epoch": 5,
}

# Save checkpoint
handler.save(
    directory="./checkpoints",
    target=state,
    step=100,      # Optional step number
    keep=5,        # Keep last 5 checkpoints
    overwrite=False,
    metadata={"description": "Training checkpoint"},
)

# Don't forget to close when done
handler.close()
```

### Restoring Checkpoints

```python
from datarax.checkpoint import OrbaxCheckpointHandler

with OrbaxCheckpointHandler() as handler:
    # Restore latest checkpoint
    state = handler.restore("./checkpoints")

    # Restore specific step
    state = handler.restore("./checkpoints", step=50)

    # List available checkpoints
    steps = handler.get_checkpoint_steps("./checkpoints")
    print(f"Available steps: {steps}")

    # Get latest step number
    latest = handler.latest_step("./checkpoints")
    print(f"Latest step: {latest}")
```

## Checkpointing Datarax Modules

Datarax modules implement the `get_state()` method for checkpointing:

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.checkpoint import OrbaxCheckpointHandler
from flax import nnx

# Create a pipeline
data = [{"value": i} for i in range(100)]
config = MemorySourceConfig()
source = MemorySource(config, data=data, rngs=nnx.Rngs(0))
pipeline = from_source(source, batch_size=10)

# Process some batches
iterator = iter(pipeline)
for i in range(3):
    batch = next(iterator)
    print(f"Batch {i}: processed")

# Checkpoint the pipeline state
with OrbaxCheckpointHandler() as handler:
    # Get state from all pipeline components
    state = {
        "pipeline_step": i,
        # Add any custom state you need to track
    }
    handler.save("./pipeline_ckpt", state, step=i)
```

## Checkpointable Iterator Pattern

Create iterators that can be checkpointed:

```python
from datarax.core.module import CheckpointableIteratorModule
from flax import nnx
import jax.numpy as jnp

class MyCheckpointableIterator(CheckpointableIteratorModule):
    def __init__(self, data, *, rngs: nnx.Rngs):
        super().__init__(rngs=rngs)
        self.data = data
        self.position = nnx.Variable(jnp.array(0))

    def __iter__(self):
        return self

    def __next__(self):
        pos = int(self.position[...])
        if pos >= len(self.data):
            raise StopIteration
        item = self.data[pos]
        self.position[...] = jnp.array(pos + 1)
        return item

    def checkpoint(self) -> dict:
        """Return checkpoint state."""
        return {
            "position": int(self.position[...]),
        }

    def restore(self, checkpoint: dict) -> None:
        """Restore from checkpoint."""
        self.position[...] = jnp.array(checkpoint["position"])

# Usage
iterator = MyCheckpointableIterator([1, 2, 3, 4, 5], rngs=nnx.Rngs(0))

# Consume some items
print(next(iterator))  # 1
print(next(iterator))  # 2

# Checkpoint
ckpt = iterator.checkpoint()
print(f"Checkpoint: {ckpt}")

# Continue
print(next(iterator))  # 3

# Restore to previous position
iterator.restore(ckpt)
print(next(iterator))  # 2 (resumed from checkpoint)
```

## PRNG State Handling

The handler automatically manages JAX PRNG keys:

```python
import jax

# PRNGKeys are automatically serialized/deserialized
state = {
    "rng_key": jax.random.key(42),
    "split_keys": jax.random.split(jax.random.key(0), 4),
}

with OrbaxCheckpointHandler() as handler:
    handler.save("./checkpoints", state, step=1)
    restored = handler.restore("./checkpoints")

# Keys are properly restored
print(type(restored["rng_key"]))  # jax.Array (key type)
```

## Checkpoint Management

### Multiple Checkpoints

```python
with OrbaxCheckpointHandler() as handler:
    # Save multiple checkpoints with keep=N
    for step in range(100):
        if step % 10 == 0:
            handler.save(
                "./checkpoints",
                {"step": step},
                step=step,
                keep=5,  # Only keep last 5 checkpoints
            )

    # List all available checkpoints
    checkpoints = handler.list_checkpoints("./checkpoints")
    print(f"Checkpoints: {checkpoints}")
```

### Overwriting Checkpoints

```python
with OrbaxCheckpointHandler() as handler:
    # First save
    handler.save("./checkpoints", {"v": 1}, step=1)

    # Overwrite existing checkpoint
    handler.save("./checkpoints", {"v": 2}, step=1, overwrite=True)
```

## Best Practices

1. **Use context manager**: Always use `with OrbaxCheckpointHandler() as handler:` to ensure proper cleanup

2. **Checkpoint regularly**: Save checkpoints at regular intervals during training

3. **Keep essential state**: Only checkpoint what's needed to resume - not derived values

4. **Use step numbers**: Use meaningful step numbers for easier checkpoint management

5. **Set keep parameter**: Limit checkpoint count to avoid disk space issues

6. **Handle PRNG state**: The handler manages PRNG keys automatically

## Error Handling

```python
from datarax.checkpoint import OrbaxCheckpointHandler
from pathlib import Path

with OrbaxCheckpointHandler() as handler:
    checkpoint_dir = Path("./checkpoints")

    # Check if checkpoints exist
    if checkpoint_dir.exists():
        steps = handler.get_checkpoint_steps(checkpoint_dir)
        if steps:
            state = handler.restore(checkpoint_dir)
            print(f"Restored from step {handler.latest_step(checkpoint_dir)}")
        else:
            print("No checkpoints found")
    else:
        print("Checkpoint directory doesn't exist")
```

## See Also

- [Troubleshooting Guide](troubleshooting_guide.md)
- [NNX Best Practices](nnx_best_practices.md)
