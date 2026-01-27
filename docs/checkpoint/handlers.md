# Checkpoint Handlers

Datarax provides checkpoint handlers for saving and restoring pipeline state using [Orbax](https://orbax.readthedocs.io/), Google's checkpointing library for JAX. The `OrbaxCheckpointHandler` offers a high-level interface with automatic handling of PRNG keys and string values.

## Key Features

| Feature | Description |
|---------|-------------|
| **Orbax integration** | Built on Orbax StandardCheckpointer |
| **PRNG key handling** | Automatic serialization of JAX random keys |
| **String support** | Converts strings to serializable format |
| **Versioned checkpoints** | Save multiple checkpoints with step numbers |
| **Metadata support** | Attach custom metadata to checkpoints |
| **Context manager** | Proper resource cleanup |

`★ Insight ─────────────────────────────────────`

- Orbax only supports `int`, `float`, `np.ndarray`, and `jax.Array`
- `OrbaxCheckpointHandler` automatically converts PRNG keys and strings
- Use versioned checkpoints for training with `step` parameter
- Always close handlers or use context manager protocol

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.checkpoint import OrbaxCheckpointHandler

# Create handler
handler = OrbaxCheckpointHandler()

# Save checkpoint
handler.save("/checkpoints", my_pipeline, step=1000)

# Restore checkpoint
handler.restore("/checkpoints", my_pipeline)

# Cleanup
handler.close()
```

## Context Manager Usage

Recommended for proper resource cleanup:

```python
with OrbaxCheckpointHandler() as handler:
    handler.save("/checkpoints", pipeline, step=100)
    handler.save("/checkpoints", pipeline, step=200)
    # Automatic cleanup on exit
```

## Saving Checkpoints

### Basic Save

```python
handler = OrbaxCheckpointHandler()

# Save without version
handler.save("/checkpoints", target)
# Creates: /checkpoints/checkpoint/

# Save with version (step number)
handler.save("/checkpoints", target, step=1000)
# Creates: /checkpoints/ckpt-1000/
```

### With Metadata

```python
handler.save(
    "/checkpoints",
    target,
    step=1000,
    metadata={
        "epoch": 10,
        "loss": 0.05,
        "config": {"lr": 1e-3},
    },
)
```

### Checkpoint Retention

Control how many checkpoints to keep:

```python
handler.save(
    "/checkpoints",
    target,
    step=1000,
    keep=5,  # Keep only last 5 checkpoints
)
```

### Overwriting

```python
handler.save(
    "/checkpoints",
    target,
    overwrite=True,  # Overwrite existing checkpoint
)
```

## Restoring Checkpoints

### Basic Restore

```python
# Restore latest checkpoint
state = handler.restore("/checkpoints")

# Restore specific step
state = handler.restore("/checkpoints", step=1000)

# Restore into existing object
handler.restore("/checkpoints", target=my_pipeline)
```

### Restore Metadata Only

```python
metadata = handler.restore(
    "/checkpoints",
    metadata_only=True,
)
print(f"Epoch: {metadata['epoch']}")
```

## Working with Datarax Objects

### Pipeline Checkpointing

```python
from datarax.dag import from_source

pipeline = from_source(source, batch_size=32)

# Train for a while...
for step, batch in enumerate(pipeline):
    loss = train_step(batch)

    if step % 1000 == 0:
        handler.save("/checkpoints", pipeline, step=step)

# Later: restore and continue
handler.restore("/checkpoints", pipeline)
```

### NNX Module Checkpointing

```python
import flax.nnx as nnx

class MyModel(nnx.Module):
    ...

model = MyModel()

# Save model state
handler.save("/checkpoints", model, step=1000)

# Restore into model
handler.restore("/checkpoints", model)
```

### Checkpointable Protocol

Any object implementing the `Checkpointable` protocol can be saved:

```python
from datarax.typing import Checkpointable

class MyCheckpointable:
    def get_state(self) -> dict:
        return {"my_data": self.data}

    def set_state(self, state: dict) -> None:
        self.data = state["my_data"]

obj = MyCheckpointable()
handler.save("/checkpoints", obj)
```

## Checkpoint Management

### List Checkpoints

```python
# Get all checkpoint steps
steps = handler.get_checkpoint_steps("/checkpoints")
# [100, 200, 300, 400, 500]

# Get latest step
latest = handler.latest_step("/checkpoints")
# 500

# List all checkpoints with paths
checkpoints = handler.list_checkpoints("/checkpoints")
# {100: '/checkpoints/ckpt-100', 200: '/checkpoints/ckpt-200', ...}
```

## PRNG Key Handling

PRNG keys are automatically serialized:

```python
import jax

state = {
    "params": model_params,
    "rng_key": jax.random.key(42),  # Automatically handled
}

handler.save("/checkpoints", state)

# Keys are restored as proper JAX PRNG keys
restored = handler.restore("/checkpoints")
new_key = jax.random.split(restored["rng_key"])
```

## String Handling

Strings are converted to character codes for serialization:

```python
state = {
    "model_name": "my_model_v2",
    "config_json": '{"lr": 0.001}',
}

handler.save("/checkpoints", state)

restored = handler.restore("/checkpoints")
assert restored["model_name"] == "my_model_v2"
```

## See Also

- [Checkpointing Guide](../user_guide/checkpointing_guide.md) - Complete checkpointing tutorial
- [Checkpoint Quick Reference](../examples/advanced/checkpointing/checkpoint-quickref.md)
- [DAG Executor](../dag/dag_executor.md) - Pipeline checkpointing
- [Orbax Documentation](https://orbax.readthedocs.io/)

---

## API Reference

::: datarax.checkpoint.handlers
