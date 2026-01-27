# Checkpoint

State persistence and recovery for pipeline checkpointing. Built on [Orbax](https://orbax.readthedocs.io/), Google's checkpointing library for JAX.

## Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **OrbaxCheckpointHandler** | Save/load state | PRNG keys, versioning, metadata |
| **Checkpointable Protocol** | Interface | `get_state()`, `set_state()` |
| **CheckpointableIterator** | Resumable iteration | Iterator + checkpointing |

`★ Insight ─────────────────────────────────────`

- Orbax handles PyTree serialization automatically
- PRNG keys and strings are converted automatically
- Use `step` parameter for versioned checkpoints
- Always use context manager or call `close()`

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.checkpoint import OrbaxCheckpointHandler

# Save and restore with context manager
with OrbaxCheckpointHandler() as handler:
    # Save versioned checkpoint
    handler.save("/checkpoints", pipeline, step=1000)

    # Restore latest
    handler.restore("/checkpoints", pipeline)
```

## Modules

- [handlers](handlers.md) - `OrbaxCheckpointHandler` for save/load operations
- [iterators](iterators.md) - Checkpointable iterator implementations

## Training Loop Example

```python
handler = OrbaxCheckpointHandler()

for step, batch in enumerate(pipeline):
    loss = train_step(batch)

    # Save every 1000 steps
    if step % 1000 == 0:
        handler.save("/checkpoints", pipeline, step=step, keep=5)

# Cleanup
handler.close()
```

## Checkpoint Management

```python
# List all checkpoints
steps = handler.get_checkpoint_steps("/checkpoints")
# [1000, 2000, 3000, 4000, 5000]

# Get latest step
latest = handler.latest_step("/checkpoints")
# 5000

# Restore specific step
handler.restore("/checkpoints", pipeline, step=3000)
```

## See Also

- [Handlers Guide](handlers.md) - Complete handler documentation
- [Checkpointing User Guide](../user_guide/checkpointing_guide.md)
- [Checkpoint Tutorial](../examples/advanced/checkpointing/checkpoint-quickref.md)
- [DAG Executor](../dag/dag_executor.md) - Pipeline checkpointing
