# Checkpoint

State persistence and recovery for pipelines, iterators and modules. Built on
[substrax](https://github.com/avitai/substrax)'s Orbax-backed checkpoint store.

## Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **Checkpointable Protocol** | Interface | `get_state()`, `set_state()` |
| **IteratorCheckpoint** | Save and restore any Checkpointable | Step addressing, retention, metadata, identity validation |

!!! note "Key points"

    - Every Datarax module, pipeline and iterator is Checkpointable
    - Arrays, typed PRNG keys and plain-Python leaves (positions, seeds, reprs) all round-trip
    - Checkpoints are addressed by integer step; Orbax keeps the most recent `max_to_keep`
    - Use the context manager, or call `close()`, to release the store

## Quick Start

```python
from datarax.checkpoint import IteratorCheckpoint

with IteratorCheckpoint("/checkpoints", max_to_keep=5) as checkpoint:
    # Save the pipeline's state under step 1000
    checkpoint.save(pipeline, step=1000)

    # Restore the latest step into a freshly built pipeline
    checkpoint.restore(pipeline)
```

## Modules

- [iterators](iterators.md) - `IteratorCheckpoint` and restore validation

## Training Loop Example

```python
with IteratorCheckpoint("/checkpoints", max_to_keep=5) as checkpoint:
    for step, batch in enumerate(pipeline):
        loss = train_step(batch)
        checkpoint.save_if_due(pipeline, step, interval=1000)
```

## Checkpoint Management

```python
# List all checkpoints
checkpoint.all_steps()
# [1000, 2000, 3000, 4000, 5000]

# Get latest step
checkpoint.latest_step()
# 5000

# Restore a specific step
checkpoint.restore(pipeline, step=3000)
```

## See Also

- [Checkpointing User Guide](../user_guide/checkpointing_guide.md)
- [Checkpoint Tutorial](../examples/advanced/checkpointing/checkpoint-quickref.md)
- [Pipeline](../dag/index.md) - Pipeline construction and checkpointing
