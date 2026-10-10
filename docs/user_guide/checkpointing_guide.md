# Checkpointing Guide

Datarax checkpoints through [substrax](https://github.com/avitai/substrax)'s
Orbax-backed checkpoint store. This guide covers how to checkpoint and restore
pipeline, iterator and module state.

## Overview

Datarax's checkpointing system is built on:

1. **The `Checkpointable` protocol**: `get_state()` returns a state dictionary,
   `set_state()` restores one. Every Datarax module, pipeline and iterator
   implements it.
2. **`IteratorCheckpoint`**: saves a Checkpointable's state under an integer step
   and restores it into a freshly built object.
3. **substrax's `OrbaxCheckpointStore`**: the storage layer. It writes the state as the
   checkpoint's `data_iterator` item, which carries arrays, typed PRNG keys and
   plain-Python leaves (positions, seeds, sampler reprs) alike, keeps the most recent
   `max_to_keep` steps, and records your `metadata` in the checkpoint's `extra`.

## Saving and Restoring

```python
from datarax.checkpoint import IteratorCheckpoint

with IteratorCheckpoint("./checkpoints", max_to_keep=5) as checkpoint:
    # Save the pipeline's state under a step; the epoch is a field of the record and
    # metadata holds free keys (one naming a record field, such as "epoch", is refused)
    checkpoint.save(pipeline, step=100, epoch=1, metadata={"description": "Training checkpoint"})

    # Restore the latest step, or a specific one, into a pipeline built the same way
    checkpoint.restore(pipeline)
    checkpoint.restore(pipeline, step=50)

    # What is on disk
    print(checkpoint.all_steps())   # [50, 100]
    print(checkpoint.latest_step())  # 100
```

`restore` reads the saved state into the object's current state as a template,
so the object must be built the way the saved one was: same structure, same
seeds. A checkpoint whose identity fields (sampler and data-source reprs,
shard and worker counts) differ from the object's is rejected with a
`ValueError` before anything is applied.

## Periodic Checkpoints in a Loop

```python
with IteratorCheckpoint("./checkpoints", max_to_keep=3) as checkpoint:
    for step, batch in enumerate(pipeline):
        train_step(model, batch)
        checkpoint.save_if_due(pipeline, step, interval=1000)
```

`save_if_due` saves when `step` is a multiple of `interval` and returns the
checkpoint path, or `None` when the step is not due.

## Checkpointing Datarax Modules

Every Datarax module other than the pipeline is Checkpointable through its NNX
state: `get_state()` is that state as a pure dictionary and `set_state()`
restores it strictly (the structure must match).

## Pipeline State

A pipeline's `get_state()` is where iteration stands, and nothing else: no
records and no stage parameters. ``for batch in pipeline`` reads each batch on
the host stage and continues where the last batch taken ended; `get_state()`
names exactly the batches already taken, whatever the read threads have read
ahead, and `set_state()` on a pipeline built the same way resumes there.

```python
import jax.numpy as jnp
from flax import nnx

from datarax.checkpoint import IteratorCheckpoint
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

data = {"value": jnp.arange(100)}
source = MemorySource(MemorySourceConfig(), data=data)
pipeline = Pipeline(source=source, stages=[], batch_size=10, rngs=nnx.Rngs(0))

for step, batch in zip(range(3), pipeline):
    pass

with IteratorCheckpoint("./pipeline_ckpt") as checkpoint:
    checkpoint.save(pipeline, step=step)

# Later: rebuild the pipeline the same way and restore
fresh = Pipeline(source=MemorySource(MemorySourceConfig(), data=data),
                 stages=[], batch_size=10, rngs=nnx.Rngs(0))
with IteratorCheckpoint("./pipeline_ckpt") as checkpoint:
    checkpoint.restore(fresh)
```

The state is a small dictionary of plain values:

| Field | Holds |
|---|---|
| `version` | The layout, `3`; any other layout is refused, naming both versions |
| `kind` | The source's record identity: `indexed`, `stream_ids` or `arrival` |
| `epoch`, `position` | The epoch and the records served in it (`None` for a stream) |
| `run_end_epoch` | The epoch the run ends at, so a resumed run ends where the uninterrupted one would |
| `stream` | A stream's pass, the records taken in it, the records arrived and the passes left (`None` for an indexed source) |
| `fingerprint` | The configuration the state is valid for: batch size, length, `drop_last`, `num_epochs`, whether it shuffles, the seed's words, the order and the shard of the records the source serves (`None` for all of them) |

`set_state()` refuses a state whose fingerprint differs from the pipeline's,
naming the field. No random stream advances while iterating: an operator keys
each record on its stable base key, so the epoch and the position decide every
draw.

### Stage parameters and statistics

Stages carry parameters training moves and statistics (batch norm) iteration
writes. They are the stage graph's NNX state, `nnx.state(pipeline.dag)`, and
are checkpointed with the model:

```python
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

with OrbaxCheckpointStore("./run") as store:
    store.save(step, {
        "model": {
            "network": nnx.to_pure_dict(nnx.state(model)),
            "stages": nnx.to_pure_dict(nnx.state(pipeline.dag)),
        },
        "data_iterator": pipeline.get_state(),
    })
```

### `step()`, `scan` and `session()`

`step()`, `scan` and the compiled session `pipeline.session()` keep their place
in the pipeline's own Variables, apart from iteration's: a run driven by them
checkpoints `nnx.state(pipeline)`, and a session's own `get_state()` /
`set_state()` resume that session. Mixing `step()` and ``for batch in pipeline``
on one pipeline is not supported; `reset()` resets both places.

## Checkpointable Iterator Pattern

A host-side iterator that must resume is a `DataSourceModule`: it is a `DataraxModule`, so its
`nnx.Variable` state is what `IteratorCheckpoint` saves. Keep the records as construction data
(`nnx.data`, never checkpointed) and the position as a Variable (checkpointed):

```python
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from datarax.checkpoint import IteratorCheckpoint
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity


@dataclass(frozen=True)
class RecordReaderConfig(StructuralConfig):
    """Configuration for a reader over in-memory records."""


class RecordReader(DataSourceModule):
    """Serves records one at a time; its position is state, so a checkpoint resumes it."""

    @property
    def record_identity(self) -> RecordIdentity:
        """A record is named by when it is read."""
        return RecordIdentity.ARRIVAL

    def __init__(self, config: RecordReaderConfig, records: list[dict]) -> None:
        super().__init__(config)
        self.records = nnx.data(records)  # construction data: never checkpointed
        self.position = nnx.Variable(jnp.int32(0))  # iteration state: checkpointed

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> "RecordReader":
        return self

    def __next__(self) -> dict:
        position = int(self.position[...])
        if position >= len(self.records):
            raise StopIteration
        self.position[...] = jnp.int32(position + 1)
        return self.records[position]


records = [{"text": f"line {i}"} for i in range(5)]
reader = RecordReader(RecordReaderConfig(), records)
print(next(reader)["text"])  # line 0
print(next(reader)["text"])  # line 1

with IteratorCheckpoint("./iterator_ckpt") as checkpoint:
    checkpoint.save(reader, step=2)
    print(next(reader)["text"])  # line 2
    resumed = RecordReader(RecordReaderConfig(), records)
    checkpoint.restore(resumed, step=2)
    print(next(resumed)["text"])  # line 2 again: resumed from the checkpoint
```

## PRNG State Handling

Typed PRNG keys are part of the state and round-trip as keys:

```python
import jax

class KeyedIterator:
    def __init__(self):
        self.key = jax.random.key(42)
        self.position = 0

    def get_state(self):
        return {"key": self.key, "position": self.position}

    def set_state(self, state):
        self.key = state["key"]
        self.position = state["position"]

with IteratorCheckpoint("./checkpoints") as checkpoint:
    checkpoint.save(KeyedIterator(), step=1)
    restored = KeyedIterator()
    checkpoint.restore(restored, step=1)
    print(jax.random.key_data(restored.key))
```

## Retention

`max_to_keep` bounds how many steps stay on disk; Orbax deletes the oldest
when a newer one is saved:

```python
with IteratorCheckpoint("./checkpoints", max_to_keep=5) as checkpoint:
    for step in range(0, 100, 10):
        checkpoint.save(pipeline, step=step)
    print(checkpoint.all_steps())  # [50, 60, 70, 80, 90]
```

## Best Practices

1. **Use the context manager**: `with IteratorCheckpoint(...) as checkpoint:` releases
   the store when the block ends

2. **Checkpoint regularly**: `save_if_due` at a fixed interval

3. **Keep essential state**: only checkpoint what is needed to resume, not derived values

4. **Use monotonic steps**: Orbax addresses checkpoints by step and keeps them in order

5. **Set `max_to_keep`**: bound the checkpoint count to avoid filling the disk

6. **Rebuild before restoring**: restore into an object built the way the saved one was

## Error Handling

```python
from datarax.checkpoint import IteratorCheckpoint

with IteratorCheckpoint("./checkpoints") as checkpoint:
    if checkpoint.has_checkpoint():
        checkpoint.restore(pipeline)
        print(f"Restored from step {checkpoint.latest_step()}")
    else:
        print("No checkpoints found")
```

`restore` raises `ValueError` when the directory holds no checkpoint at the
requested step, or when the checkpoint's identity fields do not match the
object it is being restored into.

## See Also

- [Troubleshooting Guide](troubleshooting_guide.md)
- [NNX Best Practices](nnx_best_practices.md)
