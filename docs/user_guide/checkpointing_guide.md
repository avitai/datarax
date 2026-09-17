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

A root written by datarax 0.1.11 or earlier (substrax's format 2, the state as the one
payload) restores through `IteratorCheckpoint` unchanged, and
`substrax.checkpoint.upgrade_checkpoints(source, destination,
legacy_layout=ITERATOR_STATE_FORMAT2)` rewrites it in the current format into a new root.

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

Every Datarax module is Checkpointable: `get_state()` is its NNX state as a
pure dictionary and `set_state()` restores it strictly (the structure must
match).

```python
from flax import nnx

from datarax.checkpoint import IteratorCheckpoint
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

data = [{"value": i} for i in range(100)]
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = Pipeline(source=source, stages=[], batch_size=10, rngs=nnx.Rngs(0))

for step, batch in zip(range(3), pipeline):
    pass

with IteratorCheckpoint("./pipeline_ckpt") as checkpoint:
    checkpoint.save(pipeline, step=step)

# Later: rebuild the pipeline the same way and restore
fresh = Pipeline(source=MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0)),
                 stages=[], batch_size=10, rngs=nnx.Rngs(0))
with IteratorCheckpoint("./pipeline_ckpt") as checkpoint:
    checkpoint.restore(fresh)
```

## Pipeline Iterator State

``iter(pipeline)`` over a random-access source returns a
``PipelineIterator`` — a compiled iteration session with two
checkpointing surfaces:

- **Module state**: every Variable a batch writes (position, RNG counts, and
  any stage state such as batch statistics) reaches the live pipeline module at
  every yield boundary, so checkpointing the pipeline with
  `IteratorCheckpoint` — inside the loop or after it — always captures
  exactly the batches already consumed.
- **Iterator state**: a lighter, JSON-serializable alternative for data
  checkpoints that should live outside the module snapshot:

```python
iterator = iter(pipeline)
for step, batch in enumerate(iterator):
    train_step(model, batch)
    if step % 1000 == 0:
        data_state = iterator.get_state()  # position, epoch, rng_counts, version
        save_checkpoint(model, data_state)

# Resume later: identical pipeline configuration, then restore.
iterator = iter(pipeline)
iterator.set_state(data_state)
```

``get_state()`` returns a JSON-serializable dict naming the batches the
caller has already consumed; ``set_state()`` requires a pipeline with the
same structure and seeds as the one that produced the state.

``rng_counts`` holds one count per stochastic operator, then the pipeline's
and the source's. An operator's own count stays 0: iteration keys each
record on the operator's stable base key and never draws from the operator's
private stream. A deterministic operator contributes no count, so the list's
length follows how many operators are stochastic.

``version`` names the layout those counts are in. A state saved before the
field existed is upgraded when it is restored — the counts outside operators
keep their values and their order, and every operator count restores to 0.
An upgrade needs each operator's counts to precede the rest, which holds for
operators used as pipeline stages; a pipeline holding an operator somewhere
else, such as inside its source, refuses such a state rather than resuming
from counts placed wrongly.

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

# Usage
iterator = MyCheckpointableIterator([1, 2, 3, 4, 5], rngs=nnx.Rngs(0))
print(next(iterator))  # 1
print(next(iterator))  # 2

with IteratorCheckpoint("./iterator_ckpt") as checkpoint:
    checkpoint.save(iterator, step=2)
    print(next(iterator))  # 3
    checkpoint.restore(iterator, step=2)
    print(next(iterator))  # 3 again: resumed from the checkpoint
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
