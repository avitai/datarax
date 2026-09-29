# Batch Processing Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min |
| **Prerequisites** | Basic Python, numpy |
| **Format** | Reference card |
| **Memory** | ~100 MB RAM |

## Overview

Batching is fundamental to efficient data processing in Datarax. This reference covers the `Batch` object, how batching works in pipelines, and common iteration patterns.

## What is a Batch?

A `Batch` is a frozen pytree of arrays holding records stacked along axis 0. Every field is an
array, so a batch passes through `jax.jit`, `vmap`, `scan` and sharding like a dict of arrays:

| Field | Type | Description |
|-------|------|-------------|
| `data` | pytree, leading axis `B` | The records' values (images, labels, etc.) |
| `states` | pytree, leading axis `B` | Per-record processing state |
| `indices` | `uint32 (B, 2)` | Each record's 64-bit index, as two words |
| `epochs`, `draws` | `int32 (B,)` | Each record's epoch and draw within it |
| `batch_state` | pytree | Batch-level arrays, without a record axis |

A record's identity keys its randomness in every stochastic operator. File names and other
strings stay with the source on the host.

## Iterating a Pipeline

### From a pipeline (most common)

Iterating a `Pipeline` yields plain `dict[str, jax.Array]` batches -- each
value is stacked along axis 0 with the leading `batch_size` dimension. The
pipeline does not wrap them in a `Batch` object.

```python
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.pipeline import Pipeline
import numpy as np
from flax import nnx

data = {
    "image": np.random.randn(100, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, size=(100,)),
}
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

# Pipeline groups elements into batches of the specified batch_size
pipeline = Pipeline(source=source, stages=[], batch_size=16, rngs=nnx.Rngs(0))

for batch in pipeline:
    print(batch["image"].shape)  # (16, 32, 32, 3)  -- batch is a plain dict
    break
```

## Creating Batches

### From pre-built arrays (direct construction)

When you need an explicit `Batch` -- for example to call an operator directly -- build one
from arrays with `batch_ops.from_arrays`. Row `i` is named record `(0, i)`, so each row draws
its own randomness:

```python
import jax.numpy as jnp
from datarax.core import batch_ops

batch = batch_ops.from_arrays({"image": jnp.ones((8, 32, 32, 3)), "label": jnp.zeros((8,))})
```

Records built as `Element`s are stacked and then turned into a batch:

```python
from datarax.core import Element, batch_ops

records = [Element({"x": jnp.full(3, i)}) for i in range(4)]
batch = batch_ops.from_stacked(batch_ops.stack(records))
```

## Accessing Batch Data

A `Batch` reads its data by field name; it is not a mapping itself, so `dict(batch)`,
`**batch`, `len(batch)` and iterating over it raise rather than drop its other fields:

```python
# Dict-like access (recommended)
images = batch["image"]           # jax.Array, shape (B, ...)
labels = batch["label"]           # jax.Array, shape (B,)

# Check if key exists
if "mask" in batch:
    mask = batch["mask"]

# The whole data pytree
data_dict = batch.data            # {"image": ..., "label": ...}

# Batch size, static under jit
n = batch.batch_size              # int

# One record, with its identity
first = batch_ops.element(batch, 0)
```

## Iteration Patterns

### Full epoch (iterate all data once)

```python
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))

for batch in pipeline:
    loss = train_step(batch["image"], batch["label"])
```

### Multiple epochs

```python
for epoch in range(num_epochs):
    pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))
    for batch in pipeline:
        loss = train_step(batch["image"], batch["label"])
```

### Limited iteration (first N batches)

```python
import itertools

pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))
for batch in itertools.islice(pipeline, 10):  # First 10 batches
    loss = train_step(batch["image"], batch["label"])
```

## How batch_size Works

The `Pipeline` constructor auto-batches via the `batch_size` argument:

```
Source (yields elements) --> Pipeline (groups into batches via batch_size) --> You iterate
```

- `batch_size=32` groups 32 elements into each batch
- No batch holds padding. With a random-access source (`MemorySource` and the other indexed
  sources) and `num_elements % batch_size != 0`, the batch reaching the end of an epoch is
  completed from the head of the next epoch's order, so each epoch serves every record once;
  iterating a fixed number of epochs ends with a short batch. `drop_last=True` instead skips
  the records short of a full batch and starts the next epoch
- A streaming source, which has no indexed access, yields a shorter last batch

```python
# Standard batching
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))
```

## Next Steps

- [Data Loading Quick Reference](data-loading-quickref.md) -- Load data from various sources
- [Operators Tutorial](operators-tutorial.md) -- Transform batch data with operators
- [Simple Pipeline](simple-pipeline.md) -- Complete pipeline example
