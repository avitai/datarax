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

A `Batch` is a Flax NNX Module that holds a collection of data samples stacked along axis 0. It contains three parts:

| Component | Type | Description |
|-----------|------|-------------|
| `data` | `dict[str, jax.Array]` | Stacked data arrays (images, labels, etc.) |
| `states` | `dict[str, jax.Array]` | Per-element state arrays (vmapped with data) |
| `metadata` | `list[Metadata]` | Per-element metadata (Python objects, not JIT-compiled) |

## Creating Batches

### From a pipeline (most common)

```python
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.dag import from_source
import numpy as np
from flax import nnx

data = {
    "image": np.random.randn(100, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, size=(100,)),
}
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

# from_source auto-batches with the specified batch_size
pipeline = from_source(source, batch_size=16)

for batch in pipeline:
    print(batch["image"].shape)  # (16, 32, 32, 3)
    break
```

### From pre-built arrays (direct construction)

```python
from datarax.core.element_batch import Batch
import jax.numpy as jnp

batch = Batch.from_parts(
    data={"image": jnp.ones((8, 32, 32, 3)), "label": jnp.zeros((8,))},
    states={},
)
```

## Accessing Batch Data

```python
# Dict-like access (recommended)
images = batch["image"]           # jax.Array, shape (B, ...)
labels = batch["label"]           # jax.Array, shape (B,)

# Check if key exists
if "mask" in batch:
    mask = batch["mask"]

# Get full data dict
data_dict = batch.get_data()      # {"image": ..., "label": ...}

# Batch size
n = batch.batch_size              # int
```

## Iteration Patterns

### Full epoch (iterate all data once)

```python
pipeline = from_source(source, batch_size=32)

for batch in pipeline:
    loss = train_step(batch["image"], batch["label"])
```

### Multiple epochs

```python
for epoch in range(num_epochs):
    pipeline = from_source(source, batch_size=32)
    for batch in pipeline:
        loss = train_step(batch["image"], batch["label"])
```

### Limited iteration (first N batches)

```python
import itertools

pipeline = from_source(source, batch_size=32)
for batch in itertools.islice(pipeline, 10):  # First 10 batches
    loss = train_step(batch["image"], batch["label"])
```

## How batch_size Works

The `from_source()` function adds a batching node to the pipeline:

```
Source (yields elements) --> BatchNode (groups into batches) --> You iterate
```

- `batch_size=32` groups 32 elements into each `Batch`
- The last batch may be smaller if `num_elements % batch_size != 0`
- Set `enforce_batch=False` to skip auto-batching (advanced use)

```python
# Standard batching
pipeline = from_source(source, batch_size=32)

# No auto-batching (elements yielded individually)
pipeline = from_source(source, batch_size=32, enforce_batch=False)

# With prefetching (default: 2 batches ahead)
pipeline = from_source(source, batch_size=32, prefetch_size=4)
```

## Next Steps

- [Data Loading Quick Reference](data-loading-quickref.md) -- Load data from various sources
- [Operators Tutorial](operators-tutorial.md) -- Transform batch data with operators
- [Simple Pipeline](simple-pipeline.md) -- Complete pipeline example
