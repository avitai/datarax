# Batching

Batch creation and management for data pipelines. Batching groups individual samples into batches for efficient processing.

## Components

| Component | Purpose | Features |
|-----------|---------|----------|
| **DefaultBatcher** | Standard batching | Dropping remainder |

!!! note "Key points"

    - Batching is required before most operators (batch-first)
    - Use `drop_remainder=True` for consistent batch sizes
    - Elements must share shapes — batching stacks arrays, it does not pad
    - Pipeline enforces batching by default

## Quick Start

```python
from datarax.batching import DefaultBatcher, DefaultBatcherConfig

batcher = DefaultBatcher(DefaultBatcherConfig())

# process() consumes an element iterator and yields batches
for batch in batcher.process(iter(elements), batch_size=32, drop_remainder=False):
    process(batch)
```

## Modules

- [default_batcher](default_batcher.md) - Default batching with configurable options

## With DAG Pipeline

```python
from datarax.pipeline import Pipeline

# Batching is built-in: batch_size on the Pipeline batches the source
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))

# Stages receive already-batched data
pipeline = Pipeline(source=source, stages=[transform], batch_size=32, rngs=nnx.Rngs(0))
```

## Batch Shapes

```python
# Input elements: {"image": (H, W, C)}
# After batching: {"image": (B, H, W, C)}

# Elements must share shapes — batching stacks along a new leading axis
# and does not pad. Normalize shapes upstream before batching.
```

## See Also

- [Pipeline](../dag/index.md) - Batch-first enforcement
- [Core Batcher](../core/batcher.md) - Batcher protocol
- [Pipeline](../dag/index.md) - Reshape batches with built-in DAG nodes
