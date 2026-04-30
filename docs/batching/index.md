# Batching

Batch creation and management for data pipelines. Batching groups individual samples into batches for efficient processing.

## Components

| Component | Purpose | Features |
|-----------|---------|----------|
| **DefaultBatcher** | Standard batching | Padding, dropping remainder |

`★ Insight ─────────────────────────────────────`

- Batching is required before most operators (batch-first)
- Use `drop_remainder=True` for consistent batch sizes
- Padding handles variable-length sequences
- Pipeline enforces batching by default

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.batching import DefaultBatcher

batcher = DefaultBatcher(
    batch_size=32,
    drop_remainder=False,  # Keep partial final batch
)

# Batch elements
for element in source:
    batch = batcher.add(element)
    if batch is not None:  # Full batch ready
        process(batch)

# Get remaining elements
final_batch = batcher.flush()
```

## Modules

- [default_batcher](default_batcher.md) - Default batching with configurable options

## With DAG Pipeline

```python
from datarax.pipeline import Pipeline

# Batching is built-in
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))

# Or add explicitly
pipeline = (
    source_node
    # (Pipeline(batch_size=32) handles batching; drop_remainder via partial-batch handling)
    >> transform_node
)
```

## Batch Shapes

```python
# Input elements: {"image": (H, W, C)}
# After batching: {"image": (B, H, W, C)}

# With variable lengths and padding:
# Input: {"text": (L,)} where L varies
# After batching: {"text": (B, max_L)} with padding
```

## See Also

- DAG Executor - Batch-first enforcement
- [Core Batcher](../core/batcher.md) - Batcher protocol
- DAG Rebatch - Reshape batches
