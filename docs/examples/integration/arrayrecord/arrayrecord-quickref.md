# ArrayRecord Source Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~15 min |
| **Prerequisites** | Simple Pipeline |
| **Format** | Python + Jupyter |

## Overview

Learn to use `ArrayRecordSourceModule` for loading data from Google's
ArrayRecord format. ArrayRecord is a high-performance file format used
by Google for ML datasets, similar to TFRecord but with better random access.

## Learning Goals

By the end of this quick reference, you will be able to:

1. Configure `ArrayRecordSourceConfig` for ArrayRecord files
2. Create an `ArrayRecordSourceModule` from file paths
3. Integrate ArrayRecord sources into Datarax pipelines
4. Understand checkpointing and state management

## Coming from Google Grain?

| Grain | Datarax |
|-------|---------|
| `grain.ArrayRecordDataSource(paths)` | `ArrayRecordSourceModule(config, paths)` |
| `grain.DataLoader(source)` | `from_source(source)` |
| Manual iteration | Automatic stateful iteration |
| Manual checkpointing | Built-in `get_state()` / `set_state()` |

**Key Differences:**

1. **Stateful Iteration**: Datarax tracks position automatically via NNX Variables
2. **Checkpointing**: Built-in state serialization for resume
3. **Pipeline Integration**: Direct integration with DAG-based pipelines
4. **Shuffling**: Internal shuffle handling per epoch

## Files

- **Python Script**: [`examples/integration/arrayrecord/01_arrayrecord_quickref.py`](https://github.com/avitai/datarax/blob/main/examples/integration/arrayrecord/01_arrayrecord_quickref.py)
- **Jupyter Notebook**: [`examples/integration/arrayrecord/01_arrayrecord_quickref.ipynb`](https://github.com/avitai/datarax/blob/main/examples/integration/arrayrecord/01_arrayrecord_quickref.ipynb)

## Quick Start

### Installation

ArrayRecord requires the `array_record` package:

```bash
uv pip install "datarax[data]" array-record
```

!!! note "Platform Support"
    ArrayRecord is primarily available on Linux. Check compatibility for your platform.

### Run the Python Script

```bash
python examples/integration/arrayrecord/01_arrayrecord_quickref.py
```

## Key Concepts

### ArrayRecordSourceConfig

Configuration for ArrayRecord data sources:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for shuffling |
| `num_epochs` | int | -1 | Number of epochs (-1 for infinite) |
| `shuffle_files` | bool | False | Whether to shuffle file order |

```python
from datarax.sources import ArrayRecordSourceConfig

config = ArrayRecordSourceConfig(
    seed=42,
    num_epochs=10,       # Run for 10 epochs
    shuffle_files=True,  # Shuffle each epoch
)
```

### Creating an ArrayRecord Source

```python
from datarax.sources import ArrayRecordSourceModule, ArrayRecordSourceConfig
from flax import nnx

# Single file
source = ArrayRecordSourceModule(
    ArrayRecordSourceConfig(seed=42),
    paths="/path/to/data.riegeli",
    rngs=nnx.Rngs(0),
)

# Multiple files with glob pattern
source = ArrayRecordSourceModule(
    ArrayRecordSourceConfig(seed=42, shuffle_files=True),
    paths="/path/to/data-*.riegeli",
    rngs=nnx.Rngs(0),
)

# List of specific files
source = ArrayRecordSourceModule(
    ArrayRecordSourceConfig(num_epochs=10),
    paths=[
        "/path/to/train-00000.riegeli",
        "/path/to/train-00001.riegeli",
    ],
    rngs=nnx.Rngs(0),
)
```

### Pipeline Integration

```python
from datarax import from_source
from datarax.dag.nodes import OperatorNode

# Create pipeline from ArrayRecord source
pipeline = from_source(source, batch_size=32)

# Add transformations
pipeline = pipeline.add(OperatorNode(normalize_op))

# Iterate
for batch in pipeline:
    print(f"Batch shape: {batch['data'].shape}")
```

### Checkpointing

ArrayRecordSourceModule supports full state serialization:

```python
# Save checkpoint
state = source.get_state()
# state = {"current_index": 1234, "current_epoch": 5, ...}

# Later: restore from checkpoint
source.set_state(state)
# Resumes from exact position
```

**State Contents:**

| State Key | Description |
|-----------|-------------|
| `current_index` | Current position in dataset |
| `current_epoch` | Current epoch number |
| `shuffled_indices` | Shuffle order (if enabled) |
| `prefetch_cache` | Prefetched records cache |

### Epoch Control

**Finite Epochs:**
```python
# Run for exactly 10 epochs
config = ArrayRecordSourceConfig(num_epochs=10)
source = ArrayRecordSourceModule(config, paths=paths, rngs=nnx.Rngs(0))

for batch in from_source(source, batch_size=32):
    train_step(batch)
# Automatically stops after 10 epochs
```

**Infinite Iteration:**
```python
# Run indefinitely (for step-based training)
config = ArrayRecordSourceConfig(num_epochs=-1)
source = ArrayRecordSourceModule(config, paths=paths, rngs=nnx.Rngs(0))

step = 0
for batch in from_source(source, batch_size=32):
    train_step(batch)
    step += 1
    if step >= max_steps:
        break
```

### Shuffling Behavior

When `shuffle_files=True`:

1. At initialization, indices are shuffled using `seed`
2. At each epoch boundary, indices are reshuffled using `seed + epoch`
3. This ensures reproducible but varied order across epochs

```python
config = ArrayRecordSourceConfig(
    seed=42,
    shuffle_files=True,
)
# Epoch 0: shuffled with seed=42
# Epoch 1: reshuffled with seed=43
# Epoch 2: reshuffled with seed=44
```

## Results

Running the quick reference produces:

```
============================================================
ArrayRecord Source Quick Reference
============================================================

This quick reference demonstrates the ArrayRecordSourceModule API.
Actual usage requires ArrayRecord files (*.riegeli format).

Key API Summary:

  1. Configuration:
     config = ArrayRecordSourceConfig(
         seed=42,
         num_epochs=-1,
         shuffle_files=True,
     )

  2. Source Creation:
     source = ArrayRecordSourceModule(
         config,
         paths="/path/to/*.riegeli",
         rngs=nnx.Rngs(0),
     )

  3. Pipeline Integration:
     pipeline = from_source(source, batch_size=32)

  4. Checkpointing:
     state = source.get_state()
     source.set_state(state)

============================================================
Quick reference completed!
============================================================
```

## Feature Summary

| Feature | Description |
|---------|-------------|
| **Stateful** | Tracks position via NNX Variables |
| **Checkpointing** | Full `get_state()` / `set_state()` |
| **Shuffling** | Per-epoch reshuffling with seed control |
| **Epoch Control** | Finite or infinite iteration |
| **Grain Compatible** | Wraps Grain's ArrayRecordDataSource |

## When to Use ArrayRecord

- Large datasets (>10GB)
- Need random access to records
- Working with Google's ML infrastructure
- Migrating from TFRecord to a modern format

## Next Steps

- [HuggingFace Tutorial](../huggingface/hf-tutorial.md) - Alternative data source
- [TFDS Quick Reference](../tfds/tfds-quickref.md) - TensorFlow Datasets
- [Checkpointing Guide](../../advanced/checkpointing/checkpoint-quickref.md) - Full checkpointing

## API Reference

- [`ArrayRecordSourceModule`](../../../sources/array_record_source.md)
- [`ArrayRecordSourceConfig`](../../../sources/array_record_source.md)
