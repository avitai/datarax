# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
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
"""

# %% [markdown]
"""
## Coming from Google Grain?

| Grain | Datarax |
|-------|---------|
| `grain.ArrayRecordDataSource(paths)` | `ArrayRecordSourceModule(config, paths)` |
| `grain.DataLoader(source)` | `from_source(source)` |
| Manual iteration | Automatic stateful iteration |
| Manual checkpointing | Built-in `get_state()` / `set_state()` |

## Key Differences

1. **Stateful Iteration**: Datarax tracks position automatically via NNX Variables
2. **Checkpointing**: Built-in state serialization for resume
3. **Pipeline Integration**: Direct integration with DAG-based pipelines
4. **Shuffling**: Internal shuffle handling per epoch
"""

# %% [markdown]
"""
## Setup

ArrayRecord requires the `array_record` package (Google's format):

```bash
uv pip install "datarax[data]" array-record
```

Note: ArrayRecord is primarily available on Linux. Check compatibility for your platform.
"""

# %%
# Imports
# Note: These imports would be used with actual ArrayRecord files:
# import numpy as np
# from flax import nnx
# from datarax import from_source
# from datarax.sources import ArrayRecordSourceModule, ArrayRecordSourceConfig

print("ArrayRecord Source Quick Reference")
print("=" * 50)

# %% [markdown]
"""
## Part 1: ArrayRecordSourceConfig

Configuration for ArrayRecord data sources.

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | int | 42 | Random seed for shuffling |
| `num_epochs` | int | -1 | Number of epochs (-1 for infinite) |
| `shuffle_files` | bool | False | Whether to shuffle file order |
"""

# %%
# Configuration example (conceptual - actual usage requires ArrayRecord files)
print("ArrayRecordSourceConfig Parameters:")
print()
print("  seed: int = 42")
print("    - Random seed for epoch-based shuffling")
print()
print("  num_epochs: int = -1")
print("    - Number of epochs to iterate")
print("    - -1 means infinite iteration")
print()
print("  shuffle_files: bool = False")
print("    - Whether to shuffle record order within epoch")
print("    - Re-shuffles at each epoch boundary")

# %% [markdown]
"""
## Part 2: Creating an ArrayRecord Source

### Basic Usage Pattern

```python
from datarax.sources import ArrayRecordSourceModule, ArrayRecordSourceConfig

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
"""

# %%
print()
print("Creating ArrayRecordSourceModule:")
print()
print("  # Path options:")
print('  paths = "/path/to/data.riegeli"          # Single file')
print('  paths = "/path/to/data-*.riegeli"        # Glob pattern')
print('  paths = ["file1.riegeli", "file2.riegeli"]  # List')
print()
print("  # Initialization:")
print("  source = ArrayRecordSourceModule(")
print("      ArrayRecordSourceConfig(seed=42),")
print("      paths=paths,")
print("      rngs=nnx.Rngs(0),")
print("  )")

# %% [markdown]
"""
## Part 3: Pipeline Integration

### Using with Datarax Pipelines

```python
from datarax import from_source
from datarax.dag.nodes import OperatorNode

# Create pipeline from ArrayRecord source
pipeline = from_source(source, batch_size=32)

# Add transformations
pipeline = pipeline.add(OperatorNode(normalize_op))

# Iterate
for batch in pipeline:
    # Process batch
    print(f"Batch shape: {batch['data'].shape}")
```
"""

# %%
print()
print("Pipeline Integration Pattern:")
print()
print("  # Create pipeline")
print("  pipeline = from_source(source, batch_size=32)")
print()
print("  # Add operators")
print("  pipeline = pipeline.add(OperatorNode(my_operator))")
print()
print("  # Iterate")
print("  for batch in pipeline:")
print("      train_step(batch)")

# %% [markdown]
"""
## Part 4: Checkpointing and State Management

ArrayRecordSourceModule supports full state serialization for training resume.

### State Contents

| State Key | Description |
|-----------|-------------|
| `current_index` | Current position in dataset |
| `current_epoch` | Current epoch number |
| `shuffled_indices` | Shuffle order (if enabled) |
| `prefetch_cache` | Prefetched records cache |

### Checkpointing Pattern

```python
# Save checkpoint
state = source.get_state()
# state = {"current_index": 1234, "current_epoch": 5, ...}

# Later: restore from checkpoint
source.set_state(state)
# Resumes from exact position
```
"""

# %%
print()
print("Checkpointing API:")
print()
print("  # Save state")
print("  checkpoint = {")
print('      "source_state": source.get_state(),')
print('      "model_params": model.params,')
print("  }")
print()
print("  # Restore state")
print('  source.set_state(checkpoint["source_state"])')
print("  # Iteration resumes from saved position")

# %% [markdown]
"""
## Part 5: Epoch Control

### Finite Epochs

```python
# Run for exactly 10 epochs
config = ArrayRecordSourceConfig(num_epochs=10)
source = ArrayRecordSourceModule(config, paths=paths, rngs=nnx.Rngs(0))

for epoch in range(10):
    for batch in from_source(source, batch_size=32):
        train_step(batch)
# Automatically stops after 10 epochs
```

### Infinite Iteration

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
"""

# %%
print()
print("Epoch Control:")
print()
print("  # Finite epochs")
print("  config = ArrayRecordSourceConfig(num_epochs=10)")
print("  # Stops automatically after 10 epochs")
print()
print("  # Infinite iteration (step-based)")
print("  config = ArrayRecordSourceConfig(num_epochs=-1)")
print("  # Use break/max_steps for control")

# %% [markdown]
"""
## Part 6: Shuffling Behavior

### Per-Epoch Reshuffling

When `shuffle_files=True`:

1. At initialization, indices are shuffled using `seed`
2. At each epoch boundary, indices are reshuffled using `seed + epoch`
3. This ensures reproducible but varied order across epochs

```python
# Enable shuffling with reproducible seed
config = ArrayRecordSourceConfig(
    seed=42,
    shuffle_files=True,
)
# Epoch 0: shuffled with seed=42
# Epoch 1: reshuffled with seed=43
# Epoch 2: reshuffled with seed=44
# ...
```
"""

# %%
print()
print("Shuffling Behavior:")
print()
print("  shuffle_files=True:")
print("    - Initial shuffle: seed=42")
print("    - Epoch 1 reshuffle: seed=43")
print("    - Epoch 2 reshuffle: seed=44")
print("    - Ensures varied but reproducible order")

# %% [markdown]
"""
## Results Summary

### ArrayRecordSourceModule Features

| Feature | Description |
|---------|-------------|
| **Stateful** | Tracks position via NNX Variables |
| **Checkpointing** | Full `get_state()` / `set_state()` |
| **Shuffling** | Per-epoch reshuffling with seed control |
| **Epoch Control** | Finite or infinite iteration |
| **Grain Compatible** | Wraps Grain's ArrayRecordDataSource |

### When to Use ArrayRecord

- Large datasets (>10GB)
- Need random access to records
- Working with Google's ML infrastructure
- Migrating from TFRecord to a modern format
"""

# %% [markdown]
"""
## Next Steps

- [HuggingFace Tutorial](../huggingface/hf-tutorial.ipynb) - Alternative data source
- [TFDS Quick Reference](../tfds/tfds-quickref.ipynb) - TensorFlow Datasets
- [Checkpointing Guide](../../advanced/checkpointing/checkpoint-quickref.ipynb) - Full checkpointing
"""


# %%
def main():
    """Run the ArrayRecord quick reference."""
    print("=" * 60)
    print("ArrayRecord Source Quick Reference")
    print("=" * 60)

    print()
    print("This quick reference demonstrates the ArrayRecordSourceModule API.")
    print("Actual usage requires ArrayRecord files (*.riegeli format).")

    print()
    print("Key API Summary:")
    print()
    print("  1. Configuration:")
    print("     config = ArrayRecordSourceConfig(")
    print("         seed=42,")
    print("         num_epochs=-1,")
    print("         shuffle_files=True,")
    print("     )")
    print()
    print("  2. Source Creation:")
    print("     source = ArrayRecordSourceModule(")
    print("         config,")
    print('         paths="/path/to/*.riegeli",')
    print("         rngs=nnx.Rngs(0),")
    print("     )")
    print()
    print("  3. Pipeline Integration:")
    print("     pipeline = from_source(source, batch_size=32)")
    print()
    print("  4. Checkpointing:")
    print("     state = source.get_state()")
    print("     source.set_state(state)")

    print()
    print("=" * 60)
    print("Quick reference completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
