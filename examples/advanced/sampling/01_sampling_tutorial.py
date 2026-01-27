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
# Advanced Sampling Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~25 min |
| **Prerequisites** | Pipeline Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master the sampling system in Datarax. This tutorial covers the built-in
samplers for controlling data access order, from simple sequential access
to epoch-aware shuffling with callbacks.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Use `SequentialSamplerModule` for deterministic iteration
2. Apply `ShuffleSampler` for randomized data access
3. Work with `RangeSampler` for subset selection
4. Configure `EpochAwareSamplerModule` with callbacks
5. Implement custom samplers following Datarax patterns
"""

# %% [markdown]
"""
## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `SequentialSampler` | `SequentialSamplerModule` |
| `RandomSampler` | `ShuffleSampler` |
| `SubsetRandomSampler` | `RangeSampler` |
| Custom `Sampler` class | Extend `SamplerModule` |

## Coming from TensorFlow?

| TensorFlow tf.data | Datarax |
|--------------------|---------|
| Default order | `SequentialSamplerModule` |
| `.shuffle(buffer_size)` | `ShuffleSampler(buffer_size)` |
| `.take(n)` | `RangeSampler(stop=n)` |
| Epoch callbacks | `EpochAwareSamplerModule` |
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[data]"
```
"""

# %%
# Imports
from flax import nnx

from datarax.samplers import (
    SequentialSamplerModule,
    SequentialSamplerConfig,
    ShuffleSampler,
    ShuffleSamplerConfig,
    RangeSampler,
    RangeSamplerConfig,
    EpochAwareSamplerModule,
    EpochAwareSamplerConfig,
)

print("Advanced Sampling Tutorial")
print("=" * 50)

# %% [markdown]
"""
## Part 1: Sampler Architecture

All Datarax samplers inherit from `SamplerModule`, which provides:

- NNX-based state management
- Checkpointing support via `get_state()` / `set_state()`
- Iteration protocol (`__iter__`, `__next__`)
- Consistent API across all samplers

### Sampler Hierarchy

```
SamplerModule (base)
├── SequentialSamplerModule  - Sequential indices
├── ShuffleSampler           - Buffered shuffle
├── RangeSampler             - Range-based iteration
└── EpochAwareSamplerModule  - Epoch tracking + callbacks
```
"""

# %%
print("Sampler Types:")
print()
print("  SequentialSamplerModule - Sequential indices [0, 1, 2, ...]")
print("  ShuffleSampler          - Randomized order with buffer")
print("  RangeSampler            - Custom range like Python range()")
print("  EpochAwareSamplerModule - Epoch tracking with callbacks")

# %% [markdown]
"""
## Part 2: SequentialSamplerModule

The simplest sampler - iterates through indices in order.
Ideal for evaluation and deterministic pipelines.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_records` | int | Required | Total dataset size |
| `num_epochs` | int | 1 | Number of epochs (-1 for infinite) |
"""

# %%
# Create sequential sampler
sequential_config = SequentialSamplerConfig(
    num_records=100,  # Dataset has 100 samples
    num_epochs=2,  # Iterate for 2 epochs
)
sequential_sampler = SequentialSamplerModule(sequential_config, rngs=nnx.Rngs(0))

# Collect indices from first epoch
indices_epoch1 = []
for idx in sequential_sampler:
    indices_epoch1.append(idx)
    if len(indices_epoch1) >= 10:
        break  # Just show first 10

print()
print("SequentialSamplerModule:")
print(f"  First 10 indices: {indices_epoch1}")
print(f"  Total length: {len(sequential_sampler)} (2 epochs × 100 records)")

# %% [markdown]
"""
## Part 3: ShuffleSampler

Randomizes data access order using a buffer-based approach.
Essential for training to prevent learning from data order.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `buffer_size` | int | Required | Size of shuffle buffer |
| `dataset_size` | int | None | Optional dataset size |
| `seed` | int | None | Seed for reproducibility |

### How It Works

1. Fill buffer with indices
2. Shuffle buffer randomly
3. Yield indices from buffer
4. Refill and reshuffle as needed
"""

# %%
# Create shuffle sampler with reproducible seed
shuffle_config = ShuffleSamplerConfig(
    buffer_size=50,  # Shuffle buffer size
    dataset_size=100,  # Dataset has 100 samples
    seed=42,  # For reproducibility
)
shuffle_sampler = ShuffleSampler(shuffle_config, rngs=nnx.Rngs(shuffle=42))

# Collect some shuffled indices
shuffled_indices = list(shuffle_sampler)[:10]

print()
print("ShuffleSampler:")
print("  Buffer size: 50")
print(f"  First 10 shuffled indices: {shuffled_indices}")
print("  Indices are randomized, not [0, 1, 2, ...]")

# %%
# Demonstrate reproducibility with same seed
shuffle_sampler2 = ShuffleSampler(shuffle_config, rngs=nnx.Rngs(shuffle=42))
shuffled_indices2 = list(shuffle_sampler2)[:10]

print()
print("Reproducibility test:")
print(f"  Same seed produces same order: {shuffled_indices == shuffled_indices2}")

# %% [markdown]
"""
## Part 4: RangeSampler

Creates a sequence of integers like Python's `range()`.
Useful for subset selection or custom index patterns.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | int | 0 | Start of range (inclusive) |
| `stop` | int | None | End of range (exclusive) |
| `step` | int | 1 | Step between indices |
"""

# %%
# Create range sampler for first 50 samples
range_config = RangeSamplerConfig(
    start=0,
    stop=50,  # First 50 samples
    step=1,
)
range_sampler = RangeSampler(range_config, rngs=nnx.Rngs(0))

print()
print("RangeSampler (0 to 50):")
print(f"  Length: {len(range_sampler)}")
print("  Indices: [0, 1, 2, ..., 49]")

# %%
# Range with step - useful for strided access
strided_config = RangeSamplerConfig(
    start=0,
    stop=100,
    step=10,  # Every 10th sample
)
strided_sampler = RangeSampler(strided_config, rngs=nnx.Rngs(0))

strided_indices = list(strided_sampler)
print()
print("RangeSampler with step=10:")
print(f"  Indices: {strided_indices}")
print("  Useful for: validation subsets, quick testing")

# %%
# Range for second half of dataset
second_half_config = RangeSamplerConfig(
    start=50,
    stop=100,  # Samples 50-99
)
second_half = RangeSampler(second_half_config, rngs=nnx.Rngs(0))

print()
print("RangeSampler for second half (50-100):")
print(f"  Length: {len(second_half)}")
print(f"  First few: {list(second_half)[:5]}...")

# %% [markdown]
"""
## Part 5: EpochAwareSamplerModule

Advanced sampler with explicit epoch boundary handling and callbacks.
Ideal for training loops that need epoch-level events.

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_records` | int | Required | Total dataset size |
| `num_epochs` | int | 1 | Number of epochs (-1 for infinite) |
| `shuffle` | bool | True | Shuffle each epoch |
| `seed` | int | 42 | Base seed for shuffling |

### Features

- Per-epoch reshuffling with deterministic seed
- Epoch completion callbacks
- Progress tracking
"""

# %%
# Create epoch-aware sampler with shuffle
epoch_config = EpochAwareSamplerConfig(
    num_records=100,
    num_epochs=3,  # 3 epochs
    shuffle=True,  # Shuffle each epoch differently
    seed=42,
)
epoch_sampler = EpochAwareSamplerModule(epoch_config, rngs=nnx.Rngs(sample=42))

# Add epoch completion callback
epoch_completions = []


def on_epoch_complete(epoch: int):
    """Callback fired at end of each epoch."""
    epoch_completions.append(epoch)
    print(f"    [Callback] Epoch {epoch} completed!")


epoch_sampler.add_epoch_callback(on_epoch_complete)

print()
print("EpochAwareSamplerModule:")
print("  num_records=100, num_epochs=3, shuffle=True")
print()
print("  Running through epochs:")

# %%
# Iterate and observe epoch boundaries
sample_count = 0
epoch_boundaries = []

for idx in epoch_sampler:
    sample_count += 1
    # Check progress periodically
    if sample_count % 100 == 0:
        progress = epoch_sampler.get_epoch_progress()
        print(
            f"    Progress: Epoch {progress['current_epoch']}, "
            f"Index {progress['current_index']}/{progress['records_per_epoch']}"
        )

print()
print(f"  Total samples yielded: {sample_count}")
print(f"  Epochs completed (via callback): {epoch_completions}")

# %% [markdown]
"""
## Part 6: Checkpointing Samplers

All samplers support state serialization for training resume.

### State Management API

```python
# Save state
state = sampler.get_state()

# Restore state
sampler.set_state(state)
```
"""

# %%
# Demonstrate checkpointing
seq_sampler = SequentialSamplerModule(
    SequentialSamplerConfig(num_records=100, num_epochs=1),
    rngs=nnx.Rngs(0),
)

# Iterate partway through
iter_sampler = iter(seq_sampler)
for _ in range(42):
    next(iter_sampler)

# Save checkpoint
checkpoint = seq_sampler.get_state()
print()
print("Checkpointing:")
print("  After 42 iterations:")
print(f"    current_index: {checkpoint.get('current_index')}")
print(f"    current_epoch: {checkpoint.get('current_epoch')}")

# %%
# Restore and continue
new_sampler = SequentialSamplerModule(
    SequentialSamplerConfig(num_records=100, num_epochs=1),
    rngs=nnx.Rngs(0),
)
new_sampler.set_state(checkpoint)

# Continue from checkpoint - should resume at 42
remaining = list(new_sampler)
print()
print(f"  After restore, first index: {remaining[0] if remaining else 'N/A'}")
print(f"  Resumed correctly from position 42: {remaining[0] == 42 if remaining else False}")

# %% [markdown]
"""
## Part 7: Sampler Selection Guide

| Use Case | Recommended Sampler |
|----------|---------------------|
| Evaluation / Testing | `SequentialSamplerModule` |
| Training (shuffle) | `ShuffleSampler` or `EpochAwareSamplerModule` |
| Subset selection | `RangeSampler` |
| Epoch callbacks | `EpochAwareSamplerModule` |
| Reproducible shuffle | `ShuffleSampler(seed=...)` |
| Cross-validation folds | `RangeSampler` with different ranges |
"""

# %%
print()
print("Sampler Selection Summary:")
print()
print("  Evaluation:      SequentialSamplerModule (deterministic)")
print("  Training:        ShuffleSampler (randomized)")
print("  Epoch events:    EpochAwareSamplerModule (callbacks)")
print("  Subset/Slice:    RangeSampler (range-based)")

# %% [markdown]
"""
## Results Summary

### Sampler Types

| Sampler | Stochastic | Checkpointable | Key Feature |
|---------|------------|----------------|-------------|
| `SequentialSamplerModule` | No | Yes | Deterministic order |
| `ShuffleSampler` | Yes | Yes | Buffered shuffle |
| `RangeSampler` | No | Yes | Custom ranges |
| `EpochAwareSamplerModule` | Configurable | Yes | Epoch callbacks |

### Common Patterns

```python
# Training: shuffle with reproducibility
shuffle_sampler = ShuffleSampler(
    ShuffleSamplerConfig(buffer_size=1000, seed=42),
    rngs=nnx.Rngs(shuffle=42),
)

# Evaluation: sequential for reproducibility
eval_sampler = SequentialSamplerModule(
    SequentialSamplerConfig(num_records=len(dataset)),
    rngs=nnx.Rngs(0),
)

# Training with callbacks
epoch_sampler = EpochAwareSamplerModule(
    EpochAwareSamplerConfig(num_records=len(dataset), shuffle=True),
    rngs=nnx.Rngs(sample=42),
)
epoch_sampler.add_epoch_callback(save_checkpoint)
```
"""

# %% [markdown]
"""
## Next Steps

- [Pipeline Tutorial](../../core/02_pipeline_tutorial.ipynb) - Full pipeline setup
- [Checkpointing Guide](../checkpointing/01_checkpoint_quickref.ipynb) - Resume training
- [Distributed Sharding](../distributed/01_sharding_quickref.ipynb) - Multi-device
"""


# %%
def main():
    """Run the advanced sampling tutorial."""
    print("=" * 60)
    print("Advanced Sampling Tutorial")
    print("=" * 60)

    # Demo 1: Sequential
    print()
    print("1. SequentialSamplerModule:")
    seq = SequentialSamplerModule(
        SequentialSamplerConfig(num_records=100, num_epochs=1),
        rngs=nnx.Rngs(0),
    )
    indices = list(seq)[:5]
    print(f"   First 5 indices: {indices}")

    # Demo 2: Shuffle
    print()
    print("2. ShuffleSampler:")
    shuf = ShuffleSampler(
        ShuffleSamplerConfig(buffer_size=50, seed=42),
        rngs=nnx.Rngs(shuffle=42),
    )
    shuffled = list(shuf)[:5]
    print(f"   First 5 shuffled: {shuffled}")

    # Demo 3: Range
    print()
    print("3. RangeSampler:")
    rng_samp = RangeSampler(
        RangeSamplerConfig(start=10, stop=20),
        rngs=nnx.Rngs(0),
    )
    range_indices = list(rng_samp)
    print(f"   Range 10-20: {range_indices}")

    # Demo 4: Epoch-aware
    print()
    print("4. EpochAwareSamplerModule:")
    epoch = EpochAwareSamplerModule(
        EpochAwareSamplerConfig(num_records=50, num_epochs=2, shuffle=True, seed=42),
        rngs=nnx.Rngs(sample=42),
    )
    count = sum(1 for _ in epoch)
    print(f"   Total samples (2 epochs × 50): {count}")

    print()
    print("=" * 60)
    print("Tutorial completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
