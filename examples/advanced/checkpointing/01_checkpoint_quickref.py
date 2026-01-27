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
# Pipeline Checkpointing Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~10 min |
| **Prerequisites** | Basic Datarax pipeline, JAX fundamentals |
| **Format** | Python + Jupyter |

## Overview

Save and restore data pipeline state to enable resumable processing.
This is essential for long-running data jobs that may be interrupted
and need to continue from where they left off.

## Learning Goals

By the end of this example, you will be able to:

1. Create a `CheckpointableIterator` with proper state management
2. Use `PipelineCheckpoint` to save/restore state
3. Implement resumable data processing loops
4. Handle interrupted jobs gracefully
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax
uv pip install datarax
```
"""

# %%
# Imports
import os
import shutil
import tempfile
from typing import Any

import jax
import jax.numpy as jnp

from datarax.checkpoint import PipelineCheckpoint
from datarax.typing import CheckpointableIterator

print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Step 1: Create Checkpointable Iterator

A `CheckpointableIterator` must implement:

- `get_state()` - Return current iteration state
- `set_state(state)` - Restore from saved state
"""


# %%
class SimplePipeline(CheckpointableIterator[dict[str, jax.Array]]):
    """Data stream with checkpointing support."""

    def __init__(
        self,
        data: jax.Array,
        batch_size: int = 10,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = jax.random.key(seed)
        self.epoch = 0
        self.position = 0
        self.indices = self._create_indices()

    def _create_indices(self) -> jax.Array:
        """Create iteration indices (shuffled or sequential)."""
        indices = jnp.arange(len(self.data))
        if self.shuffle:
            self.rng, key = jax.random.split(self.rng)
            indices = jax.random.permutation(key, indices)
        return indices

    def __iter__(self) -> "SimplePipeline":
        return self

    def iterator(self, pipeline_seed: int | None = None) -> "SimplePipeline":
        """Create iterator with optional new seed."""
        if pipeline_seed is not None:
            self.rng = jax.random.key(pipeline_seed)
        return self

    def __next__(self) -> dict[str, jax.Array]:
        """Get next batch."""
        if self.position >= len(self.data):
            self.epoch += 1
            self.position = 0
            self.indices = self._create_indices()
            raise StopIteration

        start = self.position
        end = min(start + self.batch_size, len(self.data))
        batch_indices = self.indices[start:end]
        self.position = end

        return {"x": self.data[batch_indices]}

    def get_state(self) -> dict[str, Any]:
        """Return checkpoint state."""
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "rng": jax.random.key_data(self.rng),  # Convert key to raw data
            "epoch": self.epoch,
            "position": self.position,
            "indices": self.indices,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint state."""
        self.batch_size = state["batch_size"]
        self.shuffle = state["shuffle"]
        self.seed = state["seed"]
        self.rng = jax.random.wrap_key_data(state["rng"])  # Convert back to key
        self.epoch = state["epoch"]
        self.position = state["position"]
        self.indices = state["indices"]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size


# Create pipeline
data = jnp.arange(50).reshape(50, 1).astype(jnp.float32)
pipeline = SimplePipeline(data, batch_size=10, shuffle=True)
print(f"Pipeline: {len(pipeline)} batches, {len(data)} samples")

# %% [markdown]
"""
## Step 2: Set Up Checkpointing

`PipelineCheckpoint` manages checkpoint files using Orbax.
"""

# %%
# Create checkpoint directory
checkpoint_dir = tempfile.mkdtemp(prefix="datarax_ckpt_")
checkpointer = PipelineCheckpoint(os.path.join(checkpoint_dir, "pipeline_state"))

print(f"Checkpoint directory: {checkpoint_dir}")

# %% [markdown]
"""
## Step 3: Process with Periodic Checkpoints

Save checkpoints at regular intervals to enable resumption.
"""

# %%
# Process data with checkpointing
step = 0
for epoch in range(2):
    print(f"\nEpoch {epoch}:")
    pipeline_iter = pipeline.iterator()

    for batch_idx, batch in enumerate(pipeline_iter):
        batch_mean = jnp.mean(batch["x"]).item()
        step += 1

        print(f"  Batch {batch_idx}: mean={batch_mean:.2f}")

        # Save checkpoint every 3 steps
        if step % 3 == 0:
            save_path = checkpointer.save(
                pipeline,
                step=step,
                metadata={"epoch": epoch, "batch": batch_idx},
                keep=2,  # Keep last 2 checkpoints
                overwrite=True,
            )
            print(f"  -> Saved checkpoint at step {step}")

print(f"\nProcessed {step} total steps")

# %% [markdown]
"""
## Step 4: Restore from Checkpoint

Demonstrate resuming from a saved checkpoint.
"""

# %%
# Create new pipeline (simulating restart)
new_pipeline = SimplePipeline(data, batch_size=10, shuffle=True)
print(f"New pipeline state: epoch={new_pipeline.epoch}, position={new_pipeline.position}")

# Restore from checkpoint
checkpointer.restore_latest(new_pipeline)
print(f"Restored state: epoch={new_pipeline.epoch}, position={new_pipeline.position}")

# Continue processing
print("\nContinuing from checkpoint:")
for batch_idx, batch in enumerate(new_pipeline):
    batch_mean = jnp.mean(batch["x"]).item()
    print(f"  Batch {batch_idx}: mean={batch_mean:.2f}")

# %% [markdown]
"""
## Step 5: Cleanup

Remove checkpoint files when done.
"""

# %%
# Clean up checkpoint directory
shutil.rmtree(checkpoint_dir)
print(f"Cleaned up: {checkpoint_dir}")

# %% [markdown]
"""
## Results Summary

| Feature | Description |
|---------|-------------|
| State Saved | RNG, position, epoch, indices |
| Checkpoint Format | Orbax (efficient, async-capable) |
| Retention | Configurable via `keep` parameter |
| Metadata | Custom fields (epoch, batch, etc.) |

Key benefits:

- **Fault tolerance**: Resume interrupted jobs
- **Incremental processing**: Process data in stages
- **Reproducibility**: Exact state restoration
"""

# %% [markdown]
"""
## Next Steps

- **CLI interface**: Integrate checkpointing with command-line tools
- **Distributed checkpoints**: Coordinate checkpoints across workers
- **Custom handlers**: Implement handlers for special data types
- **Pipeline monitoring**: [Monitoring](../monitoring/01_monitoring_quickref.ipynb)
"""


# %%
def main():
    """Run the checkpoint example."""
    print("Pipeline Checkpointing Example")
    print("=" * 50)

    # Setup
    data = jnp.arange(50).reshape(50, 1).astype(jnp.float32)
    pipeline = SimplePipeline(data, batch_size=10, shuffle=True)

    checkpoint_dir = tempfile.mkdtemp(prefix="datarax_ckpt_")
    checkpointer = PipelineCheckpoint(os.path.join(checkpoint_dir, "pipeline_state"))

    # Process with checkpoints
    step = 0
    for epoch in range(2):
        for batch in pipeline.iterator():
            step += 1
            if step % 5 == 0:
                checkpointer.save(pipeline, step=step, keep=2, overwrite=True)

    # Test restoration
    new_pipeline = SimplePipeline(data, batch_size=10, shuffle=True)
    checkpointer.restore_latest(new_pipeline)

    # Cleanup
    shutil.rmtree(checkpoint_dir)

    print(f"Processed {step} steps with checkpointing")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
