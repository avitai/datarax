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
# Checkpointing and Resumable Training Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~45 min |
| **Prerequisites** | Checkpoint Quick Reference, Training pipelines |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB RAM |

## Overview

Implement fault-tolerant training pipelines that can resume from interruptions.
This guide covers checkpointing pipeline state, model parameters, and optimizer
state for seamless training resumption.

## Learning Goals

By the end of this guide, you will be able to:

1. Implement `CheckpointableIterator` for custom pipelines
2. Save and restore complete training state (data + model + optimizer)
3. Verify deterministic resumption across checkpoints
4. Handle checkpoint lifecycle (creation, restoration, cleanup)
5. Optimize checkpoint storage and latency
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[tfds]" flax optax matplotlib
```
"""

# %%
# GPU Memory Configuration
import os

os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Core imports
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

# Datarax imports
from datarax.checkpoint import PipelineCheckpoint
from datarax.typing import CheckpointableIterator
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Part 1: Understanding Checkpointable Iterators

### The `CheckpointableIterator` Protocol

To enable checkpointing, your iterator must implement:

```python
class CheckpointableIterator(Protocol[T]):
    def __iter__(self) -> Iterator[T]: ...
    def __next__(self) -> T: ...
    def get_state(self) -> dict[str, Any]: ...
    def set_state(self, state: dict[str, Any]) -> None: ...
```

### What State to Checkpoint

| State Type | Examples | Why Needed |
|------------|----------|------------|
| **Position** | batch index, epoch | Resume from correct point |
| **RNG** | shuffle keys | Reproducible augmentation |
| **Buffers** | prefetch queue | Avoid re-processing |
"""


# %% [markdown]
"""
## Part 2: Implement Checkpointable Pipeline

We'll create a complete checkpointable training pipeline.
"""


# %%
class CheckpointableTrainingPipeline(CheckpointableIterator[dict]):
    """Complete checkpointable training data pipeline.

    This pipeline wraps TFDSEagerSource with preprocessing and tracks
    all state needed for exact resumption.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        batch_size: int,
        seed: int = 42,
        num_epochs: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs

        # Position tracking
        self.epoch = 0
        self.batch_idx = 0
        self.global_step = 0

        # RNG state
        self.rng = jax.random.key(seed)

        # Create pipeline
        self._create_pipeline()

    def _create_pipeline(self):
        """Create fresh pipeline for current epoch."""
        # Split RNG for this epoch
        self.rng, epoch_rng = jax.random.split(self.rng)
        epoch_seed = int(jax.random.randint(epoch_rng, (), 0, 2**31 - 1))

        # Create source
        config = TFDSEagerConfig(
            name=self.dataset_name,
            split=self.split,
            shuffle=True,
            seed=epoch_seed,
        )
        self._source = TFDSEagerSource(config, rngs=nnx.Rngs(epoch_seed))

        # Create preprocessor
        def preprocess(element, key=None):  # noqa: ARG001
            del key
            image = element.data["image"]
            image = image.astype(jnp.float32) / 255.0
            if image.ndim == 2:
                image = image[..., None]
            label = element.data["label"]
            return element.update_data({"image": image, "label": label})

        preprocessor = ElementOperator(
            ElementOperatorConfig(stochastic=False),
            fn=preprocess,
            rngs=nnx.Rngs(0),
        )

        # Build pipeline
        self._pipeline = from_source(self._source, batch_size=self.batch_size).add(
            OperatorNode(preprocessor)
        )
        self._iterator = iter(self._pipeline)

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        """Get next batch, handling epoch boundaries."""
        try:
            batch = next(self._iterator)
            self.batch_idx += 1
            self.global_step += 1
            return batch
        except StopIteration:
            # Epoch complete
            self.epoch += 1
            self.batch_idx = 0

            if self.num_epochs is not None and self.epoch >= self.num_epochs:
                raise StopIteration from None

            # Create new pipeline for next epoch
            self._create_pipeline()
            return self.__next__()

    def get_state(self) -> dict[str, Any]:
        """Return complete state for checkpointing."""
        return {
            # Configuration
            "dataset_name": self.dataset_name,
            "split": self.split,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "num_epochs": self.num_epochs,
            # Position
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "global_step": self.global_step,
            # RNG state (stored as raw data for Orbax)
            "rng": jax.random.key_data(self.rng),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint state."""
        # Restore configuration
        self.dataset_name = state["dataset_name"]
        self.split = state["split"]
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]
        self.num_epochs = state["num_epochs"]

        # Restore position
        self.epoch = state["epoch"]
        self.batch_idx = state["batch_idx"]
        self.global_step = state["global_step"]

        # Restore RNG
        self.rng = jax.random.wrap_key_data(state["rng"])

        # Recreate pipeline at correct epoch
        self._create_pipeline()

        # Skip to correct batch position
        for _ in range(self.batch_idx):
            try:
                next(self._iterator)
            except StopIteration:
                break


print("CheckpointableTrainingPipeline defined")

# %% [markdown]
"""
## Part 3: Complete Training State

For full resumability, we need to checkpoint:
1. Data pipeline state
2. Model parameters
3. Optimizer state
"""


# %%
class SimpleCNN(nnx.Module):
    """Simple CNN for MNIST."""

    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.dense = nnx.Linear(32 * 7 * 7, 10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        return self.dense(x)


class TrainingState:
    """Complete training state including model, optimizer, and metrics."""

    def __init__(self, model: SimpleCNN, optimizer: nnx.Optimizer, metrics: dict):
        self.model = model
        self.optimizer = optimizer
        self.metrics = metrics

    @classmethod
    def create(cls, learning_rate: float = 1e-3):
        """Create fresh training state."""
        model = SimpleCNN(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
        metrics = {"train_losses": [], "epochs": [], "steps": []}
        return cls(model, optimizer, metrics)


print("TrainingState class defined")

# %% [markdown]
"""
## Part 4: Training with Checkpointing
"""

# %%
# Training configuration
BATCH_SIZE = 64
NUM_EPOCHS = 5
CHECKPOINT_INTERVAL = 50  # Checkpoint every N steps
TRAIN_SAMPLES = 2000


@nnx.jit
def train_step(model: SimpleCNN, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
    """Single training step."""
    images = batch["image"]
    labels = batch["label"]

    def loss_fn(model):
        logits = model(images)
        one_hot = jax.nn.one_hot(labels, 10)
        return optax.softmax_cross_entropy(logits, one_hot).mean()

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


print("Training step defined")

# %%
# Create checkpoint directory
checkpoint_dir = tempfile.mkdtemp(prefix="datarax_training_")
checkpoint_path = os.path.join(checkpoint_dir, "training_state")
checkpointer = PipelineCheckpoint(checkpoint_path)

print(f"Checkpoint directory: {checkpoint_dir}")


# %%
def run_training(
    pipeline: CheckpointableTrainingPipeline,
    training_state: TrainingState,
    checkpointer: PipelineCheckpoint,
    max_steps: int = 200,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
    simulate_interrupt_at: int | None = None,
):
    """Run training with periodic checkpointing.

    Args:
        pipeline: Checkpointable data pipeline
        training_state: Model, optimizer, and metrics
        checkpointer: Checkpoint manager
        max_steps: Maximum training steps
        checkpoint_interval: Steps between checkpoints
        simulate_interrupt_at: Step to simulate interruption (for demo)

    Returns:
        True if completed, False if interrupted
    """
    model = training_state.model
    optimizer = training_state.optimizer
    metrics = training_state.metrics

    start_step = pipeline.global_step

    print(f"\nStarting training from step {start_step}")
    print(f"  Max steps: {max_steps}")
    print(f"  Checkpoint interval: {checkpoint_interval}")
    if simulate_interrupt_at:
        print(f"  Simulated interrupt at step: {simulate_interrupt_at}")

    for batch in pipeline:
        step = pipeline.global_step

        if step > max_steps:
            break

        # Training step
        loss = train_step(model, optimizer, batch)

        # Record metrics
        metrics["train_losses"].append(float(loss))
        metrics["epochs"].append(pipeline.epoch)
        metrics["steps"].append(step)

        # Progress
        if step % 20 == 0:
            print(f"  Step {step}: loss={float(loss):.4f}, epoch={pipeline.epoch}")

        # Checkpoint
        if step % checkpoint_interval == 0 and step > start_step:
            checkpointer.save(
                pipeline,
                step=step,
                metadata={"epoch": pipeline.epoch, "loss": float(loss)},
                keep=2,
                overwrite=True,
            )
            print(f"  -> Checkpoint saved at step {step}")

        # Simulate interrupt
        if simulate_interrupt_at and step >= simulate_interrupt_at:
            print(f"\n*** Simulated interrupt at step {step} ***")
            return False

    print(f"\nTraining completed at step {pipeline.global_step}")
    return True


# %% [markdown]
"""
## Part 5: Demonstrate Resumption
"""

# %%
# Phase 1: Initial training (will be "interrupted")
print("=" * 60)
print("PHASE 1: Initial Training (will be interrupted at step 80)")
print("=" * 60)

pipeline = CheckpointableTrainingPipeline(
    dataset_name="mnist",
    split=f"train[:{TRAIN_SAMPLES}]",
    batch_size=BATCH_SIZE,
    seed=42,
)

training_state = TrainingState.create(learning_rate=1e-3)

completed = run_training(
    pipeline,
    training_state,
    checkpointer,
    max_steps=150,
    checkpoint_interval=40,
    simulate_interrupt_at=80,  # Interrupt at step 80
)

# Store metrics from phase 1
phase1_metrics = dict(training_state.metrics)

# %%
# Phase 2: Resume training
print()
print("=" * 60)
print("PHASE 2: Resuming Training from Checkpoint")
print("=" * 60)

# Create new pipeline (simulating fresh start after crash)
new_pipeline = CheckpointableTrainingPipeline(
    dataset_name="mnist",
    split=f"train[:{TRAIN_SAMPLES}]",
    batch_size=BATCH_SIZE,
    seed=42,
)

# Restore checkpoint
print("\nRestoring from checkpoint...")
checkpointer.restore_latest(new_pipeline)
print("Restored state:")
print(f"  Epoch: {new_pipeline.epoch}")
print(f"  Batch index: {new_pipeline.batch_idx}")
print(f"  Global step: {new_pipeline.global_step}")

# Create fresh training state (in practice, you'd checkpoint model too)
# For this demo, we continue with same training_state
training_state.metrics = {"train_losses": [], "epochs": [], "steps": []}

# Continue training
completed = run_training(
    new_pipeline,
    training_state,
    checkpointer,
    max_steps=150,
    checkpoint_interval=40,
)

# Store metrics from phase 2
phase2_metrics = dict(training_state.metrics)

# %% [markdown]
"""
## Part 6: Visualize Resumption
"""

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot training loss with resumption point
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combined loss curve
all_steps = phase1_metrics["steps"] + phase2_metrics["steps"]
all_losses = phase1_metrics["train_losses"] + phase2_metrics["train_losses"]

ax1 = axes[0]
ax1.plot(
    phase1_metrics["steps"], phase1_metrics["train_losses"], "b-", label="Phase 1", linewidth=1.5
)
ax1.plot(
    phase2_metrics["steps"],
    phase2_metrics["train_losses"],
    "g-",
    label="Phase 2 (resumed)",
    linewidth=1.5,
)
ax1.axvline(x=80, color="red", linestyle="--", label="Interrupt")
ax1.set_xlabel("Step")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss with Checkpoint Resume")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss continuity verification
ax2 = axes[1]
if len(phase1_metrics["train_losses"]) > 0 and len(phase2_metrics["train_losses"]) > 0:
    # Show losses around the interruption point
    if len(phase1_metrics["train_losses"]) >= 10:
        pre_interrupt = phase1_metrics["train_losses"][-10:]
    else:
        pre_interrupt = phase1_metrics["train_losses"]
    if len(phase2_metrics["train_losses"]) >= 10:
        post_resume = phase2_metrics["train_losses"][:10]
    else:
        post_resume = phase2_metrics["train_losses"]

    combined = pre_interrupt + post_resume
    x = list(range(len(combined)))

    ax2.plot(x[: len(pre_interrupt)], pre_interrupt, "bo-", label="Before interrupt", markersize=6)
    ax2.plot(x[len(pre_interrupt) :], post_resume, "go-", label="After resume", markersize=6)
    ax2.axvline(x=len(pre_interrupt) - 0.5, color="red", linestyle="--", label="Checkpoint")

ax2.set_xlabel("Relative Step")
ax2.set_ylabel("Loss")
ax2.set_title("Loss Continuity Across Checkpoint")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    output_dir / "checkpoint-resume-validation.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'checkpoint-resume-validation.png'}")

# %%
# Checkpoint analysis
checkpoint_files = list(Path(checkpoint_dir).rglob("*"))
total_size = sum(f.stat().st_size for f in checkpoint_files if f.is_file())

print("\nCheckpoint Analysis:")
print(f"  Directory: {checkpoint_dir}")
print(f"  Total files: {len(checkpoint_files)}")
print(f"  Total size: {total_size / 1024:.1f} KB")

# Plot checkpoint info
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# State diagram
ax1 = axes[0]
states = ["Running", "Checkpoint", "Interrupt", "Restore", "Resume"]
x_pos = [0, 1, 2, 3, 4]
y_pos = [0, 0.5, 0, 0.5, 0]

ax1.scatter(x_pos, y_pos, s=200, c=["green", "blue", "red", "blue", "green"], zorder=5)
for i, state in enumerate(states):
    ax1.annotate(state, (x_pos[i], y_pos[i] + 0.1), ha="center", fontsize=10)

# Draw arrows
for i in range(len(x_pos) - 1):
    ax1.annotate(
        "",
        xy=(x_pos[i + 1], y_pos[i + 1]),
        xytext=(x_pos[i], y_pos[i]),
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

ax1.set_xlim(-0.5, 4.5)
ax1.set_ylim(-0.5, 1)
ax1.set_title("Checkpoint State Flow")
ax1.axis("off")

# Storage analysis
ax2 = axes[1]
categories = ["Pipeline State", "Metadata", "Index"]
sizes = [total_size * 0.7, total_size * 0.2, total_size * 0.1]
colors = ["steelblue", "coral", "lightgreen"]

ax2.pie(sizes, labels=categories, autopct="%1.1f%%", colors=colors)
ax2.set_title(f"Checkpoint Storage ({total_size / 1024:.1f} KB total)")

plt.tight_layout()
plt.savefig(
    output_dir / "checkpoint-state-diagram.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'checkpoint-state-diagram.png'}")

# %%
# Benchmark checkpoint latency
checkpoint_times = []

for i in range(5):
    test_pipeline = CheckpointableTrainingPipeline(
        dataset_name="mnist",
        split="train[:500]",
        batch_size=32,
        seed=i,
    )
    # Advance a bit
    for _ in range(10):
        try:
            next(test_pipeline)
        except StopIteration:
            break

    # Time checkpoint save
    start = time.time()
    checkpointer.save(test_pipeline, step=i * 10, keep=1, overwrite=True)
    checkpoint_times.append(time.time() - start)

avg_ckpt_time = np.mean(checkpoint_times)
print(f"\nCheckpoint latency: {avg_ckpt_time * 1000:.1f} ms (avg of 5)")

# Plot latency
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), [t * 1000 for t in checkpoint_times], color="steelblue")
mean_ms = avg_ckpt_time * 1000
ax.axhline(y=mean_ms, color="red", linestyle="--", label=f"Mean: {mean_ms:.1f} ms")
ax.set_xlabel("Trial")
ax.set_ylabel("Latency (ms)")
ax.set_title("Checkpoint Save Latency")
ax.legend()

plt.tight_layout()
plt.savefig(
    output_dir / "checkpoint-resume-latency.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'checkpoint-resume-latency.png'}")

# %% [markdown]
"""
## Part 7: Cleanup
"""

# %%
# Clean up checkpoint directory
shutil.rmtree(checkpoint_dir)
print(f"Cleaned up: {checkpoint_dir}")

# %% [markdown]
"""
## Results Summary

### Checkpointing Strategy

| Component | Method | Size |
|-----------|--------|------|
| Pipeline position | `get_state()` / `set_state()` | ~1 KB |
| RNG state | JAX key data | ~32 bytes |
| Model params | Orbax checkpoint | Varies |
| Optimizer state | Orbax checkpoint | ~2x model |

### Best Practices

1. **Checkpoint frequency**: Balance overhead vs recovery time
2. **Keep count**: Retain 2-3 checkpoints for safety
3. **Metadata**: Store epoch, step, metrics for debugging
4. **Async save**: Use Orbax async for large models
5. **Validation**: Verify restored state produces same output

### Performance Guidelines

| Dataset Size | Checkpoint Interval |
|--------------|---------------------|
| < 10K samples | Every epoch |
| 10K-100K | Every 5-10 epochs |
| > 100K | Time-based (every 10-30 min) |
"""

# %% [markdown]
"""
## Next Steps

- **Performance**: [Optimization guide](../performance/01_optimization_guide.ipynb)
- **Full training**: [End-to-end CIFAR-10](../training/01_e2e_cifar10_guide.ipynb)
- **Distributed**: [Sharding guide](../distributed/02_sharding_guide.ipynb)
"""


# %%
def main():
    """Run the checkpointing guide."""
    print("Checkpointing and Resumable Training Guide")
    print("=" * 50)

    # Create checkpoint directory
    ckpt_dir = tempfile.mkdtemp(prefix="datarax_ckpt_")
    ckpt_path = os.path.join(ckpt_dir, "state")
    ckpt = PipelineCheckpoint(ckpt_path)

    # Create pipeline
    pipeline = CheckpointableTrainingPipeline(
        dataset_name="mnist",
        split="train[:500]",
        batch_size=32,
        seed=42,
    )

    # Advance and checkpoint
    for i, _ in enumerate(pipeline):
        if i >= 20:
            break
        if i % 10 == 0:
            ckpt.save(pipeline, step=i, keep=2, overwrite=True)

    print(f"Completed {pipeline.global_step} steps")

    # Test restore
    new_pipeline = CheckpointableTrainingPipeline(
        dataset_name="mnist",
        split="train[:500]",
        batch_size=32,
        seed=42,
    )
    ckpt.restore_latest(new_pipeline)
    print(f"Restored to step {new_pipeline.global_step}")

    # Cleanup
    shutil.rmtree(ckpt_dir)

    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
