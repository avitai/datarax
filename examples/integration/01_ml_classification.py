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
# ML Classification — Datarax + NNX Tier A and Tier C

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min |
| **Prerequisites** | Pipeline Quickstart, JAX/Flax NNX basics |
| **Format** | Python + Jupyter |

## Overview

Trains a small CNN on synthetic image data, demonstrating both Tier A
(Python iterator) and Tier C (`pipeline.scan`) integration patterns. The
same Pipeline drives both paths so users can choose the integration tier
that matches their training framework.

## Setup

```bash
uv pip install datarax
```

Activate the project virtualenv:

```bash
source activate.sh
```

## Learning Goals

By the end of this example, you will be able to:

1. Build a `Pipeline` with stochastic and deterministic stages.
2. Drive training with the Tier A iterator protocol.
3. Drive training with `Pipeline.scan` for whole-epoch JIT.
4. Compare the wall-clock cost of the two integration tiers.
"""

# %%
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# %% [markdown]
"""
## 1. Model — standard nnx.Module, no datarax-specific base class
"""


# %%
class TinyCNN(nnx.Module):
    """3 conv layers + linear head; the user owns this code entirely."""

    def __init__(self, *, num_classes: int, rngs: nnx.Rngs) -> None:
        """Initialize the module."""
        self.conv1 = nnx.Conv(3, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.head = nnx.Linear(32 * 8 * 8, num_classes, rngs=rngs)

    def __call__(self, image: jax.Array) -> jax.Array:
        """Run __call__."""
        x = nnx.relu(self.conv1(image))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


# %% [markdown]
"""
## 2. Loss — plain function, datarax does not know it exists
"""


# %%
def cross_entropy_loss(model: TinyCNN, batch: dict) -> jax.Array:
    """Run cross_entropy_loss."""
    logits = model(batch["image"])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch["label"]).mean()


# %% [markdown]
"""
## 3. Pipeline stages — image augmentation
"""


# %%
class _BrightnessJitter(nnx.Module):
    """Multiply image brightness by a fixed factor (deterministic)."""

    def __init__(self, factor: float = 1.1) -> None:
        super().__init__()
        self.factor = jnp.float32(factor)

    def __call__(self, batch: dict) -> dict:
        return {**batch, "image": batch["image"] * self.factor}


class _Normalize(nnx.Module):
    """Standardise images to roughly zero-mean unit-variance."""

    def __call__(self, batch: dict) -> dict:
        return {**batch, "image": (batch["image"] - 0.5) / 0.5}


# %% [markdown]
"""
## 4. Synthetic data
"""


# %%
def build_data(num_images: int = 512) -> dict:
    """Run build_data."""
    rng = np.random.default_rng(0)
    images = rng.uniform(0, 1, size=(num_images, 32, 32, 3)).astype(np.float32)
    labels = rng.integers(0, 10, size=(num_images,)).astype(np.int32)
    return {"image": jnp.asarray(images), "label": jnp.asarray(labels)}


# %% [markdown]
"""
## 5. Tier A — Python iterator (compatible with any training framework)
"""


# %%
def train_with_iterator(
    model: TinyCNN, optimizer: nnx.Optimizer, pipeline: Pipeline, num_epochs: int
) -> list[float]:
    """Run train_with_iterator."""

    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(cross_entropy_loss)(model, batch)
        optimizer.update(model, grads)
        return loss

    # Warm-up compile: run one step so the JIT compile is excluded from timing.
    pipeline._position[...] = jnp.int32(0)
    for batch in pipeline:
        train_step(model, optimizer, batch)
        break
    jax.block_until_ready(jnp.asarray(0.0))

    losses_per_epoch: list[float] = []
    for epoch in range(num_epochs):
        pipeline._position[...] = jnp.int32(0)
        epoch_loss = jnp.float32(0.0)
        n_steps = 0
        for batch in pipeline:
            loss = train_step(model, optimizer, batch)
            epoch_loss = epoch_loss + loss
            n_steps += 1
        losses_per_epoch.append(float(epoch_loss / n_steps))
        print(f"  [iterator] epoch {epoch + 1}: loss={losses_per_epoch[-1]:.4f}")
    return losses_per_epoch


# %% [markdown]
"""
## 6. Tier C — pipeline.scan (entire epoch as one XLA graph)
"""


# %%
def train_with_scan(
    model: TinyCNN,
    optimizer: nnx.Optimizer,
    pipeline: Pipeline,
    num_epochs: int,
    steps_per_epoch: int,
) -> list[float]:
    """Run train_with_scan."""

    def step_fn(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(cross_entropy_loss)(model, batch)
        optimizer.update(model, grads)
        return loss

    # Warm-up compile: scan once with the same length so the XLA compile is
    # excluded from per-epoch timing. Subsequent scan() calls with the same
    # length hit the JIT cache.
    pipeline._position[...] = jnp.int32(0)
    pipeline.scan(step_fn, modules=(model, optimizer), length=steps_per_epoch)
    jax.block_until_ready(jnp.asarray(0.0))

    losses_per_epoch: list[float] = []
    for epoch in range(num_epochs):
        pipeline._position[...] = jnp.int32(0)
        losses = pipeline.scan(step_fn, modules=(model, optimizer), length=steps_per_epoch)
        losses_per_epoch.append(float(jnp.mean(losses)))
        print(f"  [scan]     epoch {epoch + 1}: loss={losses_per_epoch[-1]:.4f}")
    return losses_per_epoch


# %% [markdown]
"""
## 7. Driver
"""


# %%
def main() -> None:
    """Run main."""
    print("=" * 70)
    print("ML classification: TinyCNN trained on synthetic data")
    print("=" * 70)

    data = build_data(num_images=512)
    batch_size = 32
    num_epochs = 3
    steps_per_epoch = data["image"].shape[0] // batch_size

    print("\nTier A: for batch in pipeline (compatible with any training framework)")
    print("-" * 70)
    pipeline_a = Pipeline(
        source=MemorySource(MemorySourceConfig(shuffle=False), data),
        stages=[_BrightnessJitter(1.1), _Normalize()],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    model_a = TinyCNN(num_classes=10, rngs=nnx.Rngs(0))
    optimizer_a = nnx.Optimizer(model_a, optax.adam(1e-3), wrt=nnx.Param)

    t0 = time.perf_counter()
    train_with_iterator(model_a, optimizer_a, pipeline_a, num_epochs=num_epochs)
    iterator_time = time.perf_counter() - t0

    print(f"\n  Wall-clock: {iterator_time:.3f}s")

    print("\nTier C: pipeline.scan (entire epoch as one XLA graph)")
    print("-" * 70)
    pipeline_c = Pipeline(
        source=MemorySource(MemorySourceConfig(shuffle=False), data),
        stages=[_BrightnessJitter(1.1), _Normalize()],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )
    model_c = TinyCNN(num_classes=10, rngs=nnx.Rngs(0))
    optimizer_c = nnx.Optimizer(model_c, optax.adam(1e-3), wrt=nnx.Param)

    t0 = time.perf_counter()
    train_with_scan(
        model_c,
        optimizer_c,
        pipeline_c,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    scan_time = time.perf_counter() - t0

    print(f"\n  Wall-clock: {scan_time:.3f}s")

    print(
        "\nBoth tiers train the model identically. Wall-clock varies with "
        "workload size — see benchmarks/ for rigorous comparisons."
    )


if __name__ == "__main__":
    main()
