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
# Pipeline Quickstart

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min |
| **Prerequisites** | Basic Python, JAX/Flax NNX fundamentals |
| **Format** | Python + Jupyter |

## Overview

This quickstart demonstrates the recommended way to build a data
pipeline in datarax. It uses the `Pipeline` class — an `nnx.Module`
that composes a data source with a list of stages and exposes both a
Python iterator (`for batch in pipeline`) and a JIT-friendly scan-based
epoch driver (`pipeline.scan`).

## Learning goals

By the end of this example, you will be able to:

1. Create a `MemorySource` from dictionary data.
2. Build a `Pipeline` with stages that transform batches.
3. Iterate the pipeline with the standard `for batch in pipeline`
   protocol.
4. Drive the pipeline with `pipeline.scan(...)` for whole-epoch JIT.

For a more complete training-loop walkthrough, see
`examples/integration/01_ml_classification.py`.

## Setup

```bash
uv pip install datarax
```

Activate the project virtualenv:

```bash
source activate.sh
```
"""

# %%
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


# %% [markdown]
"""
## Step 1 — Create the data

datarax sources work with dictionary-keyed batches.
"""

# %%
num_samples = 256
image_shape = (28, 28, 1)

rng = np.random.default_rng(0)
data = {
    "image": rng.uniform(0, 1, size=(num_samples, *image_shape)).astype(np.float32),
    "label": rng.integers(0, 10, size=(num_samples,)).astype(np.int32),
}

print(f"data: image={data['image'].shape}, label={data['label'].shape}")


# %% [markdown]
"""
## Step 2 — Define stages

A stage is any `nnx.Module` whose `__call__(batch) -> batch` transforms
the batch. No datarax-specific base class is required.
"""


# %%
class Brightness(nnx.Module):
    """Multiply image by a fixed factor."""

    def __init__(self, factor: float = 1.1) -> None:
        """Initialize the module."""
        super().__init__()
        self.factor = jnp.float32(factor)

    def __call__(self, batch: dict) -> dict:
        """Run __call__."""
        return {**batch, "image": batch["image"] * self.factor}


class Normalize(nnx.Module):
    """Standardize images to roughly zero-mean, unit-variance."""

    def __call__(self, batch: dict) -> dict:
        """Run __call__."""
        return {**batch, "image": (batch["image"] - 0.5) / 0.5}


# %% [markdown]
"""
## Step 3 — Build the pipeline

Pass the source, the ordered list of stages, the batch size, and an
`nnx.Rngs` instance for any stochastic stages.
"""

# %%
pipeline = Pipeline(
    source=MemorySource(MemorySourceConfig(shuffle=False), data),
    stages=[Brightness(factor=1.1), Normalize()],
    batch_size=32,
    rngs=nnx.Rngs(0),
)

print(f"pipeline: source length = {len(pipeline.source)}, batch_size = {pipeline.batch_size}")


# %% [markdown]
"""
## Step 4a — Iterate (Tier A)

`for batch in pipeline` is the lowest-friction way to consume a
pipeline. Works with any training framework or custom loop.
"""

# %%
batches = list(pipeline)
print(f"iterator yielded {len(batches)} batches")
print(f"first batch shapes: image={batches[0]['image'].shape}, label={batches[0]['label'].shape}")


# %% [markdown]
"""
## Step 4b — Whole-epoch scan (Tier C)

`pipeline.scan(step_fn, length=N)` runs `N` steps under `nnx.scan` and
returns the per-step outputs stacked along axis 0. The entire epoch
compiles to one XLA graph — substantially faster than the iterator
on heavier workloads.
"""

# %%
# Reset position so we start from batch 0.
pipeline._position.value = jnp.int32(0)

steps_per_epoch = num_samples // 32


def step_fn(batch: dict) -> jax.Array:
    """Run step_fn."""
    return jnp.mean(batch["image"])


means_per_step = pipeline.scan(step_fn, length=steps_per_epoch)
print(f"scan returned {means_per_step.shape[0]} per-step means")
print(f"epoch mean: {jnp.mean(means_per_step):.4f}")


# %% [markdown]
"""
## Where to go next

- **Train a model** — see `examples/integration/01_ml_classification.py`
  for a full training loop showing both Tier A and Tier C.
- **Multi-source mixing** — `MixDataSourcesNode` composes several
  sources with weighted interleaving and works the same way as
  `MemorySource` in a pipeline.
- **Branching topology** — `Pipeline.from_dag(...)` accepts an
  explicit DAG of nodes for parallel branches and merges.
- **Custom stages with parameters** — any `nnx.Module` with
  `nnx.Param` fields is differentiable through `pipeline.scan`;
  gradients flow when you compute them inside `step_fn`.
"""
