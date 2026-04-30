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
# DAG Pipeline Fundamentals Guide

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~3 min |
| **Prerequisites** | Pipeline Quickstart, Operators Tutorial |
| **Format** | Python + Jupyter |

## Overview

The `Pipeline` class supports two composition modes:

- **Linear**: a list of stages applied in order via
  `Pipeline(source=..., stages=[...])`.
- **DAG**: an explicit graph of named nodes with edges via
  `Pipeline.from_dag(source=..., nodes={...}, edges={...}, sink=...)`.

The DAG mode supports branching (one node consumes the source, two
downstream nodes consume it independently) and merging (a node
consumes the outputs of multiple predecessors). Both modes share the
same `step`, `scan`, and iterator semantics.

## Setup

```bash
uv pip install datarax
```

Activate the project virtualenv:

```bash
source activate.sh
```

## Learning Goals

By the end of this guide, you will be able to:

1. Build a linear pipeline with `Pipeline(stages=[...])`.
2. Build a branching DAG with `Pipeline.from_dag(...)`.
3. Reason about topological execution order.
4. Choose between linear and DAG composition modes.
"""

# %%
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


print(f"JAX version: {jax.__version__}")


# %% [markdown]
"""
## Part 1: Sample Data

We build a small memory source so the rest of the guide can focus on
composition.
"""

# %%
np.random.seed(42)
data = {
    "image": np.random.rand(64, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, (64,)).astype(np.int32),
}
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
print(f"Source: {len(source)} samples")


# %% [markdown]
"""
## Part 2: Linear Pipeline

The simplest composition mode is a linear chain of stages. Each stage
receives the previous stage's output and returns the next.
"""


# %%
def normalize(element, key=None):
    """Run normalize."""
    del key
    image = element.data["image"]
    return element.update_data({"image": (image - 0.5) / 0.5})


def add_brightness(element, key=None):
    """Run add_brightness."""
    del key
    image = element.data["image"]
    return element.update_data({"image": image + 0.1})


normalize_op = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
)
brighten_op = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=add_brightness, rngs=nnx.Rngs(0)
)

linear_pipeline = Pipeline(
    source=source,
    stages=[normalize_op, brighten_op],
    batch_size=16,
    rngs=nnx.Rngs(0),
)

batch = next(iter(linear_pipeline))
print(f"Linear pipeline output: image shape={batch['image'].shape}")


# %% [markdown]
"""
## Part 3: Branching DAG

When two downstream stages need to consume the source independently
(for example, an augmentation branch and a clean-reference branch
running side by side), use `Pipeline.from_dag`. Each node declares its
predecessors via the `edges` mapping.
"""


# %%
class _Augment(nnx.Module):
    """Multiplicative brightness jitter."""

    def __init__(self, factor: float = 1.2) -> None:
        self.factor = jnp.float32(factor)

    def __call__(self, batch):
        return {**batch, "image": batch["image"] * self.factor}


class _Normalize(nnx.Module):
    """Standardise to zero-mean unit-variance."""

    def __call__(self, batch):
        return {**batch, "image": (batch["image"] - 0.5) / 0.5}


class _StackBranches(nnx.Module):
    """Merge two batches by stacking the image fields along a new axis."""

    def __call__(self, augmented, clean):
        return {
            "image": jnp.stack([augmented["image"], clean["image"]], axis=1),
            "label": augmented["label"],
        }


# Build a branching DAG:
#   source -> augment   \
#                        -> stack -> sink
#   source -> normalize /
nodes = {
    "augment": _Augment(),
    "normalize": _Normalize(),
    "stack": _StackBranches(),
}
edges = {
    "augment": [],  # consumes source directly
    "normalize": [],  # consumes source directly
    "stack": ["augment", "normalize"],  # merges both branches
}

dag_pipeline = Pipeline.from_dag(
    source=source,
    nodes=nodes,
    edges=edges,
    sink="stack",
    batch_size=16,
    rngs=nnx.Rngs(0),
)

batch = next(iter(dag_pipeline))
print(f"DAG pipeline output: image shape={batch['image'].shape}")
# Expected: (16, 2, 32, 32, 3) — two branches stacked along axis=1


# %% [markdown]
"""
## Part 4: Topological Execution Order

The DAG executor topologically sorts the nodes so each node runs after
all its predecessors. You do not need to specify execution order
manually — only the dependency edges. Cycles raise `ValueError` at
construction time.

The `pipeline.stages` property returns the resolved modules in
topological order — useful for inspection or partial replacement.
"""

# %%
print(f"Linear pipeline: {len(linear_pipeline.stages)} stages")
print(f"DAG pipeline:    {len(dag_pipeline.stages)} stages")


# %% [markdown]
"""
## Part 5: When To Use Each Mode

- **Linear (`stages=[...]`)** — when each stage produces input for the
  next stage and there is no branching. Most augmentation pipelines
  fit this shape.
- **DAG (`from_dag(...)`)** — when you need branching (one input,
  multiple downstream consumers), merging (one node consumes multiple
  predecessors), or named nodes for inspection or partial replacement.

Both modes share `pipeline.step`, `pipeline.scan`, the iterator
protocol, and JIT semantics. There is no performance difference for
identical topologies.
"""


# %%
def main():
    """Run the DAG fundamentals guide."""
    print("DAG Pipeline Fundamentals Guide")
    print("=" * 50)

    np.random.seed(42)
    data_main = {
        "image": np.random.rand(64, 16, 16, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (64,)).astype(np.int32),
    }
    src = MemorySource(MemorySourceConfig(), data=data_main, rngs=nnx.Rngs(0))

    norm = ElementOperator(ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0))
    bright = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=add_brightness, rngs=nnx.Rngs(0)
    )

    # Linear
    linear = Pipeline(source=src, stages=[norm, bright], batch_size=8, rngs=nnx.Rngs(0))
    total = sum(b["image"].shape[0] for b in linear)
    print(f"Linear: processed {total} samples")

    # DAG
    dag = Pipeline.from_dag(
        source=src,
        nodes={"a": _Augment(), "n": _Normalize(), "s": _StackBranches()},
        edges={"a": [], "n": [], "s": ["a", "n"]},
        sink="s",
        batch_size=8,
        rngs=nnx.Rngs(0),
    )
    total = sum(b["image"].shape[0] for b in dag)
    print(f"DAG: processed {total} samples")
    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
