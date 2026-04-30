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
# Branching DAG Cookbook: Branch, Merge, Parallel

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~1 min |
| **Prerequisites** | DAG Fundamentals Guide |
| **Format** | Python + Jupyter |

## Overview

Datarax pipelines compose via two surfaces:

- ``Pipeline(stages=[...])`` for linear chains
- ``Pipeline.from_dag(nodes=, edges=, sink=)`` for branching topologies

The legacy ``Branch`` / ``Merge`` / ``Parallel`` node classes were
removed because each pattern is a small ``nnx.Module`` plus the right
``edges`` declaration. This cookbook ships one runnable recipe per
pattern, plus the verified output shape so you can sanity-check
your own variant.

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

1. Express **parallel** branches as two nodes that both consume the source.
2. Express **merge** as an ``nnx.Module`` that takes multiple positional
   arguments — stack, average, concat, weighted sum, or a learned
   aggregator.
3. Express **branch (conditional)** with ``jax.lax.cond`` inside an
   ``nnx.Module``.
4. Inspect the resulting topology with ``pipeline.stages``.
"""

# %%
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


print(f"JAX version: {jax.__version__}")


# %% [markdown]
"""
## Part 1: Sample Data

A single source feeds every recipe below. The image is a 32x32 RGB
patch sampled uniformly in [0, 1] so we can see the effect of each
transform.
"""

# %%
np.random.seed(0)
data = {
    "image": np.random.uniform(0, 1, size=(64, 32, 32, 3)).astype(np.float32),
    "label": np.random.randint(0, 10, size=(64,)).astype(np.int32),
}
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
print(f"Source: {len(source)} samples")


# %% [markdown]
"""
## Part 2: Reusable Stages

Three plain ``nnx.Module``s. Each takes a dict batch and returns a
dict batch — that's the entire stage contract.
"""


# %%
class _Normalize(nnx.Module):
    """Identity-ish: the data is already in [0, 1] so we return as-is."""

    def __call__(self, batch):
        return batch


class _Brighten(nnx.Module):
    """Add a fixed +0.1 brightness, clipped to [0, 1]."""

    def __call__(self, batch):
        return {**batch, "image": jnp.clip(batch["image"] + 0.1, 0.0, 1.0)}


class _Invert(nnx.Module):
    """Photographic negative."""

    def __call__(self, batch):
        return {**batch, "image": 1.0 - batch["image"]}


# %% [markdown]
"""
## Part 3: Parallel + Stack-Merge

Two nodes whose ``edges[name] = []`` both consume the source
directly. They run independently; a downstream sink that lists both
as predecessors waits for both to finish, then receives them as
positional arguments.
"""


# %%
class _Stack(nnx.Module):
    """Merge: stack the two parallel branches along a new axis.

    Each merge declares explicitly which fields it produces. Here the
    image is genuinely merged (stacked) while the label is the same
    on both branches (neither stage mutates it), so we copy it from
    either side. Listing every output key avoids the implicit
    "metadata is invariant across branches" assumption that would
    otherwise lurk inside ``**normalized``.
    """

    def __call__(self, normalized, inverted):
        return {
            "image": jnp.stack(
                [normalized["image"], inverted["image"]],
                axis=1,
            ),
            "label": normalized["label"],
        }


parallel_pipeline = Pipeline.from_dag(
    source=source,
    nodes={
        "normalize": _Normalize(),
        "invert": _Invert(),
        "stack": _Stack(),
    },
    edges={
        "normalize": [],  # parallel branch 1: consumes source
        "invert": [],  # parallel branch 2: consumes source
        "stack": ["normalize", "invert"],  # merge: order matches __call__ args
    },
    sink="stack",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
batch = next(iter(parallel_pipeline))
print(
    "Parallel + Stack-merge:",
    "image",
    batch["image"].shape,
    "(8 batch, 2 branches, 32x32 RGB)",
)
assert batch["image"].shape == (8, 2, 32, 32, 3)


# %% [markdown]
"""
## Part 4: Average-Merge

Same topology as Part 3, different merge logic. The merge node owns
the aggregation strategy — switching from "stack" to "average" only
swaps the merge ``nnx.Module``.
"""


# %%
class _Average(nnx.Module):
    """Merge: element-wise mean of two branches."""

    def __call__(self, brightened, inverted):
        return {
            "image": (brightened["image"] + inverted["image"]) / 2,
            "label": brightened["label"],
        }


average_pipeline = Pipeline.from_dag(
    source=source,
    nodes={
        "brighten": _Brighten(),
        "invert": _Invert(),
        "merge": _Average(),
    },
    edges={"brighten": [], "invert": [], "merge": ["brighten", "invert"]},
    sink="merge",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
batch = next(iter(average_pipeline))
print("Average-merge:", "image", batch["image"].shape)
assert batch["image"].shape == (8, 32, 32, 3)


# %% [markdown]
"""
## Part 5: Branch (conditional routing)

Conditional routing uses ``jax.lax.cond`` inside a stage. The
predicate is JIT-traceable and gradients flow through both branches
(reduced to zero by the discrete predicate, but still numerically
defined).

For state-bearing branches — where one or both branches need to
update an ``nnx.Module``'s parameters — use ``nnx.cond`` instead of
``jax.lax.cond`` so the state is correctly tracked.
"""


# %%
class _BrightenIfDark(nnx.Module):
    """Apply brightening only when the batch's mean image is below 0.3."""

    def __call__(self, batch):
        is_dark = jnp.mean(batch["image"]) < 0.3
        new_image = jax.lax.cond(
            is_dark,
            lambda img: jnp.clip(img + 0.1, 0.0, 1.0),
            lambda img: img,
            batch["image"],
        )
        return {**batch, "image": new_image}


branch_pipeline = Pipeline(
    source=source,
    stages=[_BrightenIfDark()],
    batch_size=8,
    rngs=nnx.Rngs(0),
)
batch = next(iter(branch_pipeline))
print(
    "Branch (jax.lax.cond):",
    "image",
    batch["image"].shape,
    f"mean={float(batch['image'].mean()):.3f}",
)
assert batch["image"].shape == (8, 32, 32, 3)


# %% [markdown]
"""
## Part 6: Triple-Branch Parallel

Parallelism is just topology — N branches that all consume the
source. Here three transforms run in parallel and a custom merge
takes them as three positional arguments.
"""


# %%
class _Concat3(nnx.Module):
    """Merge: concatenate three branches along the channel axis."""

    def __call__(self, normalized, brightened, inverted):
        return {
            "image": jnp.concatenate(
                [normalized["image"], brightened["image"], inverted["image"]],
                axis=-1,
            ),
            "label": normalized["label"],
        }


triple_branch = Pipeline.from_dag(
    source=source,
    nodes={
        "normalize": _Normalize(),
        "brighten": _Brighten(),
        "invert": _Invert(),
        "concat": _Concat3(),
    },
    edges={
        "normalize": [],
        "brighten": [],
        "invert": [],
        "concat": ["normalize", "brighten", "invert"],
    },
    sink="concat",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
batch = next(iter(triple_branch))
print("Triple-Branch parallel + channel-concat:", "image", batch["image"].shape)
# Three RGB branches concatenated along channels → 9 channels.
assert batch["image"].shape == (8, 32, 32, 9)


# %% [markdown]
"""
## Part 7: Inspecting the Topology

The resolved execution order is exposed via ``pipeline.stages``. For
a DAG it is the topological order the executor will use; for a
linear pipeline it is just the input list.
"""

# %%
print("Triple-Branch DAG resolved order:")
for stage in triple_branch.stages:
    print("  ", type(stage).__name__)


# %% [markdown]
"""
## Results Summary

| Pattern | Topology | Merge logic | Verified output shape |
|---------|----------|-------------|-----------------------|
| Parallel + Stack | 2 branches → 1 sink | stack along new axis | (8, 2, 32, 32, 3) |
| Parallel + Average | 2 branches → 1 sink | element-wise mean | (8, 32, 32, 3) |
| Branch | linear | ``jax.lax.cond`` inside stage | (8, 32, 32, 3) |
| Triple-Branch Parallel + Concat | 3 branches → 1 sink | channel concatenation | (8, 32, 32, 9) |

### Why no dedicated `Branch`/`Merge`/`Parallel` classes?

- **Merge** can do anything — stack, mean, concat, weighted sum,
  even a learned aggregator. A fixed-strategy enum would limit what
  is naturally just a function.
- **Branch** can use any JIT-traceable predicate — including
  per-batch statistics from upstream nodes. ``jax.lax.cond`` is
  already the right primitive.
- **Parallel** is a topological property, not a wrapper class. Two
  nodes with empty ``edges[name] = []`` *are* parallel by
  definition.

Each stage you write is automatically checkpointable (it's an
``nnx.Module``), gradient-flowing (its parameters are leaves of
``nnx.state(pipeline)``), and JIT-friendly (folded into
``Pipeline.step`` and ``Pipeline.scan``).

## Next Steps

- **Linear pipelines**: see [Pipeline Tutorial](../../core/02_pipeline_tutorial.py).
- **DAG fundamentals**: see [DAG Fundamentals Guide](01_dag_fundamentals_guide.py).
- **Whole-epoch JIT**: ``Pipeline.scan(...)`` lifts the DAG into a
  single ``nnx.scan`` body — see the migration to ``Pipeline.scan``
  in DADA / ISP / DDSP examples.
"""


# %%
def main() -> None:
    """Self-test entry point."""
    print()
    print("Branching DAG Cookbook")
    print("=" * 50)
    print(f"Parallel + Stack:    {(8, 2, 32, 32, 3)}")
    print(f"Parallel + Average:  {(8, 32, 32, 3)}")
    print(f"Branch (lax.cond):   {(8, 32, 32, 3)}")
    print(f"Triple-Branch + Concat:  {(8, 32, 32, 9)}")
    print("All recipes pass.")


if __name__ == "__main__":
    main()
