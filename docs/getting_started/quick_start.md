# Quick Start Guide

This guide will help you get started with Datarax by walking through the core concepts and providing concrete examples of creating data pipelines.

## 1. The Basic Pipeline

A minimal Datarax pipeline consists of a **Data Source** and a **Pipeline Definition**.

```python
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

# 1. Prepare data — a dict of arrays sharing the leading sample dimension.
data = {
    "image": np.ones((100, 28, 28, 3), dtype=np.float32),
    "label": (np.arange(100) % 10).astype(np.int32),
}

# 2. Create the source.
config = MemorySourceConfig(shuffle=False)
source = MemorySource(config, data=data, rngs=nnx.Rngs(0))

# 3. Build the pipeline. Pipeline auto-batches via batch_size.
pipeline = Pipeline(source=source, stages=[], batch_size=10, rngs=nnx.Rngs(0))

# 4. Iterate. Pipeline yields plain dicts of jax.Array.
for i, batch in enumerate(pipeline):
    print(f"Batch {i}: image shape = {batch['image'].shape}")
    if i >= 2:
        break
```

## 2. Deterministic Operators & Immutability

Operators in Datarax work on **Elements**. A key principle is **Immutability**: operators receive an input and return a *new* output.

We use the `element.update_data()` method for clean, immutable updates.

```python
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline


def normalize(element, key=None):
    """Normalize image to [0, 1]. Key is unused for deterministic ops."""
    del key
    img = element.data["image"]
    # Return a NEW element with updated data
    return element.update_data({"image": img / 255.0})


# stochastic=False indicates this operator doesn't need a random key
normalize_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize,
    rngs=nnx.Rngs(0),
)

pipeline = Pipeline(
    source=source,
    stages=[normalize_op],
    batch_size=10,
    rngs=nnx.Rngs(0),
)
```

## 3. Stochastic Operators & JAX Control Flow

For random data augmentation, we use **Stochastic Operators**. These receive a PRNG key.

**Crucial Note**: Because Datarax runs inside JAX's JIT compilation, you must use JAX's control flow primitives (like `jax.lax.cond`) instead of Python's `if/else` when the condition depends on data or random values.

```python
import flax.nnx as nnx
import jax
import jax.numpy as jnp

def random_augment(element, key):
    """Randomly flip or add noise."""
    # Split key for different random operations
    k1, k2 = jax.random.split(key)

    # 1. Random Flip (Bernoulli)
    should_flip = jax.random.bernoulli(k1, 0.5)

    # Use jax.lax.cond because 'should_flip' is a JAX tracer, not a Python bool
    img = jax.lax.cond(
        should_flip,
        lambda x: jnp.flip(x, axis=1),  # True branch
        lambda x: x,                    # False branch
        element.data["image"]           # Operand
    )

    # 2. Random Noise
    noise = jax.random.normal(k2, img.shape) * 0.1
    img = img + noise

    return element.update_data({"image": img})

# Configure as stochastic and provide a stream name
augment_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="augment"),
    fn=random_augment,
    rngs=nnx.Rngs(augment=42),
)

pipeline = Pipeline(
    source=source,
    stages=[normalize_op, augment_op],
    batch_size=10,
    rngs=nnx.Rngs(0),
)
```

## 4. Branching DAGs with `Pipeline.from_dag`

Sequential ``stages=[...]`` covers most pipelines. When you need a
branching topology — one stage feeding two downstream stages, or a
merge that consumes both — switch to ``Pipeline.from_dag``.

```python
import jax.numpy as jnp


def invert(element, key=None):
    del key
    return element.update_data({"image": 1.0 - element.data["image"]})


invert_op = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=invert, rngs=nnx.Rngs(0)
)


class StackBranches(nnx.Module):
    """Merge the normalized and inverted streams along a new axis."""

    def __call__(self, normalized, inverted):
        return {
            "image": jnp.stack([normalized["image"], inverted["image"]], axis=1),
            "label": normalized["label"],
        }


pipeline_dag = Pipeline.from_dag(
    source=source,
    nodes={"normalize": normalize_op, "invert": invert_op, "merge": StackBranches()},
    edges={"normalize": [], "invert": [], "merge": ["normalize", "invert"]},
    sink="merge",
    batch_size=10,
    rngs=nnx.Rngs(0),
)
```

## 5. Complete Example

Putting it all together — data, source, two stages, iteration:

```python
import numpy as np
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


def normalize(element, key=None):
    del key
    img = element.data["image"]
    return element.update_data({"image": img / 255.0})


def passthrough(element, key=None):
    del key
    return element


normalize_op = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
)
augment_op = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=passthrough, rngs=nnx.Rngs(0)
)

data = {
    "image": (np.ones((100, 28, 28, 3), dtype=np.float32) * 128.0),
    "label": (np.arange(100) % 10).astype(np.int32),
}
source = MemorySource(MemorySourceConfig(shuffle=False), data=data, rngs=nnx.Rngs(0))

pipeline = Pipeline(
    source=source,
    stages=[normalize_op, augment_op],
    batch_size=32,
    rngs=nnx.Rngs(0),
)

print("Pipeline configured. Starting iteration...")
for i, batch in enumerate(pipeline):
    if i >= 2:
        break
    print(f"Batch {i}: image shape = {batch['image'].shape}")
```

## Next Steps

- **[Core Concepts](core_concepts.md)**: Deep dive into the Three-Tier Architecture.
- **[API Reference](../core/index.md)**: Full API details for Operators and Nodes.
- **[Examples](../examples/overview.md)**: Simple pipeline tutorials.
```
