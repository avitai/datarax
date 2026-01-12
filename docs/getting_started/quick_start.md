# Quick Start Guide

This guide will help you get started with Datarax by walking through the core concepts and providing concrete examples of creating data pipelines.

## 1. The Basic Pipeline

A minimal Datarax pipeline consists of a **Data Source** and a **Pipeline Definition**.

```python
import jax.numpy as jnp
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

# 1. Prepare Data
# Data is typically a list of dictionaries (elements)
data = [{"image": jnp.ones((28, 28, 3)), "label": i % 10} for i in range(100)]

# 2. Create Source
# Sources are StructuralModules that handle data ingest
# We must provide a configuration object
config = MemorySourceConfig(shuffle=False)
source = MemorySource(config, data)

# 3. Build Pipeline
# from_source creates a pipeline starting with a DataSourceNode and a BatchNode
pipeline = from_source(source, batch_size=10)

# 4. Iterate
# The pipeline yields Batches (which behave like dicts of arrays)
for i, batch in enumerate(pipeline):
    print(f"Batch {i}: shape = {batch['image'].shape}")
    if i >= 2: break
```

## 2. Deterministic Operators & Immutability

Operators in Datarax work on **Elements**. A key principle is **Immutability**: operators receive an input and return a *new* output.

We use the `element.update_data()` method for clean, immutable updates.

```python
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import OperatorNode

def normalize(element, key):
    """Normalize image to [0, 1]. Key is ignored for deterministic ops."""

    # element.data is a dict-like PyTree of the actual values
    img = element.data["image"]

    # Return a NEW element with updated data
    return element.update_data({"image": img / 255.0})

# stochastic=False indicates this operator doesn't need a random key
config = ElementOperatorConfig(stochastic=False)
normalize_op = ElementOperator(config, fn=normalize)

# Add to pipeline using the >> operator
pipeline = (
    from_source(source, batch_size=10)
    >> OperatorNode(normalize_op)
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
config = ElementOperatorConfig(stochastic=True, stream_name="augment")
rngs = nnx.Rngs(augment=42)  # Seed the stream

augment_op = ElementOperator(config, fn=random_augment, rngs=rngs)

pipeline = (
    from_source(source, batch_size=10)
    >> OperatorNode(normalize_op) # Normalize first
    >> OperatorNode(augment_op)   # Then augment
)
```

## 4. Advanced Graph Topologies

Datarax pipelines are DAGs (Directed Acyclic Graphs). You aren't limited to sequential chains; you can split, process in parallel, and merge.

### Parallel Execution & Merging

Use `Parallel` to run multiple paths and `Merge` to combine them.

```python
from datarax.dag.nodes import Parallel, Merge

# Example: Create an original and an inverted version of the image
def invert(element, key):
    return element.update_data({"image": 1.0 - element.data["image"]})
invert_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=invert)

def brighten(element, key):
    return element.update_data({"image": jnp.minimum(element.data["image"] + 0.2, 1.0)})
brighten_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=brighten)

# Graph:
#         /-> Normalize ->\
# Source -                 -> Merge (Stack)
#         \-> Invert    ->/

pipeline_nodes = (
    from_source(source, batch_size=10)
    >> Parallel([
        OperatorNode(normalize_op),
        OperatorNode(invert_op)
    ])
    # Stack results along a new axis
    >> Merge(strategy="stack", axis=1)
)
```

### Conditional Branching

Use `Branch` to route data based on a condition function.

```python
from datarax.dag.nodes import Branch

def is_dark_image(element):
    """Predicate function returning a boolean scalar."""
    return jnp.mean(element.data["image"]) < 0.3

# Graph: If dark -> Brighten, Else -> Identity
pipeline_nodes = (
    from_source(source, batch_size=10)
    >> Branch(
        condition=is_dark_image,
        true_path=OperatorNode(brighten_op), # Assume brighten_op exists
        false_path=OperatorNode(normalize_op)
    )
)
```

## 5. Complete Example

Putting it all together into a robust pipeline:

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import OperatorNode
import flax.nnx as nnx
import jax.numpy as jnp

# Define operators
def normalize(element, key):
    """Normalize image to [0, 1]."""
    img = element.data["image"]
    return element.update_data({"image": img / 255.0})

def augment(element, key):
    """Simple passthrough augmentation."""
    return element

normalize_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=normalize)
augment_op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=augment)

# Setup data
data = [{"image": jnp.ones((28, 28, 3)) * 128, "label": i % 10} for i in range(100)]
config = MemorySourceConfig(shuffle=False)
source = MemorySource(config, data)

# Pipeline using the >> operator
pipeline = (
    from_source(source, batch_size=32)
    >> OperatorNode(normalize_op)
    >> OperatorNode(augment_op)
)

print("Pipeline configured. Starting iteration...")
for i, batch in enumerate(pipeline):
    if i >= 2:  # Only process a few batches for demo
        break
    print(f"Batch {i}: shape = {batch['image'].shape}")
```

## Next Steps

- **[Core Concepts](core_concepts.md)**: Deep dive into the Three-Tier Architecture.
- **[API Reference](../core/index.md)**: Full API details for Operators and Nodes.
- **[Examples](../examples/overview.md)**: Simple pipeline tutorials.
```
