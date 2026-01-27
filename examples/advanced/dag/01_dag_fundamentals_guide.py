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
| **Level** | Advanced |
| **Runtime** | ~45 min |
| **Prerequisites** | Pipeline Tutorial, Operators Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master the Directed Acyclic Graph (DAG) pipeline architecture in Datarax.
This guide covers explicit node construction, control flow, caching strategies,
and building production-ready data pipelines with maximum performance.

## Learning Goals

By the end of this guide, you will be able to:

1. Construct explicit DAG pipelines with node types
2. Use operator-based composition (`>>` and `|` operators)
3. Implement control flow patterns (Sequential, Parallel, Branch, Merge)
4. Add caching for expensive transformations
5. Build rebatch strategies for dynamic batch sizing
6. Understand DAG execution and optimization
"""

# %% [markdown]
"""
## Coming from PyTorch?

| PyTorch | Datarax DAG |
|---------|-------------|
| `DataLoader(dataset)` | `from_source(source)` or `DataSourceNode(source)` |
| `transforms.Compose` | `Sequential([...])` or `node1 >> node2` |
| Multiple dataloaders | `Parallel([...])` or `node1 \\| node2` |
| `collate_fn` | `BatchNode` with custom logic |
| N/A (manual caching) | `Cache(node, cache_size=100)` |

## Coming from TensorFlow?

| TensorFlow tf.data | Datarax DAG |
|--------------------|-------------|
| `tf.data.Dataset` | `DataSourceNode` |
| `dataset.map()` | `OperatorNode(operator)` |
| `dataset.batch()` | `BatchNode(batch_size)` |
| `dataset.cache()` | `Cache(node)` |
| `dataset.prefetch()` | `PrefetchNode(buffer_size)` |
| `dataset.shuffle()` | `ShuffleNode(source)` |

## Coming from Google Grain?

| Grain | Datarax DAG |
|-------|-------------|
| `grain.DataLoader` | `DAGExecutor` or `from_source()` |
| `grain.MapTransform` | `OperatorNode(ElementOperator(...))` |
| `grain.Batch` | `BatchNode(batch_size)` |
| N/A | `Cache`, `Sequential`, `Parallel`, `Branch` |
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
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source, DAGExecutor
from datarax.dag.nodes import (
    Node,
    DataSourceNode,
    BatchNode,
    OperatorNode,
    Identity,
    Cache,
)
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
"""
## Part 1: DAG Node Hierarchy

Datarax DAG nodes form a hierarchy with specific responsibilities:

```
Node (base)
├── DataSourceNode     - Entry point, wraps data sources
├── BatchNode          - Creates batches from elements
├── OperatorNode       - Applies transformations
├── ShuffleNode        - Shuffles data ordering
├── PrefetchNode       - Background data loading
├── SamplerNode        - Custom sampling strategies
├── SharderNode        - Distributed sharding
├── Cache              - LRU caching for expensive ops
├── Sequential         - Chain nodes: out₁ → in₂
├── Parallel           - Apply multiple nodes to same input
├── Branch             - Route data conditionally
├── Merge              - Combine parallel branches
└── Identity           - Pass-through (useful for composition)
```
"""

# %%
# Create sample data for demonstrations
np.random.seed(42)
num_samples = 200
data = {
    "image": np.random.rand(num_samples, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

print(f"Created dataset: {num_samples} samples")
print(f"  image: {data['image'].shape}")
print(f"  label: {data['label'].shape}")

# %% [markdown]
"""
## Part 2: Simple Pipeline with from_source()

The easiest way to build a pipeline is using the `from_source()` helper.
It automatically creates the necessary nodes.
"""

# %%
# Method 1: Using from_source() helper (recommended for simple cases)
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

pipeline = from_source(source, batch_size=32)
batch = next(iter(pipeline))

print("Pipeline with from_source():")
print(f"  Batch shape: {batch['image'].shape}")
print(f"  Labels: {batch['label'][:8]}...")

# %% [markdown]
"""
## Part 3: Explicit DAG Construction

For more control, construct nodes explicitly. This allows custom
configurations and advanced patterns.
"""

# %%
# Method 2: Explicit node construction
source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))

# Create nodes explicitly
data_node = DataSourceNode(source2, name="ImageSource")
batch_node = BatchNode(batch_size=32, drop_remainder=False, name="Batcher")

# Compose using >> operator (creates Sequential)
explicit_pipeline = data_node >> batch_node

print("Explicit DAG construction:")
print(f"  Pipeline: {explicit_pipeline}")
print(f"  Type: {type(explicit_pipeline).__name__}")

# %%
# Execute the explicit pipeline
executor = DAGExecutor(explicit_pipeline, data_source=source2)

batch_count = 0
sample_count = 0
for batch in executor:
    batch_count += 1
    sample_count += batch["image"].shape[0]

print(f"  Processed {batch_count} batches, {sample_count} samples")

# %% [markdown]
"""
## Part 4: Adding Operators to the Pipeline

Use `OperatorNode` to wrap operators and add them to the DAG.
"""


# %%
# Define a normalization operator
def normalize_fn(element, key=None):
    """Normalize images to [0, 1] range."""
    del key  # Unused - deterministic
    image = element.data["image"]
    normalized = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image) + 1e-8)
    return element.update_data({"image": normalized})


normalize_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_fn,
    rngs=nnx.Rngs(0),
)

# Create brightness operator
brightness_op = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.1, 0.1),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

# Wrap in OperatorNodes
normalize_node = OperatorNode(normalize_op, name="Normalize")
brightness_node = OperatorNode(brightness_op, name="Brightness")

print("Created operator nodes:")
print(f"  - {normalize_node}")
print(f"  - {brightness_node}")

# %%
# Build pipeline with operators using >> operator
source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))
data_node3 = DataSourceNode(source3, name="Source")
batch_node3 = BatchNode(batch_size=32, name="Batch")

# Chain: Source >> Batch >> Normalize >> Brightness
pipeline_with_ops = data_node3 >> batch_node3 >> normalize_node >> brightness_node

print()
print("Pipeline with operators:")
print(f"  {pipeline_with_ops}")

# Execute
executor = DAGExecutor(pipeline_with_ops, data_source=source3)
batch = next(iter(executor))

print(f"  Output range: [{batch['image'].min():.4f}, {batch['image'].max():.4f}]")

# %% [markdown]
"""
## Part 5: Sequential and Parallel Composition

DAG nodes support operator-based composition:

- `>>` creates `Sequential`: `node1 >> node2 >> node3`
- `|` creates `Parallel`: `node1 | node2 | node3`
"""

# %%
# Sequential composition - chain operators
op1 = Identity(name="Op1")
op2 = Identity(name="Op2")
op3 = Identity(name="Op3")

# Using >> operator
sequential = op1 >> op2 >> op3
print("Sequential composition:")
print(f"  {sequential}")

# %%
# Parallel composition - multiple branches
branch_a = Identity(name="BranchA")
branch_b = Identity(name="BranchB")

# Using | operator
parallel = branch_a | branch_b
print()
print("Parallel composition:")
print(f"  {parallel}")
print("  Returns list of outputs from each branch")

# %%
# Combined composition pattern: branch then merge
# Useful for multi-view augmentation


def create_augment_branch(name: str, delta: float, seed: int):
    """Create an augmentation branch."""
    op = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(delta, delta),
            stochastic=False,
        ),
        rngs=nnx.Rngs(seed),
    )
    return OperatorNode(op, name=name)


# Create parallel augmentation branches
bright_branch = create_augment_branch("Brighten", 0.2, 1)
dark_branch = create_augment_branch("Darken", -0.2, 2)

# Parallel branches
multi_view = bright_branch | dark_branch

print()
print("Multi-view augmentation:")
print(f"  {multi_view}")

# %% [markdown]
"""
## Part 6: Caching Expensive Operations

Use `Cache` to store results of expensive transformations.
Particularly useful for deterministic operations.
"""


# %%
# Simulate expensive operation
def expensive_transform(element, key=None):
    """Simulate expensive transformation (e.g., feature extraction)."""
    del key
    image = element.data["image"]
    # Simulate computation with multiple operations
    features = jnp.mean(image, axis=(0, 1))  # Simple pooling
    features = jnp.sqrt(jnp.abs(features) + 1e-8)  # Non-linear transform
    return element.update_data({"features": features, "image": image})


expensive_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=expensive_transform,
    rngs=nnx.Rngs(0),
)

# Wrap with caching
cached_op = Cache(OperatorNode(expensive_op, name="ExpensiveOp"), cache_size=100)

print("Caching setup:")
print(f"  Wrapped: {cached_op.node}")
print(f"  Cache size: {cached_op.cache_size}")

# %% [markdown]
"""
## Part 7: Shuffle and Prefetch Nodes

Add shuffling and prefetching for training pipelines.
"""

# %%
# Create a training-ready pipeline with shuffle and prefetch
source4 = MemorySource(
    MemorySourceConfig(shuffle=True, seed=42),  # Shuffled source
    data=data,
    rngs=nnx.Rngs(3),
)

# Build training pipeline
training_pipeline = (
    from_source(source4, batch_size=32)
    .add(normalize_node)  # Normalize
    .add(brightness_node)  # Augment
)

print("Training pipeline with shuffling:")
print("  Source → Shuffle → Batch → Normalize → Brightness")

# Process a few batches
for i, batch in enumerate(training_pipeline):
    if i >= 3:
        break
    print(f"  Batch {i}: mean={batch['image'].mean():.4f}, std={batch['image'].std():.4f}")

# %% [markdown]
"""
## Part 8: Building Production Pipelines

Combine all concepts for a production-ready pipeline.
"""


# %%
def build_production_pipeline(data, batch_size=32, shuffle=True):
    """Build a complete production pipeline.

    Pipeline structure:
        Source → Shuffle? → Batch → Normalize → Augment → Output

    Args:
        data: Dictionary of arrays
        batch_size: Batch size
        shuffle: Whether to shuffle

    Returns:
        Configured pipeline
    """
    # Create source
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle, seed=42),
        data=data,
        rngs=nnx.Rngs(0),
    )

    # Create operators
    norm_op = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize_fn,
        rngs=nnx.Rngs(0),
    )

    augment_op = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.15, 0.15),
            stochastic=True,
            stream_name="aug",
        ),
        rngs=nnx.Rngs(aug=100),
    )

    # Build pipeline
    pipeline = (
        from_source(source, batch_size=batch_size)
        .add(OperatorNode(norm_op, name="Normalize"))
        .add(OperatorNode(augment_op, name="Augment"))
    )

    return pipeline


# Create and run production pipeline
prod_pipeline = build_production_pipeline(data, batch_size=64)

print("Production pipeline:")
total_batches = 0
total_samples = 0
for batch in prod_pipeline:
    total_batches += 1
    total_samples += batch["image"].shape[0]

print(f"  Batches: {total_batches}")
print(f"  Samples: {total_samples}")
print("  Batch size: 64")

# %% [markdown]
"""
## Part 9: DAG Execution Patterns

Understanding how DAG execution works helps optimize pipelines.

### Execution Model

1. **Pull-based**: Data is pulled through the graph on iteration
2. **Lazy evaluation**: Nodes only execute when outputs are needed
3. **State tracking**: NNX modules maintain state across iterations

### Key Optimization Points

| Optimization | Pattern |
|--------------|---------|
| Caching | Use `Cache` for deterministic expensive ops |
| Batching | Batch early in the pipeline for vmap efficiency |
| Operator fusion | Combine multiple light ops into single function |
| Memory reuse | Use `drop_remainder=True` for fixed batch shapes |
"""

# %%
# Demonstrate lazy evaluation
print("DAG Execution Demonstration:")
print()


class TrackedNode(Node):
    """Node that tracks when it's executed."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.call_count = nnx.Variable(0)

    def __call__(self, data, *, key=None):
        del key  # Unused
        self.call_count.value += 1
        return data


# Create tracked pipeline
tracked_a = TrackedNode("NodeA")
tracked_b = TrackedNode("NodeB")
tracked_seq = tracked_a >> tracked_b

# Execute once
_ = tracked_seq({"test": 1})

print("After 1 execution:")
print(f"  NodeA calls: {tracked_a.call_count.value}")
print(f"  NodeB calls: {tracked_b.call_count.value}")

# Execute again
_ = tracked_seq({"test": 2})
_ = tracked_seq({"test": 3})

print("After 3 executions:")
print(f"  NodeA calls: {tracked_a.call_count.value}")
print(f"  NodeB calls: {tracked_b.call_count.value}")

# %% [markdown]
"""
## Results Summary

### DAG Node Types

| Node | Purpose | Example |
|------|---------|---------|
| `DataSourceNode` | Pipeline entry point | `DataSourceNode(source)` |
| `BatchNode` | Create batches | `BatchNode(batch_size=32)` |
| `OperatorNode` | Apply transforms | `OperatorNode(operator)` |
| `ShuffleNode` | Randomize order | Via `MemorySourceConfig(shuffle=True)` |
| `Cache` | Store results | `Cache(node, cache_size=100)` |
| `Sequential` | Chain operations | `node1 >> node2` |
| `Parallel` | Multiple branches | `node1 \\| node2` |

### Composition Operators

| Operator | Creates | Usage |
|----------|---------|-------|
| `>>` | Sequential | `a >> b >> c` |
| `\\|` | Parallel | `a \\| b \\| c` |

### Best Practices

1. **Use `from_source()`** for simple pipelines
2. **Explicit construction** for complex DAGs
3. **Cache deterministic** expensive operations
4. **Batch early** for vmap efficiency
5. **Shuffle at source** level for memory efficiency
"""

# %% [markdown]
"""
## Next Steps

- [Composition Strategies](../../core/08_composition_strategies_tutorial.ipynb) - Composition
- [Sharding Guide](../distributed/02_sharding_guide.ipynb) - Distributed pipelines
- [Performance Guide](../performance/01_optimization_guide.ipynb) - Optimization tips
"""


# %%
def main():
    """Run the DAG fundamentals guide."""
    print("=" * 60)
    print("DAG Pipeline Fundamentals Guide")
    print("=" * 60)

    # Create data
    np.random.seed(42)
    data = {
        "image": np.random.rand(100, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (100,)).astype(np.int32),
    }

    # Demo 1: Simple pipeline
    print()
    print("1. Simple Pipeline (from_source):")
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    pipeline = from_source(source, batch_size=16)
    batch = next(iter(pipeline))
    print(f"   Batch shape: {batch['image'].shape}")

    # Demo 2: Operator composition
    print()
    print("2. Sequential Composition (>> operator):")
    op1 = Identity(name="A")
    op2 = Identity(name="B")
    seq = op1 >> op2
    print(f"   Pipeline: {seq}")

    # Demo 3: Parallel composition
    print()
    print("3. Parallel Composition (| operator):")
    branch_a = Identity(name="Left")
    branch_b = Identity(name="Right")
    par = branch_a | branch_b
    print(f"   Pipeline: {par}")

    # Demo 4: Production pipeline
    print()
    print("4. Production Pipeline:")
    prod = build_production_pipeline(data, batch_size=32)
    count = sum(1 for _ in prod)
    print(f"   Processed {count} batches")

    print()
    print("=" * 60)
    print("Guide completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
