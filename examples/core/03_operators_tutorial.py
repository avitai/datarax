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
# Operators Deep Dive Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~45 min |
| **Prerequisites** | Simple Pipeline, Pipeline Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master the Datarax operator system - the building blocks for data transformations.
This tutorial covers built-in operators, custom operator creation, and advanced
composition patterns for building production-ready data pipelines.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Understand operator types: deterministic vs stochastic
2. Use built-in image augmentation operators
3. Create custom operators with proper RNG handling
4. Select and transform specific data fields
5. Compose operators with different strategies (sequential, parallel)
6. Apply conditional and probabilistic transformations
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

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)

# SelectorOperator randomly selects one operator from a list
# It's different from field filtering (see Part 4 for field filtering)
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")

# %% [markdown]
"""
## Part 1: Operator Fundamentals

Operators are the transformation units in Datarax pipelines. They receive
data elements and return transformed elements.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Deterministic** | Same input always produces same output |
| **Stochastic** | Uses random keys for randomized transformations |
| **Element** | Single data sample with `.data` dictionary |
| **Batch** | Collection of elements processed together |
"""

# %%
# Create sample image data for demonstrations
np.random.seed(42)
num_samples = 100
data = {
    "image": np.random.randint(0, 256, (num_samples, 32, 32, 3)).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
    "metadata": np.random.rand(num_samples, 4).astype(np.float32),
}

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
print(f"Created dataset: {num_samples} samples")
print(f"  image: {data['image'].shape}")
print(f"  label: {data['label'].shape}")
print(f"  metadata: {data['metadata'].shape}")

# %% [markdown]
"""
## Part 2: ElementOperator - Custom Transformations

`ElementOperator` is the most flexible operator - wrap any function
to transform data elements.
"""


# %%
# Example 1: Deterministic normalization
def normalize_image(element, key=None):  # noqa: ARG001
    """Normalize image pixels to [0, 1] range."""
    del key  # Unused - deterministic operator
    image = element.data["image"]
    normalized = image / 255.0
    return element.update_data({"image": normalized})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

# Test it
pipeline = from_source(source, batch_size=16).add(OperatorNode(normalizer))
batch = next(iter(pipeline))

print("Normalization result:")
print(f"  Range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")


# %%
# Example 2: Stochastic horizontal flip
def random_flip(element, key):
    """Randomly flip image horizontally."""
    flip_key, _ = jax.random.split(key)
    should_flip = jax.random.bernoulli(flip_key, 0.5)

    image = element.data["image"]
    flipped = jax.lax.cond(
        should_flip,
        lambda x: jnp.flip(x, axis=1),
        lambda x: x,
        image,
    )
    return element.update_data({"image": flipped})


flipper = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="flip"),
    fn=random_flip,
    rngs=nnx.Rngs(flip=42),
)

print("Created stochastic flipper operator")

# %% [markdown]
"""
## Part 3: Built-in Image Operators

Datarax provides optimized image augmentation operators.
These follow a consistent pattern: Config + Operator.
"""

# %%
# Brightness adjustment
brightness_op = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.3, 0.3),  # Additive delta range
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

# Contrast adjustment
contrast_op = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image",
        contrast_range=(0.8, 1.2),  # Multiplicative factor range
        stochastic=True,
        stream_name="contrast",
    ),
    rngs=nnx.Rngs(contrast=200),
)

# Gaussian noise
noise_op = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.05,
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=300),
)

print("Built-in operators created:")
print("  - BrightnessOperator (range: -0.3 to +0.3)")
print("  - ContrastOperator (factor: 0.8-1.2)")
print("  - NoiseOperator (gaussian, std=0.05)")

# %% [markdown]
"""
## Part 4: SelectorOperator - Random Operator Selection

`SelectorOperator` randomly selects ONE of several operators to apply
per element. Useful for randomized augmentation pipelines.
"""


# %%
# Create a field filtering operator using ElementOperator
def filter_fields(element, key=None):  # noqa: ARG001
    """Keep only image and label fields."""
    del key  # Unused - deterministic operator
    filtered = {k: v for k, v in element.data.items() if k in ["image", "label"]}
    return element.update_data(filtered)


field_filter = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=filter_fields,
    rngs=nnx.Rngs(0),
)

# Test field filtering
source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))
pipeline = from_source(source2, batch_size=8).add(OperatorNode(field_filter))
batch = next(iter(pipeline))

print("After field filtering:")
print(f"  Image present: {batch['image'].shape}")
print(f"  Label present: {batch['label'].shape}")

# %% [markdown]
"""
## Part 5: CompositeOperator - Chaining Transforms

Chain multiple operators with `CompositeOperatorModule`.
Different strategies control how operators interact.

### Composition Strategies

| Strategy | Description |
|----------|-------------|
| SEQUENTIAL | Chain: out₁ → in₂ → out₂ → ... |
| PARALLEL | Apply all to same input, merge outputs |
| ENSEMBLE_MEAN | Parallel + average outputs |
"""

# %%
# Create individual operators for composition
norm_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

flip_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="flip"),
    fn=random_flip,
    rngs=nnx.Rngs(flip=42),
)

# Sequential composition: normalize → flip
sequential_augment = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[norm_op, flip_op],
        stochastic=True,
        stream_name="seq_augment",
    ),
    rngs=nnx.Rngs(seq_augment=500),
)

print("Created SEQUENTIAL composite: normalize → flip")

# %%
# Test the composite operator
source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))
pipeline = from_source(source3, batch_size=16).add(OperatorNode(sequential_augment))
batch = next(iter(pipeline))

print("Sequential composite result:")
print(f"  Image shape: {batch['image'].shape}")
print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

# %% [markdown]
"""
## Part 6: Building a Full Augmentation Pipeline

Combine everything into a production-ready augmentation pipeline.
"""

# %%
# Create fresh operators for the full pipeline
normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

flipper = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="flip"),
    fn=random_flip,
    rngs=nnx.Rngs(flip=42),
)

brightness = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.2, 0.2),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

# Build pipeline with chained operators
source4 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(3))
full_pipeline = (
    from_source(source4, batch_size=32)
    .add(OperatorNode(normalizer))
    .add(OperatorNode(flipper))
    .add(OperatorNode(brightness))
)

print("Full augmentation pipeline:")
print("  Source → Normalize → Flip → Brightness → Output")

# %%
# Process and collect statistics
stats = {"batches": 0, "samples": 0, "mean_values": []}

for batch in full_pipeline:
    stats["batches"] += 1
    stats["samples"] += batch["image"].shape[0]
    stats["mean_values"].append(float(batch["image"].mean()))

print("\nPipeline processed:")
print(f"  Batches: {stats['batches']}")
print(f"  Samples: {stats['samples']}")
print(f"  Mean pixel value: {sum(stats['mean_values']) / len(stats['mean_values']):.4f}")

# %% [markdown]
"""
## Part 7: Custom Operator Patterns

Best practices for creating robust custom operators.
"""


# %%
# Pattern 1: Multi-field transformation
def augment_image_and_mask(element, key):
    """Apply same random transform to image and corresponding mask."""
    key1, _ = jax.random.split(key)

    # Random rotation angle
    angle = jax.random.uniform(key1, minval=-15, maxval=15)

    # Apply to both fields (simplified - real rotation would use jax.scipy)
    image = element.data["image"]
    # In production, apply actual rotation here

    return element.update_data({"image": image, "rotation_angle": angle})


# %%
# Pattern 2: Conditional transformation
def conditional_augment(element, key):
    """Apply augmentation only to certain samples based on metadata."""
    key1, _ = jax.random.split(key)

    image = element.data["image"]
    label = element.data.get("label", 0)

    # Apply stronger augmentation to minority classes (e.g., label > 5)
    strength = jax.lax.cond(
        label > 5,
        lambda: 0.2,  # Strong augmentation
        lambda: 0.05,  # Weak augmentation
    )

    noise = jax.random.normal(key1, image.shape) * strength
    augmented = jnp.clip(image + noise, 0.0, 1.0)

    return element.update_data({"image": augmented})


conditional_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="cond"),
    fn=conditional_augment,
    rngs=nnx.Rngs(cond=999),
)

print("Created conditional augmentation operator")

# %% [markdown]
"""
## Results Summary

| Operator Type | Use Case | Stochastic |
|---------------|----------|------------|
| ElementOperator | Custom transforms | Configurable |
| BrightnessOperator | Image brightness | Yes |
| ContrastOperator | Image contrast | Yes |
| NoiseOperator | Add noise | Yes |
| SelectorOperator | Field filtering | No |
| CompositeOperator | Chain operators | Depends on children |

### Key Takeaways

1. **Deterministic operators**: Use `stochastic=False`, ignore `key` parameter
2. **Stochastic operators**: Use `stochastic=True`, split `key` for each random op
3. **Composition**: Use `CompositionStrategy.SEQUENTIAL` for chained transforms
4. **Field targeting**: Image operators use `target_field` parameter
5. **RNG management**: Each stochastic operator needs unique `stream_name`
"""

# %% [markdown]
"""
## Next Steps

- **Advanced composition**: Explore PARALLEL and ENSEMBLE strategies
- **Performance**: Use `jax.jit` for operator functions
- **Distributed**: [Sharding](../advanced/distributed/01_sharding_quickref.ipynb)
- **Checkpointing**: [Checkpoint](../advanced/checkpointing/01_checkpoint_quickref.ipynb)
"""


# %%
def main():
    """Run the operators tutorial."""
    print("Operators Deep Dive Tutorial")
    print("=" * 50)

    # Create data
    np.random.seed(42)
    data = {
        "image": np.random.randint(0, 256, (100, 32, 32, 3)).astype(np.float32),
        "label": np.random.randint(0, 10, (100,)).astype(np.int32),
    }
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    # Create operators
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize_image,
        rngs=nnx.Rngs(0),
    )

    brightness = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.2, 0.2),
            stochastic=True,
            stream_name="brightness",
        ),
        rngs=nnx.Rngs(brightness=100),
    )

    # Build and run pipeline
    pipeline = (
        from_source(source, batch_size=32)
        .add(OperatorNode(normalizer))
        .add(OperatorNode(brightness))
    )

    total = 0
    for batch in pipeline:
        total += batch["image"].shape[0]

    print(f"Processed {total} samples")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
