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
# Complete Pipeline Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Beginner to Intermediate |
| **Runtime** | ~30 min |
| **Prerequisites** | [Simple Pipeline Quick Reference](01_simple_pipeline.py) |
| **Format** | Python + Jupyter |

## Overview

This tutorial provides a comprehensive introduction to building data pipelines
with Datarax. You'll learn to create data sources, compose multiple operators,
handle different data modalities, and build production-ready pipelines.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Understand the DAG-based pipeline architecture
2. Create and configure different data sources
3. Build custom transformation operators
4. Compose operators using CompositeOperator
5. Apply probabilistic augmentations
6. Handle multi-field data (images + labels)
7. Build reproducible pipelines with proper RNG management
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with all dependencies
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
from datarax.operators import (
    ElementOperator,
    ElementOperatorConfig,
)
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)

# Note: ProbabilisticOperator available for conditional augmentation
# See advanced examples for probabilistic operator usage
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Part 1: Understanding the Pipeline Architecture

Datarax pipelines follow a **Directed Acyclic Graph (DAG)** pattern:

```
DataSource → OperatorNode → OperatorNode → ... → Output
     ↑            ↑              ↑
   Config      Operator       Operator
```

Key concepts:
- **Source**: Produces raw data elements
- **OperatorNode**: Wraps an operator for the pipeline DAG
- **Operator**: Transforms data elements
- **Pipeline**: Connects source and operators, handles batching
"""

# %% [markdown]
"""
## Part 2: Creating a Data Source

`MemorySource` wraps in-memory data. Data must be dictionary-based
with arrays sharing the same first dimension (sample dimension).
"""

# %%
# Create realistic training data
np.random.seed(42)  # For reproducibility

num_samples = 500
image_height, image_width, channels = 32, 32, 3

# Simulate RGB images and one-hot encoded labels
data = {
    "image": np.random.randint(0, 256, (num_samples, image_height, image_width, channels)).astype(
        np.float32
    ),
    "label": np.eye(10)[np.random.randint(0, 10, num_samples)].astype(np.float32),
    "metadata": np.random.rand(num_samples, 4).astype(np.float32),  # Extra features
}

print("Dataset structure:")
for key, value in data.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

# %%
# Create MemorySource with configuration
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

print(f"\nSource created: {len(source)} samples")

# %% [markdown]
"""
## Part 3: Building Custom Operators

Operators transform data elements. Each operator receives:
- `element`: A data element with `.data` dict
- `key`: JAX random key (for stochastic operators)

The operator returns a new element via `element.update_data(new_data)`.
"""


# %%
# Operator 1: Normalize images to [0, 1]
def normalize_image(element, key=None):
    """Normalize image pixel values to [0, 1] range."""
    image = element.data["image"]
    normalized = image / 255.0
    return element.update_data({"image": normalized})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),  # Deterministic
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)
print("Created: normalizer (deterministic)")


# %%
# Operator 2: Random horizontal flip (stochastic)
def random_flip(element, key):
    """Randomly flip image horizontally with 50% probability."""
    flip_key, _ = jax.random.split(key)
    should_flip = jax.random.bernoulli(flip_key, 0.5)

    image = element.data["image"]
    flipped = jax.lax.cond(
        should_flip,
        lambda x: jnp.flip(x, axis=1),  # Flip along width axis
        lambda x: x,
        image,
    )
    return element.update_data({"image": flipped})


flipper = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="flip"),
    fn=random_flip,
    rngs=nnx.Rngs(flip=42),
)
print("Created: flipper (stochastic)")


# %%
# Operator 3: Add Gaussian noise (stochastic)
def add_noise(element, key):
    """Add random Gaussian noise to image."""
    noise_key, _ = jax.random.split(key)
    image = element.data["image"]

    # Add small noise (std=0.1)
    noise = jax.random.normal(noise_key, image.shape) * 0.1
    noisy = jnp.clip(image + noise, 0.0, 1.0)  # Keep in valid range

    return element.update_data({"image": noisy})


noise_adder = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="noise"),
    fn=add_noise,
    rngs=nnx.Rngs(noise=123),
)
print("Created: noise_adder (stochastic)")

# %% [markdown]
"""
## Part 4: Composing Operators

`CompositeOperatorModule` chains multiple operators together,
applying them sequentially to each element.
"""

# %%
# Create composite augmentation pipeline
# CompositeOperatorConfig requires strategy and operators in the config
augmentation_config = CompositeOperatorConfig(
    strategy=CompositionStrategy.SEQUENTIAL,  # Apply operators in sequence
    operators=[flipper, noise_adder],  # List of operators to chain
    stochastic=True,
    stream_name="augment",
)

# Build composite operator from config
augmentation_pipeline = CompositeOperatorModule(
    augmentation_config,
    rngs=nnx.Rngs(augment=999),
)

print("Created composite operator with SEQUENTIAL strategy (2 operators)")

# %% [markdown]
"""
## Part 5: Additional Transformations

Add more operators to the pipeline for comprehensive augmentation.
"""


# %%
# Create a brightness adjustment operator
def adjust_brightness(element, key):
    """Adjust image brightness by a random factor."""
    brightness_key, _ = jax.random.split(key)
    factor = jax.random.uniform(brightness_key, minval=0.8, maxval=1.2)

    image = element.data["image"]
    adjusted = jnp.clip(image * factor, 0.0, 1.0)
    return element.update_data({"image": adjusted})


brightness_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="brightness"),
    fn=adjust_brightness,
    rngs=nnx.Rngs(brightness=456),
)

print("Created brightness adjustment operator (stochastic)")

# %% [markdown]
"""
## Part 6: Building the Complete Pipeline

Chain everything together using the DAG API.
"""

# %%
# Build the full pipeline
pipeline = (
    from_source(source, batch_size=32)
    .add(OperatorNode(normalizer))  # Step 1: Normalize
    .add(OperatorNode(augmentation_pipeline))  # Step 2: Flip + Noise
    .add(OperatorNode(brightness_op))  # Step 3: Brightness adjustment
)

print("Pipeline structure:")
print("  Source → Normalize → [Flip + Noise] → Brightness → Output")
print("  Batch size: 32")
print(f"  Total samples: {len(source)}")

# %% [markdown]
"""
## Part 7: Running the Pipeline

Iterate through the pipeline to process data in batches.
"""

# %%
# Process batches
print("\nProcessing batches:")
stats = {"min": [], "max": [], "mean": []}

for i, batch in enumerate(pipeline):
    if i >= 5:  # Process 5 batches for demo
        break

    image_batch = batch["image"]
    label_batch = batch["label"]

    # Collect statistics
    stats["min"].append(float(image_batch.min()))
    stats["max"].append(float(image_batch.max()))
    stats["mean"].append(float(image_batch.mean()))

    print(f"Batch {i}:")
    img_min, img_max = image_batch.min(), image_batch.max()
    print(f"  Image: shape={image_batch.shape}, range=[{img_min:.3f}, {img_max:.3f}]")
    print(f"  Label: shape={label_batch.shape}")

# %%
# Summary statistics
print("\nPipeline Statistics (5 batches):")
print(f"  Min pixel: {min(stats['min']):.4f}")
print(f"  Max pixel: {max(stats['max']):.4f}")
print(f"  Mean pixel: {sum(stats['mean']) / len(stats['mean']):.4f}")

# %% [markdown]
"""
## Part 8: Reproducibility

Datarax ensures reproducible pipelines through explicit RNG management.
Same seeds produce identical results.
"""


# %%
# Demonstrate reproducibility
def create_pipeline_with_seed(seed: int):
    """Create a fresh pipeline with specific seed."""
    src = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(seed))

    norm = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize_image, rngs=nnx.Rngs(0)
    )

    flip = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="flip"),
        fn=random_flip,
        rngs=nnx.Rngs(flip=seed),
    )

    return from_source(src, batch_size=8).add(OperatorNode(norm)).add(OperatorNode(flip))


# Create two pipelines with same seed
p1 = create_pipeline_with_seed(42)
p2 = create_pipeline_with_seed(42)

# Get first batch from each
batch1 = next(iter(p1))
batch2 = next(iter(p2))

# Check if identical
images_match = jnp.allclose(batch1["image"], batch2["image"])
print(f"Same seed produces identical results: {images_match}")

# %% [markdown]
"""
## Results Summary

| Component | Type | Purpose |
|-----------|------|---------|
| MemorySource | Source | In-memory data storage |
| ElementOperator | Operator | Element-wise transforms |
| CompositeOperator | Operator | Chain multiple operators |
| OperatorNode | DAG Node | Wrap operators for pipeline |

Pipeline features:
- **Lazy evaluation**: Data processed only when iterated
- **Reproducibility**: Deterministic with same seeds
- **Composability**: Operators can be nested and chained
- **Type safety**: Strong typing throughout
"""

# %% [markdown]
"""
## Next Steps

- **Image operators**: See `datarax.operators.modality.image` for augmentations
- **External data**: [HuggingFace](../integration/huggingface/01_hf_quickref.ipynb)
- **Distributed**: [Sharding](../advanced/distributed/01_sharding_quickref.ipynb)
- **Checkpointing**: [Checkpoint](../advanced/checkpointing/01_checkpoint_quickref.ipynb)
- **API Reference**: [Operators API](https://datarax.readthedocs.io/operators/)
"""


# %%
def main():
    """Run the complete pipeline tutorial."""
    print("Complete Pipeline Tutorial")
    print("=" * 60)

    # Create data
    np.random.seed(42)
    num_samples = 200
    data = {
        "image": np.random.randint(0, 256, (num_samples, 32, 32, 3)).astype(np.float32),
        "label": np.eye(10)[np.random.randint(0, 10, num_samples)].astype(np.float32),
    }

    # Create source and operators
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize_image, rngs=nnx.Rngs(0)
    )
    flipper = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="flip"),
        fn=random_flip,
        rngs=nnx.Rngs(flip=42),
    )

    # Build pipeline
    pipeline = (
        from_source(source, batch_size=32).add(OperatorNode(normalizer)).add(OperatorNode(flipper))
    )

    # Process all data
    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples through the pipeline")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
