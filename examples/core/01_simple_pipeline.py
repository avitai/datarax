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
# Simple Pipeline Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Python, NumPy fundamentals |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates building a basic data pipeline with Datarax.
You'll create an in-memory data source, apply transformations using operators,
and iterate through batched data - the core workflow for any Datarax pipeline.

## Learning Goals

By the end of this example, you will be able to:

1. Create a `MemorySource` from dictionary data
2. Build a pipeline using the DAG-based `from_source()` API
3. Apply deterministic and stochastic operators to data
4. Iterate through batched pipeline output
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax
uv pip install datarax
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
from datarax.sources import MemorySource, MemorySourceConfig

# %% [markdown]
"""
## Step 1: Create Sample Data

Datarax works with dictionary-based data where each key maps to an array.
The first dimension is the sample dimension.
"""

# %%
# Create sample MNIST-like data
num_samples = 1000
image_shape = (28, 28, 1)

data = {
    "image": np.random.randint(0, 255, (num_samples, *image_shape)).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

print(f"Created data: image={data['image'].shape}, label={data['label'].shape}")
# Expected output:
# Created data: image=(1000, 28, 28, 1), label=(1000,)

# %% [markdown]
"""
## Step 2: Create Data Source

`MemorySource` wraps in-memory data for pipeline consumption.
It requires a config object and random number generators (rngs).
"""

# %%
# Create source with config-based API
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

print(f"Source contains {len(source)} samples")
# Expected output:
# Source contains 1000 samples

# %% [markdown]
"""
## Step 3: Define Operators

Operators transform data elements. There are two types:

- **Deterministic**: Same input always produces same output
- **Stochastic**: Uses random keys for randomized transformations
"""


# %%
# Deterministic operator: Normalize pixel values to [0, 1]
def normalize(element, key=None):
    """Normalize image pixels to [0, 1] range."""
    return element.update_data({"image": element.data["image"] / 255.0})


normalizer_config = ElementOperatorConfig(stochastic=False)
normalizer = ElementOperator(normalizer_config, fn=normalize, rngs=nnx.Rngs(0))


# %%
# Stochastic operator: Random horizontal flip
def apply_augmentation(element, key):
    """Randomly flip image horizontally with 50% probability."""
    key1, _ = jax.random.split(key)
    flip = jax.random.bernoulli(key1, 0.5)

    def flip_image(img):
        return jnp.flip(img, axis=1)

    def no_flip(img):
        return img

    # Use jax.lax.cond for JAX-compatible branching
    new_image = jax.lax.cond(flip, flip_image, no_flip, element.data["image"])
    return element.update_data({"image": new_image})


augmenter_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
augmenter = ElementOperator(augmenter_config, fn=apply_augmentation, rngs=nnx.Rngs(augment=42))

# %% [markdown]
"""
## Step 4: Build Pipeline

Chain the source and operators using the DAG-based API.
`from_source()` creates a batched pipeline, then `.add()` appends operators.
"""

# %%
# Build the pipeline DAG
pipeline = (
    from_source(source, batch_size=32).add(OperatorNode(normalizer)).add(OperatorNode(augmenter))
)

print("Pipeline created with batch_size=32")

# %% [markdown]
"""
## Step 5: Iterate Through Data

The pipeline is iterable. Each iteration yields a batch dictionary.
"""

# %%
# Process batches
print("Processing batches:")
for i, batch in enumerate(pipeline):
    if i >= 3:  # Show first 3 batches
        break

    image_batch = batch["image"]
    label_batch = batch["label"]

    print(f"Batch {i}:")
    print(f"  Image shape: {image_batch.shape}")
    print(f"  Label shape: {label_batch.shape}")
    print(f"  Image range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")

# Expected output:
# Processing batches:
# Batch 0:
#   Image shape: (32, 28, 28, 1)
#   Label shape: (32,)
#   Image range: [0.000, 1.000]
# Batch 1:
#   Image shape: (32, 28, 28, 1)
#   ...

# %% [markdown]
"""
## Results Summary

| Component | Description |
|-----------|-------------|
| Data Source | 1000 samples of 28x28 grayscale images |
| Batch Size | 32 samples per batch |
| Operators | Normalization (deterministic) + Flip (stochastic) |
| Output Range | [0.0, 1.0] after normalization |

The pipeline processes data lazily - batches are only created when iterated.
"""

# %% [markdown]
"""
## Next Steps

- **More operators**: See [Operators Tutorial](03_operators_tutorial.ipynb)
- **External data**: [TFDS](../integration/tfds/01_tfds_quickref.ipynb) or
  [HuggingFace](../integration/huggingface/01_hf_quickref.ipynb)
- **Distributed**: [Sharding](../advanced/distributed/01_sharding_quickref.ipynb)
- **API Reference**: [MemorySource](https://datarax.readthedocs.io/sources/memory_source/)
"""


# %%
def main():
    """Run the complete pipeline example."""
    # Create data
    num_samples = 1000
    data = {
        "image": np.random.randint(0, 255, (num_samples, 28, 28, 1)).astype(np.float32),
        "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
    }

    # Create source
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    # Create operators
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )
    augmenter = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="augment"),
        fn=apply_augmentation,
        rngs=nnx.Rngs(augment=42),
    )

    # Build and run pipeline
    pipeline = (
        from_source(source, batch_size=32)
        .add(OperatorNode(normalizer))
        .add(OperatorNode(augmenter))
    )

    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples successfully!")


if __name__ == "__main__":
    main()
