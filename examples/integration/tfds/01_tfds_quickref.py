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
# TFDS Integration Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Python, NumPy fundamentals |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates loading datasets from TensorFlow Datasets (TFDS)
using Datarax's `TFDSEagerSource`. You'll load MNIST, apply transformations, and
iterate through batched data using the standard pipeline API.

## Learning Goals

By the end of this example, you will be able to:

1. Configure and create a `TFDSEagerSource` with proper config
2. Apply transformations to TFDS data
3. Build a batched pipeline with operators
4. Iterate through transformed data
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with TFDS support
uv pip install "datarax[tfds]"
```
"""

# %%
# GPU Memory Configuration
# Prevent TensorFlow from using GPU (JAX handles GPU computation)
# This MUST be set BEFORE importing tensorflow
import os

os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""  # TF-specific GPU disable
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TF logs

# Force TF to CPU-only mode BEFORE importing JAX
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Now import JAX which will handle GPU
import jax.numpy as jnp
from flax import nnx

# Conditionally import TFDS source
try:
    from datarax.sources import TFDSEagerConfig, TFDSEagerSource
except ImportError as e:
    raise ImportError(
        "This example requires TensorFlow Datasets. Install with: uv pip install datarax[tfds]"
    ) from e

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig

# %% [markdown]
"""
## Step 1: Create TFDS Data Source

`TFDSEagerSource` wraps TensorFlow Datasets for use in Datarax pipelines.
The config specifies the dataset name, split, and shuffling options.
"""

# %%
# Configure TFDS source for MNIST
config = TFDSEagerConfig(
    name="mnist",
    split="train[:500]",  # Use subset for quick demo
    shuffle=True,
    seed=42,
)

source = TFDSEagerSource(config, rngs=nnx.Rngs(42))

print("Dataset: MNIST")
print(f"Samples: {len(source)}")

# %% [markdown]
"""
## Step 2: Define Transformations

Create operators to preprocess the data. TFDS data comes as raw
uint8 images which need normalization for training.
"""


# %%
def normalize_image(element, key=None):  # noqa: ARG001
    """Normalize image to [0, 1] range."""
    del key  # Unused - deterministic operator
    image = element.data["image"]
    normalized = image.astype(jnp.float32) / 255.0
    return element.update_data({"image": normalized})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

print("Created normalizer operator")

# %% [markdown]
"""
## Step 3: Build Pipeline

Chain source and operators using the DAG-based `from_source()` API.
"""

# %%
# Build the pipeline
pipeline = from_source(source, batch_size=32).add(OperatorNode(normalizer))

print("Pipeline: TFDSEagerSource(MNIST) -> Normalize -> Output")
print("Batch size: 32")

# %% [markdown]
"""
## Step 4: Iterate Through Data

Process batches and inspect the transformed data.
"""

# %%
# Process batches
print("\nProcessing batches:")
for i, batch in enumerate(pipeline):
    if i >= 3:  # Show first 3 batches
        break

    image_batch = batch["image"]
    label_batch = batch["label"]

    print(f"Batch {i}:")
    print(f"  Image: shape={image_batch.shape}, dtype={image_batch.dtype}")
    print(f"  Image range: [{float(image_batch.min()):.3f}, {float(image_batch.max()):.3f}]")
    print(f"  Label: shape={label_batch.shape}")

# Expected output:
# Batch 0:
#   Image: shape=(32, 28, 28, 1), dtype=float32
#   Image range: [0.000, 1.000]
#   Label: shape=(32,)

# %% [markdown]
"""
## Results Summary

| Component | Description |
|-----------|-------------|
| Data Source | TFDS MNIST (500 samples) |
| Batch Size | 32 samples per batch |
| Transforms | Image normalization [0, 255] -> [0, 1] |
| Output | Normalized float32 images |

The pipeline integrates TFDS datasets into the Datarax ecosystem,
enabling the use of all standard operators and augmentations.
"""

# %% [markdown]
"""
## Next Steps

- **More datasets**: Try `cifar10`, `imagenet`, or other TFDS datasets
- **Augmentations**: Add image operators from `datarax.operators.modality.image`
- **Distributed**: [Sharding](../../advanced/distributed/01_sharding_quickref.ipynb)
- **API Reference**: [TFDSEagerSource](https://datarax.readthedocs.io/sources/tfds/)
"""


# %%
def main():
    """Run the TFDS quick reference example."""
    # Create source
    config = TFDSEagerConfig(
        name="mnist",
        split="train[:200]",
        shuffle=True,
        seed=42,
    )
    source = TFDSEagerSource(config, rngs=nnx.Rngs(42))

    # Create operator
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize_image,
        rngs=nnx.Rngs(0),
    )

    # Build and run pipeline
    pipeline = from_source(source, batch_size=32).add(OperatorNode(normalizer))

    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]
        # Verify normalization
        assert batch["image"].min() >= 0.0, "Image not normalized"
        assert batch["image"].max() <= 1.0, "Image not normalized"

    print(f"Processed {total_samples} TFDS samples successfully!")


if __name__ == "__main__":
    main()
