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
# HuggingFace Datasets Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline knowledge |
| **Format** | Python + Jupyter |

## Overview

Load and process datasets from [HuggingFace Hub](https://huggingface.co/datasets)
using Datarax's `HFSource`. This enables access to thousands of pre-built datasets
with seamless integration into your data pipelines.

## Learning Goals

By the end of this example, you will be able to:

1. Configure `HFSource` for HuggingFace datasets
2. Use streaming mode for large datasets
3. Inspect dataset structure and contents
4. Apply transformations to HuggingFace data
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with data dependencies
uv pip install "datarax[data]"
```

**Note**: First run may download dataset files from HuggingFace Hub.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import HfDataSourceConfig, HFSource

print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Step 1: Configure HuggingFace Source

`HfDataSourceConfig` specifies which dataset to load.
Key parameters:
- `name`: Dataset identifier (e.g., "mnist", "imdb", "squad")
- `split`: Which split to use ("train", "test", "validation")
- `streaming`: Enable for large datasets to avoid full download
"""

# %%
# Load MNIST dataset in streaming mode
config = HfDataSourceConfig(
    name="mnist",
    split="train",
    streaming=True,  # Stream data instead of downloading all
)

source = HFSource(config, rngs=nnx.Rngs(0))
print(f"Loaded HuggingFace dataset: {config.name}")

# Check dataset size (may not be available in streaming mode)
try:
    print(f"Dataset size: {len(source)}")
except (NotImplementedError, TypeError):
    print("Dataset size: N/A (streaming mode)")

# %% [markdown]
"""
## Step 2: Create Pipeline and Inspect Data

Build a pipeline and examine what data the dataset provides.
"""

# %%
# Create pipeline with batch_size=1 for inspection
pipeline = from_source(source, batch_size=1)

# Get first few examples
print("First 3 examples:")
example_iter = iter(pipeline)

for i in range(3):
    batch = next(example_iter)
    data = batch.get_data()

    print(f"\nExample {i + 1}:")
    print(f"  Keys: {list(data.keys())}")

    for key, value in data.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")

# Expected output (MNIST):
# Example 1:
#   Keys: ['image', 'label']
#   image: shape=(28, 28), dtype=uint8
#   label: int = 5

# %% [markdown]
"""
## Step 3: Apply Transformations

Add operators to transform the HuggingFace data.
"""


# %%
# Define a normalization transform
def normalize_image(element, key=None):
    """Normalize image to [0, 1] range and add channel dimension."""
    image = element.data.get("image")
    if image is not None and hasattr(image, "dtype"):
        # Normalize to [0, 1]
        normalized = image.astype(jnp.float32) / 255.0
        # Add channel dimension if needed
        if normalized.ndim == 2:
            normalized = normalized[..., None]
        return element.update_data({"image": normalized})
    return element


# Create operator
normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

# Build transformed pipeline (need fresh source for new iteration)
source2 = HFSource(config, rngs=nnx.Rngs(1))
transformed_pipeline = from_source(source2, batch_size=32).add(OperatorNode(normalizer))

# Process a batch
batch = next(iter(transformed_pipeline))
image_batch = batch["image"]

print("Transformed batch:")
print(f"  Image shape: {image_batch.shape}")
print(f"  Image range: [{image_batch.min():.3f}, {image_batch.max():.3f}]")

# Expected output:
# Transformed batch:
#   Image shape: (32, 28, 28, 1)
#   Image range: [0.000, 1.000]

# %% [markdown]
"""
## Results Summary

| Feature | Value |
|---------|-------|
| Dataset | MNIST from HuggingFace Hub |
| Mode | Streaming (no full download) |
| Batch Size | 32 |
| Output Shape | (32, 28, 28, 1) |
| Normalization | [0, 255] → [0, 1] |

HuggingFace integration provides:
- Access to 100,000+ datasets
- Automatic caching and versioning
- Streaming for large datasets
- Seamless Datarax pipeline integration
"""

# %% [markdown]
"""
## Next Steps

- **More datasets**: Try `"imdb"`, `"squad"`, `"cifar10"` - change the `name` parameter
- **Custom configs**: Use `HfDataSourceConfig(subset="...")` for dataset variants
- **TFDS alternative**: [TFDS](../tfds/01_tfds_quickref.ipynb)
- **Full tutorial**: [HuggingFace Tutorial](02_hf_tutorial.py) for advanced usage
"""


# %%
def main():
    """Run the HuggingFace integration example."""
    print("HuggingFace Datasets Integration Example")
    print("=" * 50)

    # Load dataset
    config = HfDataSourceConfig(name="mnist", split="train", streaming=True)
    source = HFSource(config, rngs=nnx.Rngs(0))

    # Create pipeline with normalization
    def normalize(element, key=None):
        image = element.data.get("image")
        if image is not None:
            normalized = image.astype(jnp.float32) / 255.0
            if normalized.ndim == 2:
                normalized = normalized[..., None]
            return element.update_data({"image": normalized})
        return element

    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )

    pipeline = from_source(source, batch_size=32).add(OperatorNode(normalizer))

    # Process batches
    total_samples = 0
    for i, batch in enumerate(pipeline):
        if i >= 10:  # Process 10 batches
            break
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples from HuggingFace MNIST")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
