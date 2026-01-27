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
# CIFAR-10 Pipeline Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline, TFDS setup |
| **Format** | Python + Jupyter |
| **Memory** | ~500 MB RAM |

## Overview

This quick reference demonstrates loading and processing CIFAR-10 from TensorFlow
Datasets (TFDS). CIFAR-10 is a classic benchmark dataset containing 60,000 32x32
color images in 10 classes, making it ideal for learning image classification pipelines.

## Learning Goals

By the end of this example, you will be able to:

1. Load CIFAR-10 using `TFDSEagerSource` with proper configuration
2. Apply standard CIFAR-10 normalization (ImageNet-style)
3. Build a batched pipeline ready for training
4. Understand the data shapes and preprocessing workflow
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with TFDS support
uv pip install "datarax[tfds]"
```

**Note**: First run downloads CIFAR-10 (~170 MB).
"""

# %%
# GPU Memory Configuration - prevent TensorFlow from using GPU
import os

os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Now import JAX and Datarax
import jax.numpy as jnp
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

# %% [markdown]
"""
## CIFAR-10 Preprocessing Constants

Standard normalization values for CIFAR-10, computed from the training set.
Using these values ensures compatibility with pretrained models and published results.

| Statistic | R | G | B |
|-----------|---|---|---|
| **Mean** | 0.4914 | 0.4822 | 0.4465 |
| **Std** | 0.2470 | 0.2435 | 0.2616 |
"""

# %%
# CIFAR-10 normalization constants
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2470, 0.2435, 0.2616])

# Class names for reference
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print("CIFAR-10 classes:", CIFAR10_CLASSES)

# %% [markdown]
"""
## Step 1: Create TFDS Data Source

Configure `TFDSEagerSource` to load CIFAR-10 training split. We use a subset
for this quick reference to keep runtime short.
"""

# %%
# Load CIFAR-10 training data (subset for quick demo)
config = TFDSEagerConfig(
    name="cifar10",
    split="train[:1000]",  # First 1000 samples for demo
    shuffle=True,
    seed=42,  # Integer seed for Grain's index_shuffle
    exclude_keys={"id"},  # Exclude non-numeric fields
)

source = TFDSEagerSource(config, rngs=nnx.Rngs(42))

print("Dataset: CIFAR-10")
print(f"Samples: {len(source)}")
print(f"Classes: {len(CIFAR10_CLASSES)}")

# Expected output:
# Dataset: CIFAR-10
# Samples: 1000
# Classes: 10

# %% [markdown]
"""
## Step 2: Define Preprocessing

Standard CIFAR-10 preprocessing:
1. Convert uint8 [0, 255] to float32 [0, 1]
2. Apply channel-wise normalization with CIFAR-10 statistics
"""


# %%
def preprocess_cifar10(element, key=None):  # noqa: ARG001
    """Normalize CIFAR-10 images to standard statistics."""
    del key  # Unused - deterministic operator
    image = element.data["image"]

    # Convert to float32 and scale to [0, 1]
    image = image.astype(jnp.float32) / 255.0

    # Apply CIFAR-10 normalization: (x - mean) / std
    image = (image - CIFAR10_MEAN) / CIFAR10_STD

    return element.update_data({"image": image})


normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_cifar10,
    rngs=nnx.Rngs(0),
)

print("Created CIFAR-10 normalizer with standard statistics")

# %% [markdown]
"""
## Step 3: Build Pipeline

Chain source and preprocessing into a batched pipeline.
Batch size of 32 is standard for CIFAR-10 training.
"""

# %%
# Build the training pipeline
batch_size = 32
pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(normalizer))

print("Pipeline: TFDSEagerSource(CIFAR-10) -> Normalize -> Output")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(source) // batch_size}")

# %% [markdown]
"""
## Step 4: Iterate Through Data

Process batches and verify the preprocessing is correct.
Normalized data should have approximately zero mean and unit variance.
"""

# %%
# Process and verify batches
print("\nProcessing batches:")
all_means = []
all_stds = []

for i, batch in enumerate(pipeline):
    if i >= 5:  # Show first 5 batches
        break

    image_batch = batch["image"]
    label_batch = batch["label"]

    # Compute per-channel statistics
    batch_mean = image_batch.mean(axis=(0, 1, 2))
    batch_std = image_batch.std(axis=(0, 1, 2))
    all_means.append(batch_mean)
    all_stds.append(batch_std)

    if i < 3:  # Print details for first 3 batches
        print(f"Batch {i}:")
        print(f"  Image: shape={image_batch.shape}, dtype={image_batch.dtype}")
        print(f"  Labels: {label_batch[:8]}... (first 8)")
        print(
            f"  Per-channel mean: [{batch_mean[0]:.3f}, {batch_mean[1]:.3f}, {batch_mean[2]:.3f}]"
        )

# Expected output:
# Batch 0:
#   Image: shape=(32, 32, 32, 3), dtype=float32
#   Labels: [6 9 9 4 1 1 2 7]... (first 8)
#   Per-channel mean: [-0.012, 0.034, -0.089]

# %%
# Aggregate statistics across batches
import jax.numpy as jnp

mean_of_means = jnp.stack(all_means).mean(axis=0)
mean_of_stds = jnp.stack(all_stds).mean(axis=0)

print("\nAggregate Statistics (should be ~0 mean, ~1 std):")
print(
    f"  Mean across batches: [{mean_of_means[0]:.3f}, {mean_of_means[1]:.3f}, "
    f"{mean_of_means[2]:.3f}]"
)
print(
    f"  Std across batches:  [{mean_of_stds[0]:.3f}, {mean_of_stds[1]:.3f}, {mean_of_stds[2]:.3f}]"
)

# %% [markdown]
"""
## Results Summary

| Component | Description |
|-----------|-------------|
| **Dataset** | CIFAR-10 (1000 samples for demo) |
| **Image Shape** | (32, 32, 3) RGB |
| **Batch Size** | 32 |
| **Normalization** | Channel-wise with CIFAR-10 statistics |
| **Output Range** | Approximately N(0, 1) per channel |

### Data Format

```
batch = {
    "image": Array[32, 32, 32, 3],  # (batch, height, width, channels)
    "label": Array[32]               # (batch,) integer labels 0-9
}
```

### Why Normalize?

1. **Faster convergence**: Normalized inputs improve gradient flow
2. **Compatibility**: Matches pretrained model expectations
3. **Numerical stability**: Prevents overflow/underflow in deep networks
"""

# %% [markdown]
"""
## Next Steps

- **Augmentation**: Add [image operators](05_augmentation_quickref.ipynb) for training
- **Full training**: See [MNIST Tutorial](06_mnist_tutorial.ipynb) for complete workflow
- **Advanced**: [MixUp/CutMix](../advanced/augmentation/01_mixup_cutmix_tutorial.ipynb)
- **API Reference**: [TFDSEagerSource](https://datarax.readthedocs.io/sources/tfds/)
"""


# %%
def main():
    """Run the CIFAR-10 pipeline example."""
    print("CIFAR-10 Pipeline Quick Reference")
    print("=" * 50)

    # Create source
    config = TFDSEagerConfig(
        name="cifar10",
        split="train[:500]",
        shuffle=True,
        seed=42,  # Integer seed for Grain's index_shuffle
        exclude_keys={"id"},  # Exclude non-numeric fields
    )
    source = TFDSEagerSource(config, rngs=nnx.Rngs(42))

    # Create normalizer
    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_cifar10,
        rngs=nnx.Rngs(0),
    )

    # Build and run pipeline
    pipeline = from_source(source, batch_size=32).add(OperatorNode(normalizer))

    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]
        # Verify shape
        assert batch["image"].shape[1:] == (32, 32, 3), "Unexpected image shape"

    print(f"Processed {total_samples} CIFAR-10 samples")
    print("Image shape: (batch, 32, 32, 3)")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
