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
# Image Augmentation Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline |
| **Format** | Python + Jupyter |
| **Memory** | ~200 MB RAM |

## Overview

This quick reference demonstrates Datarax's built-in image augmentation operators.
You'll learn to chain multiple operators for realistic training augmentation,
using both deterministic and stochastic transformations.

## Learning Goals

By the end of this example, you will be able to:

1. Use built-in image operators (Brightness, Contrast, Rotation, Noise)
2. Chain operators with the `>>` operator syntax
3. Understand stochastic vs deterministic modes
4. Configure operator parameters for different augmentation strengths
"""

# %% [markdown]
"""
## Setup

```bash
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
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")

# %% [markdown]
"""
## Create Sample Data

We'll create synthetic image data to demonstrate augmentations.
In practice, you'd load real images from TFDSEagerSource or HFEagerSource.
"""

# %%
# Create sample RGB images
np.random.seed(42)
num_samples = 64
image_shape = (32, 32, 3)  # CIFAR-10 like

# Create gradient images for clear visualization of transforms
data = {
    "image": np.random.rand(num_samples, *image_shape).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

print(f"Created {num_samples} sample images: {image_shape}")
print("Image range: [0.0, 1.0] (pre-normalized)")

# %% [markdown]
"""
## Built-in Image Operators

Datarax provides optimized JAX-based image operators. Each operator:

- Has a Config class for parameters
- Supports stochastic (random) or deterministic modes
- Uses named RNG streams for reproducibility

### Available Operators

| Operator | Effect | Stochastic |
|----------|--------|------------|
| `BrightnessOperator` | Additive brightness delta | Yes |
| `ContrastOperator` | Multiplicative contrast factor | Yes |
| `RotationOperator` | Rotation by angle | Yes |
| `NoiseOperator` | Gaussian/salt-pepper noise | Yes |
| `DropoutOperator` | Pixel/channel dropout | Yes |
| `PatchDropoutOperator` | Cutout-style patches | Yes |
"""

# %% [markdown]
"""
## Step 1: Individual Operators

Let's examine each operator individually before chaining.
"""

# %%
# 1. Brightness Operator
# Adds a random delta to pixel values
brightness_op = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.2, 0.2),  # Random delta in [-0.2, +0.2]
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

print("BrightnessOperator:")
print("  - Adds random delta to all pixels")
print("  - Range: [-0.2, +0.2]")
print("  - Effect: Makes images brighter or darker")

# %%
# 2. Contrast Operator
# Multiplies pixel values around the mean
contrast_op = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image",
        contrast_range=(0.8, 1.2),  # Factor between 0.8x and 1.2x
        stochastic=True,
        stream_name="contrast",
    ),
    rngs=nnx.Rngs(contrast=200),
)

print("ContrastOperator:")
print("  - Multiplies (pixel - mean) by random factor")
print("  - Range: [0.8, 1.2]")
print("  - Effect: Increases or decreases contrast")

# %%
# 3. Rotation Operator
# Rotates images by random angle
rotation_op = RotationOperator(
    RotationOperatorConfig(
        field_key="image",
        angle_range=(-15.0, 15.0),  # Degrees
        fill_value=0.0,  # Fill empty areas with black
    ),
    rngs=nnx.Rngs(0),
)

print("RotationOperator:")
print("  - Rotates image by random angle")
print("  - Range: [-15°, +15°]")
print("  - Uses bilinear interpolation")

# %%
# 4. Noise Operator
# Adds random noise to images
noise_op = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",  # or "salt_pepper", "poisson"
        noise_std=0.05,  # Standard deviation of Gaussian noise
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=300),
)

print("NoiseOperator (Gaussian mode):")
print("  - Adds zero-mean Gaussian noise")
print("  - Std: 0.05")
print("  - Effect: Simulates sensor noise")

# %% [markdown]
"""
## Step 2: Chain Operators with >>

Use the `>>` operator for fluent pipeline composition.
Operators are applied left-to-right.
"""

# %%
# Create fresh source for chained pipeline
source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))

# Create fresh operators (each needs its own RNG state)
brightness = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.15, 0.15),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=10),
)

contrast = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image",
        contrast_range=(0.85, 1.15),
        stochastic=True,
        stream_name="contrast",
    ),
    rngs=nnx.Rngs(contrast=20),
)

noise = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.03,
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=30),
)

# Chain with >> operator
augmented_pipeline = (
    from_source(source2, batch_size=16)
    >> OperatorNode(brightness)
    >> OperatorNode(contrast)
    >> OperatorNode(noise)
)

print("Augmentation Pipeline:")
print("  Source -> Brightness -> Contrast -> Noise -> Output")

# %% [markdown]
"""
## Step 3: Process Data

Run the augmented pipeline and examine results.
"""

# %%
# Process batches
print("\nProcessing augmented batches:")

for i, batch in enumerate(augmented_pipeline):
    if i >= 3:
        break

    images = batch["image"]
    labels = batch["label"]

    print(f"Batch {i}:")
    print(f"  Image shape: {images.shape}")
    print(f"  Image range: [{float(images.min()):.3f}, {float(images.max()):.3f}]")
    print(f"  Mean: {float(images.mean()):.3f}, Std: {float(images.std()):.3f}")

# Expected output:
# Batch 0:
#   Image shape: (16, 32, 32, 3)
#   Image range: [-0.123, 1.089]  # May exceed [0,1] due to augmentation
#   Mean: 0.498, Std: 0.312

# %% [markdown]
"""
## Step 4: Add Clipping (Optional)

Augmentations can push values outside [0, 1]. Add clipping if needed.
"""


# %%
def clip_image(element, key=None):  # noqa: ARG001
    """Clip image values to [0, 1] range."""
    del key
    image = element.data["image"]
    clipped = jnp.clip(image, 0.0, 1.0)
    return element.update_data({"image": clipped})


clipper = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=clip_image,
    rngs=nnx.Rngs(0),
)

# Create pipeline with clipping
source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))

brightness2 = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.15, 0.15),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=10),
)

clipped_pipeline = (
    from_source(source3, batch_size=16) >> OperatorNode(brightness2) >> OperatorNode(clipper)
)

# Verify clipping
batch = next(iter(clipped_pipeline))
img_min = float(batch["image"].min())
img_max = float(batch["image"].max())
print(f"With clipping - Image range: [{img_min:.3f}, {img_max:.3f}]")

# %% [markdown]
"""
## Deterministic vs Stochastic Mode

Operators can run in deterministic mode with fixed parameters.
"""

# %%
# Deterministic brightness (always +0.1)
deterministic_brightness = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_delta=0.1,  # Fixed delta, not range
        stochastic=False,  # Deterministic mode
    ),
    rngs=nnx.Rngs(0),
)

print("Deterministic BrightnessOperator:")
print("  - Always adds +0.1 to all pixels")
print("  - Useful for inference-time preprocessing")

# %% [markdown]
"""
## Results Summary

| Operator | Parameter | Effect |
|----------|-----------|--------|
| Brightness | `(-0.15, 0.15)` | ±15% brightness change |
| Contrast | `(0.85, 1.15)` | ±15% contrast change |
| Noise | `std=0.03` | Light Gaussian noise |
| Rotation | `(-15°, +15°)` | Mild rotation |

### Best Practices

1. **Strength matters**: Start mild, increase if needed
2. **Order matters**: Normalize last, augment first
3. **RNG streams**: Use unique `stream_name` per operator
4. **Clipping**: Add if values must stay in [0, 1]
5. **Seeds**: Set seeds for reproducibility
"""

# %% [markdown]
"""
## Next Steps

- **More operators**: See [Operators Tutorial](03_operators_tutorial.ipynb)
- **MixUp/CutMix**: [Batch augmentation](../advanced/augmentation/01_mixup_cutmix_tutorial.ipynb)
- **Full pipeline**: [MNIST Tutorial](06_mnist_tutorial.ipynb)
- **API Reference**: [Image Operators](https://datarax.readthedocs.io/operators/image/)
"""


# %%
def main():
    """Run the augmentation quick reference example."""
    print("Image Augmentation Quick Reference")
    print("=" * 50)

    # Create data
    np.random.seed(42)
    data = {
        "image": np.random.rand(64, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (64,)).astype(np.int32),
    }
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    # Create augmentation operators
    brightness = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.15, 0.15),
            stochastic=True,
            stream_name="brightness",
        ),
        rngs=nnx.Rngs(brightness=10),
    )

    contrast = ContrastOperator(
        ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.85, 1.15),
            stochastic=True,
            stream_name="contrast",
        ),
        rngs=nnx.Rngs(contrast=20),
    )

    noise = NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.03,
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=30),
    )

    # Build pipeline
    pipeline = (
        from_source(source, batch_size=16)
        >> OperatorNode(brightness)
        >> OperatorNode(contrast)
        >> OperatorNode(noise)
    )

    # Process all batches
    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} augmented samples")
    print("Operators applied: Brightness -> Contrast -> Noise")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
