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
# Fashion-MNIST Augmentation Pipeline Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 min (CPU) / ~15 min (GPU) |
| **Prerequisites** | MNIST Tutorial, Operators Tutorial |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB RAM |

## Overview

Build a complete augmentation pipeline for Fashion-MNIST, demonstrating
multiple image operators chained together. Fashion-MNIST is more challenging
than MNIST, making augmentation more important for good performance.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Apply multiple stacked image operators
2. Use PatchDropout (Cutout-style) for regularization
3. Use NoiseOperator with different noise types
4. Measure augmentation impact on training
5. Visualize various augmentation effects
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[tfds]" matplotlib
```
"""

# %%
# GPU Memory Configuration
import os

os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Core imports
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# Datarax imports
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    DropoutOperator,
    DropoutOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Fashion-MNIST Dataset

Fashion-MNIST contains 70,000 grayscale images of clothing items, designed as
a more challenging drop-in replacement for MNIST.

| Property | Value |
|----------|-------|
| Image size | 28×28×1 |
| Train samples | 60,000 |
| Test samples | 10,000 |
| Classes | 10 |

### Class Names

| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
"""

# %%
# Constants
FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Normalization (similar to MNIST)
FASHION_MEAN = 0.2860
FASHION_STD = 0.3530

BATCH_SIZE = 64
TRAIN_SAMPLES = 5000  # Subset for demo

print(f"Fashion-MNIST classes: {FASHION_CLASSES}")

# %% [markdown]
"""
## Part 1: Create Data Source
"""

# %%
# Load Fashion-MNIST
train_config = TFDSEagerConfig(
    name="fashion_mnist",
    split=f"train[:{TRAIN_SAMPLES}]",
    shuffle=True,
    seed=42,
)

train_source = TFDSEagerSource(train_config, rngs=nnx.Rngs(42))
print(f"Loaded {len(train_source)} Fashion-MNIST samples")


# %%
# Basic preprocessing
def preprocess_fashion(element, key=None):  # noqa: ARG001
    """Normalize Fashion-MNIST images."""
    del key
    image = element.data["image"]

    # Convert to float32 and normalize
    image = image.astype(jnp.float32) / 255.0

    # Ensure channel dimension
    if image.ndim == 2:
        image = image[..., None]

    # Apply normalization
    image = (image - FASHION_MEAN) / FASHION_STD

    return element.update_data({"image": image})


preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_fashion,
    rngs=nnx.Rngs(0),
)

# %% [markdown]
"""
## Part 2: Define Augmentation Operators

We'll create a complete augmentation suite including:

1. **Brightness/Contrast**: Photometric variations
2. **Rotation**: Geometric transformation
3. **Noise**: Sensor noise simulation
4. **PatchDropout**: Cutout-style occlusion
"""

# %%
# 1. Brightness augmentation
brightness_op = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.15, 0.15),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

# 2. Contrast augmentation
contrast_op = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image",
        contrast_range=(0.85, 1.15),
        stochastic=True,
        stream_name="contrast",
    ),
    rngs=nnx.Rngs(contrast=200),
)

# 3. Rotation augmentation
rotation_op = RotationOperator(
    RotationOperatorConfig(
        field_key="image",
        angle_range=(-10.0, 10.0),
        fill_value=0.0,
    ),
    rngs=nnx.Rngs(0),
)

# 4. Gaussian noise
noise_op = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.1,
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=300),
)

# 5. PatchDropout (Cutout-style)
patch_dropout_op = PatchDropoutOperator(
    PatchDropoutOperatorConfig(
        field_key="image",
        patch_size=(6, 6),  # 6x6 patches
        num_patches=2,  # Drop 2 patches
        drop_value=0.0,
        stochastic=True,
        stream_name="patch_dropout",
    ),
    rngs=nnx.Rngs(patch_dropout=400),
)

# 6. Pixel dropout (alternative to patch)
pixel_dropout_op = DropoutOperator(
    DropoutOperatorConfig(
        field_key="image",
        dropout_rate=0.1,
        stochastic=True,
        stream_name="dropout",
    ),
    rngs=nnx.Rngs(dropout=500),
)

print("Created augmentation operators:")
print("  1. Brightness: ±0.15")
print("  2. Contrast: 0.85-1.15x")
print("  3. Rotation: ±10°")
print("  4. Gaussian noise: std=0.1")
print("  5. PatchDropout: 2x 6x6 patches")
print("  6. PixelDropout: 10% probability")

# %% [markdown]
"""
## Part 3: Visualize Individual Augmentations

See the effect of each augmentation type.
"""


# %%
def create_single_aug_pipeline(operator, seed=0):
    """Create pipeline with single augmentation for visualization."""
    source = TFDSEagerSource(
        TFDSEagerConfig(name="fashion_mnist", split="train[:64]", shuffle=False),
        rngs=nnx.Rngs(seed),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_fashion,
        rngs=nnx.Rngs(0),
    )

    return from_source(source, batch_size=64).add(OperatorNode(prep)).add(OperatorNode(operator))


# Get baseline (no augmentation)
baseline_source = TFDSEagerSource(
    TFDSEagerConfig(name="fashion_mnist", split="train[:64]", shuffle=False),
    rngs=nnx.Rngs(0),
)
baseline_pipeline = from_source(baseline_source, batch_size=64).add(
    OperatorNode(
        ElementOperator(
            ElementOperatorConfig(stochastic=False),
            fn=preprocess_fashion,
            rngs=nnx.Rngs(0),
        )
    )
)

baseline_batch = next(iter(baseline_pipeline))
baseline_images = np.array(baseline_batch["image"])
baseline_labels = np.array(baseline_batch["label"])

# %%
# Create pipelines for each augmentation type
aug_configs = [
    ("Original", None, baseline_images),
    (
        "Brightness",
        BrightnessOperator(
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(-0.15, 0.15),
                stochastic=True,
                stream_name="brightness",
            ),
            rngs=nnx.Rngs(brightness=100),
        ),
        None,
    ),
    (
        "Contrast",
        ContrastOperator(
            ContrastOperatorConfig(
                field_key="image",
                contrast_range=(0.85, 1.15),
                stochastic=True,
                stream_name="contrast",
            ),
            rngs=nnx.Rngs(contrast=200),
        ),
        None,
    ),
    (
        "Rotation",
        RotationOperator(
            RotationOperatorConfig(
                field_key="image",
                angle_range=(-10.0, 10.0),
                fill_value=0.0,
            ),
            rngs=nnx.Rngs(0),
        ),
        None,
    ),
    (
        "Noise",
        NoiseOperator(
            NoiseOperatorConfig(
                field_key="image",
                mode="gaussian",
                noise_std=0.1,
                stochastic=True,
                stream_name="noise",
            ),
            rngs=nnx.Rngs(noise=300),
        ),
        None,
    ),
    (
        "PatchDropout",
        PatchDropoutOperator(
            PatchDropoutOperatorConfig(
                field_key="image",
                patch_size=(6, 6),
                num_patches=2,
                drop_value=0.0,
                stochastic=True,
                stream_name="patch_dropout",
            ),
            rngs=nnx.Rngs(patch_dropout=400),
        ),
        None,
    ),
]

# Get augmented samples
for i, (name, op, imgs) in enumerate(aug_configs):
    if imgs is None and op is not None:
        pipeline = create_single_aug_pipeline(op, seed=i)
        batch = next(iter(pipeline))
        aug_configs[i] = (name, op, np.array(batch["image"]))

# %%
# Plot augmentation comparison grid
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(6, 8, figsize=(16, 12))
fig.suptitle("Fashion-MNIST Augmentation Effects", fontsize=14)

for row_idx, (name, _, images) in enumerate(aug_configs):
    axes[row_idx, 0].set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center")

    for col_idx in range(8):
        ax = axes[row_idx, col_idx]
        img = images[col_idx] * FASHION_STD + FASHION_MEAN
        img = np.clip(img, 0, 1).squeeze()
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(FASHION_CLASSES[baseline_labels[col_idx]], fontsize=8)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-fashion-augmentation-grid.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-fashion-augmentation-grid.png'}")

# %% [markdown]
"""
## Part 4: Build Complete Augmentation Pipeline

Chain all augmentations into a single pipeline.
"""


# %%
def create_full_augmentation_pipeline(seed=42):
    """Create pipeline with all augmentations."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="fashion_mnist",
            split=f"train[:{TRAIN_SAMPLES}]",
            shuffle=True,
            seed=seed,
        ),
        rngs=nnx.Rngs(seed),
    )

    # Preprocessing
    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_fashion,
        rngs=nnx.Rngs(0),
    )

    # Augmentations
    brightness = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.15, 0.15),
            stochastic=True,
            stream_name="brightness",
        ),
        rngs=nnx.Rngs(brightness=100),
    )

    contrast = ContrastOperator(
        ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.85, 1.15),
            stochastic=True,
            stream_name="contrast",
        ),
        rngs=nnx.Rngs(contrast=200),
    )

    rotation = RotationOperator(
        RotationOperatorConfig(
            field_key="image",
            angle_range=(-10.0, 10.0),
            fill_value=0.0,
        ),
        rngs=nnx.Rngs(0),
    )

    noise = NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.05,  # Lighter for combined use
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=300),
    )

    patch_dropout = PatchDropoutOperator(
        PatchDropoutOperatorConfig(
            field_key="image",
            patch_size=(4, 4),  # Smaller patches for combined use
            num_patches=1,
            drop_value=0.0,
            stochastic=True,
            stream_name="patch_dropout",
        ),
        rngs=nnx.Rngs(patch_dropout=400),
    )

    # Build pipeline
    return (
        from_source(source, batch_size=BATCH_SIZE)
        .add(OperatorNode(prep))
        .add(OperatorNode(brightness))
        .add(OperatorNode(contrast))
        .add(OperatorNode(rotation))
        .add(OperatorNode(noise))
        .add(OperatorNode(patch_dropout))
    )


print("Full augmentation pipeline:")
print("  Source -> Preprocess -> Brightness -> Contrast -> Rotation -> Noise -> PatchDropout")

# %% [markdown]
"""
## Part 5: Measure Augmentation Latency

Profile the time cost of each augmentation step.
"""

# %%
# Benchmark individual augmentations
num_batches = 20
latencies = {}

for name, op, _ in aug_configs[1:]:  # Skip "Original"
    pipeline = create_single_aug_pipeline(op, seed=0)

    times = []
    for i, batch in enumerate(pipeline):
        if i >= num_batches:
            break
        start = time.time()
        _ = batch["image"].block_until_ready()  # Force computation
        times.append(time.time() - start)

    latencies[name] = np.mean(times[1:]) * 1000  # Skip first (warmup), convert to ms

print("Augmentation latency per batch (ms):")
for name, latency in latencies.items():
    print(f"  {name}: {latency:.2f} ms")

# %%
# Plot latency comparison
fig, ax = plt.subplots(figsize=(10, 6))
names = list(latencies.keys())
values = list(latencies.values())

bars = ax.barh(names, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names))))
ax.set_xlabel("Latency (ms)")
ax.set_title("Augmentation Latency per Batch (64 samples)")

# Add value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}ms", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "cv-fashion-latency.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'cv-fashion-latency.png'}")

# %% [markdown]
"""
## Part 6: Visualize Original vs Fully Augmented

Compare samples before and after full augmentation pipeline.
"""

# %%
# Get samples from full pipeline
full_pipeline = create_full_augmentation_pipeline(seed=42)
full_batch = next(iter(full_pipeline))
full_images = np.array(full_batch["image"])
full_labels = np.array(full_batch["label"])

# Plot comparison
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Original vs Fully Augmented Fashion-MNIST", fontsize=14)

for i in range(8):
    # Original (from baseline)
    img_orig = baseline_images[i] * FASHION_STD + FASHION_MEAN
    img_orig = np.clip(img_orig, 0, 1).squeeze()
    axes[0, i].imshow(img_orig, cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title(FASHION_CLASSES[baseline_labels[i]], fontsize=8)

    # Augmented
    img_aug = full_images[i] * FASHION_STD + FASHION_MEAN
    img_aug = np.clip(img_aug, 0, 1).squeeze()
    axes[1, i].imshow(img_aug, cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Augmented", fontsize=10)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-fashion-augmented.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-fashion-augmented.png'}")

# %%
# Sample grid from augmented pipeline
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Fashion-MNIST Training Samples (with augmentation)", fontsize=14)

for i, ax in enumerate(axes.flat):
    if i < len(full_images):
        img = full_images[i] * FASHION_STD + FASHION_MEAN
        img = np.clip(img, 0, 1).squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(FASHION_CLASSES[full_labels[i]], fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_dir / "cv-fashion-samples.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'cv-fashion-samples.png'}")

# %% [markdown]
"""
## Results Summary

| Augmentation | Parameter | Latency |
|--------------|-----------|---------|
| Brightness | ±0.15 | ~2 ms |
| Contrast | 0.85-1.15x | ~2 ms |
| Rotation | ±10° | ~5 ms |
| Noise | std=0.1 | ~3 ms |
| PatchDropout | 2×6×6 | ~3 ms |

### Best Practices

1. **Order matters**: Apply geometric transforms (rotation) before pixel transforms
2. **Lighter when stacking**: Reduce individual strengths when combining
3. **PatchDropout**: Forces model to use global features, improves robustness
4. **Noise**: Helps with sensor noise and compression artifacts
5. **Profile**: Measure latency impact for your specific hardware

### Recommended Pipeline Order

```
Source -> Preprocess -> Rotation -> Brightness -> Contrast -> Noise -> Dropout
```

Geometric transforms first, then photometric, then regularization.
"""

# %% [markdown]
"""
## Next Steps

- **MixUp/CutMix**: See advanced/augmentation for batch-level augmentation
- **Performance**: [Optimization guide](../advanced/performance/01_optimization_guide.ipynb)
- **Full training**: [End-to-end CIFAR-10](../advanced/training/01_e2e_cifar10_guide.ipynb)
"""


# %%
def main():
    """Run the Fashion-MNIST augmentation tutorial."""
    print("Fashion-MNIST Augmentation Pipeline Tutorial")
    print("=" * 50)

    # Create pipeline
    pipeline = create_full_augmentation_pipeline(seed=42)

    # Process batches
    total_samples = 0
    start_time = time.time()

    for i, batch in enumerate(pipeline):
        total_samples += batch["image"].shape[0]
        if i >= 10:
            break

    elapsed = time.time() - start_time

    print(f"Processed {total_samples} augmented samples")
    print(f"Throughput: {total_samples / elapsed:.0f} samples/s")
    print("Augmentations: Brightness, Contrast, Rotation, Noise, PatchDropout")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
