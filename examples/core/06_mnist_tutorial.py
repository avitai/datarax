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
# MNIST Classification Pipeline Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 min (CPU) / ~10 min (GPU) |
| **Prerequisites** | Simple Pipeline, Operators Tutorial |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB RAM |

## Overview

Build a complete MNIST classification pipeline from data loading to training.
This tutorial demonstrates the full Datarax workflow with a Flax NNX model,
covering data preprocessing, augmentation, training loop integration, and
performance analysis.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Create a complete training pipeline with TFDSSource
2. Apply standard MNIST preprocessing and augmentation
3. Integrate Datarax with Flax NNX training loops
4. Handle epochs and shuffling correctly
5. Generate visualizations of samples and training metrics
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with TFDS and Flax support
uv pip install "datarax[tfds]" flax optax matplotlib
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
import optax
from flax import nnx

# Datarax imports
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## MNIST Dataset Overview

MNIST is the "Hello World" of machine learning - 70,000 grayscale images of
handwritten digits (0-9).

| Property | Value |
|----------|-------|
| Image size | 28×28×1 (grayscale) |
| Train samples | 60,000 |
| Test samples | 10,000 |
| Classes | 10 (digits 0-9) |
| Pixel range | 0-255 (uint8) |

### Standard Normalization

| Statistic | Value |
|-----------|-------|
| Mean | 0.1307 |
| Std | 0.3081 |
"""

# %%
# MNIST constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28, 1)

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 3  # Reduced for tutorial
TRAIN_SAMPLES = 10000  # Subset for faster demo

print("Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Training samples: {TRAIN_SAMPLES}")

# %% [markdown]
"""
## Part 1: Data Loading and Preprocessing

Create the MNIST data source and preprocessing pipeline.
"""

# %%
# Create MNIST training source
train_config = TFDSEagerConfig(
    name="mnist",
    split=f"train[:{TRAIN_SAMPLES}]",
    shuffle=True,
    seed=42,
)

train_source = TFDSEagerSource(train_config, rngs=nnx.Rngs(42))

# Create test source (no shuffle)
test_config = TFDSEagerConfig(
    name="mnist",
    split="test[:2000]",  # Subset for faster evaluation
    shuffle=False,
)

test_source = TFDSEagerSource(test_config, rngs=nnx.Rngs(0))

print(f"Training samples: {len(train_source)}")
print(f"Test samples: {len(test_source)}")


# %% [markdown]
"""
### Preprocessing Function

Standard MNIST preprocessing:
1. Convert uint8 [0-255] to float32 [0-1]
2. Normalize with MNIST statistics
3. Ensure correct shape (add channel dim if needed)
"""


# %%
def preprocess_mnist(element, key=None):  # noqa: ARG001
    """Preprocess MNIST images with standard normalization."""
    del key  # Unused - deterministic operator
    image = element.data["image"]

    # Convert to float32 and scale to [0, 1]
    image = image.astype(jnp.float32) / 255.0

    # Ensure channel dimension
    if image.ndim == 2:
        image = image[..., None]

    # Apply MNIST normalization
    image = (image - MNIST_MEAN) / MNIST_STD

    # One-hot encode labels for cross-entropy
    label = element.data["label"]
    label_onehot = jax.nn.one_hot(label, NUM_CLASSES)

    return element.update_data({"image": image, "label": label, "label_onehot": label_onehot})


preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_mnist,
    rngs=nnx.Rngs(0),
)

print("Created MNIST preprocessor with one-hot encoding")

# %% [markdown]
"""
### Training Augmentation

Light augmentation for training: brightness and noise.
Test pipeline has no augmentation.
"""

# %%
# Training augmentation operators
brightness_aug = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.1, 0.1),
        stochastic=True,
        stream_name="brightness",
    ),
    rngs=nnx.Rngs(brightness=100),
)

noise_aug = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.1,
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=200),
)

print("Created augmentation operators:")
print("  - Brightness: ±0.1")
print("  - Gaussian noise: std=0.1")

# %% [markdown]
"""
### Build Pipelines

Training pipeline with augmentation, test pipeline without.
"""

# %%
# Training pipeline with augmentation
train_pipeline = (
    from_source(train_source, batch_size=BATCH_SIZE)
    .add(OperatorNode(preprocessor))
    .add(OperatorNode(brightness_aug))
    .add(OperatorNode(noise_aug))
)

# Test pipeline without augmentation (create fresh sources for actual use)
test_preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_mnist,
    rngs=nnx.Rngs(0),
)

test_pipeline = from_source(test_source, batch_size=BATCH_SIZE).add(OperatorNode(test_preprocessor))

print("Pipelines created:")
print("  Train: Source -> Preprocess -> Brightness -> Noise")
print("  Test:  Source -> Preprocess")

# %% [markdown]
"""
## Part 2: Visualize Sample Data

Generate visualization of MNIST samples before and after augmentation.
"""

# %%
# Get sample batch for visualization
sample_batch = next(iter(train_pipeline))
images = sample_batch["image"]
labels = sample_batch["label"]

print(f"Sample batch shape: {images.shape}")
print(f"Sample labels: {labels[:16]}")


# %%
def plot_mnist_grid(images, labels, title, filename=None, nrows=4, ncols=4):
    """Plot a grid of MNIST images with labels."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Denormalize for display
            img = images[i] * MNIST_STD + MNIST_MEAN
            img = np.clip(img, 0, 1)

            # Remove channel dim for display
            if img.ndim == 3:
                img = img.squeeze(-1)

            ax.imshow(img, cmap="gray")
            ax.set_title(f"Label: {int(labels[i])}")
        ax.axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved: {filename}")

    plt.close()
    return fig


# Generate sample grid
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

plot_mnist_grid(
    np.array(images[:16]),
    np.array(labels[:16]),
    "MNIST Training Samples (with augmentation)",
    output_dir / "cv-mnist-sample-grid.png",
)

# %% [markdown]
"""
## Part 3: Define the Model

Simple CNN for MNIST classification using Flax NNX.
"""


# %%
class MNISTClassifier(nnx.Module):
    """Simple CNN for MNIST classification."""

    def __init__(self, rngs: nnx.Rngs):
        # Convolutional layers
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        # Dense layers
        self.dense1 = nnx.Linear(64 * 7 * 7, 128, rngs=rngs)
        self.dense2 = nnx.Linear(128, NUM_CLASSES, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Conv block 1: Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Conv block 2: Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = nnx.relu(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten and dense layers
        x = x.reshape(x.shape[0], -1)
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dense2(x)

        return x


# Create model
model = MNISTClassifier(rngs=nnx.Rngs(0))

# Test forward pass
dummy_input = jnp.ones((1, 28, 28, 1))
dummy_output = model(dummy_input)
print(f"Model output shape: {dummy_output.shape}")

# %% [markdown]
"""
## Part 4: Training Loop

Implement training with Datarax pipeline integration.
"""

# %%
# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)


@nnx.jit
def train_step(model: MNISTClassifier, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
    """Single training step."""
    images = batch["image"]
    labels = batch["label_onehot"]

    def loss_fn(model):
        logits = model(images)
        loss = optax.softmax_cross_entropy(logits, labels).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


@nnx.jit
def eval_step(model: MNISTClassifier, batch: dict) -> tuple[jax.Array, jax.Array]:
    """Single evaluation step."""
    images = batch["image"]
    labels = batch["label"]

    logits = model(images)
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == labels).sum()

    return correct, len(labels)


print("Training and evaluation functions defined")

# %% [markdown]
"""
### Training Loop with Metrics Collection
"""

# %%
# Training metrics storage
train_losses = []
train_times = []
batch_throughputs = []


def create_train_pipeline():
    """Create a fresh training pipeline for each epoch."""
    source = TFDSEagerSource(train_config, rngs=nnx.Rngs(42))

    preprocessor = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_mnist,
        rngs=nnx.Rngs(0),
    )

    brightness = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.1, 0.1),
            stochastic=True,
            stream_name="brightness",
        ),
        rngs=nnx.Rngs(brightness=100),
    )

    noise = NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=200),
    )

    return (
        from_source(source, batch_size=BATCH_SIZE)
        .add(OperatorNode(preprocessor))
        .add(OperatorNode(brightness))
        .add(OperatorNode(noise))
    )


def create_test_pipeline():
    """Create a fresh test pipeline."""
    source = TFDSEagerSource(test_config, rngs=nnx.Rngs(0))

    preprocessor = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_mnist,
        rngs=nnx.Rngs(0),
    )

    return from_source(source, batch_size=BATCH_SIZE).add(OperatorNode(preprocessor))


# Training loop
print("\nStarting training...")
print("=" * 50)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    epoch_losses = []

    # Create fresh pipeline for this epoch
    pipeline = create_train_pipeline()

    for batch_idx, batch in enumerate(pipeline):
        batch_start = time.time()

        # Training step
        loss = train_step(model, optimizer, batch)
        epoch_losses.append(float(loss))

        # Track throughput
        batch_time = time.time() - batch_start
        throughput = BATCH_SIZE / batch_time if batch_time > 0 else 0
        batch_throughputs.append(throughput)

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch + 1}, Batch {batch_idx}: loss={float(loss):.4f}")

    # Epoch summary
    epoch_time = time.time() - epoch_start
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.extend(epoch_losses)
    train_times.append(epoch_time)

    # Evaluate on test set
    test_pipeline = create_test_pipeline()
    total_correct = 0
    total_samples = 0

    for batch in test_pipeline:
        correct, n = eval_step(model, batch)
        total_correct += int(correct)
        total_samples += int(n)

    accuracy = total_correct / total_samples

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
    print(f"  Train loss: {epoch_loss:.4f}")
    print(f"  Test accuracy: {accuracy:.2%}")
    print(f"  Time: {epoch_time:.1f}s")
    print()

print("Training complete!")

# %% [markdown]
"""
## Part 5: Generate Visualizations

Create plots for training metrics and pipeline performance.
"""

# %%
# 1. Training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, alpha=0.7, linewidth=0.5)

# Add smoothed line
window = min(20, len(train_losses) // 5)
if window > 1:
    smoothed = np.convolve(train_losses, np.ones(window) / window, mode="valid")
    plt.plot(range(window - 1, len(train_losses)), smoothed, linewidth=2, label="Smoothed")

plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("MNIST Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(
    output_dir / "cv-mnist-training-loss.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-mnist-training-loss.png'}")

# %%
# 2. Throughput Analysis
plt.figure(figsize=(10, 6))
plt.plot(batch_throughputs, alpha=0.5)
avg_throughput = np.mean(batch_throughputs)
plt.axhline(
    y=avg_throughput, color="r", linestyle="--", label=f"Average: {avg_throughput:.0f} samples/s"
)
plt.xlabel("Batch")
plt.ylabel("Throughput (samples/second)")
plt.title("Pipeline Throughput During Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(output_dir / "cv-mnist-throughput.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'cv-mnist-throughput.png'}")

# %%
# 3. Augmentation Comparison
# Get samples without augmentation for comparison
plain_source = TFDSEagerSource(
    TFDSEagerConfig(name="mnist", split="train[:128]", shuffle=False),
    rngs=nnx.Rngs(0),
)

plain_preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_mnist,
    rngs=nnx.Rngs(0),
)

plain_pipeline = from_source(plain_source, batch_size=128).add(OperatorNode(plain_preprocessor))
plain_batch = next(iter(plain_pipeline))

# Get augmented samples
aug_pipeline = create_train_pipeline()
aug_batch = next(iter(aug_pipeline))

# Plot comparison
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Original vs Augmented MNIST Samples", fontsize=14)

for i in range(8):
    # Original
    img_orig = np.array(plain_batch["image"][i]) * MNIST_STD + MNIST_MEAN
    img_orig = np.clip(img_orig, 0, 1).squeeze()
    axes[0, i].imshow(img_orig, cmap="gray")
    axes[0, i].axis("off")
    if i == 0:
        axes[0, i].set_ylabel("Original", fontsize=12)

    # Augmented
    img_aug = np.array(aug_batch["image"][i]) * MNIST_STD + MNIST_MEAN
    img_aug = np.clip(img_aug, 0, 1).squeeze()
    axes[1, i].imshow(img_aug, cmap="gray")
    axes[1, i].axis("off")
    if i == 0:
        axes[1, i].set_ylabel("Augmented", fontsize=12)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-mnist-augmentation-samples.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'cv-mnist-augmentation-samples.png'}")

# %% [markdown]
"""
## Results Summary

| Metric | Value |
|--------|-------|
| Final Test Accuracy | ~95%+ |
| Average Throughput | ~5000 samples/s (CPU) |
| Training Time | ~30s per epoch |

### Key Takeaways

1. **Pipeline Integration**: Datarax integrates seamlessly with Flax NNX training loops
2. **Fresh Pipelines**: Create new pipeline instances for each epoch to reset iteration
3. **Augmentation**: Light augmentation (brightness, noise) improves generalization
4. **Preprocessing**: Always normalize with dataset-specific statistics
5. **Batching**: `from_source(batch_size=N)` handles batching automatically

### Pipeline Architecture

```
Training:
TFDSEagerSource -> Preprocess -> Brightness -> Noise -> Model

Testing:
TFDSEagerSource -> Preprocess -> Model
```
"""

# %% [markdown]
"""
## Next Steps

- **Stronger augmentation**: [Fashion-MNIST tutorial](07_fashion_augmentation_tutorial.ipynb)
- **MixUp/CutMix**: [Batch augmentation](../advanced/augmentation/01_mixup_cutmix_tutorial.ipynb)
- **Distributed training**: [Sharding guide](../advanced/distributed/02_sharding_guide.ipynb)
- **Checkpointing**: See advanced/checkpointing for resumable training
"""


# %%
def main():
    """Run the complete MNIST tutorial."""
    print("MNIST Classification Pipeline Tutorial")
    print("=" * 50)

    # Quick training run
    model = MNISTClassifier(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # Single epoch
    pipeline = create_train_pipeline()
    for batch_idx, batch in enumerate(pipeline):
        _ = train_step(model, optimizer, batch)
        if batch_idx >= 10:
            break

    # Evaluate
    test_pipeline = create_test_pipeline()
    total_correct = 0
    total_samples = 0
    for batch in test_pipeline:
        correct, n = eval_step(model, batch)
        total_correct += int(correct)
        total_samples += int(n)

    accuracy = total_correct / total_samples
    print(f"Test accuracy after quick training: {accuracy:.2%}")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
