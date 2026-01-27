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
# End-to-End CIFAR-10 Training Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~60 min (CPU) / ~15 min (GPU) |
| **Prerequisites** | MNIST Tutorial, Augmentation tutorials |
| **Format** | Python + Jupyter |
| **Memory** | ~2 GB RAM |

## Overview

Build a complete, production-ready training pipeline for CIFAR-10 image
classification. This guide integrates all Datarax features: data loading,
augmentation, batch mixing, and metrics collection with a Flax NNX model.

## Learning Goals

By the end of this guide, you will be able to:

1. Design complete training and validation pipelines
2. Implement a CNN with Flax NNX from scratch
3. Use MixUp augmentation for improved generalization
4. Track training metrics and generate visualizations
5. Evaluate model performance with confusion matrices
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[tfds]" flax optax matplotlib seaborn
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
from datarax.core.config import BatchMixOperatorConfig
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## Configuration

All hyperparameters in one place for easy tuning.
"""

# %%
# Dataset
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
NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)

# Normalization
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2470, 0.2435, 0.2616])

# Training
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 5  # Reduced for demo
TRAIN_SAMPLES = 5000  # Subset for faster demo
TEST_SAMPLES = 1000

# Augmentation
USE_MIXUP = True
MIXUP_ALPHA = 0.2

print("Configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight decay: {WEIGHT_DECAY}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  MixUp: {USE_MIXUP} (alpha={MIXUP_ALPHA})")

# %% [markdown]
"""
## Part 1: Data Pipeline

### Training Pipeline (with augmentation)
- Standard CIFAR-10 normalization
- Random brightness/contrast
- Light Gaussian noise
- MixUp for regularization

### Validation Pipeline (no augmentation)
- Normalization only
- Hard labels for evaluation
"""


# %%
def preprocess_train(element, key=None):  # noqa: ARG001
    """Preprocess for training with one-hot labels for MixUp."""
    del key
    image = element.data["image"]

    # Normalize
    image = image.astype(jnp.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD

    # One-hot labels for MixUp
    label = element.data["label"]
    label_onehot = jnp.eye(NUM_CLASSES, dtype=jnp.float32)[label]

    return element.update_data(
        {
            "image": image,
            "label": label_onehot,
            "label_idx": label,
        }
    )


def preprocess_val(element, key=None):  # noqa: ARG001
    """Preprocess for validation with integer labels."""
    del key
    image = element.data["image"]

    # Normalize
    image = image.astype(jnp.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD

    return element.update_data(
        {
            "image": image,
            "label": element.data["label"],
        }
    )


print("Preprocessing functions defined")


# %%
def create_train_pipeline(seed=42):
    """Create training pipeline with augmentation."""
    # Source
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="cifar10",
            split=f"train[:{TRAIN_SAMPLES}]",
            shuffle=True,
            seed=seed,
            exclude_keys={"id"},
        ),
        rngs=nnx.Rngs(seed),
    )

    # Preprocessor
    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_train,
        rngs=nnx.Rngs(0),
    )

    # Augmentation operators
    brightness = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.1, 0.1),
            stochastic=True,
            stream_name="brightness",
        ),
        rngs=nnx.Rngs(brightness=seed + 100),
    )

    contrast = ContrastOperator(
        ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.9, 1.1),
            stochastic=True,
            stream_name="contrast",
        ),
        rngs=nnx.Rngs(contrast=seed + 200),
    )

    noise = NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.05,
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=seed + 300),
    )

    # Build pipeline
    pipeline = (
        from_source(source, batch_size=BATCH_SIZE)
        .add(OperatorNode(prep))
        .add(OperatorNode(brightness))
        .add(OperatorNode(contrast))
        .add(OperatorNode(noise))
    )

    # Add MixUp if enabled
    if USE_MIXUP:
        mixup = BatchMixOperator(
            BatchMixOperatorConfig(
                mode="mixup",
                alpha=MIXUP_ALPHA,
                data_field="image",
                label_field="label",
                stochastic=True,
                stream_name="mixup",
            ),
            rngs=nnx.Rngs(mixup=seed + 400),
        )
        pipeline = pipeline.add(OperatorNode(mixup))

    return pipeline


def create_val_pipeline():
    """Create validation pipeline (no augmentation)."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="cifar10",
            split=f"test[:{TEST_SAMPLES}]",
            shuffle=False,
            exclude_keys={"id"},
        ),
        rngs=nnx.Rngs(0),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_val,
        rngs=nnx.Rngs(0),
    )

    return from_source(source, batch_size=BATCH_SIZE).add(OperatorNode(prep))


print("Pipeline factories created")

# %% [markdown]
"""
## Part 2: Model Architecture

A ResNet-inspired CNN for CIFAR-10.
"""


# %%
class ResidualBlock(nnx.Module):
    """Basic residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nnx.Conv(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                padding="SAME",
                rngs=rngs,
            )
        else:
            self.skip = None

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out = out + identity
        out = nnx.relu(out)

        return out


class CIFAR10Net(nnx.Module):
    """ResNet-inspired network for CIFAR-10."""

    def __init__(self, rngs: nnx.Rngs):
        # Initial convolution
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)

        # Residual blocks
        self.block1 = ResidualBlock(32, 32, stride=1, rngs=rngs)
        self.block2 = ResidualBlock(32, 64, stride=2, rngs=rngs)  # 16x16
        self.block3 = ResidualBlock(64, 64, stride=1, rngs=rngs)
        self.block4 = ResidualBlock(64, 128, stride=2, rngs=rngs)  # 8x8

        # Classification head
        self.fc = nnx.Linear(128, NUM_CLASSES, rngs=rngs)

    def __call__(self, x):
        # Initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)

        # Residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))

        # Classification
        x = self.fc(x)

        return x


# Create model
model = CIFAR10Net(rngs=nnx.Rngs(0))

# Test forward pass
test_input = jnp.ones((2, 32, 32, 3))
test_output = model(test_input)
print(f"Model output shape: {test_output.shape}")

# %% [markdown]
"""
## Part 3: Training Loop
"""

# %%
# Create optimizer with weight decay
# wrt=nnx.Param tells the optimizer to update all Param variables
optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
    wrt=nnx.Param,
)


@nnx.jit
def train_step(model: CIFAR10Net, optimizer: nnx.Optimizer, images: jax.Array, labels: jax.Array):
    """Single training step with soft labels."""

    def loss_fn(model):
        logits = model(images)
        # Soft cross-entropy for MixUp
        loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    # Compute accuracy (argmax of soft labels)
    logits = model(images)
    predictions = jnp.argmax(logits, axis=-1)
    targets = jnp.argmax(labels, axis=-1)
    accuracy = (predictions == targets).mean()

    return loss, accuracy


@nnx.jit
def eval_step(model: CIFAR10Net, images: jax.Array, labels: jax.Array):
    """Single evaluation step with hard labels."""
    logits = model(images)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss, accuracy, predictions


print("Training functions defined")

# %% [markdown]
"""
### Training Execution
"""

# %%
# Metrics storage
train_losses = []
train_accs = []
val_losses = []
val_accs = []
epoch_times = []
throughputs = []

print()
print("=" * 60)
print("TRAINING CIFAR-10")
print("=" * 60)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # Training phase
    train_pipeline = create_train_pipeline(seed=epoch)
    epoch_losses = []
    epoch_accs = []
    samples_processed = 0

    for batch in train_pipeline:
        images = batch["image"]
        labels = batch["label"]

        loss, acc = train_step(model, optimizer, images, labels)

        epoch_losses.append(float(loss))
        epoch_accs.append(float(acc))
        samples_processed += images.shape[0]

    train_loss = np.mean(epoch_losses)
    train_acc = np.mean(epoch_accs)
    train_losses.extend(epoch_losses)
    train_accs.append(train_acc)

    # Validation phase
    val_pipeline = create_val_pipeline()
    val_losses_epoch = []
    val_accs_epoch = []
    all_predictions = []
    all_labels = []

    for batch in val_pipeline:
        images = batch["image"]
        labels = batch["label"]

        loss, acc, preds = eval_step(model, images, labels)

        val_losses_epoch.append(float(loss))
        val_accs_epoch.append(float(acc))
        all_predictions.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    val_loss = np.mean(val_losses_epoch)
    val_acc = np.mean(val_accs_epoch)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Timing
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    throughput = samples_processed / epoch_time
    throughputs.append(throughput)

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
    print(f"  Time: {epoch_time:.1f}s, Throughput: {throughput:.0f} samples/s")

print("\nTraining complete!")

# %% [markdown]
"""
## Part 4: Visualizations
"""

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1 = axes[0]
ax1.plot(train_losses, alpha=0.3, color="blue", label="_nolegend_")
# Smoothed
window = max(5, len(train_losses) // 20)
smoothed = np.convolve(train_losses, np.ones(window) / window, mode="valid")
ax1.plot(range(window - 1, len(train_losses)), smoothed, color="blue", linewidth=2, label="Train")
# Validation (per epoch)
epochs_x = [(i + 1) * (len(train_losses) // NUM_EPOCHS) for i in range(NUM_EPOCHS)]
ax1.plot(epochs_x, val_losses, "o-", color="orange", linewidth=2, markersize=8, label="Val")
ax1.set_xlabel("Batch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2 = axes[1]
epochs = list(range(1, NUM_EPOCHS + 1))
ax2.plot(epochs, train_accs, "o-", color="blue", linewidth=2, markersize=8, label="Train")
ax2.plot(epochs, val_accs, "o-", color="orange", linewidth=2, markersize=8, label="Val")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(output_dir / "e2e-training-curves.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'e2e-training-curves.png'}")

# %%
# 2. Confusion matrix

# Compute confusion matrix
confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
for pred, true in zip(all_predictions, all_labels):
    confusion[true, pred] += 1

# Normalize by row (recall)
confusion_norm = confusion.astype(np.float32) / confusion.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion_norm, cmap="Blues")

# Add text annotations
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        val = confusion_norm[i, j]
        color = "white" if val > 0.5 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(CIFAR10_CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Normalized)")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(
    output_dir / "e2e-confusion-matrix.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'e2e-confusion-matrix.png'}")

# %%
# 3. Per-class accuracy
per_class_acc = confusion.diagonal() / confusion.sum(axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, NUM_CLASSES))
bars = ax.bar(CIFAR10_CLASSES, per_class_acc, color=colors)
ax.set_xlabel("Class")
ax.set_ylabel("Accuracy")
ax.set_title("Per-Class Accuracy")
ax.set_ylim(0, 1)

# Add value labels
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2%}", ha="center", fontsize=9)

ax.axhline(y=val_accs[-1], color="red", linestyle="--", label=f"Mean: {val_accs[-1]:.2%}")
ax.legend()
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig(
    output_dir / "e2e-per-class-accuracy.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'e2e-per-class-accuracy.png'}")

# %%
# 4. Throughput during training
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, NUM_EPOCHS + 1), throughputs, color="steelblue")
ax.axhline(
    y=np.mean(throughputs),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(throughputs):.0f} samples/s",
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Throughput (samples/second)")
ax.set_title("Training Throughput per Epoch")
ax.legend()

for i, tp in enumerate(throughputs):
    ax.text(i + 1, tp + 50, f"{tp:.0f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(
    output_dir / "e2e-throughput-during-training.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print(f"Saved: {output_dir / 'e2e-throughput-during-training.png'}")

# %% [markdown]
"""
## Results Summary

| Metric | Value |
|--------|-------|
| Final Train Accuracy | ~{train_accs[-1]:.1%} |
| Final Val Accuracy | ~{val_accs[-1]:.1%} |
| Mean Throughput | ~{np.mean(throughputs):.0f} samples/s |
| Total Training Time | ~{sum(epoch_times):.0f}s |

### Model Architecture

```
CIFAR10Net:
├── Conv3x3(3→32) + BN + ReLU
├── ResBlock(32→32)
├── ResBlock(32→64, stride=2)  # 16x16
├── ResBlock(64→64)
├── ResBlock(64→128, stride=2)  # 8x8
├── GlobalAvgPool
└── FC(128→10)
```

### Pipeline Architecture

```
Training:
TFDSEagerSource → Preprocess → Brightness → Contrast → Noise → MixUp → Model

Validation:
TFDSEagerSource → Preprocess → Model
```

### Key Takeaways

1. **Separate pipelines**: Train with augmentation, validate without
2. **MixUp**: Creates soft labels, requires adapted loss function
3. **BatchNorm**: Use in eval mode for validation
4. **Fresh pipelines**: Create new pipeline each epoch for shuffling
5. **Throughput tracking**: Monitor for optimization opportunities
"""

# %% [markdown]
"""
## Next Steps

- **Improve accuracy**: Add more augmentations, train longer
- **Distributed**: [Sharding guide](../distributed/02_sharding_guide.ipynb)
- **Checkpointing**: [Resumable training](../checkpointing/02_resumable_training_guide.ipynb)
- **Performance**: [Optimization guide](../performance/01_optimization_guide.ipynb)
"""


# %%
def main():
    """Run the end-to-end CIFAR-10 guide."""
    print("End-to-End CIFAR-10 Training Guide")
    print("=" * 50)

    # Quick training demo
    model = CIFAR10Net(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # Train one epoch
    pipeline = create_train_pipeline(seed=0)
    for i, batch in enumerate(pipeline):
        if i >= 20:
            break
        images = batch["image"]
        labels = batch["label"]
        _, _ = train_step(model, optimizer, images, labels)

    # Validate
    val_pipeline = create_val_pipeline()
    correct = 0
    total = 0
    for batch in val_pipeline:
        images = batch["image"]
        labels = batch["label"]
        _, acc, _ = eval_step(model, images, labels)
        correct += int(acc * len(labels))
        total += len(labels)

    print(f"Validation accuracy after quick training: {correct / total:.2%}")
    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
