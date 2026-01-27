# CIFAR-10 Pipeline Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline, TFDS setup |
| **Format** | Python + Jupyter |
| **Memory** | ~500 MB RAM |

## Overview

This quick reference demonstrates loading and processing CIFAR-10 from TensorFlow Datasets (TFDS). CIFAR-10 is a classic benchmark dataset containing 60,000 32x32 color images in 10 classes, making it ideal for learning image classification pipelines.

## What You'll Learn

1. Load CIFAR-10 using `TFDSEagerSource` with proper configuration
2. Apply standard CIFAR-10 normalization (ImageNet-style statistics)
3. Build a batched pipeline ready for training
4. Understand the data shapes and preprocessing workflow
5. Verify preprocessing with statistical checks

## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `datasets.CIFAR10(root, train=True)` | `TFDSEagerSource(TFDSEagerConfig(name="cifar10", split="train"))` |
| `transforms.ToTensor()` | Automatic conversion to JAX arrays |
| `transforms.Normalize(mean, std)` | `ElementOperator` with custom normalization fn |
| `DataLoader(dataset, batch_size=32, shuffle=True)` | `from_source(source, batch_size=32)` with shuffle config |

**Key difference:** Datarax uses TFDS for dataset access and JAX arrays natively. Normalization constants are identical to PyTorch's standard values.

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tfds.load("cifar10", split="train")` | `TFDSEagerSource(TFDSEagerConfig(name="cifar10", split="train"))` |
| `dataset.batch(32).prefetch(2)` | `from_source(source, batch_size=32)` |
| `tf.keras.layers.Rescaling(1./255)` | `ElementOperator` with division by 255 |
| `tf.keras.layers.Normalization()` | `ElementOperator` with mean/std normalization |

**Key difference:** Datarax provides JAX arrays and integrates with Flax/NNX for training. The pipeline API is more functional.

## Files

- **Python Script**: [`examples/core/04_cifar10_quickref.py`](https://github.com/avitai/datarax/blob/main/examples/core/04_cifar10_quickref.py)
- **Jupyter Notebook**: [`examples/core/04_cifar10_quickref.ipynb`](https://github.com/avitai/datarax/blob/main/examples/core/04_cifar10_quickref.ipynb)

## Quick Start

```bash
# Install datarax with TFDS support
uv pip install "datarax[tfds]"

# Run the Python script
python examples/core/04_cifar10_quickref.py

# Or launch the Jupyter notebook
jupyter lab examples/core/04_cifar10_quickref.ipynb
```

**Note:** First run downloads CIFAR-10 (~170 MB).

## CIFAR-10 Preprocessing Constants

Standard normalization values for CIFAR-10, computed from the training set. Using these values ensures compatibility with pretrained models and published results.

| Statistic | R | G | B |
|-----------|---|---|---|
| **Mean** | 0.4914 | 0.4822 | 0.4465 |
| **Std** | 0.2470 | 0.2435 | 0.2616 |

```python
import jax.numpy as jnp

# CIFAR-10 normalization constants
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2470, 0.2435, 0.2616])

# Class names for reference
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

print("CIFAR-10 classes:", CIFAR10_CLASSES)
```

**Terminal Output:**
```
CIFAR-10 classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```

## Step 1: GPU Memory Configuration

Prevent TensorFlow from using GPU (reserved for JAX training):

```python
import os

# GPU Memory Configuration - prevent TensorFlow from using GPU
os.environ["CUDA_VISIBLE_DEVICES_FOR_TF"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
```

## Step 2: Create TFDS Data Source

Configure `TFDSEagerSource` to load CIFAR-10 training split. We use a subset for this quick reference to keep runtime short.

```python
from flax import nnx
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

# Load CIFAR-10 training data (subset for quick demo)
config = TFDSEagerConfig(
    name="cifar10",
    split="train[:1000]",  # First 1000 samples for demo
    shuffle=True,
    seed=42,
    exclude_keys={"id"},  # Exclude non-numeric fields
)

source = TFDSEagerSource(config, rngs=nnx.Rngs(42))

print(f"Dataset: CIFAR-10")
print(f"Samples: {len(source)}")
print(f"Classes: {len(CIFAR10_CLASSES)}")
```

**Terminal Output:**
```
Dataset: CIFAR-10
Samples: 1000
Classes: 10
```

## Step 3: Define Preprocessing

Standard CIFAR-10 preprocessing:
1. Convert uint8 [0, 255] to float32 [0, 1]
2. Apply channel-wise normalization with CIFAR-10 statistics

```python
from datarax.operators import ElementOperator, ElementOperatorConfig

def preprocess_cifar10(element, key=None):
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
```

**Terminal Output:**
```
Created CIFAR-10 normalizer with standard statistics
```

## Step 4: Build Pipeline

Chain source and preprocessing into a batched pipeline. Batch size of 32 is standard for CIFAR-10 training.

```mermaid
flowchart LR
    subgraph Source["TFDS Source"]
        T[TFDSEagerSource<br/>CIFAR-10<br/>1000 samples]
    end

    subgraph Pipeline["Pipeline"]
        FS[from_source<br/>batch_size=32]
        N[Normalizer<br/>(x - mean) / std]
    end

    subgraph Output["Output"]
        B[Batched Data<br/>(32, 32, 32, 3)]
    end

    T --> FS --> N --> B
```

```python
from datarax import from_source
from datarax.dag.nodes import OperatorNode

# Build the training pipeline
batch_size = 32
pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(normalizer))

print("Pipeline: TFDSEagerSource(CIFAR-10) -> Normalize -> Output")
print(f"Batch size: {batch_size}")
print(f"Batches per epoch: {len(source) // batch_size}")
```

**Terminal Output:**
```
Pipeline: TFDSEagerSource(CIFAR-10) -> Normalize -> Output
Batch size: 32
Batches per epoch: 31
```

## Step 5: Iterate Through Data

Process batches and verify the preprocessing is correct. Normalized data should have approximately zero mean and unit variance.

```python
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
        print(f"  Per-channel mean: [{batch_mean[0]:.3f}, {batch_mean[1]:.3f}, {batch_mean[2]:.3f}]")
```

**Terminal Output:**
```
Processing batches:
Batch 0:
  Image: shape=(32, 32, 32, 3), dtype=float32
  Labels: [6 9 9 4 1 1 2 7]... (first 8)
  Per-channel mean: [-0.012, 0.034, -0.089]
Batch 1:
  Image: shape=(32, 32, 32, 3), dtype=float32
  Labels: [3 5 8 7 0 4 5 3]... (first 8)
  Per-channel mean: [0.045, -0.021, 0.012]
Batch 2:
  Image: shape=(32, 32, 32, 3), dtype=float32
  Labels: [2 1 6 8 9 0 4 2]... (first 8)
  Per-channel mean: [-0.089, 0.015, -0.034]
```

Aggregate statistics across batches:

```python
mean_of_means = jnp.stack(all_means).mean(axis=0)
mean_of_stds = jnp.stack(all_stds).mean(axis=0)

print("\nAggregate Statistics (should be ~0 mean, ~1 std):")
print(f"  Mean across batches: [{mean_of_means[0]:.3f}, {mean_of_means[1]:.3f}, {mean_of_means[2]:.3f}]")
print(f"  Std across batches:  [{mean_of_stds[0]:.3f}, {mean_of_stds[1]:.3f}, {mean_of_stds[2]:.3f}]")
```

**Terminal Output:**
```
Aggregate Statistics (should be ~0 mean, ~1 std):
  Mean across batches: [-0.015, 0.009, -0.037]
  Std across batches:  [0.987, 1.012, 0.995]
```

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

1. **Faster convergence**: Normalized inputs improve gradient flow during training
2. **Compatibility**: Matches pretrained model expectations (e.g., ResNet, VGG)
3. **Numerical stability**: Prevents overflow/underflow in deep networks
4. **Consistent scale**: All channels have similar variance, preventing bias

## Next Steps

- [Augmentation Quick Reference](augmentation-quickref.md) - Add image operators for training
- [Operators Tutorial](operators-tutorial.md) - Deep dive into custom operators
- [MixUp/CutMix Tutorial](../advanced/augmentation/mixup-cutmix-tutorial.md) - Advanced batch augmentation
- [Full Training Example](mnist-tutorial.md) - Complete training workflow
- [API Reference: TFDSEagerSource](../../sources/tfds_source.md) - Complete TFDS API documentation
