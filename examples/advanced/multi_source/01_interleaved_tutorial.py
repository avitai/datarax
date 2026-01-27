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
# Multi-Source Data Loading Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~20 min |
| **Prerequisites** | Pipeline Tutorial, TFDS Quick Reference |
| **Format** | Python + Jupyter |
| **Memory** | ~1.5 GB RAM |

## Overview

Learn to load and combine data from multiple sources in a single pipeline.
This is essential for multi-task learning, domain adaptation, and creating
diverse training sets from heterogeneous data.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Create multiple TFDSEagerSource instances
2. Interleave data from different datasets
3. Apply source-specific preprocessing
4. Handle different data formats in the same pipeline
5. Visualize mixed dataset samples
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
from datarax.sources import TFDSEagerConfig, TFDSEagerSource

print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Use Case: Multi-Domain Digit Recognition

We'll combine MNIST and Fashion-MNIST to create a unified digit/fashion
classification dataset. This simulates scenarios like:

- Domain adaptation (source → target domain)
- Multi-task learning
- Creating diverse training sets
"""

# %% [markdown]
"""
## Part 1: Create Individual Sources

Each source has its own configuration, preprocessing, and sampling.
"""

# %%
# MNIST Source - 10 digit classes (0-9)
mnist_config = TFDSEagerConfig(
    name="mnist",
    split="train[:2000]",  # Subset for demo
    shuffle=True,
    seed=42,
)

mnist_source = TFDSEagerSource(mnist_config, rngs=nnx.Rngs(42))

# Fashion-MNIST Source - 10 fashion classes
fashion_config = TFDSEagerConfig(
    name="fashion_mnist",
    split="train[:2000]",
    shuffle=True,
    seed=43,
)

fashion_source = TFDSEagerSource(fashion_config, rngs=nnx.Rngs(43))

print(f"MNIST samples: {len(mnist_source)}")
print(f"Fashion-MNIST samples: {len(fashion_source)}")

# %% [markdown]
"""
## Dataset Properties

| Dataset | Classes | Image Size | Task |
|---------|---------|------------|------|
| MNIST | 10 digits (0-9) | 28×28×1 | Digit recognition |
| Fashion-MNIST | 10 items | 28×28×1 | Fashion classification |
"""

# %%
# Class labels
MNIST_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
FASHION_CLASSES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Boot",
]

# Normalization constants
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
FASHION_MEAN, FASHION_STD = 0.2860, 0.3530

BATCH_SIZE = 32

# %% [markdown]
"""
## Part 2: Source-Specific Preprocessing

Each source may need different preprocessing. We'll add a "source" tag
to track where each sample came from.
"""


# %%
def preprocess_mnist(element, key=None):  # noqa: ARG001
    """Preprocess MNIST with source tag."""
    del key
    image = element.data["image"]

    # Normalize
    image = image.astype(jnp.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    image = (image - MNIST_MEAN) / MNIST_STD

    # Add source indicator (0 = MNIST, 1 = Fashion)
    return element.update_data(
        {
            "image": image,
            "label": element.data["label"],
            "source": jnp.array(0, dtype=jnp.int32),  # 0 = MNIST
        }
    )


def preprocess_fashion(element, key=None):  # noqa: ARG001
    """Preprocess Fashion-MNIST with source tag."""
    del key
    image = element.data["image"]

    # Normalize
    image = image.astype(jnp.float32) / 255.0
    if image.ndim == 2:
        image = image[..., None]
    image = (image - FASHION_MEAN) / FASHION_STD

    # Add source indicator (shift labels by 10 for unified label space)
    return element.update_data(
        {
            "image": image,
            "label": element.data["label"] + 10,  # Offset for unified labels
            "original_label": element.data["label"],
            "source": jnp.array(1, dtype=jnp.int32),  # 1 = Fashion
        }
    )


mnist_preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_mnist,
    rngs=nnx.Rngs(0),
)

fashion_preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_fashion,
    rngs=nnx.Rngs(0),
)

print("Created source-specific preprocessors:")
print("  MNIST: labels 0-9, source=0")
print("  Fashion: labels 10-19, source=1")

# %% [markdown]
"""
## Part 3: Create Individual Pipelines

First, let's verify each pipeline works independently.
"""

# %%
# MNIST pipeline
mnist_pipeline = from_source(mnist_source, batch_size=BATCH_SIZE).add(
    OperatorNode(mnist_preprocessor)
)

# Fashion pipeline (need fresh source)
fashion_source2 = TFDSEagerSource(fashion_config, rngs=nnx.Rngs(43))
fashion_pipeline = from_source(fashion_source2, batch_size=BATCH_SIZE).add(
    OperatorNode(fashion_preprocessor)
)

# Test individual pipelines
mnist_batch = next(iter(mnist_pipeline))
print("MNIST batch:")
print(f"  Image shape: {mnist_batch['image'].shape}")
print(f"  Labels: {mnist_batch['label'][:8]}")
print(f"  Source: {mnist_batch['source'][:8]}")

# %% [markdown]
"""
## Part 4: Interleaving Strategy

We'll create a simple round-robin interleaving strategy that alternates
between sources. More sophisticated strategies could use weighted sampling.
"""


# %%
class InterleavedIterator:
    """Round-robin iterator that alternates between multiple sources."""

    def __init__(self, pipelines: list, weights: list[float] | None = None):
        """Initialize with list of pipelines and optional weights.

        Args:
            pipelines: List of Datarax pipelines to interleave
            weights: Optional sampling weights (default: equal)
        """
        self.pipelines = pipelines
        self.iterators = [iter(p) for p in pipelines]
        self.weights = weights or [1.0 / len(pipelines)] * len(pipelines)
        self.current_idx = 0
        self.exhausted = [False] * len(pipelines)

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch, cycling through sources."""
        if all(self.exhausted):
            raise StopIteration

        # Find next non-exhausted source
        attempts = 0
        while self.exhausted[self.current_idx] and attempts < len(self.pipelines):
            self.current_idx = (self.current_idx + 1) % len(self.pipelines)
            attempts += 1

        if attempts >= len(self.pipelines):
            raise StopIteration

        try:
            batch = next(self.iterators[self.current_idx])
            self.current_idx = (self.current_idx + 1) % len(self.pipelines)
            return batch
        except StopIteration:
            self.exhausted[self.current_idx] = True
            return self.__next__()


print("Created InterleavedIterator for round-robin sampling")


# %%
# Create interleaved pipeline
def create_interleaved_pipelines():
    """Create fresh pipelines for interleaving."""
    mnist_src = TFDSEagerSource(
        TFDSEagerConfig(
            name="mnist",
            split="train[:500]",
            shuffle=True,
            seed=42,
        ),
        rngs=nnx.Rngs(42),
    )

    fashion_src = TFDSEagerSource(
        TFDSEagerConfig(
            name="fashion_mnist",
            split="train[:500]",
            shuffle=True,
            seed=43,
        ),
        rngs=nnx.Rngs(43),
    )

    mnist_prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_mnist,
        rngs=nnx.Rngs(0),
    )

    fashion_prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_fashion,
        rngs=nnx.Rngs(0),
    )

    mnist_pipe = from_source(mnist_src, batch_size=BATCH_SIZE).add(OperatorNode(mnist_prep))
    fashion_pipe = from_source(fashion_src, batch_size=BATCH_SIZE).add(OperatorNode(fashion_prep))

    return [mnist_pipe, fashion_pipe]


# Test interleaved iteration
interleaved = InterleavedIterator(create_interleaved_pipelines())

sources_seen = []
for i, batch in enumerate(interleaved):
    if i >= 10:  # Sample first 10 batches
        break
    sources_seen.append(int(batch["source"][0]))

print(f"Sources in first 10 batches: {sources_seen}")
print("(0=MNIST, 1=Fashion)")

# %% [markdown]
"""
## Part 5: Visualize Mixed Dataset
"""

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

# Collect samples from both sources
interleaved = InterleavedIterator(create_interleaved_pipelines())
all_images = []
all_labels = []
all_sources = []
all_original_labels = []

for batch in interleaved:
    all_images.append(np.array(batch["image"]))
    all_labels.append(np.array(batch["label"]))
    all_sources.append(np.array(batch["source"]))
    # Use original_label if present, otherwise use label
    if "original_label" in batch:
        all_original_labels.append(np.array(batch["original_label"]))
    else:
        all_original_labels.append(np.array(batch["label"]))

all_images = np.concatenate(all_images)
all_labels = np.concatenate(all_labels)
all_sources = np.concatenate(all_sources)
all_original_labels = np.concatenate(all_original_labels)

print(f"Total samples collected: {len(all_images)}")
print(f"MNIST samples: {(all_sources == 0).sum()}")
print(f"Fashion samples: {(all_sources == 1).sum()}")

# %%
# Plot mixed dataset samples
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Multi-Source Dataset: MNIST + Fashion-MNIST", fontsize=14)

# Interleave MNIST and Fashion samples
mnist_indices = np.where(all_sources == 0)[0][:16]
fashion_indices = np.where(all_sources == 1)[0][:16]

for i in range(8):
    # MNIST row 1
    idx = mnist_indices[i]
    img = all_images[idx] * MNIST_STD + MNIST_MEAN
    axes[0, i].imshow(img.squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title(f"MNIST: {MNIST_CLASSES[all_original_labels[idx] % 10]}", fontsize=8)

    # Fashion row 2
    idx = fashion_indices[i]
    img = all_images[idx] * FASHION_STD + FASHION_MEAN
    axes[1, i].imshow(img.squeeze(), cmap="gray")
    axes[1, i].axis("off")
    label_idx = all_original_labels[idx] % 10
    axes[1, i].set_title(f"Fashion: {FASHION_CLASSES[label_idx]}", fontsize=8)

    # MNIST row 3
    if i + 8 < len(mnist_indices):
        idx = mnist_indices[i + 8]
        img = all_images[idx] * MNIST_STD + MNIST_MEAN
        axes[2, i].imshow(img.squeeze(), cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title(f"MNIST: {MNIST_CLASSES[all_original_labels[idx] % 10]}", fontsize=8)
    else:
        axes[2, i].axis("off")

    # Fashion row 4
    if i + 8 < len(fashion_indices):
        idx = fashion_indices[i + 8]
        img = all_images[idx] * FASHION_STD + FASHION_MEAN
        axes[3, i].imshow(img.squeeze(), cmap="gray")
        axes[3, i].axis("off")
        label_idx = all_original_labels[idx] % 10
        axes[3, i].set_title(f"Fashion: {FASHION_CLASSES[label_idx]}", fontsize=8)
    else:
        axes[3, i].axis("off")

plt.tight_layout()
plt.savefig(
    output_dir / "cv-multisource-samples.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-multisource-samples.png'}")

# %% [markdown]
"""
## Part 6: Source Distribution Analysis
"""

# %%
# Plot source distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Source distribution
source_counts = [np.sum(all_sources == 0), np.sum(all_sources == 1)]
axes[0].bar(["MNIST", "Fashion-MNIST"], source_counts, color=["steelblue", "coral"])
axes[0].set_ylabel("Sample Count")
axes[0].set_title("Source Distribution")
for i, count in enumerate(source_counts):
    axes[0].text(i, count + 10, str(count), ha="center")

# Label distribution (unified label space)
label_counts = np.bincount(all_labels, minlength=20)
x = np.arange(20)
colors = ["steelblue"] * 10 + ["coral"] * 10
axes[1].bar(x, label_counts, color=colors)
axes[1].set_xlabel("Unified Label")
axes[1].set_ylabel("Count")
axes[1].set_title("Unified Label Distribution (0-9: MNIST, 10-19: Fashion)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(x, fontsize=8)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-multisource-distribution.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-multisource-distribution.png'}")

# %% [markdown]
"""
## Part 7: Throughput Comparison
"""

# %%
import time

# Benchmark single source
mnist_src = TFDSEagerSource(
    TFDSEagerConfig(name="mnist", split="train[:1000]", shuffle=False),
    rngs=nnx.Rngs(0),
)
mnist_prep = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_mnist,
    rngs=nnx.Rngs(0),
)
single_pipeline = from_source(mnist_src, batch_size=64).add(OperatorNode(mnist_prep))

start = time.time()
single_count = 0
for batch in single_pipeline:
    _ = batch["image"].block_until_ready()
    single_count += batch["image"].shape[0]
single_time = time.time() - start
single_throughput = single_count / single_time

print(f"Single source throughput: {single_throughput:.0f} samples/s")

# Benchmark interleaved
interleaved = InterleavedIterator(create_interleaved_pipelines())

start = time.time()
multi_count = 0
for batch in interleaved:
    _ = batch["image"].block_until_ready()
    multi_count += batch["image"].shape[0]
multi_time = time.time() - start
multi_throughput = multi_count / multi_time

print(f"Multi-source throughput: {multi_throughput:.0f} samples/s")

# %%
# Plot throughput comparison
fig, ax = plt.subplots(figsize=(8, 5))

throughputs = [single_throughput, multi_throughput]
labels = ["Single Source\n(MNIST)", "Interleaved\n(MNIST + Fashion)"]
colors = ["steelblue", "coral"]

bars = ax.bar(labels, throughputs, color=colors)
ax.set_ylabel("Throughput (samples/second)")
ax.set_title("Data Loading Throughput Comparison")

for bar, val in zip(bars, throughputs):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 50, f"{val:.0f}", ha="center")

plt.tight_layout()
plt.savefig(
    output_dir / "cv-multisource-throughput.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-multisource-throughput.png'}")

# %% [markdown]
"""
## Results Summary

### Multi-Source Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Round-robin | Alternate sources equally | Balanced datasets |
| Weighted | Sample by weights | Imbalanced sources |
| Priority | Primary source + supplement | Domain adaptation |
| Batch-level | Full batches from each | Multi-task learning |

### Key Takeaways

1. **Source tagging**: Add metadata to track data provenance
2. **Label space**: Map to unified label space for multi-class
3. **Fresh pipelines**: Create new pipeline instances for each epoch
4. **Preprocessing**: Apply source-specific normalization
5. **Throughput**: Multi-source adds minimal overhead

### Unified Label Space

```
Labels 0-9:   MNIST digits
Labels 10-19: Fashion-MNIST items

Total classes: 20
```
"""

# %% [markdown]
"""
## Next Steps

- **Distributed**: [Sharding guide](../distributed/02_sharding_guide.ipynb)
- **Checkpointing**: [Resumable training](../checkpointing/02_resumable_training_guide.ipynb)
- **Full training**: [End-to-end CIFAR-10](../training/01_e2e_cifar10_guide.ipynb)
"""


# %%
def main():
    """Run the multi-source tutorial."""
    print("Multi-Source Data Loading Tutorial")
    print("=" * 50)

    # Create and iterate interleaved pipeline
    interleaved = InterleavedIterator(create_interleaved_pipelines())

    total_samples = 0
    mnist_samples = 0
    fashion_samples = 0

    for batch in interleaved:
        batch_size = batch["image"].shape[0]
        total_samples += batch_size
        source = int(batch["source"][0])
        if source == 0:
            mnist_samples += batch_size
        else:
            fashion_samples += batch_size

    print(f"Total samples processed: {total_samples}")
    print(f"  MNIST: {mnist_samples}")
    print(f"  Fashion: {fashion_samples}")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
