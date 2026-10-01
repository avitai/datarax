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
# HuggingFace Datasets Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 min |
| **Prerequisites** | HuggingFace Quick Reference, Pipeline Tutorial |
| **Format** | Python + Jupyter |

## Overview

This tutorial provides a detailed guide to using HuggingFace Datasets with Datarax.
You'll learn to work with different data modalities, configure advanced options, and
build production-ready training pipelines.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Load different dataset types (images, text, audio)
2. Configure field filtering with include/exclude keys
3. Set up reproducible shuffling
4. Build complete training pipelines with augmentation
5. Handle streaming vs downloaded modes effectively
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with data dependencies
uv pip install "datarax[data]"
```

**Note**: Some datasets may require additional dependencies.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
from flax import nnx

from datarax.operators import (
    ElementOperator,
    ElementOperatorConfig,
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image import FlipOperator, FlipOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import HFEagerConfig, HFEagerSource, HFStreamingConfig, HFStreamingSource


print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")

# %% [markdown]
"""
## Part 1: Understanding HFEagerSource Configuration

`HFEagerConfig` loads a dataset into memory; `HFStreamingConfig` reads it on the fly.

### Key Configuration Options

| Parameter | Description | Default | Config |
|-----------|-------------|---------|--------|
| `name` | Dataset identifier on HF Hub | Required | both |
| `split` | Which split to use | Required | both |
| `streaming` | Stream data on the fly | `False` | `HFStreamingConfig` |
| `shuffle` | Shuffle the stream within a buffer | `False` | `HFStreamingConfig` |
| `shuffle_buffer_size` | Buffer size for streaming shuffles | `1000` | `HFStreamingConfig` |
| `include_keys` | Only include these fields | `None` | both |
| `exclude_keys` | Exclude these fields | `None` | both |
"""

# %%
# Example: Basic configuration for MNIST
basic_config = HFEagerConfig(
    name="ylecun/mnist",
    split="train[:1000]",  # Load first 1000 samples
)

basic_source = HFEagerSource(basic_config)
print(f"Basic MNIST source: {len(basic_source)} samples")

# %% [markdown]
"""
## Part 2: Field Filtering

Use `include_keys` or `exclude_keys` to control which fields are returned.

This is useful for:

- Reducing memory usage
- Excluding metadata you don't need
- Simplifying downstream processing
"""

# %%
# Include only specific fields
filtered_config = HFEagerConfig(
    name="ylecun/mnist",
    split="train[:500]",
    include_keys={"image", "label"},  # Only return these fields
)

filtered_source = HFEagerSource(filtered_config)

# Check what fields are available
pipeline = Pipeline(source=filtered_source, stages=[], batch_size=1, rngs=nnx.Rngs(0))
batch = next(iter(pipeline))
data = batch.data

print("Filtered fields:")
for key in data.keys():
    print(f"  - {key}")

# %% [markdown]
"""
## Part 3: Shuffling Configuration

Shuffling is essential for training ML models:

- The pipeline owns the order of an eager source: `Pipeline(shuffle=True)` serves the whole
  downloaded split in a new order every epoch, reproducibly from the pipeline's `rngs`
- `HFStreamingSource` shuffles within a buffer of `shuffle_buffer_size` records
"""

# %%
# Shuffle the whole split every epoch, reproducibly from the pipeline's seed
shuffle_source = HFEagerSource(HFEagerConfig(name="ylecun/mnist", split="train[:2000]"))
shuffle_pipeline = Pipeline(
    source=shuffle_source, stages=[], batch_size=8, rngs=nnx.Rngs(42), shuffle=True
)

print("Shuffle configuration:")
print(f"  Pipeline shuffles: {shuffle_pipeline.shuffle}")

# %% [markdown]
"""
## Part 4: Streaming vs Downloaded Mode

### Streaming Mode (`streaming=True`)
- Data loaded on-the-fly from HuggingFace servers
- No disk storage required
- Ideal for large datasets
- Cannot seek to specific indices
- Dataset length may not be available

### Downloaded Mode (`streaming=False`)
- Full dataset downloaded and cached locally
- Random access to any sample
- Faster iteration after initial download
- Requires disk space
"""

# %%
# Compare streaming vs downloaded
print("Mode Comparison:")

# Streaming mode: records are read on the fly, so the length is unknown
streaming_config = HFStreamingConfig(
    name="ylecun/mnist",
    split="train",
    streaming=True,
)
streaming_source = HFStreamingSource(streaming_config, rngs=nnx.Rngs(0))

try:
    print(f"Streaming mode length: {len(streaming_source)}")
except NotImplementedError:
    print("Streaming mode length: unknown until the stream is read")
first_record = next(iter(streaming_source))
print(f"First streamed record: {sorted(first_record)}")

# Downloaded mode (using subset)
downloaded_config = HFEagerConfig(
    name="ylecun/mnist",
    split="train[:1000]",
)
downloaded_source = HFEagerSource(downloaded_config)
print(f"Downloaded mode length: {len(downloaded_source)}")

# %% [markdown]
"""
## Part 5: Building Complete Training Pipeline

Combine HFEagerSource with operators for a production-ready pipeline.

This example shows:

- Data loading from HuggingFace
- Normalization operator
- Data augmentation (random flip)
- Batched iteration
"""


# %%
# Define operators
def normalize_image(element, key=None):  # noqa: ARG001
    """Normalize image to [0, 1] and ensure proper shape."""
    del key  # Unused - deterministic
    image = element.data.get("image")
    if image is not None and hasattr(image, "dtype"):
        # Normalize to [0, 1]
        normalized = image.astype(jnp.float32) / 255.0
        # Add channel dimension if needed (for grayscale)
        if normalized.ndim == 2:
            normalized = normalized[..., None]
        return element.update_data({"image": normalized})
    return element


# Create operators
normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_image,
    rngs=nnx.Rngs(0),
)

# Random horizontal flip: FlipOperator, applied to each record with probability 0.5
flipper = ProbabilisticOperator(
    ProbabilisticOperatorConfig(probability=0.5),
    operator=FlipOperator(FlipOperatorConfig(field_key="image")),
    rngs=nnx.Rngs(augment=42),
)

# Create composite augmentation
augmentation = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        stochastic=True,
        stream_name="augment",
    ),
    operators=[normalizer, flipper],
)

print("Created operators: normalizer, flipper, augmentation")

# %%
# Build the complete pipeline
train_config = HFEagerConfig(
    name="ylecun/mnist",
    split="train[:5000]",
    include_keys={"image", "label"},
)

train_source = HFEagerSource(train_config)

# Chain: Source -> Augmentation -> Output
training_pipeline = Pipeline(
    source=train_source, stages=[augmentation], batch_size=64, rngs=nnx.Rngs(0), shuffle=True
)

print("Training pipeline:")
print("  HFEagerSource(mnist) -> Normalize -> RandomFlip -> Output")
print("  Batch size: 64")

# %%
# Process training data
print("\nProcessing training batches:")
stats = {"batches": 0, "samples": 0}

for i, batch in enumerate(training_pipeline):
    if i >= 5:  # Process 5 batches for demo
        break

    image_batch = batch["image"]
    label_batch = batch["label"]

    stats["batches"] += 1
    stats["samples"] += image_batch.shape[0]

    if i == 0:  # Print details for first batch
        print(f"Batch {i}:")
        print(f"  Image: shape={image_batch.shape}, dtype={image_batch.dtype}")
        img_min, img_max = float(image_batch.min()), float(image_batch.max())
        print(f"  Image range: [{img_min:.3f}, {img_max:.3f}]")
        print(f"  Label: shape={label_batch.shape}")

print(f"\nProcessed {stats['batches']} batches, {stats['samples']} samples")

# %% [markdown]
"""
## Part 6: Working with Different Datasets

HuggingFace Hub hosts thousands of datasets across different modalities.

### Common Dataset Examples

| Dataset | Type | Example Config |
|---------|------|----------------|
| `mnist` | Image | `split="train"` |
| `cifar10` | Image | `split="train"` |
| `imdb` | Text | `split="train"` |
| `squad` | QA | `split="train"` |
| `librispeech_asr` | Audio | `split="train.clean.100"` |

### Dataset Discovery

```python
# List available datasets
from huggingface_hub import list_datasets
datasets = list(list_datasets(limit=100))

# Get dataset info
from datasets import load_dataset_builder
builder = load_dataset_builder("mnist")
print(builder.info)
```
"""

# %%
# Example: Different split syntax
print("Split syntax examples:")
print("  'train' - Full training set")
print("  'train[:1000]' - First 1000 samples")
print("  'train[1000:2000]' - Samples 1000-2000")
print("  'train[:10%]' - First 10% of data")
print("  'train+test' - Combined splits")

# %% [markdown]
"""
## Results Summary

| Feature | Configuration |
|---------|--------------|
| Field Filtering | `include_keys` / `exclude_keys` |
| Shuffling | `Pipeline(shuffle=True)` (streaming: `shuffle=True` with `shuffle_buffer_size`) |
| Streaming | `HFStreamingSource` with `streaming=True` for large datasets |
| Reproducibility | Named RNG streams |
| Pipeline | Source -> Operators -> Output |

### Best Practices

1. **Large datasets**: Use `HFStreamingSource` with `streaming=True` to avoid memory issues
2. **Training**: Always enable shuffling
3. **Reproducibility**: Use named RNG streams (`nnx.Rngs(name=seed)`)
4. **Memory**: Use `include_keys` to filter unnecessary fields
5. **Development**: Use split syntax like `train[:1000]` for quick iteration
"""

# %% [markdown]
"""
## Next Steps

- **Image augmentation**: See [Operators Tutorial](../../core/03_operators_tutorial.ipynb)
- **TFDS alternative**: [TFDS Integration](../tfds/01_tfds_quickref.ipynb)
- **Distributed training**: [Sharding Guide](../../advanced/distributed/01_sharding_quickref.ipynb)
- **HuggingFace Hub**: Browse datasets at https://huggingface.co/datasets
"""


# %%
def main():
    """Run the HuggingFace tutorial."""
    print("HuggingFace Datasets Tutorial")
    print("=" * 50)

    # Create pipeline
    config = HFEagerConfig(
        name="ylecun/mnist",
        split="train[:2000]",
        include_keys={"image", "label"},
    )
    source = HFEagerSource(config)

    # Normalizer
    def normalize(element, key=None):  # noqa: ARG001
        del key
        image = element.data.get("image")
        if image is not None:
            normalized = image.astype(jnp.float32) / 255.0
            if normalized.ndim == 2:
                normalized = normalized[..., None]
            return element.update_data({"image": normalized})
        return element

    normalizer = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize,
        rngs=nnx.Rngs(0),
    )

    pipeline = Pipeline(
        source=source, stages=[normalizer], batch_size=64, rngs=nnx.Rngs(0), shuffle=True
    )

    # Process
    total = 0
    for batch in pipeline:
        total += batch["image"].shape[0]
        # Verify normalization
        assert batch["image"].min() >= 0.0
        assert batch["image"].max() <= 1.0

    print(f"Processed {total} samples")
    print("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
