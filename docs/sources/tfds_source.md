# TensorFlow Datasets Source

`TFDSEagerSource` provides integration with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets), giving access to hundreds of ready-to-use datasets with automatic conversion from TensorFlow tensors to JAX arrays.

> **Note:** You can also use the factory function `from_tfds(name, split, ...)` which auto-selects between eager and streaming modes based on your configuration.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | TensorFlow tensors → host NumPy columns |
| **One-time load** | Eager source converts at init, then tears down TensorFlow |
| **Supervised mode** | Optional `(image, label)` tuple unpacking |
| **Shuffling** | `Pipeline(shuffle=True)`'s O(1)-memory Feistel index shuffle (as for the HF source) |
| **Fixed prefetch** | Streaming source uses a fixed `prefetch_buffer=2`, deliberately not AUTOTUNE |

!!! note "Key points"

    - TFDS handles download and preparation automatically
    - Use `as_supervised=True` to get `{"image": ..., "label": ...}` format
    - The eager source performs a one-time conversion at init, holds the records as host NumPy columns and tears TensorFlow down afterward
    - The streaming source uses a fixed `prefetch_buffer=2` (deliberately not `tf.data.AUTOTUNE`) to avoid thread storms
    - The source keeps no iteration state: the pipeline owns the order and the position

## Installation

TFDSEagerSource requires TensorFlow and tensorflow-datasets:

```bash
pip install datarax[data]
# or
pip install tensorflow tensorflow-datasets
```

## Quick Start

```python
import flax.nnx as nnx
import numpy as np
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig

# Load MNIST dataset
config = TFDSEagerConfig(name="mnist", split="train")
source = TFDSEagerSource(config)

# Iterate over elements
for item in source:
    image = item["image"]  # NumPy array, shape (28, 28, 1)
    label = item["label"]  # NumPy scalar
    process(image, label)
```

## Supervised Mode

Get a cleaner `{"image", "label"}` structure:

```python
config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    as_supervised=True,  # Returns {"image": ..., "label": ...}
)
source = TFDSEagerSource(config)

batch = source.get_batch(to_words(np.arange(32)))  # records 0..31
images = batch["image"]  # Shape: (32, 32, 32, 3)
labels = batch["label"]  # Shape: (32,)
```

## Batch Retrieval

A training loop takes its batches from a `Pipeline`, which owns the order, the position and
the epoch:

```python
pipeline = Pipeline(source=source, stages=[], batch_size=64, rngs=nnx.Rngs(0), shuffle=True)
for batch in pipeline:
    loss = train_step(batch)
```

## Shuffling

The pipeline owns an eager source's order: `Pipeline(shuffle=True)` serves each epoch in a
new order with the same O(1)-memory Feistel index shuffle as for the HF source (no shuffle
buffer), seeded by the pipeline's `rngs`:

```python
source = TFDSEagerSource(TFDSEagerConfig(name="cifar10", split="train"))
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
```

For ImageNet-scale splits that do not fit in memory, use the streaming path via
`from_tfds(name, split, ...)` (or `TFDSStreamingConfig` directly), which streams
with a fixed prefetch buffer instead of loading everything at init.

## Custom Data Directory

Store datasets in a specific location:

```python
config = TFDSEagerConfig(
    name="imagenet2012",
    split="train",
    data_dir="/path/to/tfds_data",
)
source = TFDSEagerSource(config)
```

## Field Filtering

Select only needed fields:

```python
config = TFDSEagerConfig(
    name="coco/2017",
    split="train",
    include_keys={"image", "objects"},  # Only these fields
)

# Or exclude unwanted fields
config = TFDSEagerConfig(
    name="mnist",
    split="train",
    exclude_keys={"id"},
)
```

## Integration with DAG Pipelines

```python
from datarax.pipeline import Pipeline

config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    as_supervised=True,
)
source = TFDSEagerSource(config)

pipeline = (
    Pipeline(source=source, stages=[normalize_op, augment_op], batch_size=128, rngs=nnx.Rngs(0)))

for batch in pipeline:
    train_step(batch)
```

## Dataset Information

Access rich metadata from TFDS:

```python
info = source.get_dataset_info()
print(f"Description: {info.description}")
print(f"Features: {info.features}")
print(f"Splits: {list(info.splits.keys())}")
print(f"Citation: {info.citation}")

# Number of examples
print(f"Train examples: {info.splits['train'].num_examples}")
```

## See Also

- [Data Sources Guide](../user_guide/data_sources.md) - Complete data loading guide
- [HF Source](hf_source.md) - HuggingFace Datasets integration
- [TFDS Quick Reference](../examples/integration/tfds/tfds-quickref.md)
- [TFDS Catalog](https://www.tensorflow.org/datasets/catalog/overview)

---

## API Reference

::: datarax.sources.tfds_source
