# TensorFlow Datasets Source

`TFDSEagerSource` provides integration with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets), giving access to hundreds of ready-to-use datasets with automatic conversion from TensorFlow tensors to JAX arrays.

> **Note:** You can also use the factory function `from_tfds(name, split, ...)` which auto-selects between eager and streaming modes based on your configuration.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | TensorFlow tensors → JAX arrays |
| **One-time load** | Eager source converts TF→JAX at init, then tears down TensorFlow |
| **Supervised mode** | Optional `(image, label)` tuple unpacking |
| **Shuffling** | O(1)-memory Feistel index shuffle (same split as HF source) |
| **Fixed prefetch** | Streaming source uses a fixed `prefetch_buffer=2`, deliberately not AUTOTUNE |

!!! note "Key points"

    - TFDS handles download and preparation automatically
    - Use `as_supervised=True` to get `{"image": ..., "label": ...}` format
    - The eager source performs a one-time TF→JAX conversion at init and tears TensorFlow down afterward; iteration is then pure JAX
    - The streaming source uses a fixed `prefetch_buffer=2` (deliberately not `tf.data.AUTOTUNE`) to avoid thread storms
    - The source tracks epoch and index for stateful training loops

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
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig

# Load MNIST dataset
config = TFDSEagerConfig(name="mnist", split="train")
source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

# Iterate over elements
for item in source:
    image = item["image"]  # JAX array, shape (28, 28, 1)
    label = item["label"]  # JAX array, scalar
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

batch = source.get_batch(32)
images = batch["image"]  # Shape: (32, 32, 32, 3)
labels = batch["label"]  # Shape: (32,)
```

## Batch Retrieval

For training loops with automatic epoch cycling:

```python
# Stateful batch retrieval
for step in range(10000):
    batch = source.get_batch(64)
    loss = train_step(batch)

    # Check progress
    print(f"Epoch {source.epoch.get_value()}, "
          f"Index {source.index.get_value()}")
```

## Shuffling

The eager source uses the same O(1)-memory Feistel index shuffle as the HF source
(no shuffle buffer):

```python
config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    shuffle=True,
    seed=42,  # Integer seed for Grain's index shuffle
)
source = TFDSEagerSource(config, rngs=nnx.Rngs(42))
```

For ImageNet-scale splits that do not fit in memory, use the streaming path via
`from_tfds(name, split, ...)` (or `TFDSStreamingConfig` directly), which streams
with a fixed prefetch buffer instead of loading everything at init.

## Apply Caching

The `cacheable` flag is the module-level apply cache (memoizing the result of the
processing `apply`), not a dataset cache — the eager source already holds all data
in memory as JAX arrays, so no dataset-level caching is needed:

```python
config = TFDSEagerConfig(
    name="mnist",
    split="train",
    cacheable=True,  # Enables the module-level apply cache
)
source = TFDSEagerSource(config)
```

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
