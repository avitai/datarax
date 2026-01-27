# TensorFlow Datasets Source

`TFDSEagerSource` provides integration with [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets), giving access to hundreds of ready-to-use datasets with automatic conversion from TensorFlow tensors to JAX arrays.

> **Note:** You can also use the factory function `from_tfds(name, split, ...)` which auto-selects between eager and streaming modes based on your configuration.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | TensorFlow tensors → JAX arrays |
| **Built-in prefetching** | Uses `tf.data.AUTOTUNE` for performance |
| **Supervised mode** | Optional `(image, label)` tuple unpacking |
| **Shuffling** | TensorFlow-native shuffle with buffer |
| **Caching** | Optional dataset caching for repeated epochs |

`★ Insight ─────────────────────────────────────`

- TFDS handles download and preparation automatically
- Use `as_supervised=True` to get `{"image": ..., "label": ...}` format
- TensorFlow's prefetching is applied automatically for performance
- The source tracks epoch and index for stateful training loops

`─────────────────────────────────────────────────`

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

Enable shuffling with configurable buffer:

```python
config = TFDSEagerConfig(
    name="imagenet2012",
    split="train",
    shuffle=True,
    shuffle_buffer_size=10000,
    seed=42,
)
source = TFDSEagerSource(config, rngs=nnx.Rngs(42))
```

## Caching

Cache the dataset for faster repeated epochs:

```python
config = TFDSEagerConfig(
    name="mnist",
    split="train",
    cacheable=True,  # Cache in memory after first epoch
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
from datarax.dag import from_source
from datarax.dag.nodes import OperatorNode

config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    as_supervised=True,
)
source = TFDSEagerSource(config)

pipeline = (
    from_source(source, batch_size=128)
    >> OperatorNode(normalize_op)
    >> OperatorNode(augment_op)
)

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

- [Data Sources Guide](../user_guide/data_sources.md) - Comprehensive data loading guide
- [HF Source](hf_source.md) - HuggingFace Datasets integration
- [TFDS Quick Reference](../examples/integration/tfds/tfds-quickref.md)
- [TFDS Catalog](https://www.tensorflow.org/datasets/catalog/overview)

---

## API Reference

::: datarax.sources.tfds_source
