# TensorFlow Datasets Source

`TFDSEagerSource` reads [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) that TFDS has prepared as ArrayRecord, giving access to hundreds of ready-to-use datasets as host NumPy columns, without TensorFlow in the process.

> **Note:** You can also use the factory function `from_tfds(name, split, ...)` which auto-selects between eager and streaming modes based on your configuration.

## Key Features

| Feature | Description |
|---------|-------------|
| **No TensorFlow** | Reads through TFDS's random-access reader (`as_data_source`); TensorFlow is never imported |
| **One batched read** | The whole split is read and decoded at init, then held as host NumPy columns |
| **Provenance** | Text features (CIFAR-10's `id`) are kept per record as provenance, never served in a batch |
| **Supervised mode** | `as_supervised=True` keeps only the dataset's supervised features, under their own names |
| **Shuffling** | `Pipeline(shuffle=True)`'s O(1)-memory Feistel index shuffle (as for the HF source) |
| **Fixed prefetch** | Streaming source uses a fixed `prefetch_buffer=2`, deliberately not AUTOTUNE |

!!! note "Key points"

    - The eager source reads a copy prepared as **ArrayRecord** and never prepares one: preparing imports TensorFlow, so it runs once, in a process of its own
    - A split that is not prepared, or is prepared only as TFRecord, is refused with a `FileNotFoundError` naming the call that prepares it
    - TensorFlow in a JAX process breaks JAX's multi-GPU (NCCL) collectives; the eager source keeps it out of the training process
    - Values keep the dtype TFDS stores: a class label is int64 on the host, and int32 on a device while JAX's 64-bit types are off
    - The streaming source reads a TFRecord copy through `tf.data`, so it imports TensorFlow; it uses a fixed `prefetch_buffer=2` (deliberately not `tf.data.AUTOTUNE`) to avoid thread storms
    - The source keeps no iteration state: the pipeline owns the order and the position

## Installation

Reading needs the `data` extra (tensorflow-datasets and Pillow); preparing a dataset and the streaming source need the `tfds` extra, which adds TensorFlow:

```bash
pip install "datarax[data]"          # read prepared datasets with TFDSEagerSource
pip install "datarax[data,tfds]"     # also prepare datasets, or stream them
```

## Preparing a Dataset

Prepare each dataset once as ArrayRecord, in a process of its own (it imports TensorFlow):

```bash
python -c "import tensorflow_datasets as tfds; tfds.builder('mnist', file_format='array_record').download_and_prepare()"
```

`data_dir=` chooses where it is prepared (TFDS's default is `TFDS_DATA_DIR`, else `~/tensorflow_datasets`). A data directory holds one format per dataset version, so a dataset already prepared there as TFRecord is prepared as ArrayRecord in another directory, or after its TFRecord copy is deleted. In the datarax repository, `scripts/prepare_example_datasets.py` prepares the datasets the examples read (CIFAR-10, Fashion-MNIST, MNIST) and replaces a TFRecord copy of them in place.

## Quick Start

```python
import flax.nnx as nnx
import numpy as np
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources import TFDSEagerSource
from datarax.sources.tfds_source import TFDSEagerConfig

# Read MNIST, prepared as ArrayRecord
config = TFDSEagerConfig(name="mnist", split="train")
source = TFDSEagerSource(config)

# Iterate over elements
for item in source:
    image = item["image"]  # NumPy array, shape (28, 28, 1)
    label = item["label"]  # NumPy scalar
    process(image, label)
```

## Supervised Mode

Keep only the dataset's supervised features (`info.supervised_keys`), under their own names:

```python
config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    as_supervised=True,  # keeps {"image": ..., "label": ...}, drops "id"
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
`from_tfds(name, split, ...)` (or `TFDSStreamingConfig` directly), which streams a TFRecord
copy with a fixed prefetch buffer instead of loading everything at init.

## Custom Data Directory

Read a dataset prepared in a specific location:

```python
config = TFDSEagerConfig(
    name="cifar10",
    split="train",
    data_dir="/path/to/tfds_data",
)
source = TFDSEagerSource(config)
```

## Strings Are Provenance

A record's text features stay on the host as its provenance, aligned with the rows, and never
enter a `Batch` or a compiled program. CIFAR-10's `id` is one:

```python
source = TFDSEagerSource(TFDSEagerConfig(name="cifar10", split="train[:4]"))
sorted(source.data)  # ['image', 'label']
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
    name="cifar10",
    split="train",
    exclude_keys={"label"},
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

Access rich metadata from TFDS, read from the prepared copy:

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
