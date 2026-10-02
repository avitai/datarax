# Data Loading Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~2 min |
| **Prerequisites** | Datarax installed |
| **Format** | Reference card |
| **Memory** | ~100 MB RAM |

## Overview

Datarax provides three primary data source types for loading data into pipelines. This quick reference covers the most common patterns for each.

## At a Glance

| Source | Best For | Loads Data | Shuffling |
|--------|----------|-----------|-----------|
| `MemorySource` | In-memory numpy/JAX arrays | At init (to host NumPy columns) | O(1) Feistel cipher |
| `TFDSEagerSource` | TensorFlow Datasets (< 1GB) | At init (to host NumPy columns) | O(1) Feistel cipher |
| `HFEagerSource` | HuggingFace Datasets (< 1GB) | At init (to host NumPy columns) | O(1) Feistel cipher |

All eager sources load their data at initialization into host NumPy columns, so reading a batch is
one NumPy gather with zero framework overhead; the pipeline places it on the device. The
shuffle is the pipeline's (`Pipeline(..., shuffle=True)`).

## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `torch.utils.data.TensorDataset(X, y)` | `MemorySource(config, data={"X": X, "y": y})` |
| `torchvision.datasets.CIFAR10(root, train)` | `from_tfds("cifar10", "train")` |
| `datasets.load_dataset("stanfordnlp/imdb")` | `from_hf("stanfordnlp/imdb", "train")` |
| `DataLoader(ds, shuffle=True)` | `Pipeline(source=..., ..., shuffle=True)` |

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tf.data.Dataset.from_tensor_slices(data)` | `MemorySource(config, data=data)` |
| `tfds.load("cifar10", split="train")` | `from_tfds("cifar10", "train")` |
| `tf.data.Dataset.shuffle(buffer)` | `Pipeline(source=..., ..., shuffle=True)` (full shuffle, not buffer) |

## MemorySource

For data already in memory as numpy or JAX arrays.

```python
import numpy as np
from flax import nnx
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

# Create data as a dict of arrays (first axis = samples)
data = {
    "image": np.random.randn(1000, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, size=(1000,)),
}

# Basic usage
config = MemorySourceConfig()
source = MemorySource(config, data=data)

# Shuffling belongs to the pipeline; its seed comes from the pipeline's rngs
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
```

## TFDSEagerSource

For loading TensorFlow Datasets prepared as ArrayRecord, without TensorFlow in the process. Uses the `from_tfds()` factory for convenience.

```python
from datarax.sources import from_tfds

# cifar10 prepared as ArrayRecord: the eager source (a TFRecord copy would be streamed)
source = from_tfds("cifar10", "train")

# Specify the directory the dataset is prepared in
source = from_tfds(
    "mnist", "train",
    data_dir="/path/to/data",
)

# Load subset with split slicing
source = from_tfds("cifar10", "train[:5000]")
```

!!! note "Prepare once, read without TensorFlow"
    Reading needs `uv pip install "datarax[data]"` (tensorflow-datasets). The eager source never
    prepares a dataset: prepare it once as ArrayRecord, in a process of its own with the `tfds`
    extra, e.g. `python -c "import tensorflow_datasets as tfds; tfds.builder('cifar10', file_format='array_record').download_and_prepare()"`.
    A copy that is not prepared, or is prepared only as TFRecord, is refused naming that call.

## HFEagerSource

For loading HuggingFace Datasets. Uses the `from_hf()` factory.

```python
from datarax.sources import from_hf

# Load a HuggingFace dataset
source = from_hf("ylecun/mnist", "train")

# Filter specific columns
source = from_hf(
    "imdb", "train",
    include_keys={"text", "label"},
)

# Force streaming for large datasets
source = from_hf("allenai/c4", "train", streaming=True)
```

!!! note "HF Datasets requires `datasets`"
    Install with `uv pip install datasets`. Like TFDS, Datarax lazy-imports
    the HuggingFace `datasets` library.

## Using Sources in Pipelines

All sources plug into `Pipeline(source=..., stages=[...], batch_size=N, rngs=...)` to create iterable pipelines:

```python
from datarax.pipeline import Pipeline

pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))

for batch in pipeline:
    images = batch["image"]   # shape: (32, 32, 32, 3)
    labels = batch["label"]   # shape: (32,)
    # ... process batch
```

## Next Steps

- [Batch Processing Basics](batch-processing-quickref.md) -- Understand how batches work
- [Simple Pipeline](simple-pipeline.md) -- Build your first complete pipeline
- [Operators Tutorial](operators-tutorial.md) -- Add transformations to your pipeline
