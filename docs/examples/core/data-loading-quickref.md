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
| `MemorySource` | In-memory numpy/JAX arrays | At init (from arrays) | O(1) Feistel cipher |
| `TFDSEagerSource` | TensorFlow Datasets (< 1GB) | At init (to JAX arrays) | O(1) Feistel cipher |
| `HFEagerSource` | HuggingFace Datasets (< 1GB) | At init (to JAX arrays) | O(1) Feistel cipher |

All eager sources convert data to JAX arrays at initialization, so iteration is pure JAX with zero framework overhead.

## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `torch.utils.data.TensorDataset(X, y)` | `MemorySource(config, data={"X": X, "y": y})` |
| `torchvision.datasets.CIFAR10(root, train)` | `from_tfds("cifar10", "train")` |
| `datasets.load_dataset("imdb")` | `from_hf("imdb", "train")` |
| `DataLoader(ds, shuffle=True)` | `MemorySourceConfig(shuffle=True)` |

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tf.data.Dataset.from_tensor_slices(data)` | `MemorySource(config, data=data)` |
| `tfds.load("cifar10", split="train")` | `from_tfds("cifar10", "train")` |
| `tf.data.Dataset.shuffle(buffer)` | `MemorySourceConfig(shuffle=True)` (full shuffle, not buffer) |

## MemorySource

For data already in memory as numpy or JAX arrays.

```python
import numpy as np
from flax import nnx
from datarax.sources import MemorySource, MemorySourceConfig

# Create data as a dict of arrays (first axis = samples)
data = {
    "image": np.random.randn(1000, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, size=(1000,)),
}

# Basic usage
config = MemorySourceConfig()
source = MemorySource(config, data=data, rngs=nnx.Rngs(0))

# With shuffling (seed comes from rngs, not config)
config = MemorySourceConfig(shuffle=True)
source = MemorySource(config, data=data, rngs=nnx.Rngs(42))
```

## TFDSEagerSource

For loading TensorFlow Datasets. Uses the `from_tfds()` factory for convenience.

```python
from datarax.sources import from_tfds
import flax.nnx as nnx

# Auto-detect eager vs streaming (< 1GB = eager)
source = from_tfds("cifar10", "train", shuffle=True, rngs=nnx.Rngs(0))

# Specify custom data directory
source = from_tfds(
    "mnist", "train",
    data_dir="/path/to/data",
    shuffle=True,
    rngs=nnx.Rngs(0),
)

# Load subset with split slicing
source = from_tfds("cifar10", "train[:5000]", rngs=nnx.Rngs(0))

# Load from Google Cloud Storage (bypasses local preparation)
source = from_tfds("nsynth/gansynth_subset", "train", try_gcs=True)
```

!!! note "TFDS requires `tensorflow-datasets`"
    Install with `uv pip install tensorflow-datasets`. Datarax lazy-imports TFDS
    to avoid slowing down startup when it's not needed.

## HFEagerSource

For loading HuggingFace Datasets. Uses the `from_hf()` factory.

```python
from datarax.sources import from_hf
import flax.nnx as nnx

# Load a HuggingFace dataset
source = from_hf("mnist", "train", shuffle=True, rngs=nnx.Rngs(0))

# Filter specific columns
source = from_hf(
    "imdb", "train",
    include_keys={"text", "label"},
    rngs=nnx.Rngs(0),
)

# Force streaming for large datasets
source = from_hf("allenai/c4", "train", streaming=True, rngs=nnx.Rngs(0))
```

!!! note "HF Datasets requires `datasets`"
    Install with `uv pip install datasets`. Like TFDS, Datarax lazy-imports
    the HuggingFace `datasets` library.

## Using Sources in Pipelines

All sources plug into `from_source()` to create iterable pipelines:

```python
from datarax.dag import from_source

pipeline = from_source(source, batch_size=32)

for batch in pipeline:
    images = batch["image"]   # shape: (32, 32, 32, 3)
    labels = batch["label"]   # shape: (32,)
    # ... process batch
```

## Next Steps

- [Batch Processing Basics](batch-processing-quickref.md) -- Understand how batches work
- [Simple Pipeline](simple-pipeline.md) -- Build your first complete pipeline
- [Operators Tutorial](operators-tutorial.md) -- Add transformations to your pipeline
