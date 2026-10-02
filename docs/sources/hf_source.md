# HuggingFace Source

`HFEagerSource` provides seamless integration with [HuggingFace Datasets](https://huggingface.co/docs/datasets), allowing you to load any of the 100,000+ datasets available on the Hub directly into your Datarax pipelines: array columns become host NumPy columns, and text and other objects the records' provenance.

> **Note:** The factory `from_hf(name, split, ...)` builds `HFEagerSource`, or `HFStreamingSource` with `streaming=True`.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | Images and numeric columns → host NumPy columns |
| **Streaming support** | `HFStreamingSource` / `from_hf(streaming=True)` loads large datasets without downloading everything |
| **Shuffling** | `Pipeline(shuffle=True)`: eager, an O(1)-memory Feistel index shuffle; streaming, HF's buffer shuffle seeded by the pipeline |
| **Key filtering** | Include/exclude specific dataset fields |
| **Stateless reads** | Iteration in order and `get_batch(indices)`; the pipeline owns the position |

!!! note "Key points"

    - HFEagerSource wraps the `datasets` library for JAX-native workflows
    - PIL images are converted to NumPy arrays once, at load
    - Text and other non-array columns are kept as each record's provenance, beside the batches, never refused
    - For datasets larger than your disk, use `HFStreamingSource` or `from_hf(streaming=True)` (streaming is not a field on `HFEagerConfig`)
    - Both are shuffled by their pipeline (`Pipeline(shuffle=True)`): the eager source by an O(1)-memory Feistel index shuffle, the stream by HuggingFace's buffer shuffle seeded from the pipeline's key, each pass in its own order
    - The stream names records by their arrival and hands text out beside each batch (`get_batch(n, with_provenance=True)`); the eager source serves it by index (`provenance(indices)`)
    - `get_batch(indices)` reads the named records as a `Batch` with one host gather

## Installation

HFEagerSource requires the `datasets` package:

```bash
pip install datarax[data]
# or
pip install datasets
```

## Quick Start

```python
import flax.nnx as nnx
from datarax.pipeline import Pipeline
from datarax.sources import HFEagerSource
from datarax.sources.hf_source import HFEagerConfig

# Load IMDB sentiment dataset
config = HFEagerConfig(name="stanfordnlp/imdb", split="train")
source = HFEagerSource(config)

# Iterate over elements: their array columns (the text is the records' provenance)
for item in source:
    label = item["label"]
    process(label)
```

## Batch Retrieval

`get_batch(indices)` reads the records its indices name as a `Batch`, statelessly; a
training loop takes its batches from a `Pipeline`, which owns the order and the position:

```python
import numpy as np
from datarax.core.index_words import to_words

# Records 0..31 as a host Batch; batch["label"] has shape (32,)
batch = source.get_batch(to_words(np.arange(32)))

# Training: the pipeline serves every epoch in a new order
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
for batch in pipeline:
    train_step(batch)
```

## Streaming Large Datasets

For datasets too large to download completely:

```python
import flax.nnx as nnx
from datarax.sources import from_hf

# Streaming is selected via the factory, not HFEagerConfig
source = from_hf("allenai/c4", "train", streaming=True)

# Data is fetched on-demand
for item in source:
    process(item)
```

## Shuffling

The pipeline owns an eager source's order: `Pipeline(shuffle=True)` serves each epoch in a
new order, an O(1)-memory index shuffle seeded by the pipeline's `rngs`:

```python
source = HFEagerSource(HFEagerConfig(name="mnist", split="train"))
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
```

## Field Filtering

Select only the fields you need:

```python
# Include only specific fields
config = HFEagerConfig(
    name="glue",
    split="train",
    include_keys={"sentence", "label"},  # Only these fields
)

# Or exclude unwanted fields
config = HFEagerConfig(
    name="imdb",
    split="train",
    exclude_keys={"idx"},  # Everything except idx
)
```

## Integration with DAG Pipelines

```python
from datarax.pipeline import Pipeline

# Build a pipeline
config = HFEagerConfig(name="ylecun/mnist", split="train")
source = HFEagerSource(config)

pipeline = (
    Pipeline(source=source, stages=[normalize_op, augment_op], batch_size=64, rngs=nnx.Rngs(0)))

for batch in pipeline:
    train_step(batch)
```

## Dataset Information

Access metadata about the loaded dataset:

```python
# Get HuggingFace DatasetInfo
info = source.get_dataset_info()
print(f"Description: {info.description}")
print(f"Features: {info.features}")

# Check length (if available)
print(f"Dataset length: {len(source)}")
```

## See Also

- [Data Sources Guide](../user_guide/data_sources.md) - Detailed data loading guide
- [TFDS Source](tfds_source.md) - TensorFlow Datasets integration
- [HuggingFace Quick Reference](../examples/integration/huggingface/hf-quickref.md)
- [HuggingFace Tutorial](../examples/integration/huggingface/hf-tutorial.md)

---

## API Reference

::: datarax.sources.hf_source
