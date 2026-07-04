# HuggingFace Source

`HFEagerSource` provides seamless integration with [HuggingFace Datasets](https://huggingface.co/docs/datasets), allowing you to load any of the 100,000+ datasets available on the Hub directly into your Datarax pipelines with automatic JAX array conversion.

> **Note:** You can also use the factory function `from_hf(name, split, ...)` which auto-selects between eager and streaming modes based on your configuration.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | TensorFlow/NumPy tensors → JAX arrays |
| **Streaming support** | `HFStreamingSource` / `from_hf(streaming=True)` loads large datasets without downloading everything |
| **Shuffling** | Eager: O(1)-memory Feistel index shuffle; streaming: buffer-based shuffle |
| **Key filtering** | Include/exclude specific dataset fields |
| **Stateful iteration** | Track position, epoch, and support batch retrieval |

!!! note "Key points"

    - HFEagerSource wraps the `datasets` library for JAX-native workflows
    - PIL images are automatically converted to JAX arrays
    - For datasets larger than your disk, use `HFStreamingSource` or `from_hf(streaming=True)` (streaming is not a field on `HFEagerConfig`)
    - Eager shuffling is an O(1)-memory Feistel index shuffle; only the streaming source uses a buffer-based shuffle
    - The `get_batch()` method enables efficient batch retrieval

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
from datarax.sources import HFEagerSource
from datarax.sources.hf_source import HFEagerConfig

# Load IMDB sentiment dataset
config = HFEagerConfig(name="stanfordnlp/imdb", split="train")
source = HFEagerSource(config, rngs=nnx.Rngs(0))

# Iterate over elements
for item in source:
    text = item["text"]
    label = item["label"]
    process(text, label)
```

## Batch Retrieval

For training loops, use the stateful `get_batch()` method:

```python
# Get batches of 32 samples
batch = source.get_batch(32)
# batch["text"] has shape (32,)
# batch["label"] has shape (32,)

# Automatic epoch cycling
for step in range(1000):
    batch = source.get_batch(32)
    train_step(batch)
```

## Streaming Large Datasets

For datasets too large to download completely:

```python
import flax.nnx as nnx
from datarax.sources import from_hf

# Streaming is selected via the factory, not HFEagerConfig
source = from_hf("allenai/c4", "train", streaming=True, rngs=nnx.Rngs(0))

# Data is fetched on-demand
for item in source:
    process(item)
```

## Shuffling

Enable shuffling with configurable buffer size:

```python
config = HFEagerConfig(
    name="mnist",
    split="train",
    shuffle=True,
    seed=42,  # Integer seed for Grain's O(1)-memory index shuffle
)
source = HFEagerSource(config, rngs=nnx.Rngs(42))
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
