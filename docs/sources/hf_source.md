# HuggingFace Source

`HFSource` provides seamless integration with [HuggingFace Datasets](https://huggingface.co/docs/datasets), allowing you to load any of the 100,000+ datasets available on the Hub directly into your Datarax pipelines with automatic JAX array conversion.

## Key Features

| Feature | Description |
|---------|-------------|
| **Automatic conversion** | TensorFlow/NumPy tensors → JAX arrays |
| **Streaming support** | Load large datasets without downloading everything |
| **Shuffling** | Built-in shuffle with configurable buffer size |
| **Key filtering** | Include/exclude specific dataset fields |
| **Stateful iteration** | Track position, epoch, and support batch retrieval |

`★ Insight ─────────────────────────────────────`

- HFSource wraps the `datasets` library for JAX-native workflows
- PIL images are automatically converted to JAX arrays
- Use `streaming=True` for datasets larger than your disk
- The `get_batch()` method enables efficient batch retrieval

`─────────────────────────────────────────────────`

## Installation

HFSource requires the `datasets` package:

```bash
pip install datarax[hf]
# or
pip install datasets
```

## Quick Start

```python
import flax.nnx as nnx
from datarax.sources import HFSource
from datarax.sources.hf_source import HfDataSourceConfig

# Load IMDB sentiment dataset
config = HfDataSourceConfig(name="imdb", split="train")
source = HFSource(config, rngs=nnx.Rngs(0))

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
config = HfDataSourceConfig(
    name="c4",  # Common Crawl dataset (800GB+)
    split="train",
    streaming=True,
)
source = HFSource(config)

# Data is fetched on-demand
for item in source:
    process(item)
```

## Shuffling

Enable shuffling with configurable buffer size:

```python
config = HfDataSourceConfig(
    name="mnist",
    split="train",
    shuffle=True,
    shuffle_buffer_size=10000,  # Buffer for streaming shuffle
)
source = HFSource(config, rngs=nnx.Rngs(42))
```

## Field Filtering

Select only the fields you need:

```python
# Include only specific fields
config = HfDataSourceConfig(
    name="glue",
    split="train",
    include_keys={"sentence", "label"},  # Only these fields
)

# Or exclude unwanted fields
config = HfDataSourceConfig(
    name="imdb",
    split="train",
    exclude_keys={"idx"},  # Everything except idx
)
```

## Integration with DAG Pipelines

```python
from datarax.dag import from_source
from datarax.dag.nodes import OperatorNode

# Build a pipeline
config = HfDataSourceConfig(name="mnist", split="train")
source = HFSource(config)

pipeline = (
    from_source(source, batch_size=64)
    >> OperatorNode(normalize_op)
    >> OperatorNode(augment_op)
)

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

- [Data Sources Guide](../user_guide/data_sources.md) - Comprehensive data loading guide
- [TFDS Source](tfds_source.md) - TensorFlow Datasets integration
- [HuggingFace Quick Reference](../examples/integration/huggingface/hf-quickref.md)
- [HuggingFace Tutorial](../examples/integration/huggingface/hf-tutorial.md)

---

## API Reference

::: datarax.sources.hf_source
