# Sources

Data source adapters for loading data from various formats and libraries. Sources provide a unified interface for accessing datasets, with automatic conversion to JAX arrays.

## Available Sources

| Source | Backend | Best For |
|--------|---------|----------|
| **HFEagerSource** | HuggingFace Datasets | Small/medium Hub datasets |
| **HFStreamingSource** | HuggingFace Datasets | Large datasets (streaming) |
| **TFDSEagerSource** | TensorFlow Datasets | Small/medium TFDS catalog |
| **TFDSStreamingSource** | TensorFlow Datasets | Large datasets (streaming) |
| **MemorySource** | In-memory arrays | Testing, small data |
| **ArrayRecordSource** | ArrayRecord format | Large-scale training |
| **MixedSource** | Multiple sources | Multi-dataset training |

!!! tip "Factory functions with auto-selection"
    Use `from_hf(name, split, ...)` and `from_tfds(name, split, ...)` for automatic eager/streaming mode selection. These factory functions choose between eager and streaming mode based on dataset size (threshold: 1GB). You can override with `eager=True` or `eager=False`.

## Quick Start

```python
from datarax.sources import HFEagerSource, TFDSEagerSource
from datarax.sources.hf_source import HFEagerConfig

# HuggingFace dataset
config = HFEagerConfig(name="mnist", split="train")
source = HFEagerSource(config)

# Iterate or get batches
for item in source:
    process(item)

# Or use stateful batching
batch = source.get_batch(32)
```

## Modules

- [hf_source](hf_source.md) - HuggingFace Datasets integration (recommended)
- [tfds_source](tfds_source.md) - TensorFlow Datasets integration
- [memory_source](memory_source.md) - In-memory data for testing
- [array_record_source](array_record_source.md) - ArrayRecord format (Google)
- [mixed_source](mixed_source.md) - Combine multiple data sources

## Common Patterns

### Streaming Large Datasets

```python
config = HFEagerConfig(
    name="c4",
    split="train",
    streaming=True,  # Don't download everything
)
```

### Shuffling

```python
config = HFEagerConfig(
    name="imagenet",
    split="train",
    shuffle=True,
    shuffle_buffer_size=10000,
    seed=42,
)
```

### Field Filtering

```python
config = HFEagerConfig(
    name="coco",
    split="train",
    include_keys={"image", "label"},  # Only these fields
)
```

## See Also

- [HFEagerSource Guide](hf_source.md) - HuggingFace integration details
- [TFDSEagerSource Guide](tfds_source.md) - TensorFlow Datasets details
- [Data Sources User Guide](../user_guide/data_sources.md)
- [HuggingFace Examples](../examples/integration/huggingface/hf-quickref.md)
