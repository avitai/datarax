# Sources

Data source adapters for loading data from various formats and libraries. Sources provide a unified interface for accessing datasets, with automatic conversion to JAX arrays.

## Available Sources

| Source | Backend | Best For |
|--------|---------|----------|
| **HFSource** | HuggingFace Datasets | 100K+ Hub datasets |
| **TFDSSource** | TensorFlow Datasets | TFDS catalog |
| **MemorySource** | In-memory arrays | Testing, small data |
| **ArrayRecordSource** | ArrayRecord format | Large-scale training |
| **MixedSource** | Multiple sources | Multi-dataset training |

`★ Insight ─────────────────────────────────────`

- All sources auto-convert to JAX arrays
- Use `streaming=True` for datasets larger than disk
- `get_batch()` provides stateful batch retrieval
- Sources track epoch and index for training loops

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.sources import HFSource, TFDSSource
from datarax.sources.hf_source import HfDataSourceConfig

# HuggingFace dataset
config = HfDataSourceConfig(name="mnist", split="train")
source = HFSource(config)

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
config = HfDataSourceConfig(
    name="c4",
    split="train",
    streaming=True,  # Don't download everything
)
```

### Shuffling

```python
config = HfDataSourceConfig(
    name="imagenet",
    split="train",
    shuffle=True,
    shuffle_buffer_size=10000,
)
```

### Field Filtering

```python
config = HfDataSourceConfig(
    name="coco",
    split="train",
    include_keys={"image", "label"},  # Only these fields
)
```

## See Also

- [HFSource Guide](hf_source.md) - HuggingFace integration details
- [TFDSSource Guide](tfds_source.md) - TensorFlow Datasets details
- [Data Sources User Guide](../user_guide/data_sources.md)
- [HuggingFace Examples](../examples/integration/huggingface/hf-quickref.md)
