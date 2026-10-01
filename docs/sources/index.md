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
| **ArrayRecordSourceModule** | ArrayRecord format | Large-scale training |
| **MixDataSourcesNode** | Multiple sources | Multi-dataset training |

!!! tip "Factory functions with auto-selection"
    Use `from_hf(name, split, ...)` and `from_tfds(name, split, ...)` for eager/streaming mode selection. `from_tfds` picks by split size (`< 1GB` → eager), while `from_hf` defaults to eager — pass `streaming=True` to force HuggingFace streaming. You can also override with `eager=True` or `eager=False`.

## Quick Start

```python
import numpy as np
from datarax.core.index_words import to_words
from datarax.sources import HFEagerSource, TFDSEagerSource
from datarax.sources.hf_source import HFEagerConfig

# HuggingFace dataset
config = HFEagerConfig(name="ylecun/mnist", split="train")
source = HFEagerSource(config)

# Iterate in order, or read the records named by their indices as a Batch
for item in source:
    process(item)

batch = source.get_batch(to_words(np.arange(32)))
```

## Modules

- [eager_source](eager_source.md) - The in-memory base: host columns, provenance and the host read
- [hf_source](hf_source.md) - HuggingFace Datasets integration (recommended)
- [source_ops](source_ops.md) - The helpers a source is built from: wrapped index resolution, worker shares, config validation and streaming
- [tfds_source](tfds_source.md) - TensorFlow Datasets integration
- [memory_source](memory_source.md) - In-memory data for testing
- [array_record_source](array_record_source.md) - ArrayRecord format (Google)
- [mixed_source](mixed_source.md) - Combine multiple data sources

## Common Patterns

### Streaming Large Datasets

```python
import flax.nnx as nnx
from datarax.sources import from_hf

# Streaming is selected via the factory, not HFEagerConfig
source = from_hf("allenai/c4", "train", streaming=True)
```

### Shuffling

The pipeline owns an eager source's order (an O(1)-memory Feistel index shuffle):

```python
source = HFEagerSource(HFEagerConfig(name="mnist", split="train"))
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
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
