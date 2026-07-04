# Working with Data Sources

This guide explains how to use data sources in Datarax to load and prepare data for your machine learning pipelines.

## Introduction to Data Sources

Data sources are the entry point for data in Datarax pipelines. They provide a way to iterate through datasets from various origins, such as in-memory data, files, TensorFlow Datasets, or Hugging Face datasets.

All data sources in Datarax inherit from `DataSourceModule`, which is an NNX module that implements the iterator protocol.

## Built-in Data Sources

Datarax includes several built-in data sources for common use cases.

### MemorySource

The simplest data source is `MemorySource`, which works with data already loaded in memory:

```python
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx
import jax.numpy as jnp

# Create sample data
data = [{"image": jnp.ones((28, 28)), "label": i % 10} for i in range(100)]

# Create data source with config
config = MemorySourceConfig()
source = MemorySource(config, data)

# Use in a pipeline
pipeline = Pipeline(source=source, stages=[], batch_size=10, rngs=nnx.Rngs(0))

# Iterate through batches
for i, batch in enumerate(pipeline):
    print(f"Batch shape: {batch['image'].shape}")
    if i >= 2:
        break
```

`MemorySource` accepts a dict of arrays or a list/sequence of elements, such as a list of dictionaries.

### TFDSEagerSource

For data from TensorFlow Datasets, use `TFDSEagerSource`:

```python
from datarax.pipeline import Pipeline
from datarax.sources import TFDSEagerSource, TFDSEagerConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from flax import nnx

# Load MNIST from TensorFlow Datasets
config = TFDSEagerConfig(name="mnist", split="train")
train_source = TFDSEagerSource(config)

# Define normalization as an operator
def normalize(element, key=None):
    img = element.data["image"]
    # Normalize to [0, 1] range
    return element.update_data({"image": img / 255.0})

normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize
)

# Create training pipeline
train_pipeline = (
    Pipeline(source=train_source, stages=[normalizer], batch_size=32, rngs=nnx.Rngs(0)))

# Iterate
for i, batch in enumerate(train_pipeline):
    # Train model
    print(f"Batch {i}: {batch['image'].shape}")
    if i >= 2:
        break
```

`TFDSEagerSource` handles downloading, caching, and preprocessing datasets from the TensorFlow Datasets catalog.

> **Tip:** Use `from_tfds(name, split, ...)` factory function for automatic eager/streaming mode selection.

### HFEagerSource

For data from Hugging Face datasets, use `HFEagerSource`:

```python
from datarax.pipeline import Pipeline
from datarax.sources import HFStreamingSource, HFStreamingConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from flax import nnx

# Stream a large dataset from HuggingFace (streaming keeps memory bounded)
config = HFStreamingConfig(
    name="stanfordnlp/sst2",
    split="train",
    streaming=True,
)
train_source = HFStreamingSource(config, rngs=nnx.Rngs(0))

# Define field extraction as an operator
def extract_fields(element, key=None):
    return element.update_data({
        "text": element.data["sentence"],
        "label": element.data["label"]
    })

extractor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=extract_fields
)

# Create pipeline
pipeline = (
    Pipeline(source=train_source, stages=[extractor], batch_size=16, rngs=nnx.Rngs(0)))

# Iterate
for i, batch in enumerate(pipeline):
    # Process batch
    print(f"Batch {i}: {batch['text'][:2]}...")  # Print first 2 texts
    if i >= 2:
        break
```

`HFEagerSource` loads the entire dataset into JAX arrays at initialization, so it is best for datasets that fit in memory. For datasets too large to hold in memory, use `HFStreamingSource` (shown above), which wraps HuggingFace's streaming iterator.

> **Note:** Dataset configs/variants (for example selecting `"sst2"` within the `"glue"` dataset) are currently unsupported — pass the standalone dataset name to `name`. There is no `config_name` (or subset) field on the HF configs.

> **Tip:** Use `from_hf(name, split, streaming=True, rngs=...)` to select eager or streaming mode, or construct `HFEagerConfig`/`HFStreamingConfig` directly.

### ArrayRecordSourceModule

For array record format data (commonly used in large-scale ML training), use `ArrayRecordSourceModule`:

```python
from datarax.pipeline import Pipeline
from datarax.sources import ArrayRecordSourceModule, ArrayRecordSourceConfig
from flax import nnx

# Create source from array record file (config first, then path)
config = ArrayRecordSourceConfig()
source = ArrayRecordSourceModule(config, "path/to/arrayrecord/file")

# Use in pipeline
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))
```

## Creating Custom Data Sources

You can create custom data sources by subclassing `DataSourceModule`:

```python
import csv
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule


@dataclass(frozen=True)
class CSVDataSourceConfig(StructuralConfig):
    """Configuration for CSVDataSource."""

    file_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.file_path is None:
            raise ValueError("file_path is required")


class CSVDataSource(DataSourceModule):
    """Random-access data source that reads numeric rows from a CSV file."""

    # Narrow config type for pyright (base stores via nnx.static)
    config: CSVDataSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: CSVDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        # A StructuralConfig-derived config is the required first argument.
        super().__init__(config, rngs=rngs, name=name)

        # Load all rows once at construction (skip the header row).
        with open(config.file_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = [[float(value) for value in row] for row in reader]

        # Wrap array data with nnx.data so NNX treats it as pytree data,
        # not trainable parameters.
        self.data = nnx.data(jnp.asarray(rows))

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def supports_indexed_access(self) -> bool:
        # Random-access sources return True so Pipeline drives them via
        # the jitted, indexed step() path.
        return True

    def get_batch_at(self, start: int | Any, size: int, key: Any | None = None) -> dict[str, Any]:
        # Return `size` rows starting at `start`, wrapping at the end.
        indices = (jnp.arange(size) + start) % len(self)
        return {"features": self.data[indices]}
```

When creating custom data sources, ensure:

1. Your class extends `DataSourceModule`
2. You pass a `StructuralConfig`-derived config as the required first positional
   argument to `super().__init__(config, ...)`
3. You implement the Pipeline contract: for random access, implement
   `get_batch_at(start, size, key)` and return `True` from
   `supports_indexed_access()`; for forward-only streaming, implement
   `get_batch()` instead
4. Any mutable state is managed appropriately for checkpointing

## Using Data Sources in Pipelines

Data sources plug directly into a `Pipeline`:

```python
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

# Create data and source
data = [{"x": i} for i in range(100)]
config = MemorySourceConfig()
source = MemorySource(config, data)

# Define a simple operator
def identity(element, key):
    return element

op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=identity)

# Build pipeline
pipeline = (
    Pipeline(source=source, stages=[op], batch_size=32, rngs=nnx.Rngs(0)))
```

## Data Source Features

### Metadata Support

Datarax data sources support metadata tracking through the `RecordMetadata` system:

```python
import jax.numpy as jnp
from datarax.sources import MemorySource, MemorySourceConfig

# Create sample images and labels
images = [jnp.ones((28, 28)) for _ in range(10)]
labels = [i % 10 for i in range(10)]

# Create source with metadata tracking (opt-in)
data = [{"image": image, "label": label} for image, label in zip(images, labels)]
config = MemorySourceConfig(track_metadata=True)
source = MemorySource(config, data)

# Metadata tracking is opt-in via track_metadata=True; when enabled, each
# element carries associated metadata (index, source info, etc.)
for element in source:
    # Element has associated metadata (index, source info, etc.)
    pass
```

### State Management

All data sources inherit state management from `DataSourceModule`:

```python
from datarax.sources import MemorySource, MemorySourceConfig

# Create source
data = [{"x": i} for i in range(100)]
config = MemorySourceConfig()
source = MemorySource(config, data)

# Iterate through some elements
iterator = iter(source)
for i in range(10):
    element = next(iterator)

# Source maintains iteration state
# Can be used for checkpointing
```

## Best Practices for Data Sources

When working with data sources:

1. **Use appropriate source types**: Choose the right data source for your data to optimize loading and processing
2. **Leverage shuffling**: For training, enable shuffling on the source config, e.g. `TFDSEagerConfig(name="mnist", split="train", shuffle=True, seed=42)`. Eager sources shuffle in O(1) memory via Grain's index_shuffle (a Feistel permutation) — there is no shuffle buffer to size.
3. **Batch appropriately**: Batching is the Pipeline's job — set `Pipeline(source=source, stages=[], batch_size=N, rngs=nnx.Rngs(0))`. Sources do not expose a `.batch()` method.
4. **Handle state properly**: Ensure your custom data sources properly manage their state
5. **Monitor performance**: Watch for bottlenecks in data loading, especially with large datasets
6. **Use JAX arrays**: Convert to JAX arrays early in the pipeline for better performance

## Available Sources Summary

Datarax provides the following data sources:

- **MemorySource**: For data already in memory (lists, arrays)
- **TFDSEagerSource**: For TensorFlow Datasets
- **HFEagerSource**: For Hugging Face datasets
- **ArrayRecordSourceModule**: For array record format files
- **Custom sources**: Subclass `DataSourceModule` for your own sources

> **Factory Functions:** Use `from_tfds()` and `from_hf()` for automatic eager/streaming mode selection based on your configuration.

## Next Steps

Now that you understand data sources, explore:

- [Quick Start](../getting_started/quick_start.md) - See data sources in action
- [Core Concepts](../getting_started/core_concepts.md) - Understand the full pipeline architecture
- [API Reference](../core/index.md) - Detailed API documentation
