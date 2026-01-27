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
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
import jax.numpy as jnp

# Create sample data
data = [{"image": jnp.ones((28, 28)), "label": i % 10} for i in range(100)]

# Create data source with config
config = MemorySourceConfig()
source = MemorySource(config, data)

# Use in a pipeline
pipeline = from_source(source, batch_size=10)

# Iterate through batches
for i, batch in enumerate(pipeline):
    print(f"Batch shape: {batch['image'].shape}")
    if i >= 2:
        break
```

`MemorySource` accepts any iterable of elements, such as a list of dictionaries or arrays.

### TFDSEagerSource

For data from TensorFlow Datasets, use `TFDSEagerSource`:

```python
from datarax import from_source
from datarax.sources import TFDSEagerSource, TFDSEagerConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import OperatorNode

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
    from_source(train_source, batch_size=32)
    >> OperatorNode(normalizer)
)

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
from datarax import from_source
from datarax.sources import HFEagerSource, HFEagerConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import OperatorNode

# Load dataset from HuggingFace (streaming mode for large datasets)
config = HFEagerConfig(
    name="glue",
    config_name="sst2",  # Use SST-2 for simpler example
    split="train",
    streaming=True
)
train_source = HFEagerSource(config)

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
    from_source(train_source, batch_size=16)
    >> OperatorNode(extractor)
)

# Iterate
for i, batch in enumerate(pipeline):
    # Process batch
    print(f"Batch {i}: {batch['text'][:2]}...")  # Print first 2 texts
    if i >= 2:
        break
```

`HFEagerSource` supports both downloaded and streaming modes, allowing you to work with datasets of any size.

> **Tip:** Use `from_hf(name, split, ...)` factory function for automatic eager/streaming mode selection.

### ArrayRecordSourceModule

For array record format data (commonly used in large-scale ML training), use `ArrayRecordSourceModule`:

```python
from datarax import from_source
from datarax.sources import ArrayRecordSourceModule

# Create source from array record file
source = ArrayRecordSourceModule("path/to/arrayrecord/file")

# Use in pipeline
pipeline = from_source(source, batch_size=32)
```

## Creating Custom Data Sources

You can create custom data sources by subclassing `DataSourceModule`:

```python
import flax.nnx as nnx
import jax.numpy as jnp
from datarax.core.data_source import DataSourceModule
from typing import Iterator
from datarax.typing import Element

class CSVDataSource(DataSourceModule):
    """Data source that reads from a CSV file."""

    def __init__(self, file_path: str, name: str = "csv_source"):
        super().__init__(name=name)
        self.file_path = file_path
        self.data = None  # Will be loaded later
        self.index = 0

    def __iter__(self) -> Iterator[Element]:
        # Load data on first iteration
        if self.data is None:
            import csv
            import numpy as np

            rows = []
            with open(self.file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                for row in reader:
                    rows.append([float(x) for x in row])

            self.data = np.array(rows)

        # Reset iterator state
        self.index = 0
        return self

    def __next__(self) -> Element:
        if self.index >= len(self.data):
            raise StopIteration

        # Get current element
        element = {"features": jnp.array(self.data[self.index])}

        # Update state
        self.index += 1

        return element
```

When creating custom data sources, ensure:

1. Your class extends `DataSourceModule`
2. The `__iter__` method returns `self` and resets any iteration state
3. The `__next__` method returns the next element or raises `StopIteration`
4. Any mutable state is managed appropriately for checkpointing

## Using Data Sources in Pipelines

Data sources can be used with the DAG API in two ways:

### Method 1: Using from_source (Recommended)

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import OperatorNode

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
    from_source(source, batch_size=32)
    >> OperatorNode(op)
)
```

### Method 2: Using the >> operator

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig

# Create data source
data = [{"x": i} for i in range(100)]
config = MemorySourceConfig()
source = MemorySource(config, data)

# Create operator
def double(element, key):
    return element.update_data({"x": element.data["x"] * 2})

op = ElementOperator(ElementOperatorConfig(stochastic=False), fn=double)

# Build pipeline using >> operator
pipeline = (
    from_source(source, batch_size=32)
    >> OperatorNode(op)
)
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

# Create source with metadata
data = [{"image": image, "label": label} for image, label in zip(images, labels)]
config = MemorySourceConfig()
source = MemorySource(config, data)

# Metadata is automatically tracked per element
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
2. **Leverage shuffling**: For training, use `.shuffle(buffer_size)` with a sufficiently large buffer
3. **Batch appropriately**: Use `.batch(batch_size)` or `from_source(source, batch_size=N)` for efficient processing
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
