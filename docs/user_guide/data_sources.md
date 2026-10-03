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

# Create sample data: one array per field, records along the first axis
data = {"image": jnp.ones((100, 28, 28)), "label": jnp.arange(100) % 10}

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

`MemorySource` accepts a dict of arrays, one per field with records along the first axis, held as
host NumPy columns that a `Pipeline` batches on the device. It also accepts a list of records
(dictionaries, `Element`s, numbers), which it turns into columns once at construction: the
numeric fields of equal shape are stacked, and strings and other objects are kept as each
record's provenance, never batched (see [Records as Columns](#records-as-columns)).

`Pipeline.from_arrays(data, batch_size=..., seed=..., shuffle=...)` is this source and a pipeline
with no stages in one call; `drop_last` and `num_epochs` reach the pipeline as in its constructor.

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

`TFDSEagerSource` reads a dataset from the TensorFlow Datasets catalog that TFDS has prepared as ArrayRecord, without TensorFlow in the process; prepare it once, in a process of its own, with `tfds.builder("mnist", file_format="array_record").download_and_prepare()` (the `tfds` extra). Text features, such as CIFAR-10's `id`, are kept as each record's provenance.

> **Tip:** `from_tfds(name, split, ...)` picks the source by the copy's prepared format: `TFDSEagerSource` for ArrayRecord, `TFDSStreamingSource` (no TensorFlow) for TFRecord.

### HFEagerSource

For data from Hugging Face datasets, use `HFEagerSource`:

```python
from datarax.pipeline import Pipeline
from datarax.sources import HFStreamingSource, HFStreamingConfig
from flax import nnx

# Stream a large dataset from HuggingFace (streaming keeps memory bounded)
train_source = HFStreamingSource(HFStreamingConfig(name="stanfordnlp/sst2", split="train"))

# A stream's batches hold its numeric columns; its text travels beside each batch
batch, provenance = train_source.get_batch(16, with_provenance=True)
print(batch["label"][:2], [record["sentence"] for record in provenance[:2]])

# Or iterate a pipeline over the stream, shuffled from the pipeline's seed
pipeline = Pipeline(
    source=train_source, stages=[], batch_size=16, rngs=nnx.Rngs(0), shuffle=True
)
for i, batch in enumerate(pipeline):
    print(f"Batch {i}: labels {batch['label'][:4]}")
    if i >= 2:
        break
```

`HFEagerSource` loads the entire dataset into host NumPy columns at initialization (text and other objects as each record's provenance), so it is best for datasets that fit in memory. For datasets too large to hold in memory, use `HFStreamingSource` (shown above), which reads with HuggingFace's streaming mode and names records by their arrival.

> **Note:** A dataset's configuration (for example `"sst2"` within `"nyu-mll/glue"`) is `load_dataset`'s `name`, passed through `download_kwargs`: `HFEagerConfig(name="nyu-mll/glue", split="train", download_kwargs={"name": "sst2"})`. There is no `config_name` or `subset` field on the HF configs.

> **Tip:** `from_hf(name, split)` builds the eager source and `from_hf(name, split, streaming=True)` the stream; or construct `HFEagerConfig`/`HFStreamingConfig` directly.

### ArrayRecordSourceModule

For array record format data (commonly used in large-scale ML training), use `ArrayRecordSourceModule`:

```python
import numpy as np
from datarax.pipeline import Pipeline
from datarax.sources import ArrayRecordSourceModule, ArrayRecordSourceConfig
from flax import nnx


def decode(record: bytes) -> dict[str, np.ndarray]:
    # ArrayRecord records are bytes; turn one into a dict of arrays.
    return {"features": np.frombuffer(record, dtype=np.float32)}


# Create source from array record file (config first, then path)
config = ArrayRecordSourceConfig()
source = ArrayRecordSourceModule(config, "path/to/arrayrecord/file", decode=decode)

# Each pass over the pipeline covers one epoch of decoded batches
pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0))
```

## Creating Custom Data Sources

A source whose records fit in host memory subclasses `EagerSource`, loads its records once and
stores them with `_store`: indexing, iteration, the host read, the traced gather, the order and
the spec are the base's.

```python
import csv
from dataclasses import dataclass

import numpy as np

from datarax.core.config import StructuralConfig
from datarax.sources import EagerSource


@dataclass(frozen=True)
class CSVDataSourceConfig(StructuralConfig):
    """Configuration for CSVDataSource."""

    file_path: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.file_path is None:
            raise ValueError("file_path is required")


class CSVDataSource(EagerSource):
    """Numeric rows of a CSV file, held on the host as one column."""

    def __init__(self, config: CSVDataSourceConfig, *, name: str | None = None) -> None:
        super().__init__(config, name=name)
        with open(config.file_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = [[float(value) for value in row] for row in reader]
        self._store({"features": np.asarray(rows, dtype=np.float32)})
```

Any other source subclasses `DataSourceModule` and declares what its record index means,
`record_identity`:

1. Your class extends `DataSourceModule` (or `EagerSource`, which declares `INDEXED`)
2. You pass a `StructuralConfig`-derived config as the required first positional
   argument to `super().__init__(config, ...)`
3. You declare the kind with a `record_identity` property returning `RecordIdentity.INDEXED`
   (a stable position in the source), `STREAM_IDS` (an id the stream reports) or `ARRIVAL`
   (the arrival ordinal); a source without one is refused at construction. The kind routes it: an `INDEXED` source
   implements a stateless, JAX-traceable `get_records(indices)` (what `step()` and `scan`
   call) and the host read `get_batch(indices, *, epochs, contiguous)` (what
   `for batch in pipeline` calls on the host stage); a `STREAM_IDS` or `ARRIVAL` source builds on
   `datarax.sources.StreamingSourceBase`, implements `_open_pass(pass_index, key, read_size)`
   (a generator of `StreamChunk`s: host columns, provenance and ids, read `read_size` records
   at a time; `key` is the pipeline's key as uint32 words on the host, or `None` for the
   stream's own order, and a pass's order is drawn from `pass_seed(key, pass_index)`), and is
   read forward by the host stage. An indexed source that
   partitions or mixes records also overrides `record_indices_at(start, size, key)` to return
   the stable index of the record at each position of the order the key selects (the
   sequential order when the key is `None`), uint32 `(size, 2)` with each 64-bit index as its
   words `(hi, lo)` (`datarax.core.index_words`). `start` is a Python int, its two uint32
   words `(hi, lo)` (a position of the order, below its length: how the host names positions
   past `2**31`), or a traced int32; the pipeline computes those indices once per
   batch, gathers them with `get_records`, and stochastic operators key each record's
   randomness on the same indices. The default names records by position, shuffled by the
   key when the pipeline shuffles
4. `element_spec()` describes exactly the records your batches carry: the same
   keys, per-element shapes and dtypes. For a stream, `Pipeline` checks
   every batch against it, as the device will hold it, with `datarax.core.spec.validate_batch` before running
   the DAG, and names the field that disagrees. It reads the declaration once per
   source and x64 setting, so keep it fixed after construction. Declare dtypes the
   device holds as declared: while `jax_enable_x64` is off, a declared `float64` or
   `int64` field is refused rather than narrowed, so cast in the source or enable
   x64. See [Element Specs](../core/spec.md).
5. Any mutable state is managed appropriately for checkpointing

## Using Data Sources in Pipelines

Data sources plug directly into a `Pipeline`:

```python
import jax.numpy as jnp

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx

# Create data and source
data = {"x": jnp.arange(100.0)}
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

### Records as Columns

An in-memory source holds its records as host NumPy columns and keeps no iteration state: the
pipeline owns the order, the position and the epoch, and checkpoints them. A list of records is
turned into columns once, at construction: numbers become columns, and strings and other
objects become each record's provenance, kept beside the columns and never part of a batch.
Every record must hold the same numeric fields with the same shapes; a field whose shape varies
is refused, naming the field, so pad it to a fixed length (with its mask or length in `data`) or
pack records with segment ids.

```python
import numpy as np
from datarax.core.index_words import to_words
from datarax.sources import MemorySource, MemorySourceConfig

data = [{"x": i, "name": f"record {i}"} for i in range(100)]
source = MemorySource(MemorySourceConfig(), data)

source[3]  # {"x": np.int64(3)}: the record's numbers
batch = source.get_batch(to_words(np.asarray([7, 2, 9])))  # a Batch of those records
```

## Best Practices for Data Sources

When working with data sources:

1. **Use appropriate source types**: Choose the right data source for your data to optimize loading and processing
2. **Leverage shuffling**: For training, build the pipeline with `shuffle=True`, e.g. `Pipeline(source=TFDSEagerSource(TFDSEagerConfig(name="mnist", split="train")), stages=[], batch_size=128, rngs=nnx.Rngs(42), shuffle=True)`. The pipeline owns the order and shuffles in O(1) memory via a keyed Feistel bijection — there is no shuffle buffer to size.
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

> **Factory Functions:** `from_tfds()` picks the TFDS source by the copy's prepared format; `from_hf()` builds the eager source, or the stream with `streaming=True`.

## Next Steps

Now that you understand data sources, explore:

- [Quick Start](../getting_started/quick_start.md) - See data sources in action
- [Core Concepts](../getting_started/core_concepts.md) - Understand the full pipeline architecture
- [API Reference](../core/index.md) - Detailed API documentation
