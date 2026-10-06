# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# ArrayRecord Source Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5 min |
| **Prerequisites** | Simple Pipeline |
| **Format** | Python + Jupyter |

## Overview

ArrayRecord is Google's record file format with random access by position, the format TFDS
prepares datasets in for Grain. `ArrayRecordSourceModule` serves ArrayRecord files to a
Datarax pipeline as an indexed source: a record's index is its position in the files, the
pipeline chooses which records each batch holds, and the source reads them with one batched
read and decodes them with one call per batch. Nothing is held in memory between batches, so
the files can be larger than RAM.

## Learning Goals

By the end of this quick reference, you will be able to:

1. Write a decoder that turns a batch of `bytes` records into arrays and provenance
2. Read named records with `get_batch` and their non-numeric values with `provenance`
3. Serve ArrayRecord files through a shuffled `Pipeline` and resume it from its state
4. Read a TFDS split prepared as ArrayRecord per batch with `from_tfds(..., in_memory=False)`
"""

# %% [markdown]
"""
## Coming from Google Grain?

| Grain | Datarax |
|-------|---------|
| `grain.sources.ArrayRecordDataSource(paths)` | `ArrayRecordSourceModule(config, paths, decode=decode)` |
| `MapDataset.source(source).shuffle(seed)` | `Pipeline(source=source, ..., shuffle=True)` |
| `.map(parse)` per record | `decode(records)`, one call per batch |
| `iterator.get_state()` / `set_state()` | `pipeline.get_state()` / `set_state()` |

The source reads through ArrayRecord's `ArrayRecordDataSource`, the reader Grain wraps; the order, the epochs and the
position belong to the pipeline, as for every indexed source.
"""

# %% [markdown]
"""
## Setup

`array_record` and Grain are installed with Datarax on Linux.

```bash
uv pip install datarax
```
"""

# %%
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx

from datarax.core.index_words import from_words, to_words
from datarax.pipeline import Pipeline
from datarax.sources import ArrayRecordSourceConfig, ArrayRecordSourceModule


# %% [markdown]
"""
## Part 1: ArrayRecord files

Each record here is 4 float32 features, an int32 label and a text id, serialized to `bytes`.
Two files hold 40 and 24 records; the source numbers them 0 to 63 across the files in order.
"""

# %%
FEATURES = 4
SHARDS = (40, 24)


def encode(index: int) -> bytes:
    """One record: its features, its label and its id."""
    features = np.full(FEATURES, index / 10, dtype=np.float32)
    label = np.int32(index % 3)
    return features.tobytes() + label.tobytes() + f"record-{index:03d}".encode()


directory = Path(tempfile.mkdtemp(prefix="arrayrecord_quickref_"))
paths, start = [], 0
for shard, count in enumerate(SHARDS):
    path = directory / f"train-{shard:05d}-of-{len(SHARDS):05d}.array_record"
    writer = ArrayRecordWriter(str(path), "group_size:1")
    for index in range(start, start + count):
        writer.write(encode(index))
    writer.close()
    paths.append(str(path))
    start += count
print(f"Wrote {sum(SHARDS)} records to {len(paths)} files")

# %% [markdown]
"""
## Part 2: A decoder for a batch of records

`decode` receives the `bytes` of every record in a batch, in the order named, and returns one
mapping of values per record. Numeric values become the batch's columns; strings and other
objects become the record's provenance, which the source serves by index beside the batch.
"""


# %%
def decode(records: Sequence[bytes]) -> list[dict[str, Any]]:
    """The batch's records as features, label and id."""
    split = 4 * FEATURES
    return [
        {
            "features": np.frombuffer(record[:split], dtype=np.float32),
            "label": np.frombuffer(record[split : split + 4], dtype=np.int32)[0],
            "id": record[split + 4 :].decode(),
        }
        for record in records
    ]


source = ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths, decode=decode)
print(f"Records: {len(source)}")
print(f"Element spec: {source.element_spec()}")

# %% [markdown]
"""
## Part 3: Reading named records

`get_batch` takes record indices as their uint32 words (`to_words` names plain positions),
reads the named records with one batched read, decodes them with one call and returns a host
`Batch` named with them. `provenance` returns the same records' non-numeric values.
"""

# %%
words = to_words(np.array([45, 2, 39, 40], dtype=np.uint64))
batch = source.get_batch(words)
print(f"Labels: {batch['label']}")
print(f"Features of record 45: {batch['features'][0]}")
print(f"Indices: {from_words(np.asarray(batch.indices))}")
print(f"Ids: {[p['id'] for p in source.provenance(words)]}")

# %% [markdown]
"""
## Part 4: A shuffled pipeline

The pipeline owns the order: with `shuffle=True` each epoch serves a new permutation of the 64
records, drawn from the pipeline's key, and every batch names its records and their epoch.
"""


# %%
def build() -> Pipeline:
    """A shuffled pipeline over the files, two epochs of batches of 16."""
    return Pipeline(
        source=source, stages=[], batch_size=16, rngs=nnx.Rngs(0), shuffle=True, num_epochs=2
    )


pipeline = build()
served = [(from_words(np.asarray(b.indices)), np.asarray(b.epochs)) for b in pipeline]
first_epoch = np.concatenate([indices for indices, epochs in served if epochs[0] == 0])
print(f"Batches served: {len(served)}")
print(f"First batch: {served[0][0]}")
print(f"Epoch 0 serves every record once: {sorted(first_epoch.tolist()) == list(range(64))}")

# %% [markdown]
"""
## Part 5: Resuming from the pipeline's state

The pipeline's state is its cursor: the epoch, the position and a fingerprint of its
configuration. A pipeline built the same way and given that state serves the rest of the run.
"""

# %%
interrupted = build()
batches = iter(interrupted)
for _ in range(3):
    next(batches)
state = interrupted.get_state()
print(f"State after three batches: epoch {state['epoch']}, position {state['position']}")

resumed = build()
resumed.set_state(state)
rest = [from_words(np.asarray(b.indices)) for b in resumed]
same = all(np.array_equal(a, b) for a, (b, _) in zip(rest, served[3:], strict=True))
print(f"The resumed run serves the uninterrupted run's last {len(rest)} batches: {same}")
interrupted.close()

# %% [markdown]
"""
## Part 6: A TFDS split prepared as ArrayRecord

`from_tfds` decodes an ArrayRecord split into memory by default. With `in_memory=False` it
returns an `ArrayRecordSourceModule` over the split's files, with TFDS's decoder, for a split
larger than RAM; every record is then decoded each epoch.

```python
from datarax.sources import from_tfds

source = from_tfds("imagenet2012", "train", data_dir=data_dir, in_memory=False)
```
"""

# %%
source.close()
shutil.rmtree(directory)
print("Closed the source's file handles and removed the files.")

# %% [markdown]
"""
## Results Summary

| Aspect | `ArrayRecordSourceModule` |
|--------|---------------------------|
| Record index | position in the files, across files in order |
| Read | one batched read of a batch's records (a parallel read per file) |
| Decode | one `decode` call per batch: numbers to columns, other values to provenance |
| Order, epochs, resume | the pipeline's (`shuffle`, `num_epochs`, `get_state` / `set_state`) |
| Memory | no records held between batches |
| File handles | `close()` or a `with` block |

## Next Steps

- [TFDS Quick Reference](../tfds/tfds-quickref.ipynb) - TFDS splits in memory or per batch
- [Resumed Training Guide](../../comparison/resumed-training-guide.ipynb) - checkpointing a run
"""
