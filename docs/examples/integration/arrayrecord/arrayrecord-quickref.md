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

## Coming from Google Grain?

| Grain | Datarax |
|-------|---------|
| `grain.sources.ArrayRecordDataSource(paths)` | `ArrayRecordSourceModule(config, paths, decode=decode)` |
| `MapDataset.source(source).shuffle(seed)` | `Pipeline(source=source, ..., shuffle=True)` |
| `.map(parse)` per record | `decode(records)`, one call per batch |
| `iterator.get_state()` / `set_state()` | `pipeline.get_state()` / `set_state()` |

The source reads through ArrayRecord's `ArrayRecordDataSource`, the reader Grain wraps; the order, the epochs and the
position belong to the pipeline, as for every indexed source.

## Files

- **Python Script**: [`examples/integration/arrayrecord/01_arrayrecord_quickref.py`](https://github.com/avitai/datarax/blob/main/examples/integration/arrayrecord/01_arrayrecord_quickref.py)
- **Jupyter Notebook**: [`examples/integration/arrayrecord/01_arrayrecord_quickref.ipynb`](https://github.com/avitai/datarax/blob/main/examples/integration/arrayrecord/01_arrayrecord_quickref.ipynb)

## Quick Start

`array_record` and Grain are installed with Datarax on Linux.

```bash
uv pip install datarax
python examples/integration/arrayrecord/01_arrayrecord_quickref.py
```

The example writes 64 records to two ArrayRecord files in a temporary directory and removes
them at the end.

## Key Concepts

### A decoder for a batch of records

`decode` receives the `bytes` of every record in a batch, in the order named, and returns one
mapping of values per record. Numeric values become the batch's columns; strings and other
objects become the record's provenance, served by index beside the batch.

```python
def decode(records: Sequence[bytes]) -> list[dict[str, Any]]:
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
```

`paths` is a path, a sequence of paths, or TFDS `FileInstruction`s that read part of each
file. `ArrayRecordSourceConfig(local_files_only=True)` refuses a missing path by name at
construction.

### Reading named records

`get_batch` takes record indices as their uint32 words and returns a host `Batch` named with
them; `provenance` returns the same records' non-numeric values.

```python
words = to_words(np.array([45, 2, 39, 40], dtype=np.uint64))
batch = source.get_batch(words)
ids = [p["id"] for p in source.provenance(words)]
```

### A shuffled pipeline, and resuming it

With `shuffle=True` each epoch serves a new permutation of the records, drawn from the
pipeline's key. The pipeline's state is its cursor; a pipeline built the same way and given
that state serves the rest of the run.

```python
pipeline = Pipeline(
    source=source, stages=[], batch_size=16, rngs=nnx.Rngs(0), shuffle=True, num_epochs=2
)
state = pipeline.get_state()  # after some batches
resumed = Pipeline(...)       # built the same way
resumed.set_state(state)
```

### A TFDS split prepared as ArrayRecord

`from_tfds` decodes an ArrayRecord split into memory by default. With `in_memory=False` it
returns an `ArrayRecordSourceModule` over the split's files, with TFDS's decoder, for a split
larger than RAM; every record is then decoded each epoch.

```python
source = from_tfds("imagenet2012", "train", data_dir=data_dir, in_memory=False)
```

## Example Output

```
Wrote 64 records to 2 files
Records: 64
Element spec: {'features': ShapeDtypeStruct(shape=(4,), dtype=float32), 'label': ShapeDtypeStruct(shape=(), dtype=int32)}
Labels: [0 2 0 1]
Features of record 45: [4.5 4.5 4.5 4.5]
Indices: [45  2 39 40]
Ids: ['record-045', 'record-002', 'record-039', 'record-040']
Batches served: 8
First batch: [27 46 38 59 43 60 18 26 57  5 23 45 34  2 17 35]
Epoch 0 serves every record once: True
State after three batches: epoch 0, position 48
The resumed run serves the uninterrupted run's last 5 batches: True
Closed the source's file handles and removed the files.
```

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

- [TFDS Quick Reference](../tfds/tfds-quickref.md) - TFDS splits in memory or per batch
- [Resumed Training Guide](../../comparison/resumed-training-guide.md) - checkpointing a run

## API Reference

- [`ArrayRecordSourceModule`](../../../sources/array_record_source.md)
- [`ArrayRecordSourceConfig`](../../../sources/array_record_source.md)
