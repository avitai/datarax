# Streaming Disk Source

An indexed source over columns of arrays larger than RAM, each read through a memory map. A
column, `DiskColumn(name, file, offset, dtype, shape)`, is an array stored raw in a file at a byte
offset, its leading axis the record axis: `DiskColumn.from_npy(name, path)` reads a `.npy` file's
header for its offset, dtype and shape, and an uncompressed contiguous HDF5 dataset is read at the
offset its file reports (`h5py`'s `dataset.id.get_offset()`). A column's name is its field's path,
`/` between levels (`"meta/score"`), so records keep their nesting. No column is loaded whole: each
read touches only the rows it names, and each column pickles as its place on disk.

Two reads serve it:

- `get_batch(indices, *, epochs=0, contiguous=False)` reads on the host, as the eager sources
  do, and is what `for batch in pipeline` calls: one NumPy gather of the named rows of every
  column (a run declared contiguous is a view of the memory maps), returned as a `Batch` named
  with the given index words and epochs. It creates no device array, and refuses the padding
  index and rows outside the columns.
- `get_records(indices)` reads inside a compiled program through `io_callback`, for
  `pipeline.step()`, `scan` and `pipeline.session()`; its output is wrapped in `stop_gradient`.

With `StreamingDiskSourceConfig(provenance=DiskProvenance(file, records))` the source also serves
its records' provenance (their strings and objects) by index from one ArrayRecord file, as a
[decoded copy](decoded_copy.md) holds it; `source_identity()` declares its columns (and the copy
they are), which a copy written from it is checked against.

```python
import numpy as np
from datarax.core.index_words import to_words
from datarax.sources import DiskColumn, StreamingDiskSource, StreamingDiskSourceConfig

columns = (
    DiskColumn.from_npy("image", "images.npy"),
    DiskColumn.from_npy("label", "labels.npy"),
    DiskColumn(name="meta/score", file="scores.h5", offset=4096, dtype="float32", shape=(N, 2)),
)
source = StreamingDiskSource(StreamingDiskSourceConfig(columns=columns))
batch = source.get_batch(to_words(np.arange(256, 512, dtype=np.uint64)), contiguous=True)
```

## See Also

- [Sources Overview](index.md) - All data sources
- [eager_source](eager_source.md) - The host read the in-memory sources share
- [decoded_copy](decoded_copy.md) - A source's records decoded once into these columns

---

::: datarax.sources.streaming_disk_source
