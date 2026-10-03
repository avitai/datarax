# Streaming Disk Source

An indexed source over a `.npy` array larger than RAM, read through a memory map. The array is
never loaded whole: each read touches only the rows it names.

Two reads serve it:

- `get_batch(indices, *, epochs=0, contiguous=False)` reads on the host, as the eager sources
  do, and is what `for batch in pipeline` calls: one NumPy gather of the named rows (a run declared contiguous is a view of the memory
  map), returned as a `Batch` named with the given index words and epochs. It creates no device
  array, and refuses the padding index and rows outside the array.
- `get_records(indices)` reads inside a compiled program through `io_callback`, for
  `pipeline.step()`, `scan` and `pipeline.session()`; its output is wrapped in `stop_gradient`.

```python
import numpy as np
from datarax.core.index_words import to_words
from datarax.sources import StreamingDiskSource, StreamingDiskSourceConfig

source = StreamingDiskSource(StreamingDiskSourceConfig(path="features.npy", feature_key="x"))
batch = source.get_batch(to_words(np.arange(256, 512, dtype=np.uint64)), contiguous=True)
```

## See Also

- [Sources Overview](index.md) - All data sources
- [eager_source](eager_source.md) - The host read the in-memory sources share

---

::: datarax.sources.streaming_disk_source
