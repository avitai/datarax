# Data Source Protocol

Core protocol for data sources.

## Provenance and record keys

A record's strings and Python objects (a caption, a file name, an id) are its provenance: a
source keeps them on the host, never in a `Batch` and never in a trace. What a source can do with
them depends on what its record index means (`record_identity`):

| kind | `source.provenance(indices)` | `source.record_keys(batch)` |
|---|---|---|
| `INDEXED` (in-memory sources, `StreamingDiskSource`, a mix) | each named record's mapping, by index | `batch.indices` |
| `STREAM_IDS` (a stream naming records by the ids it reports) | by the id | `batch.indices` |
| `ARRIVAL` (a stream naming records by arrival) | refused: `TypeError` naming the kind | refused |

`provenance` returns one immutable mapping per index, in the order named (an empty one for a record
carrying nothing but arrays), and refuses the padding index and indices outside the source.
`record_keys` gives the identities a per-record table is keyed by; it reads only the source's
static kind, so under `jit` the batch's indices pass straight through. A record's index has the top
bit clear, and a batch built without identities names its rows `(2**31, row)`, so a device table
that reads row `lo` only where `hi == 0 and lo < N` (filling the read and dropping the update
elsewhere, or refusing with `checkify`) never touches a row for them or for padding, under every
transform. `provenance` and the host reads refuse such rows, naming why.

```python
import numpy as np
from datarax.core.index_words import to_words
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(
    MemorySourceConfig(),
    [{"x": np.float32(i), "caption": f"image {i}"} for i in range(4)],
)
batch = source.get_batch(to_words(np.asarray([2, 0], np.uint64)))
[p["caption"] for p in source.provenance(batch.indices)]  # ['image 2', 'image 0']
```

## See Also

- [Core Overview](index.md) - All core protocols
- [Sources](../sources/index.md) - Source implementations
- [HF Source](../sources/hf_source.md) - HuggingFace integration
- [Data Sources Guide](../user_guide/data_sources.md)

---

::: datarax.core.data_source
