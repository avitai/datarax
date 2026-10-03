# Mixed Source

`MixDataSourcesNode` mixes indexed sources in fixed proportions, the way Grain's
`MapDataset.mix` does, and names every record of every source in one 64-bit index space.

```python
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig, MixDataSourcesConfig, MixDataSourcesNode

small = MemorySource(MemorySourceConfig(), {"x": np.zeros((300, 8), np.float32)})
large = MemorySource(MemorySourceConfig(), {"x": np.ones((700, 8), np.float32)})
mix = MixDataSourcesNode(MixDataSourcesConfig(weights=(0.3, 0.7)), [small, large])
pipeline = Pipeline(source=mix, stages=[], batch_size=32, rngs=nnx.Rngs(0), shuffle=True)
```

## Which source serves each position

Grain turns the weights into integer proportions. It scales the smallest weight to 100, scales
the others by the same factor and truncates them, so `(0.3, 0.7)` becomes `100:233`. The
selection interleaves the sources in those proportions and repeats every `sum(proportions)`
positions. A source's `j`-th position in the mix is its own position `j`. The selection draws
nothing at random.

## How long an epoch is

An epoch is Grain's length: the most positions that serve each record of each source at most
once, `min(len(source) * S / p)` over the sources, for proportions `p` summing to `S`. Every
record an epoch serves is therefore distinct, and no two rows of an epoch share a random key.
Sources at `(0.5, 0.5)` with 1,000 and 10 records give an epoch of 20 positions, 10 from each.
A larger weight draws more from a larger source.

## The order within each source

The pipeline owns the order. With `Pipeline(shuffle=True)` each source orders its own records by
the epoch's key folded with the source's position in the mix, so each epoch reads a different
part of every source that holds more records than an epoch takes from it. With `shuffle=False`
the mix is a fixed interleave of each source's records in order, the same records every epoch:
a source that an epoch does not exhaust serves the same first records each time, as an
unshuffled `drop_last` epoch skips the same tail. One pass over every record of several sources
is a concatenation of the sources, not a weighted mix.

## Record identity

A mixed record's index is its source's offset, the number of records in the sources before it,
plus its index within that source, as two `uint32` words. `mix.provenance(indices)` serves each
record's strings and objects from the source that owns it. When a source grows, the indices of
the sources after it move and the epoch lengthens.

## What a mix takes

Every source must be an `INDEXED` source with at least one record. The mix refuses a stream
(`STREAM_IDS`, `ARRIVAL`), which has no stable positions, and one worker's shard of a
`MemorySource`. A mix is itself an `INDEXED` source, so mixes nest. Weights must be positive. The
mix refuses weights so far apart that Grain's proportions sum past `2**32 - 1`.

## See Also

- [Sources Overview](index.md) - All data sources
- [Data Sources Guide](../user_guide/data_sources.md) - Full guide
- [Samplers](../samplers/index.md) - Sampling strategies
- [HF Source](hf_source.md) - HuggingFace integration

---

::: datarax.sources.mixed_source
