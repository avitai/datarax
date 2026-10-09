# source_ops

The operations a data source is built from. Sources compose these instead of
re-implementing index resolution (the wrapped, partitioned order a key selects), a worker's
share of the records, key filters and a named-dataset source's repr. A named-dataset config
validates its own fields (`SourceConfigBase`). An in-memory source builds
on [`EagerSource`](eager_source.md), which holds its records and reads them.

::: datarax.sources.source_ops
