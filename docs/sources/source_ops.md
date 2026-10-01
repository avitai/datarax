# source_ops

The operations a data source is built from. Sources compose these instead of
re-implementing index resolution (the wrapped, partitioned order a key selects), a worker's
share of the records, config validation and the streaming helpers. An in-memory source builds
on [`EagerSource`](eager_source.md), which holds its records and reads them.

::: datarax.sources.source_ops
