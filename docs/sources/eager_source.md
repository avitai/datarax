# Eager Source

The base of every in-memory source (`MemorySource`, `TFDSEagerSource`, `HFEagerSource`): records
held on the host as NumPy columns, each record's strings and objects kept beside them as its
provenance, and one stateless host read, `get_batch(indices, *, epochs=0)`, that returns a
`Batch` named with the given indices and epochs. A source that loads records from somewhere
new subclasses `EagerSource` and stores its columns with `_store`.

## See Also

- [Sources Overview](index.md) - All data sources
- [Memory Source](memory_source.md) - A dict of arrays, a list of records or an array
- [Data Sources Guide](../user_guide/data_sources.md) - In-depth guide

---

::: datarax.sources.eager_source
