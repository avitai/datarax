# Host Values

What a source or a pipeline keeps on the host, out of NNX state and out of every compiled
program: a source's provenance, a dataset's info, a stream's position, the pipeline's host stage,
and the memory map of an on-disk array.

A holder is a plain object, so NNX treats it as a static of the module graph. It is never module
state, never a jit argument and never traced. Its equality decides whether two modules share a
graphdef, and with it a compiled program:

- `HostValue` and its subclasses (`HostProvenance`, and the stream and pipeline holders built on
  it) compare equal by kind, whatever they hold, so sources that differ only in such values share
  one compiled program. Nothing a program runs reads them.
- `HostArray` compares equal only to itself, because a compiled program's host callback reads its
  memory map. It pickles as its path, so a worker process opens the file itself.

## See Also

- [data_source](data_source.md) - The source contract
- [Eager Source](../sources/eager_source.md) - Provenance beside in-memory columns
- [Streaming Disk Source](../sources/streaming_disk_source.md) - An on-disk array read through a memory map

---

::: datarax.core.host_values
