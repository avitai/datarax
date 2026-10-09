# Host Values

What a source or a pipeline keeps on the host, out of NNX state and out of every compiled
program: a source's provenance, a dataset's info, the pipeline's host stage (where iteration
stands, a stream's read included), and the memory maps of a disk source's columns.

A holder is a plain object, so NNX treats it as a static of the module graph. It is never module
state, never a jit argument and never traced. Its equality decides whether two modules share a
graphdef, and with it a compiled program:

- `HostValue` and its subclasses (`HostProvenance`, and the pipeline's holder built on it)
  compare equal by kind, whatever they hold, so sources that differ only in such values share
  one compiled program. Nothing a program runs reads them.
- `HostArray` (one column's memory map) and `HostArrays` (a source's maps, one static of its graph)
  compare equal only to themselves, because a compiled program's host callback reads the maps. A
  map pickles as its place on disk (file, offset, dtype and shape), so a worker process maps the
  file itself.

The rule both follow is that a holder's equality reflects what compiled code reads. No program
takes a kind-equal holder off a module, reads what it holds, or captures it in a host callback:
a program that did would serve one source's values to every source sharing its graphdef, without
compiling again and without an error. A holder a program must read compares equal only to itself,
as `HostArray` does. A test records every kind-equal holder taken, read or captured while JAX
traces, over every source and every traced path
(`tests/pipeline/test_host_values_stay_out_of_programs.py`).

## See Also

- [data_source](data_source.md) - The source contract
- [Eager Source](../sources/eager_source.md) - Provenance beside in-memory columns
- [Streaming Disk Source](../sources/streaming_disk_source.md) - Columns on disk read through memory maps

---

::: datarax.core.host_values
