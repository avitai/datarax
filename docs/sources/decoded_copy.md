# Decoded Copy

A decoded copy holds an indexed source's records as they decode, once, on disk: one `.npy` file
per numeric field and the records' provenance (their strings and objects) in one ArrayRecord
file, beside a `manifest.json` written last. It serves a source whose read decodes every record
on every epoch (a TFDS split read per batch, ArrayRecord files of encoded images) from memory
maps instead: a gather, with no decode. The copy is opt-in; it has no default directory, and
`datarax.pipeline.materialize(source, directory=...)` writes it: on the calling thread, or in the
RAM budget's worker processes for a read that decodes in Python, each worker writing its units'
rows in place. It returns a complete copy of the same source already in the directory as it is,
so every later run, and every other training process on the host, reuses it.

```python
from datarax.pipeline import materialize
from datarax.sources import from_tfds

materialize(from_tfds("cifar10", "train", in_memory=False), host_resources=resources,
            directory="/data/cifar10-train")
```

`open_decoded_copy(directory)` opens a finished copy as a `StreamingDiskSource`, an `INDEXED`
source serving what the source serves, record for record, its provenance by index:

```python
from datarax.sources import from_tfds, open_decoded_copy

# The split's copy, checked against the split it was written from:
source = from_tfds("cifar10", "train", in_memory=False, decoded_dir="/data/cifar10-train")

# The copy on its manifest alone, on a host without the TFDS files:
source = open_decoded_copy("/data/cifar10-train")
```

## Layout

| file | what it holds |
|---|---|
| `column-<i>.npy` | one numeric field of every record, at its stored dtype (an int64 label stays int64) |
| `provenance.array_record` | one record per record: its strings and objects as UTF-8 JSON, `bytes` base64-encoded |
| `manifest.json` | the format, record count, units, each column's field, file, dtype, shape and size, the source's identity, the decoder's libraries and the content digest |
| `.lock` | the writer's lock; it stays in a finished copy |

A file named `*.partial`, or a directory without `manifest.json`, is a copy being written or
whose writer stopped: every reader refuses it, naming it incomplete.

## Identity

A copy is checked against what its source declares (`source_identity()`), never against where
the source's files are:

- an ArrayRecord source: each file's base name and size, the record count and its decode;
- a TFDS split read per batch: also the dataset, version, config, split, the kept features,
  the SHA-256 of `features.json`, and the versions of the libraries that decode it
  (`tensorflow-datasets`, `protobuf` with its runtime, `numpy`, `pillow`, `opencv`). A decoder
  of another version decodes to other bytes in general, so it invalidates the copy;
- a `StreamingDiskSource`: its columns' fields, dtypes, shapes, offsets, files' base names and
  sizes, and the content digest of the copy it is.

A copy moved with its source to another directory or host still matches; a source file renamed,
resized or re-encoded does not, and touching a file's modification time alone changes nothing.
`open_decoded_copy(directory, identity=source.source_identity())` refuses a copy of another
source, naming every field that differs.

The copy is also identified by its own bytes: the manifest's content digest is the SHA-256 of the
units' SHA-256 digests, in unit order. `open_decoded_copy(directory, verify=True)` reads every
column and refuses a copy whose bytes no longer give it (a full read).

## Writing and reading safely

- One process writes a copy at a time (an exclusive `fcntl.flock` on `.lock`, freed by the kernel
  when its holder dies); a process that waited returns the copy the first one finished.
- A rebuild unlinks the old manifest first, so a reader opening the copy meanwhile is refused; a
  reader that already mapped the old files keeps reading them, and the rebuild writes new files.
- A copy read cold from disk needs concurrent readers to keep a GPU fed: a pipeline over it with
  `host_resources` reads it on several threads.
- A copy larger than the page cache, read in a shuffled order, costs a device page per record.
  `Pipeline(..., shuffle=True, block_order=BlockOrder(...))` reads it a few blocks of consecutive
  rows at a time
  ([data larger than the page cache](../user_guide/data_sources.md#data-larger-than-the-page-cache)).

## Refusals

- no `manifest.json` (an incomplete copy), or a copy of another format, naming both;
- a column file missing, or of another size than the manifest records;
- an identity that differs, naming the fields; a content digest that does not match, with
  `verify=True`;
- writing: a field whose dtype or record shape varies between records (a ragged field), a TFDS
  `Video` feature (decoded frames are far larger than their stored size), a provenance value JSON
  cannot hold exactly (text, bytes, numbers, bools and `None` are kept), a target on a networked
  or FUSE filesystem (writing there through shared memory maps is not verified; reading a
  finished copy from one is allowed), a target directory holding a file the format does not
  name, and a disk without the room: every file is reserved to its full size with
  `os.posix_fallocate` before a row is written, and the refusal names the column, the bytes
  needed and the bytes free.

## See Also

- [streaming_disk_source](streaming_disk_source.md) - The reader a copy opens as
- [tfds_source](tfds_source.md) - The TFDS split a copy is most often of

---

::: datarax.sources.decoded_copy
