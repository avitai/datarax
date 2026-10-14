"""A source's records decoded once into columns on disk, or in memory: the one per-unit writer.

A decoded copy holds an indexed source's records as they decode, one ``.npy`` file per numeric
field (its column) and the records' provenance in one ArrayRecord file of UTF-8 JSON, beside a
``manifest.json`` written last. It is read back by :func:`open_decoded_copy` as a
:class:`~datarax.sources.streaming_disk_source.StreamingDiskSource`, which serves what the source
serves, record for record, from memory maps. The copy is opt-in and has no default directory;
``datarax.pipeline.materialize`` writes one, through the training process's Grain workers or on
the calling thread.

Records are decoded unit by unit (:data:`UNIT_RECORDS` records) by one writer,
:func:`write_unit`: it reads the records with the source's own host read and assigns each field
to its rows of a given column, refusing a field unlike the copy's :class:`ColumnLayout` instead of
casting it. The same writer fills in-memory columns (:func:`build_in_memory`, the one-thread
build ``TFDSEagerSource`` uses) and a copy's memory-mapped ``.partial`` files, so a source has one
decode path.

Writing a copy follows one protocol:

1. :class:`CopyLock` holds an exclusive ``fcntl.flock`` on ``<copy>/.lock`` for the whole write,
   so one process writes a copy at a time; the kernel frees it when its holder dies.
2. :func:`clear_copy` removes an old copy in an order safe for readers: its manifest first, the
   directory fsynced, then its other files; a reader that mapped them keeps their old inodes.
3. :func:`begin_copy` creates every ``.partial`` file fresh and reserves its full size with
   ``os.posix_fallocate`` before a row is written, so a full disk is refused in the writing
   process (a sparse memory map written on a full disk is a ``SIGBUS``).
4. Each unit's rows are written in place (:meth:`CopyFiles.rows`) and digested
   (:func:`unit_digest`); the provenance is appended unit by unit (:meth:`CopyFiles.provenance`).
5. :func:`finish_copy` fsyncs each ``.partial`` once, renames it, fsyncs the directory, and writes
   the manifest through a fsynced ``manifest.json.tmp`` replaced into place, fsyncing the
   directory last. A directory without a manifest is an incomplete copy, refused by every reader.

A copy is identified by what its source declares
(:class:`~datarax.core.data_source.SourceIdentified`), never by where the source's files are, and
by its own bytes: the manifest's content digest is the SHA-256 of the units' SHA-256 digests in
unit order. A copy moved to another directory or host still matches its
source; an edited source or a decoder library of another version does not.

Rows are written through shared memory maps, which is safe for disjoint writers on a local disk;
a target on a networked or FUSE filesystem is refused for writing
(:func:`refuse_unsupported_filesystem`), while a finished copy is read from anywhere.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datarax.sources._disk_provenance import DiskProvenance
from datarax.sources.decoded_copy._manifest import (
    identity_differences,
    read_manifest,
    verify_content,
)
from datarax.sources.streaming_disk_source import (
    DiskColumn,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


def open_decoded_copy(
    directory: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any] | None = None,
    verify: bool = False,
) -> StreamingDiskSource:
    """Open a finished decoded copy as a ``StreamingDiskSource``, with or without its source.

    Args:
        directory: The copy's directory, on any filesystem.
        identity: The source's ``source_identity()``, which the copy must have been written
            from; ``None`` opens the copy on its manifest alone (a copy carried to a host
            without its source).
        verify: Whether every column is read to check the content digest.

    Returns:
        An ``INDEXED`` source serving the copy's records and their provenance by index.

    Raises:
        FileNotFoundError: If the manifest or a column's file is missing.
        ValueError: If the copy is of another format, a file's size is not the manifest's, the
            identity differs (naming the fields), or the content digest does not match.
    """
    root = Path(directory)
    manifest = read_manifest(root)
    if identity is not None:
        differing = identity_differences(manifest, identity)
        if differing:
            raise ValueError(
                f"{root} is a copy of another source than the one given: it differs in "
                f"{differing}; rebuild it with datarax.pipeline.materialize(..., rebuild=True)"
            )
    columns = []
    for entry in manifest["columns"]:
        path = root / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(
                f"{root}: column {entry['name']!r}'s file {entry['file']} is missing"
            )
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(
                f"{root}: column {entry['name']!r}'s file {entry['file']} holds {size} bytes, "
                f"not the {entry['bytes']} its manifest records: rebuild the copy"
            )
        column = DiskColumn.from_npy(entry["name"], path)
        columns.append(column)
    if verify:
        verify_content(root, manifest, columns)
    stored = manifest["provenance"]
    provenance = (
        None
        if stored is None
        else DiskProvenance(file=str(root / stored["file"]), records=stored["records"])
    )
    return StreamingDiskSource(
        StreamingDiskSourceConfig(
            columns=tuple(columns),
            provenance=provenance,
            content_digest=manifest["content"]["digest"],
        )
    )


__all__ = ["open_decoded_copy"]
