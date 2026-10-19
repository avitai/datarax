"""A source's records decoded once into columns on disk, or in memory: the one per-unit writer.

A decoded copy holds an indexed source's records as they decode, one ``.npy`` file per numeric
field (its column) and the records' provenance in one ArrayRecord file of UTF-8 JSON, beside a
``manifest.json`` written last. A ragged field (its records differ in length along their first
axis) is two ``.npy`` files, its values ``column-k.values.npy`` of ``(T, *inner)`` and its int64
offsets ``column-k.offsets.npy`` of ``(N + 1,)``, the layout of Megatron-LM's indexed dataset and
levanter's ``JaggedArrayStore``; a source holding one is read field by field through its ragged
read, and its copy serves that read alone. It is read back by :func:`open_decoded_copy` as a
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
3. :func:`begin_copy` creates every ``.partial`` file fresh and reserves each fixed column's full
   size with ``os.posix_fallocate`` before a row is written, so a full disk is refused in the
   writing process (a sparse memory map written on a full disk is a ``SIGBUS``).
4. Each unit's fixed rows are written in place (:meth:`CopyFiles.rows`), its ragged fields
   appended in unit order by the training process (:meth:`CopyFiles.ragged`), and the unit
   digested (:func:`unit_digest`); the provenance is appended unit by unit
   (:meth:`CopyFiles.provenance`).
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

A copy also keeps the pack plans planned over it (``_plans.copy_plan``): one ``plan-<digest>``
directory per planning parameters and content, planned once and read by every later run through
memory maps; rebuilding the copy removes them.
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
    DiskRagged,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


def open_decoded_copy(  # noqa: DOC503 - _column raises the FileNotFoundError
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
    fields = [_field(root, entry) for entry in manifest["columns"]]
    if verify:
        verify_content(root, manifest, fields)
    stored = manifest["provenance"]
    provenance = (
        None
        if stored is None
        else DiskProvenance(file=str(root / stored["file"]), records=stored["records"])
    )
    return StreamingDiskSource(
        StreamingDiskSourceConfig(
            columns=tuple(f for f in fields if isinstance(f, DiskColumn)),
            ragged=tuple(f for f in fields if isinstance(f, DiskRagged)),
            provenance=provenance,
            content_digest=manifest["content"]["digest"],
        )
    )


def _field(root: Path, entry: Mapping[str, Any]) -> DiskColumn | DiskRagged:
    """A manifest column's field on disk: a fixed column, or a ragged field's values and offsets.

    Args:
        root: The copy's directory.
        entry: The manifest's entry.

    Returns:
        The field, its files checked against the manifest.
    """
    values = _column(root, entry["name"], entry["file"], entry["bytes"])
    offsets = entry["offsets"]
    if offsets is None:
        return values
    return DiskRagged(
        values=values, offsets=_column(root, entry["name"], offsets["file"], offsets["bytes"])
    )


def _column(root: Path, name: str, file: str, size: int) -> DiskColumn:
    """The ``.npy`` file ``file`` of the column ``name``, refused unless it is ``size`` bytes.

    Args:
        root: The copy's directory.
        name: The column's field.
        file: The file's name.
        size: Its bytes, as the manifest records them.

    Returns:
        The column.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If it holds another number of bytes.
    """
    path = root / file
    if not path.is_file():
        raise FileNotFoundError(f"{root}: column {name!r}'s file {file} is missing")
    held = path.stat().st_size
    if held != size:
        raise ValueError(
            f"{root}: column {name!r}'s file {file} holds {held} bytes, not the {size} its "
            "manifest records: rebuild the copy"
        )
    return DiskColumn.from_npy(name, path)


__all__ = ["open_decoded_copy"]
