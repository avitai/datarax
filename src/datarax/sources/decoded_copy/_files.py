"""A decoded copy's files: the lock, clearing an old copy, reserving, writing and finishing.

:class:`CopyLock` holds the copy's ``fcntl.flock`` for the whole write; :func:`clear_copy` removes
an old copy in an order safe for readers, its pack plans with it; :func:`begin_copy` creates every
``.partial`` file, reserving each fixed column; :func:`finish_copy` fsyncs and renames them and
writes the manifest last. A fixed column is written in place, through memory maps of its reserved
file (:meth:`CopyFiles.rows`); a ragged field's values and offsets are appended in unit order by
the training process (:class:`RaggedWriter`), so their total is never needed before the build. A
target on a networked or FUSE filesystem is refused for writing
(:func:`refuse_unsupported_filesystem`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from array_record.python.array_record_module import ArrayRecordWriter

from datarax.core.ragged import RaggedColumn
from datarax.sources._disk_provenance import encode_provenance, PROVENANCE_ENCODING
from datarax.sources.decoded_copy._layout import (
    COLUMN,
    ColumnLayout,
    COPY_FORMAT,
    DIGEST,
    LayoutColumn,
    LOCK,
    MANIFEST,
    MANIFEST_TMP,
    PARTIAL,
    PLAN,
    PROVENANCE,
    unit_count,
    UNIT_RECORDS,
    unit_rows,
)
from datarax.sources.decoded_copy._units import content_digest, raw_bytes


REFUSED_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "9p", "ceph", "glusterfs", "lustre", "gpfs", "beegfs", "fuse"}
)
"""Networked and FUSE filesystems, on which concurrent writes through shared maps are not
verified (any ``fuse.*`` type too)."""


def free_bytes(directory: Path) -> int:
    """The bytes the filesystem holding ``directory`` has free for an unprivileged writer."""
    stats = os.statvfs(directory)
    return stats.f_bavail * stats.f_frsize


def close_files(stack: contextlib.ExitStack, exc_info: tuple[Any, ...]) -> None:
    """Close the files ``stack`` holds; while an error propagates, keep that error.

    A build that fails (a full disk, a refused record) unwinds through its writers, whose own
    close may then fail for the same cause (array_record's writer flushes its last chunk on
    close). That second error would replace the first, so a caller sees the symptom, not the
    cause: it is added to the propagating error as a note instead (PEP 678). With no error
    propagating, a failing close raises as usual.

    Args:
        stack: The writer's open files.
        exc_info: What the ``with`` holding the writer exits with.

    Raises:
        OSError: If a close fails while no error propagates (a full disk on the last flush).
        RuntimeError: Likewise, from array_record's writer.
    """
    error = exc_info[1]
    if error is None:
        stack.__exit__(None, None, None)
        return
    try:
        stack.__exit__(*exc_info)
    except (OSError, RuntimeError) as secondary:
        if secondary is error:
            raise
        error.add_note(f"closing the decoded copy's files then failed too: {secondary}")


class ProvenanceWriter:
    """Appends records' provenance, in order, to one ArrayRecord file; a context manager.

    Records are written one per ArrayRecord record (``group_size:1``, so each is read alone by
    index). The writer is closed by the ``with`` that holds it.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Open the file for writing.

        Args:
            path: The file.
        """
        self._stack = contextlib.ExitStack()
        # array_record's writer has close() and no __enter__, so it is held by closing().
        self._writer = self._stack.enter_context(
            contextlib.closing(ArrayRecordWriter(os.fspath(path), "group_size:1"))
        )

    def write(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Append ``records``' provenance, in order.

        Args:
            records: One mapping per record.
        """
        for record in records:
            self._writer.write(encode_provenance(record))

    def __enter__(self) -> ProvenanceWriter:
        """The writer."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the file (see :func:`close_files`)."""
        close_files(self._stack, exc_info)


class CopyLock:
    """One writer per copy: an exclusive ``fcntl.flock`` on ``<copy>/.lock``; a context manager.

    A second writer waits until the first releases it. The kernel releases it when its holder
    dies, and a spawned worker does not inherit the descriptor, so a killed writer never leaves
    the copy locked. The file is never unlinked, so no process locks an unlinked inode. The
    ``filelock`` package wraps the same ``flock``, but is not a core dependency, unlinks the file
    on release and falls back to a lock file a dead holder does not free; the standard library's
    call is used instead.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        """Name the copy whose lock is taken on entry.

        Args:
            directory: The copy's directory, created if missing.
        """
        self._path = Path(directory) / LOCK
        self._stack = contextlib.ExitStack()

    def __enter__(self) -> CopyLock:
        """Wait for the lock and hold it.

        Returns:
            The lock.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._stack.enter_context(self._path.open("a+b"))
        fcntl.flock(handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Release the lock by closing its file, as the stack holding it exits."""
        self._stack.__exit__(*exc_info)


def refuse_unsupported_filesystem(directory: str | os.PathLike[str]) -> str:
    """Refuse a copy written to a networked or FUSE filesystem; return the target's type.

    The type is that of the mount whose mountpoint is the longest prefix of the target's real
    path, in whole path components, from ``psutil.disk_partitions(all=True)`` (every mount,
    ``tmpfs``, FUSE and NFS included).

    Args:
        directory: Where the copy is to be written.

    Returns:
        The target's filesystem type.

    Raises:
        ValueError: If the type is networked or FUSE, naming it.
    """
    target = Path(os.path.realpath(directory))
    best: tuple[int, str] = (-1, "")
    for partition in psutil.disk_partitions(all=True):
        mount = Path(partition.mountpoint)
        if (target == mount or mount in target.parents) and len(mount.parts) > best[0]:
            best = (len(mount.parts), partition.fstype)
    kind = best[1]
    if kind in REFUSED_FILESYSTEMS or kind.startswith("fuse."):
        raise ValueError(
            f"{directory} is on a {kind} filesystem, where workers writing a copy's rows in place "
            "through shared memory maps are not known to be safe: write the copy on a local disk "
            "and copy the finished directory there; reading a finished copy from it is allowed"
        )
    return kind


def is_copy_file(name: str) -> bool:
    """Whether ``name`` is a file a decoded copy's directory holds.

    Args:
        name: A file name.

    Returns:
        Whether the copy format names it.
    """
    base = name.removesuffix(PARTIAL)
    return (
        name in {MANIFEST, MANIFEST_TMP, LOCK}
        or base == PROVENANCE
        or bool(COLUMN.fullmatch(base) or PLAN.fullmatch(base))
    )


@contextlib.contextmanager
def read_descriptor(path: Path) -> Iterator[int]:
    """A read-only descriptor of ``path`` (a directory too), closed when the block ends."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def fsync_directory(directory: Path) -> None:
    """Make the names ``directory`` holds durable: ``fsync`` it."""
    with read_descriptor(directory) as descriptor:
        os.fsync(descriptor)


def fsync_file(path: Path) -> None:
    """Make ``path`` durable: ``fsync`` every dirty page of its inode, whoever mapped it."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def clear_copy(directory: str | os.PathLike[str]) -> None:
    """Remove the copy in ``directory``, in an order safe for its readers; ``.lock`` stays.

    A file the format does not name is refused before anything is removed. The manifest goes
    first and the directory is fsynced, so a reader starting now is refused (an incomplete
    copy); then every other file the format names, and every pack plan (planned over the old
    content, so stale). A reader that already mapped them keeps reading their old inodes.

    Args:
        directory: The copy's directory.

    Raises:
        ValueError: If the directory holds a file the copy format does not name, naming it.
    """
    root = Path(directory)
    if not root.exists():
        return
    names = sorted(entry.name for entry in root.iterdir())
    foreign = [name for name in names if not is_copy_file(name)]
    if foreign:
        raise ValueError(
            f"{root} holds {foreign}, which no decoded copy writes: a copy's directory holds the "
            "copy alone; choose an empty directory or remove them"
        )
    if MANIFEST in names:
        (root / MANIFEST).unlink()
        fsync_directory(root)
    for name in names:
        if name in {MANIFEST, LOCK}:
            continue
        if (root / name).is_dir():  # a pack plan
            shutil.rmtree(root / name)
        else:
            (root / name).unlink()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CopyFiles:
    """The ``.partial`` files of a copy being written; it pickles, for a worker process.

    Attributes:
        directory: The copy's directory.
        layout: Its columns.
        unit_records: Records per unit.
        offsets: Each column file's header length, the byte offset of its first row, in
            :meth:`column_files` order.
    """

    directory: Path
    layout: ColumnLayout
    unit_records: int
    offsets: tuple[int, ...]

    @property
    def units(self) -> int:
        """The copy's units."""
        return unit_count(self.layout.length, self.unit_records)

    def unit_rows(self, unit: int) -> range:
        """The records of ``unit``."""
        return unit_rows(unit, self.layout.length, self.unit_records)

    def column_files(self) -> tuple[str, ...]:
        """Each column's final file names, in layout order: a ragged field's values, offsets."""
        return tuple(
            name
            for position, column in enumerate(self.layout.columns)
            for name in column_names(position, column)
        )

    def header_lengths(self) -> dict[str, int]:
        """Each column file's header length, by final name."""
        return dict(zip(self.column_files(), self.offsets, strict=True))

    @contextlib.contextmanager
    def rows(self, unit: int) -> Iterator[dict[str, np.ndarray]]:
        """Map ``unit``'s rows of every column's ``.partial`` file for writing.

        Args:
            unit: The unit.

        Yields:
            One writable memory map per column, keyed by field path in layout order; released
            when the block ends.
        """
        rows = self.unit_rows(unit)
        headers = self.header_lengths()
        maps: dict[str, np.ndarray] = {}
        try:
            for position, column in enumerate(self.layout.columns):
                if column.ragged:
                    continue
                (name,) = column_names(position, column)
                maps[column.name] = np.memmap(
                    self.directory / f"{name}{PARTIAL}",
                    dtype=np.dtype(column.dtype),
                    mode="r+",
                    offset=headers[name] + rows.start * column.row_bytes,
                    shape=(len(rows), *column.shape),
                )
            yield maps
        finally:
            del maps  # numpy's memmap has no close: dropping the maps unmaps them

    def provenance(self) -> ProvenanceWriter:
        """The writer of the records' provenance, a context manager over its ``.partial`` file.

        Returns:
            The writer.
        """
        return ProvenanceWriter(self.directory / f"{PROVENANCE}{PARTIAL}")

    def ragged(self) -> RaggedWriter:
        """The appender of the ragged fields' values and offsets, a context manager.

        Returns:
            The writer, over the ragged fields' ``.partial`` files.
        """
        return RaggedWriter(self)


def column_names(position: int, column: LayoutColumn) -> tuple[str, ...]:
    """The file names of the layout's column at ``position``: a ragged field's values, offsets.

    Args:
        position: The column's place in the layout.
        column: The column.

    Returns:
        ``column-k.npy``, or ``column-k.values.npy`` and ``column-k.offsets.npy``.
    """
    if column.ragged:
        return f"column-{position}.values.npy", f"column-{position}.offsets.npy"
    return (f"column-{position}.npy",)


class RaggedWriter:
    """Appends each unit's ragged fields, in unit order, to their ``.partial`` files.

    Values are appended as they come and each unit's offsets shifted by the cells already
    written, as levanter's ``JaggedArrayStore`` appends batch by batch, so the total is never
    needed before the build. The files are written by ``write``, never mapped, so a full disk
    raises an ``OSError`` here rather than a ``SIGBUS``. Held by the training process alone; a
    context manager.
    """

    def __init__(self, files: CopyFiles) -> None:
        """Open every ragged field's two ``.partial`` files for appending.

        Args:
            files: The copy's files.
        """
        self._stack = contextlib.ExitStack()
        self._directory = files.directory
        self._files: dict[str, tuple[Any, Any]] = {}
        self._cells: dict[str, int] = {}
        for position, column in enumerate(files.layout.columns):
            if not column.ragged:
                continue
            values, offsets = (
                self._stack.enter_context((files.directory / f"{name}{PARTIAL}").open("ab"))
                for name in column_names(position, column)
            )
            self._files[column.name] = (values, offsets)
            self._cells[column.name] = 0

    def write(self, parts: Mapping[str, RaggedColumn]) -> None:
        """Append one unit's ragged fields, the next unit in order.

        Args:
            parts: Each ragged field's values and offsets from 0, by path.

        Raises:
            OSError: If the filesystem cannot hold them, naming the directory, the field and the
                bytes free.
        """
        for name, (values, offsets) in self._files.items():
            part = parts[name]
            try:
                values.write(raw_bytes(part.values))
                offsets.write(
                    (np.asarray(part.offsets[1:], np.int64) + self._cells[name]).tobytes()
                )
            except OSError as error:
                raise OSError(
                    error.errno,
                    f"the decoded copy in {self._directory} cannot append ragged field {name!r}: "
                    f"the filesystem has {free_bytes(self._directory)} bytes free "
                    f"({os.strerror(error.errno or 0)})",
                ) from error
            self._cells[name] += len(part.values)

    def __enter__(self) -> RaggedWriter:
        """The writer."""
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Close the files (see :func:`close_files`)."""
        close_files(self._stack, exc_info)


def reserve(path: Path, size: int, column: str, directory: Path) -> None:
    """Allocate ``size`` bytes of ``path`` on disk now, refusing a disk without them.

    Args:
        path: The file, created.
        size: Its full size.
        column: The column it holds, for the refusal.
        directory: The copy's directory, for the refusal.

    Raises:
        OSError: If the filesystem cannot hold them (or has no ``posix_fallocate``), naming the
            directory, the column, the bytes needed and the bytes free.
    """
    reserve = getattr(os, "posix_fallocate", None)
    if reserve is None:
        raise OSError(
            errno.ENOTSUP,
            f"{sys.platform} has no os.posix_fallocate, with which a decoded copy reserves its "
            "files before writing them in place (an unreserved map written on a full disk "
            "kills its writer): write the copy on Linux, or build in memory",
        )
    with path.open("r+b") as handle:
        try:
            reserve(handle.fileno(), 0, size)
        except OSError as error:
            raise OSError(
                error.errno,
                f"the decoded copy in {directory} cannot reserve column {column!r}: it needs "
                f"{size} bytes, and the filesystem has {free_bytes(directory)} bytes free "
                f"({os.strerror(error.errno or 0)})",
            ) from error


def npy_header(dtype: str, shape: tuple[int, ...]) -> dict[str, Any]:
    """The ``.npy`` header of a C-ordered array of ``dtype`` and ``shape``."""
    return {
        "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": shape,
    }


def create_column(path: Path, dtype: str, shape: tuple[int, ...]) -> int:
    """Create ``path`` fresh with its ``.npy`` header; return the header's length.

    numpy's writer leaves room in the header for the first axis to grow to any 64-bit size
    (``numpy.lib.format.GROWTH_AXIS_MAX_DIGITS``), so a ragged field's values file, created with
    no cell, has its header rewritten in place with its final length (:func:`finish_copy`).
    """
    path.unlink(missing_ok=True)
    with path.open("xb") as handle:
        np.lib.format.write_array_header_1_0(handle, npy_header(dtype, shape))
        return handle.tell()


def create_files(root: Path, layout: ColumnLayout, created: list[Path]) -> list[int]:  # noqa: DOC502 - reserve raises the OSError
    """Create every column's ``.partial`` file fresh, a fixed column's reserved to its size.

    A ragged field's values file starts with no cell; its offsets file holds its first offset,
    0. Each path is added to ``created`` as soon as it exists.

    Args:
        root: The copy's directory.
        layout: Its columns.
        created: The files created so far, extended here.

    Returns:
        Each file's header length, in :meth:`CopyFiles.column_files` order.

    Raises:
        OSError: If a fixed column cannot be reserved.
    """
    offsets: list[int] = []
    for position, column in enumerate(layout.columns):
        names = column_names(position, column)
        paths = [root / f"{name}{PARTIAL}" for name in names]
        if not column.ragged:
            offset = create_column(paths[0], column.dtype, (layout.length, *column.shape))
            created.append(paths[0])
            reserve(paths[0], offset + layout.length * column.row_bytes, column.name, root)
            offsets.append(offset)
            continue
        offsets.append(create_column(paths[0], column.dtype, (0, *column.shape)))
        created.append(paths[0])
        offsets.append(create_column(paths[1], "<i8", (layout.length + 1,)))
        created.append(paths[1])
        with paths[1].open("ab") as handle:
            handle.write(np.zeros(1, np.int64).tobytes())
    return offsets


@contextlib.contextmanager
def begin_copy(  # noqa: DOC503 - reserve raises the OSError
    directory: str | os.PathLike[str], layout: ColumnLayout, *, unit_records: int = UNIT_RECORDS
) -> Iterator[CopyFiles]:
    """Create a copy's ``.partial`` files, each fresh, a fixed column's reserved to its full size.

    Every file is a new inode with its ``.npy`` header; a fixed column's is reserved with
    ``os.posix_fallocate`` before the block runs, so a disk without the room is refused here (a
    ragged field's files are appended, and a full disk refuses an append). When the block ends,
    every ``.partial`` still present is unlinked: :func:`finish_copy`, called inside it, has renamed
    those of a finished copy, so only an unfinished copy's are removed.

    Args:
        directory: The copy's directory, created if missing.
        layout: Its columns.
        unit_records: Records per unit.

    Yields:
        The copy's files.

    Raises:
        FileExistsError: If the directory holds a finished copy (clear it first).
        OSError: If a file cannot be reserved, naming the directory, the column, the bytes
            needed and the bytes free.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    if (root / MANIFEST).exists():
        raise FileExistsError(f"{root} holds a finished copy's manifest: clear it first")
    created: list[Path] = []
    try:
        offsets = create_files(root, layout, created)
        yield CopyFiles(
            directory=root, layout=layout, unit_records=unit_records, offsets=tuple(offsets)
        )
    finally:
        for path in [*created, root / f"{PROVENANCE}{PARTIAL}"]:
            if path.exists():  # a finished copy's were renamed
                path.unlink()


def complete_values_headers(files: CopyFiles) -> dict[str, int]:
    """Rewrite each ragged field's values header with its cells, in place; return the cells.

    Args:
        files: The copy's files, every unit appended.

    Returns:
        Each ragged field's cell count, by path.

    Raises:
        ValueError: If a header would change length, or a file holds a partial cell.
    """
    headers = files.header_lengths()
    cells: dict[str, int] = {}
    for position, column in enumerate(files.layout.columns):
        if not column.ragged:
            continue
        name = column_names(position, column)[0]
        path = files.directory / f"{name}{PARTIAL}"
        stored = path.stat().st_size - headers[name]
        count, rest = divmod(stored, column.row_bytes) if column.row_bytes else (0, stored)
        if rest:
            raise ValueError(f"{path} holds {stored} bytes of cells of {column.row_bytes} bytes")
        with path.open("r+b") as handle:
            np.lib.format.write_array_header_1_0(
                handle, npy_header(column.dtype, (count, *column.shape))
            )
            if handle.tell() != headers[name]:
                raise ValueError(
                    f"{path}'s header grew from {headers[name]} to {handle.tell()} bytes"
                )
        cells[column.name] = count
    return cells


def manifest_column(
    root: Path, position: int, column: LayoutColumn, length: int, cells: Mapping[str, int]
) -> dict[str, Any]:
    """The manifest's entry for a column: its file and shape, and a ragged field's offsets.

    Args:
        root: The copy's directory, its files renamed.
        position: The column's place in the layout.
        column: The column.
        length: The copy's records.
        cells: Each ragged field's cell count.

    Returns:
        ``name``, ``file``, ``dtype``, ``shape`` (the file's array), ``bytes`` and ``offsets``
        (``None`` for a fixed field; a ragged field's offsets ``file`` and ``bytes``).
    """
    names = column_names(position, column)
    leading = cells[column.name] if column.ragged else length
    return {
        "name": column.name,
        "file": names[0],
        "dtype": column.dtype,
        "shape": [leading, *column.shape],
        "bytes": (root / names[0]).stat().st_size,
        "offsets": (
            {"file": names[1], "bytes": (root / names[1]).stat().st_size} if column.ragged else None
        ),
    }


def finish_copy(
    files: CopyFiles, *, unit_digests: Sequence[bytes], identity: Mapping[str, Any]
) -> None:
    """Make a written copy durable and complete: its files renamed, its manifest written last.

    A ragged field's values header is rewritten in place with the cells appended. Each
    ``.partial`` (the columns and the provenance) is fsynced once and renamed, the directory
    fsynced; the manifest is written to ``manifest.json.tmp``, fsynced, replaced into place, and
    the directory fsynced last. The provenance and ragged writers are closed first, by their
    ``with``.

    Args:
        files: The copy's files, every unit written.
        unit_digests: Each unit's digest, in unit order.
        identity: What the source declares (``source_identity()``).

    Raises:
        ValueError: If there is not one digest per unit.
    """
    if len(unit_digests) != files.units:
        raise ValueError(f"{len(unit_digests)} unit digests for a copy of {files.units} units")
    root = files.directory
    cells = complete_values_headers(files)
    names = [*files.column_files()]
    provenance_partial = root / f"{PROVENANCE}{PARTIAL}"
    if provenance_partial.exists():
        names.append(PROVENANCE)
    for name in names:
        fsync_file(root / f"{name}{PARTIAL}")
        (root / f"{name}{PARTIAL}").rename(root / name)
    fsync_directory(root)
    manifest = {
        "format": COPY_FORMAT,
        "length": files.layout.length,
        "unit_records": files.unit_records,
        "columns": [
            manifest_column(root, position, column, files.layout.length, cells)
            for position, column in enumerate(files.layout.columns)
        ],
        "provenance": (
            {"file": PROVENANCE, "records": files.layout.length, "encoding": PROVENANCE_ENCODING}
            if PROVENANCE in names
            else None
        ),
        "source": identity["source"],
        "decoder": identity["decoder"],
        "content": {"algorithm": DIGEST, "digest": content_digest(unit_digests)},
    }
    temporary = root / MANIFEST_TMP
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(root / MANIFEST)
    fsync_directory(root)
