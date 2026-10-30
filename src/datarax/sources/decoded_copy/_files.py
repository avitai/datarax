"""A decoded copy's files: the lock, clearing an old copy, reserving, writing and finishing.

:class:`CopyLock` holds the copy's ``fcntl.flock`` for the whole write; :func:`clear_copy` removes
an old copy in an order safe for readers; :func:`begin_copy` creates and reserves every ``.partial``
file; :func:`finish_copy` fsyncs and renames them and writes the manifest last. A target on a
networked or FUSE filesystem is refused for writing (:func:`refuse_unsupported_filesystem`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import fcntl
import json
import os
import struct
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from array_record.python.array_record_module import ArrayRecordWriter

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
    PROVENANCE,
    unit_count,
    UNIT_RECORDS,
    unit_rows,
)
from datarax.sources.decoded_copy._units import content_digest


REFUSED_FILESYSTEMS = frozenset(
    {"nfs", "nfs4", "cifs", "smb3", "9p", "ceph", "glusterfs", "lustre", "gpfs", "beegfs", "fuse"}
    # macOS's names for its networked and FUSE mounts, as mpv's stream/stream_file.c lists them.
    | {"smbfs", "afpfs", "webdav", "macfuse", "osxfuse", "osxfusefs"}
)
"""Networked and FUSE filesystems, on which concurrent writes through shared maps are not
verified (any ``fuse.*`` or ``fusefs.*`` type too)."""
_REFUSED_PREFIXES = ("fuse.", "fusefs.")

# macOS has no posix_fallocate; fcntl(F_PREALLOCATE) reserves the blocks. CPython's fcntl module
# exports none of these names (fcntlmodule.c on main, 3.12 and 3.13), and the one PyPI binding,
# `fallocate`, last released in 2018, so the values are xnu's (bsd/sys/fcntl.h). The call follows
# rustix's Apple `fallocate` and fs4-rs's `allocate`, which skips blocks already reserved.
_F_PREALLOCATE = 42
_F_ALLOCATEALL = 0x4
"""All of the requested space or none of it."""
_F_PEOFPOSMODE = 3
"""The region starts at the file's physical end."""
_FSTORE = struct.Struct("Iiqqq")
"""xnu's ``fstore_t``: flags, position mode, offset, length, bytes allocated (written back)."""


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
        """Close the file, as the stack holding it exits."""
        self._stack.__exit__(*exc_info)


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
    if kind in REFUSED_FILESYSTEMS or kind.startswith(_REFUSED_PREFIXES):
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
        name in {MANIFEST, MANIFEST_TMP, LOCK} or base == PROVENANCE or bool(COLUMN.fullmatch(base))
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
    copy); then every other file the format names. A reader that already mapped them keeps
    reading their old inodes.

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
        if name not in {MANIFEST, LOCK}:
            (root / name).unlink()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CopyFiles:
    """The ``.partial`` files of a copy being written; it pickles, for a worker process.

    Attributes:
        directory: The copy's directory.
        layout: Its columns.
        unit_records: Records per unit.
        offsets: Each column file's header length, the byte offset of its first row.
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
        """Each column's final file name, in layout order."""
        return tuple(f"column-{position}.npy" for position in range(len(self.layout.columns)))

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
        maps: dict[str, np.ndarray] = {}
        try:
            for column, name, offset in zip(
                self.layout.columns, self.column_files(), self.offsets, strict=True
            ):
                maps[column.name] = np.memmap(
                    self.directory / f"{name}{PARTIAL}",
                    dtype=np.dtype(column.dtype),
                    mode="r+",
                    offset=offset + rows.start * column.row_bytes,
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


def reserve(path: Path, size: int, column: str, directory: Path) -> None:
    """Allocate ``size`` bytes of ``path`` on disk now, refusing a disk without them.

    Args:
        path: The file, created.
        size: Its full size.
        column: The column it holds, for the refusal.
        directory: The copy's directory, for the refusal.

    Raises:
        OSError: If the filesystem cannot hold them, naming the directory, the column, the bytes
            needed and the bytes free; or if the platform has no way to reserve them.
    """
    allocate = _reservation()
    if allocate is None:
        raise OSError(
            errno.ENOTSUP,
            f"{sys.platform} has neither os.posix_fallocate nor macOS's F_PREALLOCATE, with which "
            "a decoded copy reserves its files before writing them in place (an unreserved map "
            "written on a full disk kills its writer): write the copy on Linux or macOS, or build "
            "in memory",
        )
    with path.open("r+b") as handle:
        try:
            allocate(handle.fileno(), size)
        except OSError as error:
            stats = os.statvfs(directory)
            raise OSError(
                error.errno,
                f"the decoded copy in {directory} cannot reserve column {column!r}: it needs "
                f"{size} bytes, and the filesystem has {stats.f_bavail * stats.f_frsize} bytes "
                f"free ({os.strerror(error.errno or 0)})",
            ) from error


def _reservation() -> Callable[[int, int], None] | None:
    """The call that allocates a file's first bytes on this platform, or ``None`` if it has none."""
    fallocate = getattr(os, "posix_fallocate", None)
    if fallocate is not None:
        return lambda fd, size: fallocate(fd, 0, size)
    if sys.platform == "darwin":
        return _preallocate
    return None


def _preallocate(fd: int, size: int) -> None:
    """Allocate the first ``size`` bytes of ``fd`` on macOS, then make the file that long.

    Blocks already allocated are not asked for again: macOS can answer ENOSPC to that.
    """
    if os.fstat(fd).st_blocks * 512 >= size:
        return
    request = _FSTORE.pack(_F_ALLOCATEALL, _F_PEOFPOSMODE, 0, size, 0)
    fcntl.fcntl(fd, _F_PREALLOCATE, request)
    os.ftruncate(fd, size)


def create_column(path: Path, column: LayoutColumn, length: int) -> int:
    """Create ``path`` fresh with its ``.npy`` header; return the header's length."""
    path.unlink(missing_ok=True)
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(column.dtype)),
        "fortran_order": False,
        "shape": (length, *column.shape),
    }
    with path.open("xb") as handle:
        np.lib.format.write_array_header_1_0(handle, header)
        return handle.tell()


@contextlib.contextmanager
def begin_copy(  # noqa: DOC503 - reserve raises the OSError
    directory: str | os.PathLike[str], layout: ColumnLayout, *, unit_records: int = UNIT_RECORDS
) -> Iterator[CopyFiles]:
    """Create a copy's ``.partial`` files, each fresh and reserved to its full size.

    Every file is a new inode with its ``.npy`` header, reserved (:func:`reserve`) before the
    block runs, so a disk without the room is refused here. When the block ends, every
    ``.partial`` still present is unlinked: :func:`finish_copy`, called inside it, has renamed
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
        offsets = []
        for position, column in enumerate(layout.columns):
            path = root / f"column-{position}.npy{PARTIAL}"
            offset = create_column(path, column, layout.length)
            created.append(path)
            reserve(path, offset + layout.length * column.row_bytes, column.name, root)
            offsets.append(offset)
        yield CopyFiles(
            directory=root, layout=layout, unit_records=unit_records, offsets=tuple(offsets)
        )
    finally:
        for path in [*created, root / f"{PROVENANCE}{PARTIAL}"]:
            if path.exists():  # a finished copy's were renamed
                path.unlink()


def finish_copy(
    files: CopyFiles, *, unit_digests: Sequence[bytes], identity: Mapping[str, Any]
) -> None:
    """Make a written copy durable and complete: its files renamed, its manifest written last.

    Each ``.partial`` (the columns and the provenance) is fsynced once and renamed, the
    directory fsynced; the manifest is written to ``manifest.json.tmp``, fsynced, replaced into
    place, and the directory fsynced last. The provenance writer is closed first, by its ``with``.

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
            {
                "name": column.name,
                "file": name,
                "dtype": column.dtype,
                "shape": [files.layout.length, *column.shape],
                "bytes": (root / name).stat().st_size,
            }
            for column, name in zip(files.layout.columns, files.column_files(), strict=True)
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
