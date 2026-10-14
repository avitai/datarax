"""Reading TFRecord shard files without TensorFlow: frames, their payloads and their CRCs.

A TFRecord file is a sequence of frames, each a payload's length, the length's masked CRC32C, the
payload and the payload's masked CRC32C. :class:`ShardIndex` holds where each record's payload sits
in its file, built once from the frame headers (:func:`record_index`); a payload is read at its
offset and checked against its CRC (:func:`payload`), as tf.data's TFRecord reader checks it, and
a damaged frame raises :class:`DamagedRecordError`. A run read by worker processes shares one
read-only copy of its indexes (:func:`share_index`). A record's id holds its shard's number in the
high word and its offset in the shard in the low word (:func:`split_shard`).

Imports no TensorFlow Datasets; ``google_crc32c`` comes with the ``data`` extra.
"""

from __future__ import annotations

import contextlib
import functools
import os
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


try:
    import google_crc32c
except ImportError as error:
    msg = 'datarax.sources._tfrecord needs google_crc32c: uv pip install "datarax[data]"'
    raise ImportError(msg) from error


LARGEST_OFFSET = 2**32 - 1
"""The largest record offset in a shard file: an id keeps the offset in its low word."""

LARGEST_SHARD = 2**31 - 1
"""The largest shard number: an id keeps it in its high word, whose top bit marks no record."""

TFRECORD_HEADER = struct.Struct("<QI")
"""A TFRecord frame's header: the payload's length (uint64) and its masked CRC (uint32)."""

TFRECORD_CRC = struct.Struct("<I")
"""A TFRecord frame's trailer, after the payload: the payload's masked CRC (uint32)."""

CRC_MASK_DELTA = 0xA282EAD8
"""The constant TFRecord adds to a rotated CRC32C to mask it (TensorFlow's ``crc32c::Mask``)."""


@dataclass(frozen=True, slots=True)
class Shard:
    """The records of one shard file a split reads: offsets ``skip .. skip + take``."""

    path: str
    shard: int
    skip: int
    take: int


@dataclass(frozen=True, slots=True, eq=False)
class ShardIndex:
    """Where each record of a TFRecord shard file sits: its payload's byte offset and length.

    An index read from the copy shared with worker processes (:func:`share_index`) pickles as
    its place in that copy, so a worker maps the one copy instead of receiving its own.

    Attributes:
        payloads: int64 byte offset of each record's payload in the file, by record offset.
        lengths: int64 byte length of each record's payload, by record offset.
        shared: ``(file, first)``: the shared copy holding the index and its first entry there,
            or ``None`` for an index held by this process alone.
    """

    payloads: np.ndarray
    lengths: np.ndarray
    shared: tuple[str, int] | None = None

    def __reduce__(self) -> tuple[Any, ...]:
        """A shared index by its place in the shared copy; any other by its arrays."""
        if self.shared is None:
            return ShardIndex, (self.payloads, self.lengths)
        path, first = self.shared
        return shared_shard_index, (path, first, len(self.lengths))

    @property
    def nbytes(self) -> int:
        """The bytes of its two arrays."""
        return self.payloads.nbytes + self.lengths.nbytes


# Memory every process maps; mkstemp makes each copy's file unique and its owner's only.
SHARED_DIRECTORY = Path("/dev/shm")  # nosec B108


def share_index(indexes: Sequence[ShardIndex]) -> tuple[str, tuple[ShardIndex, ...]]:
    """One read-only copy of ``indexes`` for every worker of a run: a file in shared memory.

    The offsets and lengths are written once, as one ``(2, records)`` int64 ``.npy`` file in
    ``/dev/shm`` (memory every process maps once) or the temporary directory where there is none,
    and read back as a memory map of it, which replaces the indexes this process held. The map
    is read through once, so the copy is resident in this process, as its own copy was.

    Args:
        indexes: The split's shard indexes, in shard order.

    Returns:
        The file's path, and each index as a view of the one copy.
    """
    total = sum(len(index.lengths) for index in indexes)
    directory = SHARED_DIRECTORY if SHARED_DIRECTORY.is_dir() else Path(tempfile.gettempdir())
    descriptor, path = tempfile.mkstemp(prefix="datarax-tfds-index-", suffix=".npy", dir=directory)
    os.close(descriptor)
    table = np.lib.format.open_memmap(path, mode="w+", dtype=np.int64, shape=(2, total))
    first = 0
    for index in indexes:
        count = len(index.lengths)
        table[0, first : first + count] = index.payloads
        table[1, first : first + count] = index.lengths
        first += count
    table.flush()
    del table
    mapped = np.load(path, mmap_mode="r")
    int(np.asarray(mapped).sum())  # every page mapped in, so this process holds the one copy
    shared, first = [], 0
    for index in indexes:
        count = len(index.lengths)
        shared.append(
            ShardIndex(
                mapped[0, first : first + count], mapped[1, first : first + count], (path, first)
            )
        )
        first += count
    return path, tuple(shared)


@functools.cache
def mapped_index(path: str) -> np.ndarray:
    """A worker's one memory map of a shared index copy, however many shards read from it."""
    return np.load(path, mmap_mode="r")


def shared_shard_index(path: str, first: int, count: int) -> ShardIndex:
    """A shard's index read from the shared copy at ``path``: what a worker unpickles."""
    table = mapped_index(path)
    payloads, lengths = table[0, first : first + count], table[1, first : first + count]
    return ShardIndex(payloads, lengths, (path, first))


def remove(path: str) -> None:
    """Remove a shared index copy once its source is gone."""
    with contextlib.suppress(FileNotFoundError):
        Path(path).unlink()


def pread(file: Any, size: int, position: int) -> bytes:
    """Read ``size`` bytes of an open file from ``position``, with no buffering around it."""
    return os.pread(file.fileno(), size, position)


def masked_crc(data: bytes) -> int:
    """TFRecord's masked CRC32C of ``data``: rotated right by 15 bits, plus a constant."""
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + CRC_MASK_DELTA) & 0xFFFFFFFF


class DamagedRecordError(ValueError):
    """A TFRecord frame that fails tf.data's checks: its CRCs, or a file ending inside it."""

    def __init__(self, path: str, record: int, damage: str) -> None:
        """Name the shard file, the record and the damage.

        Args:
            path: The shard file.
            record: The record's offset in it.
            damage: What is wrong with the frame.
        """
        super().__init__(f"shard file {path}: record {record} {damage}; the file is damaged")


def record_index(path: str) -> ShardIndex:
    """The payload offset and length of every record in a TFRecord file, from its frame headers.

    Neither TFDS, Grain nor ArrayRecord indexes a TFRecord file (searched: ``tfds.core.reader``,
    ``grain.experimental.TFRecordIterDataset``, ``array_record``), so the 12-byte frame headers
    are read here, each with one unbuffered positioned read, the payloads skipped. Each length is
    checked against its masked CRC32C and each frame against the file's size, as tf.data's
    TFRecord reader checks them.

    Args:
        path: The shard file.

    Returns:
        The shard's index.

    Raises:
        DamagedRecordError: If a frame's length fails its CRC check or the file ends inside it.
    """
    payloads, lengths, position = [], [], 0
    with Path(path).open("rb", buffering=0) as file:
        size = os.fstat(file.fileno()).st_size
        while header := pread(file, TFRECORD_HEADER.size, position):
            record = len(lengths)
            if len(header) < TFRECORD_HEADER.size:
                raise DamagedRecordError(
                    path, record, f"is cut short: the file ends at byte {size}"
                )
            length, length_crc = TFRECORD_HEADER.unpack(header)
            if masked_crc(header[:8]) != length_crc:
                raise DamagedRecordError(path, record, "has a length that fails its CRC check")
            end = position + TFRECORD_HEADER.size + length + TFRECORD_CRC.size
            if end > size:
                raise DamagedRecordError(
                    path, record, f"is cut short: the file ends at byte {size}"
                )
            payloads.append(position + TFRECORD_HEADER.size)
            lengths.append(length)
            position = end
    return ShardIndex(np.asarray(payloads, np.int64), np.asarray(lengths, np.int64))


def payload(file: Any, path: str, index: ShardIndex, offset: int) -> bytes:
    """The payload of record ``offset`` of an open shard file, checked against its CRC.

    One positioned read takes the payload and the CRC after it.

    Args:
        file: The shard file, open unbuffered.
        path: Its path, for the refusal.
        index: Its record index.
        offset: The record's offset in the file.

    Returns:
        The serialized record.

    Raises:
        DamagedRecordError: If the payload fails its CRC check.
    """
    length = int(index.lengths[offset])
    frame = pread(file, length + TFRECORD_CRC.size, int(index.payloads[offset]))
    data = frame[:length]
    if (
        len(frame) != length + TFRECORD_CRC.size
        or masked_crc(data) != (TFRECORD_CRC.unpack_from(frame, length)[0])
    ):
        raise DamagedRecordError(path, offset, "has a payload that fails its CRC check")
    return data


def split_shard(path: str, shard: int, skip: int, take: int) -> Shard:
    """The records ``skip .. skip + take`` a split reads of a shard file, as a :class:`Shard`.

    Args:
        path: The shard file.
        shard: Its number among the dataset's files.
        skip: The split's first record offset in the file.
        take: How many records the split reads of it.

    Returns:
        The shard.

    Raises:
        ValueError: If the split's last offset does not fit an id's low word, or the shard's
            number its high word.
    """
    if shard > LARGEST_SHARD:
        raise ValueError(
            f"shard file {path} is number {shard}, but an id's high word holds a shard's number "
            f"below 2**31 ({LARGEST_SHARD} at most): its top bit marks no record"
        )
    last = skip + take - 1
    if last > LARGEST_OFFSET:
        raise ValueError(
            f"shard file {path}: the record at offset {last} does not fit an id, whose low "
            f"word holds a record's offset in its shard, at most {LARGEST_OFFSET}"
        )
    return Shard(path, shard, skip, take)
