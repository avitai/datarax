"""The record order of a TFDS stream's pass, computed as int64 arrays of record words.

Given the pipeline's key, a pass is ordered as TFDS's training read orders it: the shard files in a
keyed order, read 16 at a time in blocks of 16 (tf.data's ``interleave``), through tf.data's buffer
shuffle, the buffer holding record ids rather than records. A record is the word
``(position << 32) | offset``: the place of its shard among the split's shards and its offset in
the file.

The order is computed a chunk of records at a time with array operations, the shape MosaicML
Streaming (``streaming/base/dataset.py``, an epoch's sample ids as one array) and FFCV
(``traversal_order``, the epoch's order as one int64 array) give an epoch's order, so no worker
walks it record by record; computing it chunk by chunk keeps what a worker holds to the buffer and
a chunk, whatever the pass's length. The draws are tf.data's: one uniform pick from the buffer per
record served, taken from NumPy in bulk, which draws as many scalar calls would. Imports no
TensorFlow Datasets.
"""

from __future__ import annotations

import itertools
import struct
import sys
from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from datarax.sources._tfrecord import Shard


CYCLE_LENGTH = 16
"""Shard files read at once when a pass is shuffled: TFDS's ``interleave_cycle_length``."""

BLOCK_LENGTH = 16
"""Records read from one file before the next: TFDS's ``interleave_block_length``."""

ORDER_CHUNK = 1 << 16
"""Records of a pass's order computed together: 512 KiB per int64 array of them."""

OFFSET_BITS = 32
"""A record word's low bits: the record's offset in its shard file."""

OFFSET_MASK = (1 << OFFSET_BITS) - 1


type _Run = tuple[int, int]
"""Consecutive records of one file: the first's word and how many."""


def pass_order(
    shards: Sequence[Shard],
    generator: np.random.Generator | None,
    buffer_size: int,
    *,
    chunk: int = ORDER_CHUNK,
) -> Iterator[np.ndarray]:
    """A pass's records in its order, as int64 arrays of at most ``chunk`` record words.

    Args:
        shards: The split's shards in file order; a record's position is its shard's place here.
        generator: The pass's generator (:func:`~datarax.core.prng.pass_generator`), or ``None``
            for the files in order.
        buffer_size: Records the shuffle buffer holds.
        chunk: Records computed together.

    Returns:
        The order, chunk by chunk.
    """
    files = [
        (position << OFFSET_BITS | shard.skip, shard.take) for position, shard in enumerate(shards)
    ]
    if generator is None:
        return _chunked(files, chunk)
    records = sum(shard.take for shard in shards)
    read = [files[i] for i in generator.permutation(len(files))]
    return _buffer_shuffled(
        _chunked(_interleaved(read), chunk), min(buffer_size, records), generator, chunk
    )


def _interleaved(files: Sequence[_Run]) -> Iterator[_Run]:
    """The runs of ``files`` as TFDS's training read interleaves them, a block at a time.

    ``CYCLE_LENGTH`` files are open at once and ``BLOCK_LENGTH`` records are taken from each in
    turn; a file whose block comes back short (empty included) is replaced by the next one in its
    place in the cycle (tf.data's ``interleave(cycle_length, block_length)``, which
    ``tfds.core.reader`` applies).
    """
    pending = iter(files)
    cycle = [list(file) for file in itertools.islice(pending, CYCLE_LENGTH)]
    while cycle:
        slot = 0
        while slot < len(cycle):
            first, left = cycle[slot]
            block = min(left, BLOCK_LENGTH)
            if block:
                yield first, block
            if block < BLOCK_LENGTH:
                following = next(pending, None)
                if following is None:
                    del cycle[slot]
                    continue
                cycle[slot] = list(following)
            else:
                cycle[slot] = [first + block, left - block]
            slot += 1


def _chunked(runs: Iterable[_Run], chunk: int) -> Iterator[np.ndarray]:
    """The records of ``runs`` as int64 arrays of ``chunk`` words, the last one short."""
    firsts: list[int] = []
    lengths: list[int] = []
    count = 0
    for run_first, run_length in runs:
        first, length = run_first, run_length
        while length:
            taken = min(length, chunk - count)
            firsts.append(first)
            lengths.append(taken)
            count += taken
            first, length = first + taken, length - taken
            if count == chunk:
                words = _words(firsts, lengths)
                firsts, lengths, count = [], [], 0
                yield words
    if count:
        words = _words(firsts, lengths)
        firsts, lengths = [], []
        yield words


def _words(firsts: list[int], lengths: list[int]) -> np.ndarray:
    """The words of the runs starting at ``firsts``, each ``lengths`` long, in order."""
    sizes = np.fromiter(lengths, np.int64, len(lengths))
    starts = np.fromiter(firsts, np.int64, len(firsts))
    starts -= np.cumsum(sizes) - sizes  # each run's first word less its place in the chunk
    words = np.repeat(starts, sizes)
    words += np.arange(words.size, dtype=np.int64)
    return words


def _buffer_shuffled(
    chunks: Iterator[np.ndarray], held: int, generator: np.random.Generator, chunk: int
) -> Iterator[np.ndarray]:
    """``chunks`` through tf.data's buffer shuffle of ``held`` records, chunk by chunk.

    The buffer fills with the first ``held`` records; then each record served is a uniform pick
    from the buffer, whose place the next record takes; at the end the buffer is served in
    uniform picks, the last record moving into the place of each one served.
    """
    buffer = np.empty(held, np.int64)
    filled = 0
    for words in chunks:
        incoming = words
        if filled < held:
            fill = min(held - filled, incoming.size)
            buffer[filled : filled + fill] = incoming[:fill]
            filled += fill
            incoming = incoming[fill:]
        if incoming.size:
            yield _steady(buffer, incoming, generator)
    yield from _drained(buffer, generator, chunk)


def _steady(buffer: np.ndarray, incoming: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """The records served while ``incoming`` enters the full ``buffer``, which it updates.

    Step ``k`` serves the slot ``picks[k]`` and writes ``incoming[k]`` there. A step serves what
    the slot held before the chunk, unless an earlier step of the chunk wrote it: then the record
    that step wrote. Sorting the steps by slot, stably, puts each step right after the previous
    write to its slot; every write below goes to distinct places, since NumPy leaves the outcome
    of repeated indices in one assignment unspecified.
    """
    picks = generator.integers(buffer.size, size=incoming.size)
    served = buffer[picks]
    steps = np.argsort(picks, kind="stable")
    slots = picks[steps]
    del picks
    repeat = slots[1:] == slots[:-1]  # step steps[k + 1] writes the slot step steps[k] wrote
    served[steps[1:][repeat]] = incoming[steps[:-1][repeat]]
    last = np.flatnonzero(np.append(~repeat, True))  # the last write to each slot written
    buffer[slots[last]] = incoming[steps[last]]
    return served


def _drained(
    buffer: np.ndarray, generator: np.random.Generator, chunk: int
) -> Iterator[np.ndarray]:
    """The buffer served once no record is left to enter it, in chunks of at most ``chunk``."""
    if not buffer.size:
        return
    held = buffer.tolist()
    picks = generator.integers(np.arange(buffer.size, 0, -1, dtype=np.int64)).tolist()
    served: list[int] = []
    for live, pick in zip(range(buffer.size, 0, -1), picks, strict=True):
        served.append(held[pick])
        held[pick] = held[live - 1]
        if len(served) == chunk:
            yield np.asarray(served, np.int64)
            served = []
    if served:
        yield np.asarray(served, np.int64)


class OrderCursor:
    """A place in a pass's order, moved by runs of records taken or skipped.

    Attributes:
        at: Records of the order taken or skipped so far.
    """

    def __init__(self, chunks: Iterator[np.ndarray]) -> None:
        """Start at the order's head.

        Args:
            chunks: The pass's order (:func:`pass_order`).
        """
        self._chunks = chunks
        self._head = np.empty(0, np.int64)
        self.at = 0

    def skip(self, count: int) -> None:
        """Move past ``count`` records, or to the end of the order."""
        while count and self._refilled():
            moved = min(count, self._head.size)
            self._head = self._head[moved:]
            self.at += moved
            count -= moved

    def take(self, count: int) -> np.ndarray:
        """The next ``count`` record words, fewer at the end of the order."""
        parts: list[np.ndarray] = []
        while count and self._refilled():
            moved = min(count, self._head.size)
            parts.append(self._head[:moved])
            self._head = self._head[moved:]
            self.at += moved
            count -= moved
        return np.concatenate(parts) if parts else np.empty(0, np.int64)

    def _refilled(self) -> bool:
        """Whether a record is left, reading the next chunk once the current one is used."""
        if not self._head.size:
            self._head = next(self._chunks, self._head)
        return bool(self._head.size)


_WORD = np.dtype(np.int64).itemsize
_POINTER = struct.calcsize("P")
_RECORD_INT = sys.getsizeof(1 << 62)
"""A record word as a Python int: a shard position below 2**31 above a 32-bit offset."""
_SMALL_INT = sys.getsizeof((1 << 30) - 1)
"""A count or a pick below 2**30 as a Python int."""


def _list_bytes(items: int) -> int:
    """A list grown to ``items`` by appends: CPython over-allocates by an eighth plus 6 slots."""
    return sys.getsizeof([]) + _POINTER * (items + (items >> 3) + 6)


def pass_order_bytes(buffer_size: int, *, chunk: int = ORDER_CHUNK, files: int = 0) -> int:
    """The most a worker holds while computing a pass's order, its consumer keeping one chunk.

    A bound from what each phase holds at once, besides the consumer's chunk and the buffer of
    ``buffer_size`` words:

    - a chunk being built: its runs' first words and lengths in two lists (one run per word at
      most), then five int64 arrays of a word per run or record (sizes, starts, the cumulative
      sizes and their difference, the words and their ramp);
    - the buffer's steady state: the incoming chunk, the picks, the served records, the steps
      sorted by slot (stable argsort's merge buffer up to half of them more), the slots, two bool
      masks, and the pairs of repeated writes with their gathered records;
    - the drain: the buffer and its picks as lists, the picks as an array, and the served chunk
      as a list and an array.

    Args:
        buffer_size: Records the shuffle buffer holds.
        chunk: Records computed together.
        files: The split's shards, held as one run each while the files are interleaved.

    Returns:
        The bytes, independent of the pass's length.
    """
    consumer = _WORD * chunk
    building = 2 * _list_bytes(chunk) + chunk * (_RECORD_INT + _SMALL_INT) + 6 * _WORD * chunk
    steady = _WORD * chunk * 9 + _WORD * chunk // 2 + 2 * chunk
    drain = (
        2 * _list_bytes(buffer_size)
        + buffer_size * (_RECORD_INT + _SMALL_INT + _WORD)
        + _list_bytes(chunk)
        + _WORD * chunk
    )
    interleaving = files * (_POINTER + sys.getsizeof((0, 0)) + _RECORD_INT + _SMALL_INT + _WORD)
    interleaving += 2 * _list_bytes(files) + CYCLE_LENGTH * _list_bytes(2)
    return consumer + _WORD * buffer_size + max(building, steady, drain) + interleaving
