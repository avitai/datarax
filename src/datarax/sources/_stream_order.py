"""The record order of a TFDS stream's pass, computed over record ids.

Given the pipeline's key, a pass is ordered as TFDS's training read orders it: the shard files in a
keyed order, read 16 at a time in blocks of 16 (:func:`interleaved`), through tf.data's buffer
shuffle (:func:`buffer_shuffled`), the buffer holding record ids rather than records. Imports no
TensorFlow Datasets.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from datarax.sources._tfrecord import Shard


CYCLE_LENGTH = 16
"""Shard files read at once when a pass is shuffled: TFDS's ``interleave_cycle_length``."""

BLOCK_LENGTH = 16
"""Records read from one file before the next: TFDS's ``interleave_block_length``."""


class RecordId(NamedTuple):
    """A record of a pass: the position of its shard in the pass's read, and its offset there."""

    position: int
    offset: int


def shard_ids(position: int, shard: Shard) -> Iterator[RecordId]:
    """The records ``shard`` holds for the split, in file order, as ids."""
    return (RecordId(position, offset) for offset in range(shard.skip, shard.skip + shard.take))


def interleaved(streams: Sequence[Iterator[RecordId]]) -> Iterator[RecordId]:
    """The records of ``streams``, one per shard, as TFDS's training read interleaves its files.

    ``CYCLE_LENGTH`` files are open at once and ``BLOCK_LENGTH`` records are taken from each in
    turn; a file that ends is replaced by the next one in its place in the cycle (tf.data's
    ``interleave(cycle_length, block_length)``, which ``tfds.core.reader`` applies).
    """
    pending = iter(streams)
    cycle = list(itertools.islice(pending, CYCLE_LENGTH))
    while cycle:
        slot = 0
        while slot < len(cycle):
            block = list(itertools.islice(cycle[slot], BLOCK_LENGTH))
            yield from block
            if len(block) < BLOCK_LENGTH:
                following = next(pending, None)
                if following is None:
                    del cycle[slot]
                    continue
                cycle[slot] = following
            slot += 1


def buffer_shuffled(
    ids: Iterator[RecordId], size: int, generator: np.random.Generator
) -> Iterator[RecordId]:
    """``ids`` through tf.data's buffer shuffle of ``size`` records.

    The buffer fills with the first ``size`` records; then each record served is a uniform pick
    from the buffer, whose place the next record takes; at the end the buffer is served in
    uniform picks. The buffer holds record ids, so its state is the ids and the generator's.
    """
    buffer = list(itertools.islice(ids, size))
    for record in ids:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = record
    while buffer:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = buffer[-1]
        buffer.pop()
