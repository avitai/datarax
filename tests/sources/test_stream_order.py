"""A TFDS stream pass's order as int64 arrays: the definition, its draws, and what it holds.

The order is defined record by record below (``_defined_order``), as TFDS's training read orders
an epoch: the shard files in a keyed order, read ``CYCLE_LENGTH`` at a time in blocks of
``BLOCK_LENGTH`` (tf.data's ``interleave``), through tf.data's buffer shuffle with one draw per
record served. :func:`pass_order` computes the same order as int64 arrays of record words, the
shape MosaicML Streaming and FFCV give an epoch's order, one chunk at a time, so no worker walks
the order record by record and none holds a whole pass.
"""

from __future__ import annotations

import itertools
import tracemalloc
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from datarax.core.prng import pass_generator
from datarax.sources._stream_order import (
    BLOCK_LENGTH,
    CYCLE_LENGTH,
    OrderCursor,
    pass_order,
    pass_order_bytes,
)
from datarax.sources._tfrecord import Shard


def _defined_interleave(streams: Sequence[Iterator[tuple[int, int]]]) -> Iterator[tuple[int, int]]:
    """tf.data's ``interleave(cycle_length, block_length)``, a file that ends replaced in place."""
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


def _defined_shuffle(
    ids: Iterator[tuple[int, int]], size: int, generator: np.random.Generator
) -> Iterator[tuple[int, int]]:
    """tf.data's buffer shuffle: a uniform pick from the buffer per record, the next filling it."""
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


def _file_records(position: int, shard: Shard) -> Iterator[tuple[int, int]]:
    """A file's records in file order; a function, so each stream keeps its own position."""
    return ((position, offset) for offset in range(shard.skip, shard.skip + shard.take))


def _defined_order(
    shards: Sequence[Shard], generator: np.random.Generator | None, size: int
) -> list[int]:
    """The pass's records as words ``(position << 32) | offset``, one record at a time."""
    streams = [_file_records(position, shard) for position, shard in enumerate(shards)]
    if generator is None:
        records = itertools.chain.from_iterable(streams)
    else:
        files = generator.permutation(len(streams))
        records = _defined_shuffle(
            _defined_interleave([streams[i] for i in files]), size, generator
        )
    return [(position << 32) | offset for position, offset in records]


def _shards(takes: Sequence[int], skip: int = 0) -> tuple[Shard, ...]:
    return tuple(Shard(f"shard-{n}", n, skip + n % 3, take) for n, take in enumerate(takes))


def _served(chunks: Iterator[np.ndarray]) -> list[int]:
    served: list[int] = []
    for chunk in chunks:
        assert chunk.dtype == np.int64
        served.extend(chunk.tolist())
    return served


def _key(seed: int) -> np.ndarray:
    return np.asarray([seed, seed + 1], np.uint32)


# Files shorter than a block, exactly a block, a block and one, files that outnumber the cycle,
# and one file holding the whole split.
_LAYOUTS = {
    "one file": (37,),
    "short files": (3, 1, 5, 2),
    "whole blocks": (16, 32, 16, 48),
    "blocks and one": (17, 33, 1, 15, 16),
    "more files than the cycle": tuple(5 + (7 * n) % 23 for n in range(40)),
    "uneven": (200, 3, 120, 64, 1, 90),
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("size", [1, 7, 16, 100, 10_000])
@pytest.mark.parametrize("chunk", [1, 5, 64, None])
@pytest.mark.parametrize(("seed", "pass_index"), [(0, 0), (11, 1), (2026, 5)])
def test_the_order_is_the_defined_order(
    layout: str, size: int, chunk: int | None, seed: int, pass_index: int
) -> None:
    """Whatever the chunk, its boundaries fall in the buffer's fill, its steady state or drain."""
    shards = _shards(_LAYOUTS[layout])
    sizing = {} if chunk is None else {"chunk": chunk}

    served = _served(pass_order(shards, pass_generator(_key(seed), pass_index), size, **sizing))

    assert served == _defined_order(shards, pass_generator(_key(seed), pass_index), size)


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("chunk", [1, 7, None])
def test_without_a_generator_the_order_is_the_files_in_order(
    layout: str, chunk: int | None
) -> None:
    shards = _shards(_LAYOUTS[layout], skip=4)
    sizing = {} if chunk is None else {"chunk": chunk}

    assert _served(pass_order(shards, None, 8, **sizing)) == _defined_order(shards, None, 8)


def test_a_pass_of_many_files_and_records_is_the_defined_order() -> None:
    """A split where the buffer's steady state spans several default chunks."""
    shards = _shards([2_000 + (37 * n) % 400 for n in range(64)])

    served = _served(pass_order(shards, pass_generator(_key(7), 3), 1_000))

    assert len(served) > 2 * 65_536
    assert served == _defined_order(shards, pass_generator(_key(7), 3), 1_000)


def test_each_record_is_served_once_in_chunks_of_at_most_their_size() -> None:
    shards = _shards(_LAYOUTS["uneven"])

    chunks = list(pass_order(shards, pass_generator(_key(1), 0), 50, chunk=9))

    assert max(chunk.size for chunk in chunks) <= 9
    every = [
        (position << 32) | offset
        for position, shard in enumerate(shards)
        for offset in range(shard.skip, shard.skip + shard.take)
    ]
    assert sorted(np.concatenate(chunks).tolist()) == every


class TestTheCursor:
    """Positions of a pass's order taken and skipped in runs, as a stream's iterator moves."""

    @pytest.mark.parametrize("chunk", [1, 4, 64])
    def test_runs_taken_and_skipped_are_the_order_s_slices(self, chunk: int) -> None:
        shards = _shards(_LAYOUTS["blocks and one"])
        whole = _defined_order(shards, pass_generator(_key(5), 2), 7)
        cursor = OrderCursor(pass_order(shards, pass_generator(_key(5), 2), 7, chunk=chunk))

        first = cursor.take(10)
        cursor.skip(13)
        second = cursor.take(9)
        cursor.skip(0)
        rest = cursor.take(len(whole))

        assert first.tolist() == whole[:10]
        assert second.tolist() == whole[23:32]
        assert rest.tolist() == whole[32:]
        assert cursor.at == len(whole)

    def test_a_take_past_the_end_returns_what_is_left(self) -> None:
        cursor = OrderCursor(pass_order(_shards((5,)), None, 1, chunk=2))

        assert cursor.take(3).tolist() == [0, 1, 2]
        assert cursor.take(10).tolist() == [3, 4]
        assert cursor.take(1).size == 0


@pytest.mark.parametrize(("records", "size", "chunk"), [
    (200_000, 1_000, 4_096),
    (200_000, 50_000, 4_096),
    (60_000, 100_000, 65_536),
    (3_000, 16, 256),
])  # fmt: skip
def test_the_working_bytes_bound_what_computing_the_order_holds(
    records: int, size: int, chunk: int
) -> None:
    """What a worker is charged for its order covers the peak computing a pass allocates.

    The charge depends on the buffer and the chunk, not on the pass's length: the consumer here
    keeps one chunk at a time, as a stream's iterator does.
    """
    shards = _shards([records // 20] * 20)
    generator = pass_generator(_key(3), 0)

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        served = 0
        for block in pass_order(shards, generator, size, chunk=chunk):
            served += block.size
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert served == records
    assert peak <= pass_order_bytes(size, chunk=chunk, files=len(shards))
