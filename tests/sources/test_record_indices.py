"""Every indexed source names its records with 64-bit indices, uint32 ``(size, 2)`` words.

``record_indices_at(start, size, key)`` returns ``(hi, lo)`` per record, the layout of
``Batch.indices``, on every source and the default, with no int32 form left. The wrapped and
partitioned order (``resolve_wrapped_indices``) is exact past ``2**31`` and ``2**32`` records,
computed from Python-integer positions on the host and from traced positions alike, and composed
with the epoch plan it serves the shuffle's order across an epoch boundary. Only positions are
computed here: no source of that size, and no data, is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, known_length, RecordIdentity
from datarax.core.index_shuffle import shuffle_positions_host
from datarax.core.index_words import from_words, MAX_RECORDS, to_words
from datarax.pipeline.epochs import EpochPlan
from datarax.sources.eager_source import EagerSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.source_ops import partition_length, resolve_wrapped_indices
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Sized(DataSourceModule):
    """A source with a length and nothing else: the default names records by position."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self, length: int) -> None:
        super().__init__(_Config())
        self.rows = length

    def __len__(self) -> int:
        return self.rows


class _Unsized(DataSourceModule):
    """A source without a length: its positions are never wrapped."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(_Config())


class _Eager(EagerSource):
    def __init__(self, length: int) -> None:
        super().__init__(StructuralConfig())
        self._store({"x": np.arange(length, dtype=np.float32)})


def _memory(length: int, *, num_workers: int = 1, shard_id: int | None = None) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(num_workers=num_workers, shard_id=shard_id),
        {"x": np.arange(length, dtype=np.float32)},
    )


def _mixed() -> MixDataSourcesNode:
    return MixDataSourcesNode(
        MixDataSourcesConfig(weights=(0.5, 0.5)),
        [_memory(4), _memory(6)],
    )


def _disk(tmp_path: Path) -> StreamingDiskSource:
    path = tmp_path / "data.npy"
    np.save(path, np.arange(16, dtype=np.float32))
    return StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))


def _sources(tmp_path: Path) -> dict[str, DataSourceModule]:
    return {
        "default": _Sized(10),
        "unsized": _Unsized(),
        "memory": _memory(10),
        "memory-worker": _memory(10, num_workers=3, shard_id=1),
        "eager": _Eager(10),
        "mixed": _mixed(),
        "disk": _disk(tmp_path),
    }


_NAMES = [
    "default",
    "unsized",
    "memory",
    "memory-worker",
    "eager",
    "mixed",
    "disk",
]


class TestEverySource:
    @pytest.mark.parametrize("name", _NAMES)
    @pytest.mark.parametrize("start", [3, jnp.int32(3)], ids=["int", "traced"])
    def test_record_indices_are_uint32_words(
        self, tmp_path: Path, name: str, start: int | jax.Array
    ) -> None:
        source = _sources(tmp_path)[name]
        # A source without a length has no order to shuffle, so it is named sequentially.
        key = None if name == "unsized" else jax.random.key(1)
        indices = source.record_indices_at(start, 4, key)
        assert indices.shape == (4, 2)
        assert indices.dtype == jnp.uint32

    @pytest.mark.parametrize("name", [n for n in _NAMES if n not in {"default", "unsized"}])
    def test_the_records_named_are_the_records_gathered(self, tmp_path: Path, name: str) -> None:
        """``get_records`` takes the words ``record_indices_at`` gives and gathers those rows."""
        source = _sources(tmp_path)[name]
        key = jax.random.key(2)
        indices = source.record_indices_at(5, 6, key)
        served = np.asarray(source.get_records(indices)["x"])
        assert served.shape == (6,)
        if name != "mixed":  # a mixed index is an offset into the concatenated sources
            np.testing.assert_array_equal(served, from_words(indices).astype(np.float32))

    def test_a_source_without_a_length_names_records_by_position(self) -> None:
        start = (1 << 32) - 2
        np.testing.assert_array_equal(
            from_words(_Unsized().record_indices_at(start, 4)),
            np.arange(start, start + 4, dtype=np.uint64),
        )


def _share(length: int, workers: int, shard: int) -> int:
    """Positions ``p < length`` with ``p % workers == shard``."""
    return -(-(length - shard) // workers)


def _reference(
    start: int, size: int, length: int, *, workers: int = 1, shard: int = 0
) -> list[int]:
    """The worker's wrapped positions as global positions, in Python integers."""
    worker_length = _share(length, workers, shard)
    return [shard + ((start + offset) % worker_length) * workers for offset in range(size)]


_WIDE = ((1 << 31) + 1, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, 1 << 40, MAX_RECORDS)


class TestWrappedOrder:
    @pytest.mark.parametrize("length", [1, 2, 7, 10, 11, *_WIDE])
    @pytest.mark.parametrize("workers", [1, 3, 8])
    def test_the_workers_shares_partition_the_records(self, length: int, workers: int) -> None:
        """Shares sum to the length and differ by at most one, the first ones larger."""
        shares = [partition_length(length, workers, shard) for shard in range(workers)]
        assert sum(shares) == length
        assert shares == [
            length // workers + (1 if shard < length % workers else 0) for shard in range(workers)
        ]
        if length < 1 << 20:
            assert shares == [len(range(shard, length, workers)) for shard in range(workers)]

    @pytest.mark.parametrize("length", _WIDE)
    @pytest.mark.parametrize(("workers", "shard"), [(1, 0), (3, 2), (8, 5)])
    def test_wrapped_positions_are_exact_past_int32(
        self, length: int, workers: int, shard: int
    ) -> None:
        worker_length = _share(length, workers, shard)
        for start in (0, worker_length - 3, worker_length + 5, 5 * worker_length - 1):
            names = resolve_wrapped_indices(
                start, 8, length, None, num_workers=workers, shard_id=shard
            )
            assert names.dtype == jnp.uint32
            assert [int(v) for v in from_words(names)] == _reference(
                start, 8, length, workers=workers, shard=shard
            )

    @pytest.mark.parametrize("length", _WIDE)
    def test_shuffled_positions_are_the_order_s_records(self, length: int) -> None:
        seed, epoch = 3, 2
        key = jax.random.fold_in(jax.random.key(seed), epoch)
        names = resolve_wrapped_indices(length - 4, 8, length, key)
        expected = shuffle_positions_host(_reference(length - 4, 8, length), length, seed, epoch)
        np.testing.assert_array_equal(from_words(names), expected)

    def test_a_traced_start_names_what_an_integer_start_names(self) -> None:
        length = (1 << 32) + 1
        at = jax.jit(lambda start: resolve_wrapped_indices(start, 8, length, None))
        np.testing.assert_array_equal(at(jnp.int32(7)), resolve_wrapped_indices(7, 8, length, None))

    def test_a_batch_wrapping_a_short_source_several_times_is_exact(self) -> None:
        names = resolve_wrapped_indices(2, 10, 3, None)
        assert [int(v) for v in from_words(names)] == _reference(2, 10, 3)


@pytest.mark.parametrize("length", [(1 << 31) + 1, (1 << 32) + 1, 1 << 40])
@pytest.mark.parametrize("drop_last", [False, True])
def test_the_epoch_plan_composed_with_the_shuffle_crosses_wide_epochs(
    length: int, drop_last: bool
) -> None:
    """Three batches across an epoch's end name the plan's rows in each row's epoch's order.

    Every row's ``(epoch, position)`` comes from the plan in Python integers; a row past the
    epoch's end takes the next epoch's head. The names come from ``resolve_wrapped_indices``,
    one call per epoch touched, as the pipeline names a crossing batch.
    """
    seed, size = 11, 256
    plan = EpochPlan(length=length, batch_size=size, drop_last=drop_last, num_epochs=None)
    position, epoch = length - 300, 4
    for _ in range(3):
        start, epoch = plan.batch_start(position, epoch)
        rows = [(epoch + (start + i) // length, (start + i) % length) for i in range(size)]
        named = []
        for offset in range(plan.epochs_touched(size)):
            first = start if offset == 0 else 0
            key = jax.random.fold_in(jax.random.key(seed), epoch + offset)
            named.append(from_words(resolve_wrapped_indices(first, size, length, key)))
        served = []
        for row, (row_epoch, row_position) in enumerate(rows):
            later = row_epoch - epoch
            # The first epoch's names start at ``start``; a later epoch's start at its head.
            served.append(int(named[later][row if later == 0 else row_position]))
        expected = [
            int(shuffle_positions_host([pos], length, seed, row_epoch)[0])
            for row_epoch, pos in rows
        ]
        assert served == expected
        if drop_last:
            assert {row_epoch for row_epoch, _ in rows} == {epoch}
        position, epoch = plan.advance(start, epoch, size)


class TestWordStart:
    """A start given as two uint32 words, the host stage's form, names what an integer start names.

    The host stage names positions past ``2**31`` (an int32 cannot hold them) with ``(hi, lo)``
    words, as NumPy words or traced inside its naming program; every source's override takes it.
    """

    @pytest.mark.parametrize("name", [n for n in _NAMES if n != "unsized"])
    @pytest.mark.parametrize("keyed", [False, True])
    def test_every_source_names_a_word_start_as_its_integer_start(
        self, tmp_path: Path, name: str, keyed: bool
    ) -> None:
        source = _sources(tmp_path)[name]
        key = jax.random.key(4) if keyed else None
        named = jax.jit(lambda start, key: source.record_indices_at(start, 5, key))
        length = known_length(source)
        assert length is not None
        for start in range(length):
            expected = np.asarray(source.record_indices_at(start, 5, key))
            words = to_words(start)
            np.testing.assert_array_equal(source.record_indices_at(words, 5, key), expected)
            np.testing.assert_array_equal(named(jnp.asarray(words), key), expected)

    def test_a_source_without_a_length_names_a_word_start_past_uint32(self) -> None:
        start = (1 << 32) + 7
        np.testing.assert_array_equal(
            from_words(_Unsized().record_indices_at(to_words(start), 4)),
            np.arange(start, start + 4, dtype=np.uint64),
        )

    @pytest.mark.parametrize("length", _WIDE)
    @pytest.mark.parametrize(("workers", "shard"), [(1, 0), (3, 2)])
    def test_a_word_start_past_int32_and_uint32_is_exact(
        self, length: int, workers: int, shard: int
    ) -> None:
        worker_length = _share(length, workers, shard)
        traced = jax.jit(
            lambda start: resolve_wrapped_indices(
                start, 8, length, None, num_workers=workers, shard_id=shard
            )
        )
        for start in (0, (1 << 31) + 3, (1 << 32) - 2, worker_length - 3):
            if start >= worker_length:
                continue  # a word start is a position of the order
            words = to_words(start)
            expected = _reference(start, 8, length, workers=workers, shard=shard)
            assert [int(v) for v in from_words(traced(jnp.asarray(words)))] == expected
            assert [
                int(v)
                for v in from_words(
                    resolve_wrapped_indices(
                        words, 8, length, None, num_workers=workers, shard_id=shard
                    )
                )
            ] == expected

    @pytest.mark.parametrize("name", ["default", "memory", "mixed"])
    def test_under_vmap_over_keys_each_key_names_its_order(self, tmp_path: Path, name: str) -> None:
        source = _sources(tmp_path)[name]
        keys = jnp.stack([jax.random.key(seed) for seed in range(4)])
        start = jnp.asarray(to_words(3))
        named = jax.vmap(lambda key: source.record_indices_at(start, 5, key))(keys)
        for row, seed in enumerate(range(4)):
            np.testing.assert_array_equal(
                named[row], source.record_indices_at(3, 5, jax.random.key(seed))
            )

    @pytest.mark.parametrize("name", ["default", "memory", "mixed"])
    def test_under_scan_over_word_starts_each_step_names_its_batch(
        self, tmp_path: Path, name: str
    ) -> None:
        source = _sources(tmp_path)[name]
        key = jax.random.key(6)
        starts = jnp.asarray(to_words(np.arange(0, 8, 2, dtype=np.uint64)))

        def body(carry: None, start: jax.Array) -> tuple[None, jax.Array]:
            return carry, source.record_indices_at(start, 3, key)

        _, named = jax.lax.scan(body, None, starts)
        for step, start in enumerate(range(0, 8, 2)):
            np.testing.assert_array_equal(named[step], source.record_indices_at(start, 3, key))
