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
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.index_words import from_words, MAX_RECORDS
from datarax.pipeline.epochs import EpochPlan
from datarax.samplers.index_shuffle import shuffle_positions_host
from datarax.sources._source_base import EagerSourceBase
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.source_ops import partition_length, resolve_wrapped_indices
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Sized(DataSourceModule):
    """A source with a length and nothing else: the default names records by position."""

    def __init__(self, length: int) -> None:
        super().__init__(_Config())
        self.rows = length

    def __len__(self) -> int:
        return self.rows


class _Unsized(DataSourceModule):
    """A source without a length: its positions are never wrapped."""

    def __init__(self) -> None:
        super().__init__(_Config())


class _Eager(EagerSourceBase):
    def __init__(self, length: int, *, shuffle: bool) -> None:
        super().__init__(StructuralConfig())
        self.data = nnx.data({"x": jnp.arange(length, dtype=jnp.float32)})
        self.index = nnx.Variable(jnp.int32(0))
        self.epoch = nnx.Variable(jnp.int32(0))
        self._seed = 0
        self._is_random_order = shuffle
        self.dataset_name = "eager"
        self.split_name = "all"
        self._dataset_info = None


def _memory(
    length: int, *, shuffle: bool = False, num_workers: int = 1, shard_id: int | None = None
) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=shuffle, num_workers=num_workers, shard_id=shard_id),
        {"x": np.arange(length, dtype=np.float32)},
        rngs=nnx.Rngs(0),
    )


def _mixed() -> MixDataSourcesNode:
    return MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
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
        "memory-shuffled": _memory(10, shuffle=True),
        "memory-worker": _memory(10, shuffle=True, num_workers=3, shard_id=1),
        "eager": _Eager(10, shuffle=False),
        "eager-shuffled": _Eager(10, shuffle=True),
        "mixed": _mixed(),
        "disk": _disk(tmp_path),
    }


_NAMES = [
    "default",
    "unsized",
    "memory",
    "memory-shuffled",
    "memory-worker",
    "eager",
    "eager-shuffled",
    "mixed",
    "disk",
]


class TestEverySource:
    @pytest.mark.parametrize("name", _NAMES)
    @pytest.mark.parametrize("start", [3, jnp.int32(3)], ids=["int", "traced"])
    def test_record_indices_are_uint32_words(
        self, tmp_path: Path, name: str, start: object
    ) -> None:
        source = _sources(tmp_path)[name]
        indices = source.record_indices_at(start, 4, jax.random.key(1))
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
                start, 8, length, False, None, num_workers=workers, shard_id=shard
            )
            assert names.dtype == jnp.uint32
            assert [int(v) for v in from_words(names)] == _reference(
                start, 8, length, workers=workers, shard=shard
            )

    @pytest.mark.parametrize("length", _WIDE)
    def test_shuffled_positions_are_the_order_s_records(self, length: int) -> None:
        seed, epoch = 3, 2
        key = jax.random.fold_in(jax.random.key(seed), epoch)
        names = resolve_wrapped_indices(length - 4, 8, length, True, key)
        expected = shuffle_positions_host(_reference(length - 4, 8, length), length, seed, epoch)
        np.testing.assert_array_equal(from_words(names), expected)

    def test_a_traced_start_names_what_an_integer_start_names(self) -> None:
        length = (1 << 32) + 1
        at = jax.jit(lambda start: resolve_wrapped_indices(start, 8, length, False, None))
        np.testing.assert_array_equal(
            at(jnp.int32(7)), resolve_wrapped_indices(7, 8, length, False, None)
        )

    def test_a_batch_wrapping_a_short_source_several_times_is_exact(self) -> None:
        names = resolve_wrapped_indices(2, 10, 3, False, None)
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
            named.append(from_words(resolve_wrapped_indices(first, size, length, True, key)))
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
