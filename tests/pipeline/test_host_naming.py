"""A run's batches are named on the host: the records each batch holds, from where it starts.

:func:`~datarax.pipeline.epochs.batch_records` is the one rule naming a batch's records from the
epoch plan: the rows of the epoch the batch starts in, then the heads of the following epochs, each
row named by its own epoch's order. The compiled session names its batches with it, and
:class:`~datarax.pipeline.epochs.HostNaming` runs it as one ``jax.jit`` program on the CPU device
for the host stage, taking the start as two uint32 words, so positions past ``2**31`` and ``2**32``
are named exactly. The names are checked against each epoch's order asked of the source directly,
one consecutive run of positions per epoch, and against the session's names.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.index_shuffle import shuffle_positions
from datarax.core.index_words import from_words, to_words
from datarax.pipeline import Pipeline
from datarax.pipeline.epochs import batch_records, EpochPlan, HostNaming
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.step_jaxpr import host_callbacks
from tests.test_common.transfers import implicit_upload_raises


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Length(DataSourceModule):
    """An indexed source with a length and nothing else: no record is built at any size."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self, length: int) -> None:
        super().__init__(_Config())
        self.rows = length

    def __len__(self) -> int:
        return self.rows


def _memory(length: int) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), {"x": np.arange(length * 2, dtype=np.float32).reshape(length, 2)}
    )


def _mix(length: int) -> MixDataSourcesNode:
    first = length // 3 + 1
    return MixDataSourcesNode(
        MixDataSourcesConfig(weights=(1.0, 2.0)), [_memory(first), _memory(length - first + 2)]
    )


def _pipeline(
    source: DataSourceModule,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_epochs: int | None = None,
) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(5),
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
    )


def _key_words(pipe: Pipeline) -> np.ndarray | None:
    """The pipeline's key base as host words, or ``None`` for a pipeline that does not shuffle."""
    if not pipe.shuffle:
        return None
    return np.asarray(jax.device_get(pipe._epoch_key_base[...]), np.uint32)  # noqa: SLF001


def _expected(pipe: Pipeline, start: int, epoch: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Each row's record and epoch, from each epoch's order asked of the source directly.

    A batch's rows are consecutive positions of the plan: the rows of one epoch are a run, named
    by one ``record_indices_at`` call under that epoch's key.
    """
    plan = pipe.epoch_plan
    length = plan.length
    assert length is not None
    rows = [(epoch + (start + i) // length, (start + i) % length) for i in range(size)]
    base = pipe._epoch_key_base[...]  # noqa: SLF001
    names, epochs = [], []
    index = 0
    while index < size:
        row_epoch, first = rows[index]
        count = 1
        while index + count < size and rows[index + count][0] == row_epoch:
            count += 1
        key = (
            jax.random.fold_in(jax.random.wrap_key_data(base), row_epoch) if pipe.shuffle else None
        )
        names.append(np.asarray(pipe.source.record_indices_at(first, count, key), np.uint32))
        epochs.extend([row_epoch] * count)
        index += count
    return np.concatenate(names), np.asarray(epochs, np.int32)


def _batches(plan: EpochPlan, count: int) -> Iterator[tuple[int, int, int]]:
    """``(start, epoch, size)`` of a run's first ``count`` batches, by the plan's rule."""
    extent = plan.run_extent(0)
    total, final = (count, plan.batch_size) if extent is None else extent
    position, epoch = 0, 0
    for served in range(min(count, total)):
        size = final if served == total - 1 else plan.batch_size
        start, epoch = plan.batch_start(position, epoch)
        yield start, epoch, size
        position, epoch = plan.advance(start, epoch, size)


def _cases() -> list[tuple[int, int, bool, bool]]:
    cases = []
    for length in (7, 10, 64, 50_000):
        for batch in (1, 3, 4, 256):
            for shuffle in (False, True):
                for drop_last in (False, True):
                    if drop_last and batch > length:
                        continue  # the plan refuses it: every record would be skipped
                    cases.append((length, batch, shuffle, drop_last))
    return cases


@pytest.mark.parametrize(("length", "batch", "shuffle", "drop_last"), _cases())
def test_host_naming_names_each_row_by_its_own_epoch_s_order(
    length: int, batch: int, shuffle: bool, drop_last: bool
) -> None:
    """Every batch of a run, epoch crossings included, as each epoch's order names its rows."""
    pipe = _pipeline(
        _memory(length), batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_epochs=3
    )
    naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=shuffle)
    key = _key_words(pipe)
    count = 0
    for start, epoch, size in _batches(pipe.epoch_plan, 12):
        indices, epochs = naming(start, epoch, size, key)
        expected_indices, expected_epochs = _expected(pipe, start, epoch, size)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(epochs, expected_epochs)
        count += 1
    extent = pipe.epoch_plan.run_extent(0)
    assert extent is not None
    assert count == min(12, extent[0])


@pytest.mark.parametrize(("length", "batch", "shuffle", "drop_last"), _cases()[::3])
def test_host_naming_names_what_the_compiled_session_names(
    length: int, batch: int, shuffle: bool, drop_last: bool
) -> None:
    """The host's names equal the session's, row for row: one rule names both."""
    pipe = _pipeline(
        _memory(length), batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_epochs=3
    )
    naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=shuffle)
    key = _key_words(pipe)
    for start, epoch, size in _batches(pipe.epoch_plan, 8):
        session = pipe._records_at(jnp.int32(start), jnp.int32(epoch), size)  # noqa: SLF001
        indices, epochs = naming(start, epoch, size, key)
        np.testing.assert_array_equal(indices, np.asarray(session.indices))
        np.testing.assert_array_equal(epochs, np.asarray(session.epochs))


@pytest.mark.parametrize("shuffle", [False, True])
def test_a_mix_is_named_on_the_host_as_each_epoch_s_order_names_it(shuffle: bool) -> None:
    pipe = _pipeline(_mix(40), batch_size=16, shuffle=shuffle, drop_last=False, num_epochs=4)
    naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=shuffle)
    key = _key_words(pipe)
    for start, epoch, size in _batches(pipe.epoch_plan, 10):
        indices, epochs = naming(start, epoch, size, key)
        expected_indices, expected_epochs = _expected(pipe, start, epoch, size)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(epochs, expected_epochs)


def test_batch_records_traced_with_an_int32_start_names_what_word_starts_name() -> None:
    """The session's int32 start and the host stage's two-word start name the same records."""
    pipe = _pipeline(_memory(10), batch_size=4, shuffle=True, drop_last=False)
    base = pipe._epoch_key_base[...]  # noqa: SLF001
    plan = pipe.epoch_plan

    @jax.jit
    def names(start: jax.Array, epoch: jax.Array) -> tuple[jax.Array, jax.Array]:
        records = batch_records(pipe.source, plan, key_base=base, start=start, epoch=epoch, size=4)
        return records.indices, records.epochs

    for start in range(10):
        a = names(jnp.int32(start), jnp.int32(2))
        b = names(jnp.asarray(to_words(start)), jnp.int32(2))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestPast2To31:
    """A run over more than ``2**31`` and ``2**32`` records is named exactly on the host."""

    @pytest.mark.parametrize("length", [(1 << 31) + 5, (1 << 32) + 3, 1 << 40])
    @pytest.mark.parametrize("shuffle", [False, True])
    def test_batches_past_the_int32_and_uint32_positions_are_named_exactly(
        self, length: int, shuffle: bool
    ) -> None:
        pipe = _pipeline(_Length(length), batch_size=64, shuffle=shuffle, drop_last=False)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=shuffle)
        key = _key_words(pipe)
        base = pipe._epoch_key_base[...]  # noqa: SLF001
        for start in ((1 << 31) - 7, (1 << 31) + 1, (1 << 32) - 20, length - 30):
            if start >= length:
                continue  # a start is a position of the epoch
            indices, epochs = naming(start, 3, 64, key)
            positions = np.asarray([(start + i) % length for i in range(64)], dtype=np.uint64)
            row_epochs = np.asarray([3 + (start + i) // length for i in range(64)])
            expected = to_words(positions)
            if shuffle:
                for row_epoch in np.unique(row_epochs):
                    rows = row_epochs == row_epoch
                    key_e = jax.random.fold_in(jax.random.wrap_key_data(base), int(row_epoch))
                    expected[rows] = np.asarray(
                        shuffle_positions(jnp.asarray(expected[rows]), length, key_e)
                    )
            np.testing.assert_array_equal(indices, expected)
            assert list(epochs) == list(row_epochs)

    @pytest.mark.parametrize("length", [(1 << 31) + 5, 1 << 40])
    def test_an_epoch_s_names_are_distinct_records_of_the_source(self, length: int) -> None:
        """A sampled bijection check: 4096 consecutive positions name 4096 distinct records."""
        pipe = _pipeline(_Length(length), batch_size=4096, shuffle=True, drop_last=False)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=True)
        indices, _ = naming((1 << 32) - 2048, 0, 4096, _key_words(pipe))
        values = from_words(indices)
        assert len(np.unique(values)) == 4096
        assert int(values.max()) < length


class TestProgram:
    """The host naming is one ``jax.jit`` program on the CPU device, per batch shape."""

    def test_one_compile_per_batch_shape_and_none_in_steady_state(self) -> None:
        pipe = _pipeline(_memory(50), batch_size=8, shuffle=True, drop_last=False, num_epochs=4)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=True)
        key = _key_words(pipe)
        with expect_first_call_compiles("jit(_names)"):
            naming(0, 0, 8, key)
        with expect_compiles(0):
            for start, epoch, _ in _batches(pipe.epoch_plan, 40):
                naming(start, epoch, 8, key)
        with expect_first_call_compiles("jit(_names)"):
            naming(46, 3, 4, key)  # the run's short final batch: one more shape

    def test_the_source_s_columns_are_never_an_argument(self) -> None:
        """Naming transfers none of the source's data: it is closed over, never passed."""
        assert implicit_upload_raises(), "the guard must fire on an implicit upload"
        pipe = _pipeline(_memory(50), batch_size=8, shuffle=True, drop_last=False)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=True)
        key = _key_words(pipe)
        naming(0, 0, 8, key)
        before = jax.live_arrays()
        known = {id(a) for a in before}  # ``before`` holds them, so no id is reused
        with jax.transfer_guard("disallow"):
            for start in range(0, 48, 8):
                naming(start, 1, 8, key)
        created = [a for a in jax.live_arrays() if id(a) not in known]
        assert all(a.shape[0] < 50 if a.ndim else True for a in created)
        assert isinstance(pipe.source, MemorySource)
        assert isinstance(pipe.source.data["x"], np.ndarray)

    def test_the_names_are_computed_on_the_cpu_device_with_no_host_callback(self) -> None:
        pipe = _pipeline(_mix(40), batch_size=16, shuffle=True, drop_last=False)
        base = pipe._epoch_key_base[...]  # noqa: SLF001
        jaxpr = jax.make_jaxpr(
            lambda start, epoch: batch_records(
                pipe.source, pipe.epoch_plan, key_base=base, start=start, epoch=epoch, size=16
            )
        )(jnp.asarray(to_words(5)), jnp.int32(2))
        assert host_callbacks(jaxpr) == []
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=True)
        naming(5, 2, 16, _key_words(pipe))
        before = jax.live_arrays()
        known = {id(a) for a in before}  # ``before`` holds them, so no id is reused
        indices, epochs = naming(21, 2, 16, _key_words(pipe))
        created = [a for a in jax.live_arrays() if id(a) not in known]
        cpu = jax.devices("cpu")[0]
        assert all(a.devices() == {cpu} for a in created)
        assert isinstance(indices, np.ndarray) and indices.dtype == np.uint32
        assert isinstance(epochs, np.ndarray) and epochs.dtype == np.int32

    def test_a_pipeline_that_does_not_shuffle_takes_no_key(self) -> None:
        pipe = _pipeline(_memory(10), batch_size=4, shuffle=False, drop_last=False)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=False)
        indices, _ = naming(8, 0, 4, None)
        assert [int(v) for v in from_words(indices)] == [8, 9, 0, 1]
        with pytest.raises(ValueError, match="does not shuffle"):
            naming(0, 0, 4, np.zeros(2, np.uint32))

    def test_a_shuffling_pipeline_needs_its_key(self) -> None:
        pipe = _pipeline(_memory(10), batch_size=4, shuffle=True, drop_last=False)
        naming = HostNaming(pipe.source, pipe.epoch_plan, shuffled=True)
        with pytest.raises(ValueError, match="key"):
            naming(0, 0, 4, None)
