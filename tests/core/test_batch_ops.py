"""Batch operations: pure functions over the ``Batch`` pytree.

Inside a compiled step they fuse into their consumers and transfer nothing; on a host batch of
NumPy arrays the construction operations stay on the host, as NumPy views, until placement.
Padding rows carry ``PADDING_INDEX`` and weight 0, so a filtered batch keeps its static shape.
"""

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from substrax.spmd import place_batch_on_shards
from substrax.testing.compiles import expect_compiles

from datarax.core import batch_ops, Maybe
from datarax.core.element_batch import Batch, Element, PADDING_INDEX
from datarax.core.state_keys import WEIGHT


B = 8


def host_batch(size: int = B, offset: int = 0, epoch: int = 2) -> Batch:
    """A batch of NumPy arrays, as a host reader builds it before placement."""
    rows = np.arange(size, dtype=np.float32) + offset
    return Batch(
        {"x": np.stack([rows, -rows], axis=1), "nested": {"y": (rows * 10).astype(np.int32)}},
        states={"score": rows / 100},
        indices=np.stack([np.zeros(size, np.uint32), np.arange(size, dtype=np.uint32) + offset], 1),
        epochs=np.full(size, epoch, np.int32),
        draws=np.zeros(size, np.int32),
        batch_state={"seen": np.int32(offset)},
    )


def device_batch(size: int = B, offset: int = 0, epoch: int = 2) -> Batch:
    return jax.device_put(host_batch(size, offset, epoch))


def rows_of(batch: Batch) -> list[int]:
    """The low words of a batch's indices: the record each row holds."""
    return [int(i) for i in np.asarray(batch.indices)[:, 1]]


def is_padding(batch: Batch) -> np.ndarray:
    return np.all(np.asarray(batch.indices) == PADDING_INDEX, axis=1)


class TestConstruction:
    """The two ways a batch is built: from stacked records, and from arrays without identities."""

    def test_from_arrays_keys_rows_on_their_positions(self) -> None:
        batch = batch_ops.from_arrays({"x": np.ones((5, 3), np.float32)})

        np.testing.assert_array_equal(batch.indices, np.stack([np.zeros(5), np.arange(5)], 1))
        assert batch.indices.dtype == np.uint32
        np.testing.assert_array_equal(batch.epochs, np.zeros(5, np.int32))
        np.testing.assert_array_equal(batch.draws, np.zeros(5, np.int32))
        assert batch.epochs.dtype == np.int32 and batch.draws.dtype == np.int32
        assert batch.states == {} and batch.batch_state == {}
        assert batch.batch_size == 5

    def test_from_arrays_keeps_states_and_batch_state(self) -> None:
        states = {"score": np.zeros(4, np.float32)}
        batch_state = {"step": np.int32(3)}

        batch = batch_ops.from_arrays(
            {"x": np.ones((4,), np.float32)}, states=states, batch_state=batch_state
        )

        assert batch.states is states and batch.batch_state is batch_state

    def test_from_arrays_stays_on_the_host_for_numpy_data(self) -> None:
        with jax.transfer_guard("disallow"):
            batch = batch_ops.from_arrays({"x": np.ones((4, 2), np.float32)})

        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param({"x": np.ones((4, 2)), "y": np.ones((5,))}, id="leading-axes-differ"),
            pytest.param({}, id="no-leaves"),
            pytest.param({"x": np.ones((4,)), "s": np.float32(1.0)}, id="a-scalar-leaf"),
        ],
    )
    def test_from_arrays_refuses_data_without_one_record_axis(self, data: dict) -> None:
        with pytest.raises(ValueError, match="record axis"):
            batch_ops.from_arrays(data)

    def test_from_stacked_takes_a_grain_batched_element(self) -> None:
        """Grain stacks every field of the records, identities included, along a new axis."""
        records = [
            Element(
                {"x": np.full(3, i, np.float32)},
                state={"seen": np.int32(i)},
                index=np.array([1, 10 + i], np.uint32),
                epoch=np.array(4, np.int32),
                draw=np.array(i % 2, np.int32),
            )
            for i in range(6)
        ]
        stacked = next(iter(grain.MapDataset.source(records).batch(6)))

        batch = batch_ops.from_stacked(stacked)

        assert batch.batch_size == 6
        np.testing.assert_array_equal(batch.data["x"][:, 0], np.arange(6))
        np.testing.assert_array_equal(batch.states["seen"], np.arange(6))
        np.testing.assert_array_equal(batch.indices, [[1, 10 + i] for i in range(6)])
        np.testing.assert_array_equal(batch.epochs, np.full(6, 4))
        np.testing.assert_array_equal(batch.draws, [0, 1, 0, 1, 0, 1])
        assert batch.batch_state == {}

    def test_records_without_an_index_are_named_by_their_rows(self) -> None:
        """Stacked records built without identities never share a key: row i is (0, i)."""
        records = [Element({"x": np.full(2, i, np.float32)}) for i in range(4)]

        batch = batch_ops.from_stacked(batch_ops.stack(records))

        np.testing.assert_array_equal(batch.indices, [[0, i] for i in range(4)])
        np.testing.assert_array_equal(batch.epochs, np.zeros(4))
        np.testing.assert_array_equal(batch.draws, np.zeros(4))

    def test_records_with_and_without_an_index_do_not_stack(self) -> None:
        records = [
            Element({"x": np.ones(2)}),
            Element({"x": np.ones(2)}, index=np.zeros(2, np.uint32)),
        ]

        with pytest.raises(ValueError):
            batch_ops.stack(records)

    def test_from_stacked_refuses_records_without_one_record_axis(self) -> None:
        unstacked = Element({"x": np.ones(3, np.float32)})

        with pytest.raises(ValueError, match="record axis"):
            batch_ops.from_stacked(unstacked)

    def test_from_stacked_refuses_an_index_that_is_not_two_words_per_record(self) -> None:
        records = Element({"x": np.ones((4, 3), np.float32)}, index=np.zeros((4,), np.uint32))
        records = records.replace(epoch=np.zeros(4, np.int32), draw=np.zeros(4, np.int32))

        with pytest.raises(ValueError, match=r"\(4, 2\) uint32 words"):
            batch_ops.from_stacked(records)

    @pytest.mark.parametrize("combine", [batch_ops.concatenate, batch_ops.stack])
    def test_combining_nothing_is_refused(self, combine) -> None:
        with pytest.raises(ValueError, match="at least one"):
            combine([])


class TestRows:
    """Reading and regrouping rows."""

    def test_element_reads_one_row_with_its_identity(self) -> None:
        batch = host_batch(offset=20)

        element = batch_ops.element(batch, 3)

        np.testing.assert_array_equal(element.data["x"], [23.0, -23.0])
        assert int(element.data["nested"]["y"]) == 230
        np.testing.assert_allclose(element.state["score"], 0.23)
        assert element.index is not None  # a row of a batch always carries its identity
        np.testing.assert_array_equal(element.index, [0, 23])
        assert int(element.epoch) == 2 and int(element.draw) == 0

    def test_slice_rows_and_take(self) -> None:
        batch = host_batch()

        assert rows_of(batch_ops.slice_rows(batch, 2, 5)) == [2, 3, 4]
        taken = batch_ops.take(batch, np.array([6, 1, 1]))
        assert rows_of(taken) == [6, 1, 1]
        np.testing.assert_array_equal(taken.data["x"][:, 0], [6.0, 1.0, 1.0])
        assert taken.batch_state["seen"] == batch.batch_state["seen"]

    def test_split_gives_each_part_the_batch_state(self) -> None:
        parts = batch_ops.split(host_batch(offset=5), 4)

        assert [rows_of(p) for p in parts] == [[5, 6], [7, 8], [9, 10], [11, 12]]
        assert all(int(p.batch_state["seen"]) == 5 for p in parts)

    def test_split_refuses_a_batch_that_does_not_divide(self) -> None:
        with pytest.raises(ValueError, match="divide"):
            batch_ops.split(host_batch(size=6), 4)

    def test_concatenate_joins_rows_and_keeps_the_first_batch_state(self) -> None:
        joined = batch_ops.concatenate([host_batch(4, 0), host_batch(3, 100)])

        assert rows_of(joined) == [0, 1, 2, 3, 100, 101, 102]
        assert joined.batch_size == 7
        assert int(joined.batch_state["seen"]) == 0

    def test_stack_of_split_restores_the_batch(self) -> None:
        """A ``(K, B, ...)`` chunk holds the batch's rows in order, one part per step."""
        batch = host_batch()

        chunk = batch_ops.stack(batch_ops.split(batch, 4))

        assert chunk.data["x"].shape == (4, 2, 2)
        assert chunk.batch_state["seen"].shape == (4,)
        restored = jax.tree.map(lambda x: x.reshape(B, *x.shape[2:]), chunk.replace(batch_state={}))
        for got, want in zip(
            jax.tree.leaves(restored), jax.tree.leaves(batch.replace(batch_state={})), strict=True
        ):
            np.testing.assert_array_equal(got, want)

    def test_host_operations_on_numpy_data_stay_on_the_host(self) -> None:
        """A host reader builds batches as NumPy views; nothing reaches a device yet."""
        batch = host_batch()

        with jax.transfer_guard("disallow"):
            results = [
                batch_ops.element(batch, 2),
                batch_ops.slice_rows(batch, 1, 4),
                batch_ops.take(batch, np.array([3, 0])),
                *batch_ops.split(batch, 2),
                batch_ops.concatenate([batch, batch]),
                batch_ops.stack([batch, batch]),
            ]

        for result in results:
            assert all(
                isinstance(leaf, np.ndarray | np.generic) for leaf in jax.tree.leaves(result)
            )

    def test_a_host_slice_is_a_view(self) -> None:
        batch = host_batch()

        sliced = batch_ops.slice_rows(batch, 2, 6)

        assert np.shares_memory(sliced.data["x"], batch.data["x"])


class TestPadding:
    """Padding rows: the reserved index and weight 0, at a static shape."""

    def test_mask_turns_rows_not_kept_into_padding(self) -> None:
        keep = np.array([True, False, True, True, False, True, True, False])

        masked = jax.jit(batch_ops.mask)(device_batch(), keep)

        np.testing.assert_array_equal(is_padding(masked), ~keep)
        np.testing.assert_array_equal(masked.states[WEIGHT], keep.astype(np.float32))
        np.testing.assert_array_equal(masked.data["x"], host_batch().data["x"])
        np.testing.assert_array_equal(np.asarray(masked.indices)[keep][:, 1], np.flatnonzero(keep))

    def test_mask_scales_an_existing_weight(self) -> None:
        batch = device_batch()
        batch = batch.replace(states={**batch.states, WEIGHT: jnp.full(B, 0.5)})
        keep = np.arange(B) < 3

        masked = jax.jit(batch_ops.mask)(batch, keep)

        np.testing.assert_array_equal(masked.states[WEIGHT], np.where(keep, 0.5, 0.0))

    def test_mask_leaves_padding_as_padding(self) -> None:
        once = jax.jit(batch_ops.mask)(device_batch(), np.arange(B) < 4)

        twice = jax.jit(batch_ops.mask)(once, np.ones(B, bool))

        np.testing.assert_array_equal(is_padding(twice), np.arange(B) >= 4)
        np.testing.assert_array_equal(twice.states[WEIGHT], (np.arange(B) < 4).astype(np.float32))

    @pytest.mark.parametrize(
        "keep",
        [
            pytest.param(np.ones(B, bool), id="all-kept"),
            pytest.param(np.zeros(B, bool), id="none-kept"),
            pytest.param(np.arange(B) % 2 == 1, id="alternating"),
            pytest.param(np.array([0, 0, 1, 0, 1, 1, 0, 1], bool), id="scattered"),
        ],
    )
    def test_compact_moves_kept_rows_first_in_order(self, keep: np.ndarray) -> None:
        compacted, count = jax.jit(batch_ops.compact)(device_batch(), keep)

        kept = int(keep.sum())
        assert int(count) == kept
        assert rows_of(compacted)[:kept] == list(np.flatnonzero(keep))
        np.testing.assert_array_equal(is_padding(compacted), np.arange(B) >= kept)
        np.testing.assert_array_equal(
            compacted.states[WEIGHT], (np.arange(B) < kept).astype(np.float32)
        )
        np.testing.assert_array_equal(
            compacted.data["x"][:kept, 0], np.flatnonzero(keep).astype(np.float32)
        )

    def test_compact_never_keeps_a_padding_row(self) -> None:
        padded = jax.jit(batch_ops.mask)(device_batch(), np.arange(B) < 5)

        compacted, count = jax.jit(batch_ops.compact)(padded, np.ones(B, bool))

        assert int(count) == 5
        assert rows_of(compacted)[:5] == [0, 1, 2, 3, 4]

    @pytest.mark.parametrize(
        ("padding_rows", "expected"),
        [(np.zeros(B, bool), B), (np.ones(B, bool), 0), (np.arange(B) % 2 == 0, B // 2)],
        ids=["no-padding", "all-padding", "alternating"],
    )
    def test_record_count_excludes_padding_rows(
        self, padding_rows: np.ndarray, expected: int
    ) -> None:
        batch = host_batch()
        indices = np.where(padding_rows[:, None], PADDING_INDEX, batch.indices)

        count = jax.jit(batch_ops.record_count)(jax.device_put(batch.replace(indices=indices)))

        assert int(count) == expected

    def test_an_index_with_one_all_ones_word_is_a_record(self) -> None:
        """Only both words all ones is padding; one all-ones word is still a record."""
        indices = np.array([[0, 2**32 - 1], [2**32 - 1, 0], [2**32 - 1, 2**32 - 1]], np.uint32)
        batch = batch_ops.from_arrays({"x": np.ones(3, np.float32)}).replace(indices=indices)

        assert int(jax.jit(batch_ops.record_count)(batch)) == 2


class TestInsideAStep:
    """Every operation traces into one program: no host transfer, no compile per batch."""

    @staticmethod
    def _every_operation(batch: Batch, keep: jax.Array, i: jax.Array) -> tuple:
        parts = batch_ops.split(batch, 2)
        compacted, count = batch_ops.compact(batch, keep)
        return (
            batch_ops.element(batch, i),
            batch_ops.slice_rows(batch, 1, 5),
            batch_ops.take(batch, jnp.array([4, 2])),
            batch_ops.concatenate(parts),
            batch_ops.stack(parts),
            batch_ops.mask(batch, keep),
            compacted,
            count,
            batch_ops.record_count(batch),
        )

    def test_device_data_transfers_nothing_and_compiles_once(self) -> None:
        step = jax.jit(self._every_operation)
        keep = jax.device_put(np.arange(B) % 3 != 0)
        i = jax.device_put(np.int32(5))
        step(device_batch(), keep, i)

        with jax.transfer_guard("disallow"), expect_compiles(0):
            for offset in (8, 16, 24):
                out = step(device_batch(offset=offset), keep, i)

        assert rows_of(out[3]) == list(range(24, 32))
        assert int(out[7]) == int((np.arange(B) % 3 != 0).sum())

    def test_the_transfer_guard_positive_control_raises(self) -> None:
        """The guard is live: a NumPy batch passed to the compiled step is a host transfer."""
        step = jax.jit(self._every_operation)
        keep = jax.device_put(np.ones(B, bool))
        i = jax.device_put(np.int32(0))
        step(device_batch(), keep, i)

        with jax.transfer_guard("disallow"), pytest.raises(RuntimeError, match="[Dd]isallowed"):
            step(host_batch(), keep, i)


def test_shardings_places_rows_on_the_data_axis_and_the_batch_state_replicated() -> None:
    """One ``P('data')`` for the whole batch fails on its 0-d batch-level leaf; the prefix works."""
    mesh = Mesh(np.array(jax.devices()), ("data",))
    rows, replicated = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
    batch = host_batch(size=len(jax.devices()))

    prefix = batch_ops.shardings(batch, rows, replicated)
    placed = place_batch_on_shards(batch, prefix)

    assert placed.data["x"].sharding == rows and placed.indices.sharding == rows
    assert placed.epochs.sharding == rows and placed.draws.sharding == rows
    assert placed.states["score"].sharding == rows
    assert placed.batch_state["seen"].sharding == replicated
    np.testing.assert_array_equal(placed.indices, batch.indices)
    with pytest.raises(ValueError, match="batch_state"):
        place_batch_on_shards(batch, rows)


def maybe_batch(size: int = B, offset: int = 0) -> Batch:
    """A host batch with a nested missing-capable field: row ``i`` has a depth when ``i % 3``."""
    batch = host_batch(size, offset)
    present = (np.arange(size) + offset) % 3 != 0
    values = np.where(present[:, None], np.arange(size, dtype=np.float32)[:, None] + offset, 0.0)
    depth = Maybe(np.repeat(values, 2, axis=1).astype(np.float32), present)
    return batch.replace(data={**batch.data, "nested": {**batch.data["nested"], "depth": depth}})


def depth_of(batch: Batch | Element) -> Maybe:
    field = batch.data["nested"]["depth"]
    assert isinstance(field, Maybe)
    return field


class TestMissingValues:
    """A ``Maybe`` at any depth moves through every batch operation with its rows."""

    def test_element_gives_one_record_its_presence(self) -> None:
        record = batch_ops.element(maybe_batch(), 4)

        field = depth_of(record)
        assert field.present.shape == () and bool(field.present)
        np.testing.assert_array_equal(field.value, [4.0, 4.0])
        missing = depth_of(batch_ops.element(maybe_batch(), 3))
        assert not bool(missing.present)
        np.testing.assert_array_equal(missing.value, [0.0, 0.0])

    def test_slice_take_split_and_concatenate_keep_presence_with_its_rows(self) -> None:
        batch = maybe_batch()
        present = np.arange(B) % 3 != 0

        np.testing.assert_array_equal(
            depth_of(batch_ops.slice_rows(batch, 2, 5)).present, present[2:5]
        )
        np.testing.assert_array_equal(
            depth_of(batch_ops.take(batch, np.array([6, 1, 0]))).present, present[[6, 1, 0]]
        )
        parts = batch_ops.split(batch, 4)
        for k, part in enumerate(parts):
            np.testing.assert_array_equal(depth_of(part).present, present[2 * k : 2 * k + 2])
        joined = batch_ops.concatenate(parts)
        np.testing.assert_array_equal(depth_of(joined).present, present)
        np.testing.assert_array_equal(depth_of(joined).value, depth_of(batch).value)

    def test_stack_of_split_restores_the_batch(self) -> None:
        batch = maybe_batch()

        chunk = batch_ops.stack(batch_ops.split(batch, 4))

        assert depth_of(chunk).value.shape == (4, 2, 2)
        assert depth_of(chunk).present.shape == (4, 2)
        np.testing.assert_array_equal(depth_of(chunk).present.reshape(B), depth_of(batch).present)

    def test_stacked_records_become_a_batch(self) -> None:
        records = [
            Element(
                {"depth": Maybe(np.full(3, i, np.float32), np.array(i % 2 == 0))},
                index=np.array([0, i], np.uint32),
            )
            for i in range(4)
        ]

        batch = batch_ops.from_stacked(next(iter(grain.MapDataset.source(records).batch(4))))

        field = batch.data["depth"]
        assert isinstance(field, Maybe)
        np.testing.assert_array_equal(field.present, [True, False, True, False])
        assert field.value.shape == (4, 3)

    def test_host_operations_keep_presence_on_the_host(self) -> None:
        batch = maybe_batch()

        with jax.transfer_guard("disallow"):
            results = [
                batch_ops.element(batch, 2),
                batch_ops.take(batch, np.array([3, 0])),
                *batch_ops.split(batch, 2),
                batch_ops.concatenate([batch, batch]),
                batch_ops.stack([batch, batch]),
            ]

        for result in results:
            assert isinstance(depth_of(result).present, np.ndarray | np.generic)

    @pytest.mark.parametrize(
        "keep", [np.arange(B) % 2 == 1, np.array([0, 0, 1, 0, 1, 1, 0, 1], bool)]
    )
    def test_mask_and_compact_keep_values_and_presence(self, keep: np.ndarray) -> None:
        batch = jax.device_put(maybe_batch())

        masked = jax.jit(batch_ops.mask)(batch, keep)
        compacted, count = jax.jit(batch_ops.compact)(batch, keep)

        np.testing.assert_array_equal(depth_of(masked).present, depth_of(batch).present)
        np.testing.assert_array_equal(depth_of(masked).value, depth_of(batch).value)
        kept = int(count)
        np.testing.assert_array_equal(
            depth_of(compacted).present[:kept], np.asarray(depth_of(batch).present)[keep]
        )
        np.testing.assert_array_equal(
            depth_of(compacted).value[:kept], np.asarray(depth_of(batch).value)[keep]
        )

    def test_every_operation_in_one_step_compiles_once_and_transfers_nothing(self) -> None:
        step = jax.jit(TestInsideAStep._every_operation)
        keep = jax.device_put(np.arange(B) % 3 != 0)
        i = jax.device_put(np.int32(5))
        step(jax.device_put(maybe_batch()), keep, i)

        with jax.transfer_guard("disallow"), expect_compiles(0):
            for offset in (1, 2, 3):
                out = step(jax.device_put(maybe_batch(offset=offset)), keep, i)

        np.testing.assert_array_equal(depth_of(out[3]).present, (np.arange(B) + 3) % 3 != 0)

    def test_placement_puts_presence_on_the_row_sharding(self) -> None:
        mesh = Mesh(np.array(jax.devices()), ("data",))
        rows, replicated = NamedSharding(mesh, P("data")), NamedSharding(mesh, P())
        batch = maybe_batch(size=len(jax.devices()))

        placed = place_batch_on_shards(batch, batch_ops.shardings(batch, rows, replicated))

        assert depth_of(placed).present.sharding == rows
        assert depth_of(placed).value.sharding == rows
        np.testing.assert_array_equal(depth_of(placed).present, depth_of(batch).present)
