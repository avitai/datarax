"""datarax's own state is typed: the weight, the marks and the mix are attributes, not state keys.

``Element.weight`` and ``Batch.weights`` are always arrays: 1 for a record, 0 for a row whose index
is ``PADDING_INDEX``, set wherever an index is written. ``masked`` and ``imputed`` are bool mappings
in the data's own nesting, empty until an operator marks a field. ``Batch.mix`` is ``None`` until a
whole-batch mixing operator writes ``Mix(partners, lams)``, one coefficient per row. Every typed
attribute has the record axis first, so the row operations treat it as rows, and ``state`` and
``batch_state`` hold only the user's names.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

import datarax
from datarax import core as datarax_core
from datarax.core import batch_ops, element_batch
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import ArrayValue, Batch, Element, PADDING_INDEX
from datarax.core.index_words import to_words
from datarax.core.operator import OperatorModule
from datarax.pipeline.dag import name_records
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _data(size: int = 4) -> dict[str, Any]:
    return {
        "x": np.arange(size * 3, dtype=np.float32).reshape(size, 3),
        "meta": {"score": np.arange(size, dtype=np.float32)},
    }


def _batch(size: int = 4) -> Batch:
    return batch_ops.from_arrays(_data(size))


def _with_padding(rows: list[bool]) -> np.ndarray:
    """uint32 ``(B, 2)`` words: record ``i`` where ``rows[i]``, else ``PADDING_INDEX``."""
    words = np.asarray(to_words(np.arange(len(rows))), np.uint32)
    return np.where(np.asarray(rows)[:, None], words, PADDING_INDEX)


class TestTheWeight:
    """Always an array: 1 for a record, 0 for padding, set wherever an index is written."""

    def test_a_record_s_weight_is_one_by_default(self) -> None:
        element = Element({"x": np.zeros(3, np.float32)})
        assert element.weight.dtype == np.float32
        assert element.weight.shape == ()
        assert float(element.weight) == 1.0

    def test_from_arrays_gives_every_record_weight_one_as_a_host_array(self) -> None:
        batch = _batch()
        assert isinstance(batch.weights, np.ndarray)
        assert batch.weights.dtype == np.float32
        assert batch.weights.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_a_padding_index_is_weight_zero_at_construction(self) -> None:
        batch = batch_ops.from_arrays(_data(), indices=_with_padding([True, True, False, True]))
        assert batch.weights.tolist() == [1.0, 1.0, 0.0, 1.0]

    def test_from_stacked_keeps_each_record_s_weight_and_zeroes_padding(self) -> None:
        records = [
            Element(
                {"x": np.zeros(3, np.float32)}, index=index, weight=np.asarray(weight, np.float32)
            )
            for index, weight in zip(
                _with_padding([True, True, False]), [0.5, 1.0, 0.7], strict=True
            )
        ]
        batch = batch_ops.from_stacked(batch_ops.stack(records))
        assert np.asarray(batch.weights).tolist() == pytest.approx([0.5, 1.0, 0.0])

    def test_mask_and_compact_write_weights(self) -> None:
        keep = jnp.asarray([True, False, True, False])
        masked = batch_ops.mask(_batch(), keep)
        assert np.asarray(masked.weights).tolist() == [1.0, 0.0, 1.0, 0.0]
        compacted, count = batch_ops.compact(_batch(), keep)
        assert int(count) == 2
        assert np.asarray(compacted.weights).tolist() == [1.0, 1.0, 0.0, 0.0]

    def test_name_records_zeroes_a_padding_index(self) -> None:
        batch = name_records(_batch(), _with_padding([True, False, True, True]), 0)
        assert np.asarray(batch.weights).tolist() == [1.0, 0.0, 1.0, 1.0]

    def test_a_host_batch_stays_numpy_in_every_leaf(self) -> None:
        batch = batch_ops.from_arrays(_data(), indices=_with_padding([True, False, True, True]))
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))

    def test_a_padding_row_keeps_the_batch_s_structure(self) -> None:
        plain = _batch()
        padded = batch_ops.from_arrays(_data(), indices=_with_padding([True, True, True, False]))
        assert jax.tree.structure(plain) == jax.tree.structure(padded)


class TestTheMarks:
    """``masked`` and ``imputed``: bool mappings in the data's own nesting, empty by default."""

    def test_records_and_batches_start_with_no_marks(self) -> None:
        assert Element({"x": np.zeros(3)}).masked == {}
        assert Element({"x": np.zeros(3)}).imputed == {}
        assert _batch().masked == {}
        assert _batch().imputed == {}

    def test_a_nested_mark_sits_at_its_field_s_path(self) -> None:
        batch = _batch().replace(masked={"meta": {"score": np.ones(4, bool)}})
        row = batch_ops.element(batch, 1)
        assert bool(row.masked["meta"]["score"])
        assert row.weight.shape == ()


class TestTheMix:
    """``Batch.mix``: ``None`` until a mixing operator writes one coefficient per row."""

    def test_a_built_batch_has_no_mix(self) -> None:
        assert _batch().mix is None

    def test_mix_is_exported_beside_element_and_batch(self) -> None:
        assert datarax.Mix is element_batch.Mix
        assert datarax_core.Mix is element_batch.Mix

    def test_concatenate_keeps_each_batch_s_coefficients(self) -> None:
        def mixed(lam: float) -> Batch:
            mix = element_batch.Mix(
                partners=np.arange(4, dtype=np.int32)[::-1].copy(),
                lams=np.full(4, lam, np.float32),
            )
            return _batch().replace(mix=mix)

        joined = batch_ops.concatenate([mixed(0.7), mixed(0.2)])
        assert joined.mix is not None
        assert np.asarray(joined.mix.lams).tolist() == pytest.approx([0.7] * 4 + [0.2] * 4)


class TestRowOperations:
    """Every typed attribute is a row leaf: each row operation carries it with its row."""

    @staticmethod
    def _marked() -> Batch:
        mix = element_batch.Mix(
            partners=np.array([1, 0, 3, 2], np.int32), lams=np.linspace(0, 1, 4, dtype=np.float32)
        )
        return batch_ops.from_arrays(
            _data(), indices=_with_padding([True, True, True, False])
        ).replace(
            masked={"x": np.array([True, False, True, False])},
            imputed={"meta": {"score": np.array([False, True, False, True])}},
            mix=mix,
        )

    def test_split_and_slice_rows(self) -> None:
        first, second = batch_ops.split(self._marked(), 2)
        assert second.mix is not None
        assert np.asarray(second.weights).tolist() == [1.0, 0.0]
        assert np.asarray(second.masked["x"]).tolist() == [True, False]
        assert np.asarray(second.mix.lams).tolist() == pytest.approx([2 / 3, 1.0])
        assert np.asarray(first.imputed["meta"]["score"]).tolist() == [False, True]

    def test_as_chunk_and_take(self) -> None:
        chunk = batch_ops.as_chunk(self._marked(), 2)
        assert chunk.mix is not None
        assert np.asarray(chunk.weights).shape == (2, 2)
        assert np.asarray(chunk.mix.partners).shape == (2, 2)
        taken = batch_ops.take(self._marked(), np.array([3, 0]))
        assert np.asarray(taken.weights).tolist() == [0.0, 1.0]
        assert np.asarray(taken.masked["x"]).tolist() == [False, True]

    def test_element_carries_the_row_s_weight_and_marks_and_no_mix(self) -> None:
        row = batch_ops.element(self._marked(), 3)
        assert float(row.weight) == 0.0
        assert bool(row.imputed["meta"]["score"])
        assert not hasattr(row, "mix")

    def test_shardings_put_every_typed_attribute_on_the_row_sharding(self) -> None:
        mesh = jax.make_mesh((1,), ("data",), axis_types=(jax.sharding.AxisType.Auto,))
        row = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        shardings = batch_ops.shardings(self._marked(), row, replicated)
        assert shardings.mix is not None
        assert shardings.weights == row
        assert shardings.mix.lams == row
        assert shardings.masked["x"] == row


class _Reweights(OperatorModule):
    """Scales each record's weight by a learned factor and marks its ``x``."""

    calls = 0

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.scale = nnx.Param(jnp.asarray(2.0))

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        type(self).calls += 1
        return element.replace(
            weight=element.weight * self.scale[...],
            masked={**element.masked, "x": element.data["x"] > 4.0},
        )


def _weighted_loss(batch: Batch) -> jax.Array:
    return jnp.sum(batch.weights * jnp.sum(batch["x"], axis=-1))


@pytest.mark.parametrize("graph", [True, False])
class TestUnderNnxTransforms:
    """The typed attributes trace as today's fields do, in both NNX modes."""

    def test_nnx_jit_traces_once_and_once_more_for_a_new_batch_size(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch).weights), graph=graph)
        op = _Reweights()
        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(op, _batch())
            for _ in range(3):
                apply(_Reweights(), _batch())
        with counter.expect(new_traces=1):
            apply(op, _batch(8))

    def test_a_batch_carrying_marks_is_another_program(self, graph: bool) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda op, batch: op(batch).weights), graph=graph)
        with counter.expect(new_traces=2):  # the different-spec control
            apply(_Reweights(), _batch())
            apply(_Reweights(), _batch().replace(imputed={"x": np.zeros(4, bool)}))

    def test_nnx_vmap_and_nnx_scan_over_a_chunk(self, graph: bool) -> None:
        stacked = batch_ops.stack([_batch(), _batch()])
        weights = nnx.vmap(lambda op, batch: op(batch).weights, in_axes=(None, 0), graph=graph)(
            _Reweights(), stacked
        )
        assert weights.shape == (2, 4)

        def body(op: _Reweights, total: jax.Array, batch: Batch) -> jax.Array:
            return total + _weighted_loss(op(batch))

        scan = nnx.scan(body, in_axes=(None, nnx.Carry, 0), out_axes=nnx.Carry, graph=graph)
        expected = 2 * 2.0 * float(np.sum(_data()["x"]))
        assert float(scan(_Reweights(), jnp.float32(0), stacked)) == pytest.approx(expected)

    def test_value_and_grad_through_the_weight(self, graph: bool) -> None:
        def loss(op: _Reweights, batch: Batch) -> jax.Array:
            return _weighted_loss(op(batch))

        value, grads = nnx.value_and_grad(loss, graph=graph)(_Reweights(), _batch())
        total = float(np.sum(_data()["x"]))
        assert float(value) == pytest.approx(2.0 * total)
        # d/d scale of sum(scale * sum_x) is sum_x, the closed form of the loss.
        assert float(jax.tree.leaves(grads)[0]) == pytest.approx(total)


class TestUnderJaxTransforms:
    """The typed attributes under plain jax: one program per structure, gradients exact."""

    def test_jax_jit_over_a_static_graphdef_and_with_a_closure(self) -> None:
        graphdef, state = nnx.split(_Reweights())
        counter = TraceCounter()

        @jax.jit
        @counter.wrap
        def apply(state: Any, batch: Batch) -> ArrayValue:
            return nnx.merge(graphdef, state)(batch).weights

        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(state, _batch())
            apply(nnx.split(_Reweights())[1], _batch())
        op = _Reweights()
        closure = jax.jit(lambda batch: op(batch).weights)
        assert np.asarray(closure(_batch())).tolist() == [2.0] * 4

    def test_jax_vmap_lax_scan_and_grad(self) -> None:
        graphdef, params, rest = nnx.split(_Reweights(), nnx.Param, ...)
        stacked = batch_ops.stack([_batch(), _batch(), _batch()])

        def apply(params: Any, batch: Batch) -> Batch:
            return nnx.merge(graphdef, params, rest)(batch)

        weights = jax.jit(jax.vmap(lambda batch: apply(params, batch).weights))(stacked)
        assert weights.shape == (3, 4)

        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, None]:
            return total + _weighted_loss(apply(params, batch)), None

        total, _ = jax.lax.scan(body, jnp.float32(0), stacked)
        assert float(total) == pytest.approx(3 * 2.0 * float(np.sum(_data()["x"])))
        grads = jax.jit(jax.grad(lambda p, b: _weighted_loss(apply(p, b))))(params, _batch())
        assert float(grads["scale"][...]) == pytest.approx(float(np.sum(_data()["x"])))

    def test_equal_batches_share_one_treedef_and_its_hash(self) -> None:
        first, second = (
            _batch(),
            batch_ops.from_arrays(
                {"x": np.ones((4, 3), np.float32), "meta": {"score": np.ones(4, np.float32)}}
            ),
        )
        assert jax.tree.structure(first) == jax.tree.structure(second)
        assert hash(jax.tree.structure(first)) == hash(jax.tree.structure(second))


class TestInAPipeline:
    """A stage writing the weight and a mark, through each of the pipeline's paths."""

    @pytest.mark.parametrize("path", ["step", "iteration", "scan", "session"])
    def test_every_path_serves_the_written_weight_and_mark(self, path: str) -> None:
        data = {"x": np.arange(12, dtype=np.float32).reshape(4, 3)}
        pipe = Pipeline(
            source=MemorySource(MemorySourceConfig(), data),
            stages=[_Reweights()],
            batch_size=2,
            rngs=nnx.Rngs(0),
        )
        try:
            if path == "step":
                batch = pipe.step()
            elif path == "iteration":
                batch = next(iter(pipe))
            elif path == "scan":
                total = pipe.scan(lambda batch: jnp.sum(batch.weights), length=2)
                assert float(jnp.sum(total)) == pytest.approx(2 * 2 * 2.0)
                return
            else:
                batch = next(iter(pipe.session()))
        finally:
            pipe.close()
        assert np.asarray(batch.weights).tolist() == [2.0, 2.0]
        assert np.asarray(batch.masked["x"]).shape == (2, 3)
