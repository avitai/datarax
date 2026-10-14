"""``MemorySource.of_columns``: decoded columns, nested by field, held as given with provenance.

The columns a build decodes (``materialize``'s RAM result) are nested by field and come with each
record's provenance, so ``of_columns`` takes them as they are, not copied. They are held as every
eager source's columns are: in a host holder, out of NNX state, so no dataset-shaped leaf is in
``nnx.state`` and no transform of a module holding the source transfers them, in any of Flax's
modes. A column held as ``nnx.data`` is the control: it is state, transferred as a program's
input. The traced read takes the columns' device copy as an argument, as every eager source's
does (``device_columns()``), and is refused without it, naming it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources.decoded_copy import _units
from datarax.sources.decoded_copy._units import build_in_memory, source_layout
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_RECORDS = 40


def _columns() -> dict[str, Any]:
    return {
        "image": np.arange(_RECORDS * 6, dtype=np.uint8).reshape(_RECORDS, 2, 3),
        "label": np.arange(_RECORDS, dtype=np.int64),
        "meta": {"score": np.linspace(0, 1, _RECORDS * 2, dtype=np.float32).reshape(-1, 2)},
    }


def _flat_columns() -> dict[str, Any]:
    """The same columns as one flat mapping, as ``MemorySource`` takes them."""
    columns = _columns()
    return {"image": columns["image"], "label": columns["label"], "score": columns["meta"]["score"]}


def _provenance() -> tuple[dict[str, Any], ...]:
    return tuple({"name": f"r{i}"} for i in range(_RECORDS))


class _Holder(nnx.Module):
    """A module holding a source, as a pipeline does."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.w = nnx.Param(jnp.ones(()))


class _StateHolder(nnx.Module):
    """The control: a module holding the columns as ``nnx.data``, NNX state."""

    def __init__(self, columns: Any) -> None:
        self.columns = nnx.data(columns)
        self.w = nnx.Param(jnp.ones(()))


_MODES = [(True, True), (True, False), (False, False)]
"""Flax's ``(graph, graph_updates)`` modes: the default, the Trainer's binding, tree mode."""


def _step(graph: bool, graph_updates: bool) -> Any:
    """A step over a module holding a source; every array of its state is an argument.

    ``keep_unused`` keeps an argument the step does not read, so a dataset held as state is
    transferred as a program's input would be.
    """
    return nnx.jit(lambda m: m.w * 2, graph=graph, graph_updates=graph_updates, keep_unused=True)


class TestOfColumns:
    def test_serves_the_columns_and_provenance_it_is_given(self) -> None:
        columns = _columns()
        source = MemorySource.of_columns(columns, _provenance())

        assert source.record_identity is RecordIdentity.INDEXED
        assert len(source) == _RECORDS
        words = to_words(np.array([7, 1, 39], dtype=np.uint64))
        batch = source.get_batch(words)
        np.testing.assert_array_equal(batch.data["image"], columns["image"][[7, 1, 39]])
        np.testing.assert_array_equal(
            batch.data["meta"]["score"], columns["meta"]["score"][[7, 1, 39]]
        )
        assert batch.data["label"].dtype == np.int64
        assert [dict(p) for p in source.provenance(words)] == [
            {"name": f"r{i}"} for i in (7, 1, 39)
        ]
        held = source.get_batch(to_words(np.arange(3, dtype=np.uint64)), contiguous=True)
        assert np.shares_memory(held.data["label"], columns["label"]), "held without a copy"

    def test_holds_no_dataset_shaped_leaf_in_nnx_state(self) -> None:
        source = MemorySource.of_columns(_columns(), _provenance())

        for leaf in jax.tree.leaves(nnx.state(source)):
            assert np.ndim(leaf) == 0 or np.shape(leaf)[0] != _RECORDS

    def test_positive_control_columns_held_as_nnx_data_are_state(self) -> None:
        holder = _StateHolder(_flat_columns())

        assert any(
            np.ndim(leaf) and np.shape(leaf)[0] == _RECORDS
            for leaf in jax.tree.leaves(nnx.state(holder))
        )

    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_no_transform_of_a_module_holding_it_transfers_the_columns(
        self, graph: bool, graph_updates: bool
    ) -> None:
        holder = _Holder(MemorySource.of_columns(_columns(), _provenance()))
        step = _step(graph, graph_updates)
        step(holder)  # the first call places the parameter

        with jax.transfer_guard_host_to_device("disallow"):
            for _ in range(3):
                step(holder)
        words = to_words(np.arange(1, dtype=np.uint64))
        assert isinstance(holder.source.get_batch(words).data["label"], np.ndarray)

    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_positive_control_columns_as_state_are_transferred(
        self, graph: bool, graph_updates: bool
    ) -> None:
        holder = _StateHolder(_flat_columns())
        step = _step(graph, graph_updates)

        with pytest.raises(jax.errors.JaxRuntimeError, match="Disallowed host-to-device"):
            with jax.transfer_guard_host_to_device("disallow"):
                for _ in range(3):
                    step(holder)

    def test_the_traced_read_takes_the_columns_device_copy_and_is_refused_without_it(
        self,
    ) -> None:
        source = MemorySource.of_columns(_columns(), _provenance())
        words = jnp.asarray(to_words(np.array([5, 2], dtype=np.uint64)))
        columns = source.device_columns()

        read = jax.jit(lambda w, c: source.get_records(w, columns=c))(words, columns)
        np.testing.assert_array_equal(read["meta"]["score"], _columns()["meta"]["score"][[5, 2]])
        with pytest.raises(TypeError, match="device_columns"):
            jax.jit(source.get_records)(words)

    def test_a_pipeline_over_it_serves_what_one_over_the_same_records_does(self) -> None:
        held = MemorySource.of_columns(_columns())
        state = MemorySource(MemorySourceConfig(), {"label": _columns()["label"]})

        def epoch(source: Any) -> list[np.ndarray]:
            pipe = Pipeline(source=source, stages=[], batch_size=8, rngs=nnx.Rngs(3), shuffle=True)
            return [np.asarray(batch["label"]) for batch in pipe]

        served = epoch(held)
        assert [list(b) for b in served] == [list(b) for b in epoch(state)]
        assert sorted(np.concatenate(served).tolist()) == list(range(_RECORDS))

    def test_refuses_columns_of_different_lengths_and_provenance_of_another_count(self) -> None:
        with pytest.raises(ValueError, match="lengths"):
            MemorySource.of_columns({"a": np.zeros(3), "b": np.zeros(4)})
        with pytest.raises(ValueError, match="provenance"):
            MemorySource.of_columns({"a": np.zeros(3)}, ({"n": 1},))


def test_the_one_thread_build_writes_every_unit_through_the_one_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemorySource(
        MemorySourceConfig(), [{"x": np.full(2, i, np.int16), "n": f"{i}"} for i in range(11)]
    )
    written: list[range] = []
    write_unit = _units.write_unit

    def counting(src: Any, rows: range, *args: Any, **kwargs: Any) -> Any:
        written.append(rows)
        return write_unit(src, rows, *args, **kwargs)

    monkeypatch.setattr(_units, "write_unit", counting)

    columns, provenance = build_in_memory(source, unit_records=4)

    assert written == [range(0, 4), range(4, 8), range(8, 11)]
    np.testing.assert_array_equal(
        columns["x"], np.repeat(np.arange(11, dtype=np.int16), 2).reshape(11, 2)
    )
    assert provenance == tuple({"n": f"{i}"} for i in range(11))
    assert source_layout(source).nbytes == columns["x"].nbytes
