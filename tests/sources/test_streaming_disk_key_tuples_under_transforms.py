"""A disk source's traced read under the transforms it meets, for one-key and key-tuple columns.

The traced read (``get_records``) issues the disk read through ``io_callback`` and rebuilds the
record from the columns' key tuples at trace time. Each cell runs over two column sets: one-key
names (``"score"``, the common case, served the same before and after key-tuple naming) and key
tuples (``("meta", "score")`` nested, ``("a/b",)`` one key). It holds under ``jax.jit`` (a
closure, and a static graphdef with a rebuilt module), ``nnx.jit`` in Flax's three modes,
``jax.vmap`` over ``(K, B)``, ``lax.scan`` over a chunk, ``nnx.vmap``, ``nnx.scan``, ``jax.grad``
and ``nnx.value_and_grad``, and a Pipeline's step, scan and session. A cached call rebuilds
nothing. Two builds of one source have equal configs and compile apart, because the holder of
their memory maps compares by identity (D3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import by_package, counted_calls, TraceCounter
from substrax.testing.compiles import compiled_programs

from datarax.core.element_batch import Batch
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources.streaming_disk_source import (
    DiskColumn,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)
from tests.test_common.compiles import expect_first_call_compiles


_ROWS = 16
_MODES = [(True, True), (True, False), (False, False)]
"""Flax's ``(graph, graph_updates)`` modes: the default, the Trainer's binding, tree mode."""


def _score() -> np.ndarray:
    return np.linspace(-1.0, 1.0, _ROWS * 2, dtype=np.float32).reshape(_ROWS, 2)


def _literal() -> np.ndarray:
    return np.arange(_ROWS * 3, dtype=np.float32).reshape(_ROWS, 3) / 10


def _label() -> np.ndarray:
    return np.arange(_ROWS, dtype=np.int32) % 4


_NAMES: dict[str, tuple[Any, ...]] = {
    "one-key": ("score", "ab", "label"),
    "key-tuples": (("meta", "score"), ("a/b",), "label"),
}
"""Each column set's names, in the order of :func:`_arrays`."""


def _arrays() -> tuple[np.ndarray, ...]:
    return _score(), _literal(), _label()


def _write(directory: Path, names: tuple[Any, ...], rows: int = _ROWS) -> tuple[DiskColumn, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    columns = []
    for index, (name, array) in enumerate(zip(names, _arrays(), strict=True)):
        np.save(directory / f"column-{index}.npy", array[:rows])
        columns.append(DiskColumn.from_npy(name, directory / f"column-{index}.npy"))
    return tuple(columns)


@pytest.fixture(params=list(_NAMES))
def columns(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[DiskColumn, ...]:
    return _write(tmp_path / "columns", _NAMES[request.param])


def _keys(column: DiskColumn) -> tuple[str, ...]:
    """The field a column names: a string is one key, a tuple its keys."""
    return (column.name,) if isinstance(column.name, str) else tuple(column.name)


def _get(records: Any, column: DiskColumn) -> Any:
    node = records
    for key in _keys(column):
        node = node[key]
    return node


def _source(columns: tuple[DiskColumn, ...]) -> StreamingDiskSource:
    return StreamingDiskSource(StreamingDiskSourceConfig(columns=columns))


def _indices(rows: list[int]) -> jax.Array:
    return jnp.asarray(to_words(np.asarray(rows, dtype=np.uint64)))


def _served(records: Any, rows: list[int], columns: tuple[DiskColumn, ...]) -> None:
    """The record's structure is the columns' key tuples, and its values the rows'."""
    served = {jax.tree_util.keystr(path) for path, _ in jax.tree.leaves_with_path(records)}
    expected = {jax.tree_util.keystr(tuple(map(jax.tree_util.DictKey, _keys(c)))) for c in columns}
    assert served == expected
    for column, array in zip(columns, _arrays(), strict=True):
        np.testing.assert_array_equal(np.asarray(_get(records, column)), array[rows])


class _Reader(nnx.Module):
    """A module holding the source and one parameter, as a training step holds them."""

    def __init__(self, source: StreamingDiskSource) -> None:
        self.source = source
        self.scale = nnx.Param(jnp.float32(1.5))

    def __call__(self, indices: jax.Array) -> jax.Array:
        records = self.source.get_records(indices)
        score, literal = (_get(records, c) for c in self.source.config.columns[:2])
        return jnp.mean((self.scale[...] * score) ** 2) + jnp.mean(self.scale[...] * literal)


def _loss_in_memory(scale: jax.Array, rows: list[int]) -> jax.Array:
    """The same loss over the same arrays held in memory: the gradient's reference."""
    return jnp.mean((scale * jnp.asarray(_score()[rows])) ** 2) + jnp.mean(
        scale * jnp.asarray(_literal()[rows])
    )


class TestJaxJit:
    def test_a_closure_serves_the_key_tuples_and_traces_once(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        source = _source(columns)
        counter = TraceCounter()
        read = jax.jit(counter.wrap(source.get_records))

        with counter.expect(new_traces=1):
            _served(read(_indices([3, 0, 7])), [3, 0, 7], columns)
        with counter.expect(new_traces=0):
            _served(read(_indices([1, 2, 15])), [1, 2, 15], columns)
        with counter.expect(new_traces=1):  # positive control: another B is another trace
            _served(read(_indices([4, 5])), [4, 5], columns)

    def test_a_static_graphdef_and_a_rebuilt_module_trace_once(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        source = _source(columns)
        graphdef, state = nnx.split(source)
        counter = TraceCounter()

        @jax.jit
        @counter.wrap
        def read(state: Any, indices: jax.Array) -> Any:
            return nnx.merge(graphdef, state).get_records(indices)

        with counter.expect(new_traces=1):
            _served(read(state, _indices([2, 9])), [2, 9], columns)
        with counter.expect(new_traces=0):
            rebuilt = nnx.split(nnx.merge(graphdef, state))[1]
            _served(read(rebuilt, _indices([5, 6])), [5, 6], columns)


class TestNnxJit:
    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_three_calls_and_three_on_rebuilt_modules_trace_once(
        self, columns: tuple[DiskColumn, ...], graph: bool, graph_updates: bool
    ) -> None:
        reader = _Reader(_source(columns))
        counter = TraceCounter()
        step = nnx.jit(counter.wrap(lambda m, i: m(i)), graph=graph, graph_updates=graph_updates)

        with counter.expect(new_traces=1):
            losses = [step(reader, _indices([1, 2, 3])) for _ in range(3)]
        graphdef, state = nnx.split(reader)
        with counter.expect(new_traces=0):
            losses += [step(nnx.merge(graphdef, state), _indices([1, 2, 3])) for _ in range(3)]

        expected = _loss_in_memory(jnp.float32(1.5), [1, 2, 3])
        for loss in losses:
            np.testing.assert_allclose(np.asarray(loss), np.asarray(expected), rtol=0, atol=1e-6)

    def test_a_second_build_of_one_source_has_its_config_and_compiles_apart(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        """D3: the memory maps' holder compares by identity, so two builds are two programs."""
        first, second = _source(columns), _source(columns)
        counter = TraceCounter()
        read = nnx.jit(counter.wrap(lambda s, i: s.get_records(i)))

        assert first.config == second.config
        assert hash(first.config) == hash(second.config)
        assert nnx.graphdef(first) != nnx.graphdef(second)
        with counter.expect(new_traces=1):
            read(first, _indices([0, 1]))
            read(first, _indices([2, 3]))
        with counter.expect(new_traces=1):
            _served(read(second, _indices([4, 5])), [4, 5], columns)

    def test_another_spec_traces_again(
        self, columns: tuple[DiskColumn, ...], tmp_path: Path
    ) -> None:
        source = _source(columns)
        names = tuple(column.name for column in columns)
        shorter = _source(_write(tmp_path / "shorter", names, rows=_ROWS // 2))
        counter = TraceCounter()
        read = nnx.jit(counter.wrap(lambda s, i: s.get_records(i)))

        with counter.expect(new_traces=1):
            read(source, _indices([0, 1]))
        with counter.expect(new_traces=1):
            _served(read(shorter, _indices([0, 1])), [0, 1], columns)


class TestVectorisedAndScanned:
    def test_jax_vmap_over_k_and_b(self, columns: tuple[DiskColumn, ...]) -> None:
        source = _source(columns)
        indices = _indices(list(range(6))).reshape(2, 3, 2)

        records = jax.vmap(source.get_records)(indices)

        assert _get(records, columns[1]).shape == (2, 3, 3)
        np.testing.assert_array_equal(
            np.asarray(_get(records, columns[0])), _score()[:6].reshape(2, 3, 2)
        )
        np.testing.assert_array_equal(
            np.asarray(_get(records, columns[1])), _literal()[:6].reshape(2, 3, 3)
        )

    def test_lax_scan_over_a_chunk_equals_its_reads(self, columns: tuple[DiskColumn, ...]) -> None:
        source = _source(columns)
        chunk = _indices(list(range(6))).reshape(2, 3, 2)

        _, scanned = jax.lax.scan(lambda c, i: (c, source.get_records(i)), None, chunk)

        for k in range(2):
            step = jax.tree.map(lambda leaf, k=k: leaf[k], scanned)
            _served(step, list(range(3 * k, 3 * k + 3)), columns)

    def test_nnx_vmap_and_nnx_scan_over_the_module(self, columns: tuple[DiskColumn, ...]) -> None:
        reader = _Reader(_source(columns))
        chunk = _indices(list(range(6))).reshape(2, 3, 2)

        vmapped = nnx.vmap(lambda m, i: m(i), in_axes=(None, 0))(reader, chunk)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def scanned(m: _Reader, i: jax.Array) -> tuple[_Reader, jax.Array]:
            return m, m(i)

        _, scanned_losses = scanned(reader, chunk)
        for j in range(2):
            expected = _loss_in_memory(jnp.float32(1.5), list(range(3 * j, 3 * j + 3)))
            np.testing.assert_allclose(np.asarray(vmapped[j]), np.asarray(expected), atol=1e-6)
            np.testing.assert_allclose(
                np.asarray(scanned_losses[j]), np.asarray(expected), atol=1e-6
            )


class TestGradients:
    """DD-D12: gradients over a ``Param`` and named float fields, the read's rows constants."""

    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_a_param_s_gradient_through_the_read_equals_the_in_memory_one(
        self, columns: tuple[DiskColumn, ...], graph: bool, graph_updates: bool
    ) -> None:
        reader = _Reader(_source(columns))
        rows = [3, 8, 12]

        @nnx.jit(graph=graph, graph_updates=graph_updates)
        def step(m: _Reader, indices: jax.Array) -> tuple[jax.Array, Any]:
            return nnx.value_and_grad(lambda m: m(indices))(m)

        loss, grads = step(reader, _indices(rows))
        expected_loss, expected_grad = jax.value_and_grad(_loss_in_memory)(jnp.float32(1.5), rows)

        np.testing.assert_allclose(np.asarray(loss), np.asarray(expected_loss), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(grads.scale[...]), np.asarray(expected_grad), rtol=1e-6, atol=1e-6
        )

    def test_a_named_float_field_s_gradient_downstream_of_the_read(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        source = _source(columns)
        rows = [0, 5]
        records = jax.jit(source.get_records)(_indices(rows))

        def loss(score: jax.Array, literal: jax.Array) -> jax.Array:
            return jnp.sum(score**2) + 3.0 * jnp.sum(literal)

        d_score, d_literal = jax.grad(loss, argnums=(0, 1))(
            _get(records, columns[0]), _get(records, columns[1])
        )

        np.testing.assert_allclose(np.asarray(d_score), 2.0 * _score()[rows], atol=1e-6)
        np.testing.assert_array_equal(np.asarray(d_literal), np.full((2, 3), 3.0, np.float32))


def test_a_cached_call_runs_the_read_s_body_only_at_trace(columns: tuple[DiskColumn, ...]) -> None:
    """``get_records``' Python body runs once, at trace; a cached call runs only the disk read.

    The callback's host read is Python on every call by design, so the body itself is counted.
    """
    source = _source(columns)
    read = jax.jit(source.get_records)
    keys = by_package("datarax", named=["datarax"])
    body = "datarax:StreamingDiskSource.get_records"

    with counted_calls(keys) as traced:
        read(_indices([1, 2]))
    jax.block_until_ready(read(_indices([3, 4])))
    with counted_calls(keys) as cached:
        jax.block_until_ready(read(_indices([5, 6])))

    assert traced[body] == 1, "positive control: the trace runs the body"
    assert cached[body] == 0


class TestAPipelineOverTheSource:
    def _pipeline(self, columns: tuple[DiskColumn, ...]) -> Pipeline:
        return Pipeline(
            source=_source(columns), stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=False
        )

    def test_its_step_serves_the_key_tuples(self, columns: tuple[DiskColumn, ...]) -> None:
        batch = self._pipeline(columns).step()

        _served(batch.data, [0, 1, 2, 3], columns)

    def test_its_scan_equals_its_steps_and_a_second_scan_compiles_nothing(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        def body(batch: Batch) -> jax.Array:
            return _get(batch.data, columns[1])

        pipe = self._pipeline(columns)  # before the block: building compiles programs too
        with expect_first_call_compiles("jit(iota)", "jit(scan)"):
            first = pipe.scan(body, length=3)
        with compiled_programs() as built:  # the same build: a second build compiles apart (D3)
            second = pipe.scan(body, length=3)
        reference = self._pipeline(columns)
        steps = np.stack([np.asarray(_get(reference.step().data, columns[1])) for _ in range(6)])

        assert "jit(scan)" not in built, built
        np.testing.assert_array_equal(np.asarray(first), _literal()[:12].reshape(3, 4, 3))
        np.testing.assert_array_equal(np.concatenate([first, second]), steps)

    def test_its_session_compiles_once_and_serves_the_key_tuples(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        session = self._pipeline(columns).session()
        with expect_first_call_compiles("jit(session_step)"):
            first = next(session)
        _served(first.data, [0, 1, 2, 3], columns)
        _served(next(session).data, [4, 5, 6, 7], columns)
