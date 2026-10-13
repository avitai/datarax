"""An eager source's columns stay on the host, out of NNX state; programs take them as an argument.

``EagerSource`` (``MemorySource``, ``TFDSEagerSource``, ``HFEagerSource``) keeps its columns in a
host holder that compares equal by kind, so no dataset-sized array is NNX state and sources that
differ only in their data share one graph definition and one compiled program. Each program
datarax compiles over a pipeline (``step()``, a session, ``Pipeline.scan``) takes the columns'
device copy, uploaded once, as an explicit argument. A caller's own transform passes them the same
way: ``pipeline.device_columns()`` outside the transform, ``step(columns=...)`` or
``scan(..., columns=...)`` inside it; without them it is refused, naming the fix. A source's
traced read takes them as ``get_records(indices, columns=...)``.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.element_batch import Batch
from datarax.core.host_values import HostProvenance
from datarax.core.index_words import from_words
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


_N = 64
_BATCH = 8


def _columns(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "x": np.arange(_N * 3, dtype=np.float32).reshape(_N, 3) + offset,
        "y": np.arange(_N, dtype=np.int32),
    }


def _memory(offset: float = 0.0) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _columns(offset))


class _Scale(nnx.Module):
    """Scales ``x`` by a learnable factor."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.ones(()))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "x": batch["x"] * self.factor[...]})


def _pipeline(source: Any, *, stages: list[nnx.Module] | None = None) -> Pipeline:
    return Pipeline(source=source, stages=stages or [], batch_size=_BATCH, rngs=nnx.Rngs(0))


def _dataset_sized(tree: Any) -> list[tuple[int, ...]]:
    return [
        np.shape(leaf)
        for leaf in jax.tree.leaves(tree)
        if np.ndim(leaf) and np.shape(leaf)[0] == _N
    ]


def _rows(batch: Batch) -> list[int]:
    return [int(i) for i in from_words(np.asarray(batch.indices))]


class TestOutOfNnxState:
    def test_an_eager_source_s_columns_are_not_nnx_state(self) -> None:
        source = _memory()

        assert _dataset_sized(nnx.state(source)) == []
        assert all(isinstance(column, np.ndarray) for column in source.data.values())
        np.testing.assert_array_equal(source.data["x"], _columns()["x"])

    def test_a_pipeline_s_state_holds_no_dataset_sized_array_after_it_runs(self) -> None:
        pipeline = _pipeline(_memory())
        pipeline.step()
        next(iter(pipeline.session()))
        pipeline.scan(lambda batch: jnp.sum(batch["x"]), length=2)

        assert _dataset_sized(nnx.state(pipeline)) == []

    def test_sources_differing_only_in_their_data_share_one_graph_definition(self) -> None:
        assert nnx.graphdef(_memory(0.0)) == nnx.graphdef(_memory(1000.0))


class TestDatarax_sPrograms:
    """``step()``, a session and ``Pipeline.scan`` take the device columns as an argument."""

    def test_a_second_source_of_the_structure_compiles_nothing_and_serves_its_own_data(
        self,
    ) -> None:
        _pipeline(_memory(0.0)).step()
        other = _pipeline(_memory(1000.0))

        with compiled_programs() as built:
            batch = other.step()

        assert built == []
        np.testing.assert_array_equal(np.asarray(batch["x"]), _columns(1000.0)["x"][_rows(batch)])

    def test_steady_steps_sessions_and_scans_upload_nothing(self) -> None:
        pipeline = _pipeline(_memory())
        session = iter(pipeline.session())
        pipeline.step()
        next(session)
        pipeline.scan(lambda b: jnp.sum(b["x"]), length=2)

        with jax.transfer_guard_host_to_device("disallow"):
            jax.block_until_ready(pipeline.step())
            jax.block_until_ready(next(session)["x"])
            jax.block_until_ready(pipeline.scan(lambda b: jnp.sum(b["x"]), length=2))


class TestACaller_sTransform:
    """A caller's own transform passes the columns as an argument, as data is passed to a step."""

    def test_without_the_columns_it_is_refused_naming_device_columns(self) -> None:
        pipeline = _pipeline(_memory())

        with pytest.raises(TypeError, match=r"device_columns\(\)"):
            nnx.jit(lambda p: p.step())(pipeline)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
    def test_with_the_columns_it_serves_the_batches_datarax_s_own_step_serves(
        self, graph: bool
    ) -> None:
        reference = _pipeline(_memory())
        expected = [reference.step() for _ in range(3)]
        pipeline = _pipeline(_memory())
        columns = pipeline.device_columns()
        step = nnx.jit(lambda p, c: p.step(columns=c), graph=graph)

        served = [step(pipeline, columns)]
        with compiled_programs() as built:
            served += [step(pipeline, columns) for _ in range(2)]

        assert built == []
        for got, want in zip(served, expected, strict=True):
            assert _rows(got) == _rows(want)
            np.testing.assert_array_equal(np.asarray(got["x"]), np.asarray(want["x"]))

    def test_with_the_columns_it_uploads_no_column_on_any_call(self) -> None:
        pipeline = _pipeline(_memory())
        columns = pipeline.device_columns()
        step = nnx.jit(lambda p, c: p.step(columns=c), graph=False)
        step(pipeline, columns)

        with jax.transfer_guard_host_to_device("disallow"):
            jax.block_until_ready(step(pipeline, columns))

    @pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
    def test_value_and_grad_through_scan_inside_its_jit_reaches_the_stage_s_parameter(
        self, graph: bool
    ) -> None:
        """The randomness tutorial's pattern: differentiate a learnable stage through the scan."""
        pipeline = _pipeline(_memory(), stages=[_Scale()])
        columns = pipeline.device_columns()

        def total(pipeline: Pipeline, columns: Any) -> jax.Array:
            return jnp.sum(pipeline.scan(lambda b: jnp.sum(b["x"]), length=2, columns=columns))

        # The outer jit runs in either mode; the gradient is graph-mode in both, since a tree-mode
        # gradient differentiates every leaf it is given, the pipeline's integer counters too.
        @nnx.jit(graph=graph)
        def train(pipeline: Pipeline, columns: Any) -> tuple[jax.Array, Any]:
            return nnx.value_and_grad(lambda p: total(p, columns))(pipeline)

        value, grads = train(pipeline, columns)

        rows = np.arange(2 * _BATCH)
        assert float(value) == pytest.approx(float(_columns()["x"][rows].sum()), rel=1e-6)
        (factor_grad,) = jax.tree.leaves(grads)
        assert float(factor_grad) == pytest.approx(float(value), rel=1e-6)


class TestTheTracedRead:
    def test_a_traced_get_records_reads_the_columns_it_is_given(self) -> None:
        source = _memory()
        indices = source.record_indices_at(0, _BATCH)
        columns = source.device_columns()

        records = jax.jit(lambda i, c: source.get_records(i, columns=c))(indices, columns)

        np.testing.assert_array_equal(np.asarray(records["x"]), _columns()["x"][:_BATCH])

    def test_a_traced_get_records_without_columns_is_refused_naming_them(self) -> None:
        source = _memory()
        indices = source.record_indices_at(0, _BATCH)

        with pytest.raises(TypeError, match="columns="):
            jax.jit(source.get_records)(indices)


def _run(pipeline: Pipeline, path: str) -> None:
    """Serve one batch through ``path``: datarax's ``step()``, a session, or ``Pipeline.scan``."""
    if path == "step":
        pipeline.step()
    elif path == "session":
        next(iter(pipeline.session()))
    else:
        pipeline.scan(lambda b: jnp.sum(b["x"]), length=2)


def _provenance_holders_tagged(tag: str) -> int:
    """Live provenance holders whose first record carries ``tag``.

    Counted, not weakly referenced: the rows are mappings in a tuple, which take no weak reference.
    """
    gc.collect()
    return sum(
        1
        for holder in gc.get_objects()
        if isinstance(holder, HostProvenance) and holder.value and holder.value[0].get("tag") == tag
    )


class TestRetention:
    """A dropped pipeline's host values are freed: no program datarax caches keeps them.

    The programs are keyed by the pipeline's structure, so a cache entry outlives the pipeline
    that built it; it must keep that structure only (the host naming's rule, its stand-in),
    never the columns, the provenance or the host stage of a pipeline already dropped.
    """

    @pytest.mark.parametrize("path", ["step", "session", "scan"])
    def test_a_dropped_pipeline_s_columns_are_freed(self, path: str) -> None:
        def run_and_drop() -> weakref.ref[np.ndarray]:
            source = _memory()
            column = weakref.ref(source.data["x"])
            batch_size = {"step": 2, "session": 4, "scan": 6}[path]  # a new structure, see below
            _run(Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(0)), path)
            return column

        columns = [run_and_drop() for _ in range(3)]
        gc.collect()

        assert [column() is None for column in columns] == [True] * 3

    @pytest.mark.parametrize("path", ["step", "session", "scan"])
    def test_a_dropped_pipeline_s_provenance_and_host_stage_are_freed(self, path: str) -> None:
        """Each path's batch size is its own, so its pipelines' structure is a new cache entry.

        Only the first pipeline of a structure keys a cache entry; one built earlier in the
        process would key it instead, and this test's pipelines would not be measured.
        """
        tag = f"retention-{path}"
        batch_size = {"step": 3, "session": 5, "scan": 7}[path]

        def run_and_drop() -> weakref.ref[Any]:
            records = [{"x": np.float32(i), "y": np.int32(i), "tag": tag} for i in range(_N)]
            source = MemorySource(MemorySourceConfig(), records)
            pipeline = Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(0))
            stage = weakref.ref(pipeline.host_stage)
            _run(pipeline, path)
            return stage

        stages = [run_and_drop() for _ in range(3)]
        gc.collect()

        assert [stage() is None for stage in stages] == [True] * 3
        assert _provenance_holders_tagged(tag) == 0
