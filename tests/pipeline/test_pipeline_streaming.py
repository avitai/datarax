"""The pipeline's stream route: the epoch rule, the pipeline's key, validation and the compiled DAG.

A stream names its records and counts its passes; the pipeline keeps no counter of its own for it
(D4). The pipeline passes its key to the stream when it shuffles and ``None`` otherwise, applies
the epoch rule to the stream's passes (D15: ``drop_last`` skips a pass's short tail, otherwise the
tail is completed from the next pass's head; ``num_epochs`` passes, or ``None`` for no end; no
padding), checks each pulled batch against the declared spec before the DAG sees it, and runs the
DAG as one compiled step per batch shape over the stream's ``Batch``. The DAG over a stream-read
``Batch`` is differentiable as it is over any other.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

from datarax.core.config import OperatorConfig
from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import from_words
from datarax.core.operator import OperatorModule, require_key
from datarax.core.spec import SpecMismatchError
from datarax.pipeline.epochs import EpochPlan, stream_batches
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig, StreamChunk
from tests.test_common.streams import RecordStream
from tests.test_common.transfers import device_to_host_raises


_N = 10


def _columns(size: int = _N) -> dict[str, np.ndarray]:
    return {
        "x": np.arange(size * 3, dtype=np.float32).reshape(size, 3),
        "y": np.arange(size, dtype=np.int32),
    }


def _stream(**kwargs: Any) -> RecordStream:
    return RecordStream(_columns(), **kwargs)


def _served(batches: list[Batch]) -> list[tuple[list[int], list[int]]]:
    return [
        ([int(i) for i in from_words(np.asarray(b.indices))], [int(e) for e in b.epochs])
        for b in batches
    ]


class _Counting(nnx.Module):
    def __init__(self) -> None:
        self.calls = nnx.Variable(jnp.zeros((), jnp.int32))

    def __call__(self, batch: Batch) -> Batch:
        self.calls[...] = self.calls[...] + 1
        return batch


class _Statistics(nnx.Module):
    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.zeros((), jnp.float32))

    def __call__(self, batch: Batch) -> Batch:
        self.total[...] = self.total[...] + batch["x"].sum()
        return batch


class _Growing(nnx.Module):
    def __call__(self, batch: Batch) -> Batch:
        self.seen = nnx.Variable(jnp.zeros((), jnp.int32))
        return batch


_TRACED: list[tuple[int, ...]] = []


class _Tracing(nnx.Module):
    def __call__(self, batch: Batch) -> Batch:
        _TRACED.append(tuple(batch["x"].shape))
        return batch


class _JitteredScale(OperatorModule):
    """Stochastic and learnable: ``x * scale * jitter``, the jitter from the record's key."""

    def __init__(self) -> None:
        super().__init__(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=3)
        )
        self.scale = nnx.Param(jnp.asarray(0.75))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        del stats
        jitter = jax.random.uniform(require_key(key, self), (), minval=0.5, maxval=1.5)
        data = element.data
        return element.replace(data={**data, "x": data["x"] * self.scale[...] * jitter})


def _pipeline(source: Any, stages: list[nnx.Module] | None = None, **kwargs: Any) -> Pipeline:
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("rngs", nnx.Rngs(0))
    return Pipeline(source=source, stages=stages or [], **kwargs)


# ---------------------------------------------------------------------------
# The epoch rule (D15, DD-G11, W2b-72)
# ---------------------------------------------------------------------------


class TestTheEpochRule:
    def test_drop_last_serves_full_batches_only_in_every_pass(self) -> None:
        for epochs, batches in ((1, 2), (2, 4)):
            served = _served(list(_pipeline(_stream(), drop_last=True, num_epochs=epochs)))
            assert [len(names) for names, _ in served] == [4] * batches
            assert [names for names, _ in served][:2] == [[0, 1, 2, 3], [4, 5, 6, 7]]

    def test_a_short_tail_is_completed_from_the_next_pass_s_head(self) -> None:
        served = _served(list(_pipeline(_stream(), num_epochs=2)))

        assert served == [
            ([0, 1, 2, 3], [0, 0, 0, 0]),
            ([4, 5, 6, 7], [0, 0, 0, 0]),
            ([8, 9, 0, 1], [0, 0, 1, 1]),
            ([2, 3, 4, 5], [1, 1, 1, 1]),
            ([6, 7, 8, 9], [1, 1, 1, 1]),
        ]

    def test_one_epoch_ends_with_its_short_tail(self) -> None:
        served = _served(list(_pipeline(_stream(), num_epochs=1)))

        assert served[-1] == ([8, 9], [0, 0])
        assert len(served) == 3

    def test_no_epoch_count_never_ends(self) -> None:
        batches = _pipeline(_stream(), num_epochs=None)
        iterator = iter(batches)

        served = _served([next(iterator) for _ in range(8)])

        assert all(len(names) == 4 for names, _ in served)
        assert served[7][1] == [2, 2, 3, 3]  # records 28-31: pass 2's 8, 9, pass 3's 0, 1

    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("num_epochs", [1, 2, 3])
    def test_the_rule_is_the_indexed_sources_rule(self, drop_last: bool, num_epochs: int) -> None:
        settings = {"drop_last": drop_last, "num_epochs": num_epochs}
        streamed = _served(list(_pipeline(_stream(), **settings)))
        indexed = _served(
            list(_pipeline(MemorySource(MemorySourceConfig(), _columns()), **settings))
        )

        assert streamed == indexed

    def test_the_pipeline_keeps_no_counter_for_a_stream(self) -> None:
        pipeline = _pipeline(_stream(), num_epochs=2)

        list(pipeline)

        assert int(pipeline._position[...]) == 0
        assert int(pipeline._epoch[...]) == 0

    def test_the_rule_takes_any_pull(self) -> None:
        """Pulls of any size, empty at a pass's end: the rule needs nothing else."""
        stream = _stream(chunk=7)

        served = _served(
            list(stream_batches(stream.get_batch, batch_size=4, drop_last=False, num_epochs=2))
        )

        assert [names for names, _ in served] == [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 0, 1],
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ]

    def test_a_stream_holding_no_record_serves_nothing(self) -> None:
        empty = RecordStream({"x": np.zeros((0, 3), np.float32)})

        assert list(stream_batches(empty.get_batch, 4, drop_last=False, num_epochs=None)) == []

    @pytest.mark.parametrize("num_epochs", [None, 1, 2])
    def test_a_run_starting_at_a_pass_s_unseen_end_serves_the_next_pass(
        self, num_epochs: int | None
    ) -> None:
        """A run that did not see a pass start cannot read its empty end as an empty stream."""
        stream = RecordStream({"x": np.arange(8, dtype=np.float32)})
        assert stream.get_batch(8).batch_size == 8  # the pass's records, not yet its end

        run = stream_batches(stream.get_batch, 4, drop_last=False, num_epochs=num_epochs)
        served = _served(list(itertools.islice(run, 2)))

        # The pass whose end the run starts at counts as served, as an exhausted epoch does.
        expected = [] if num_epochs == 1 else [([0, 1, 2, 3], [1] * 4), ([4, 5, 6, 7], [1] * 4)]
        assert served == expected


type _Loops = list[list[tuple[list[int], list[int]]]]


def _epoch_loops(pipeline: Pipeline, loops: int, steps: int) -> _Loops:
    """``for step, batch in zip(range(steps), pipeline)`` run ``loops`` times, each loop's batches.

    ``zip`` stops on ``range`` without advancing the pipeline again, so a loop taking exactly an
    epoch's batches leaves the source at the epoch's end, unseen.
    """
    return [_served([batch for _, batch in zip(range(steps), pipeline)]) for _ in range(loops)]


def _plan_loops(plan: EpochPlan, loops: int, steps: int) -> _Loops:
    """What :func:`_epoch_loops` serves from a source in its own order, by ``plan``'s rule.

    Each loop is a session from where the last stopped: :meth:`EpochPlan.run_extent` bounds it,
    :meth:`EpochPlan.batch_start` and :meth:`EpochPlan.advance` place each batch.
    """
    length = plan.length
    if length is None:
        raise ValueError("the table is of a source with a length")
    position = epoch = 0
    table = []
    for _ in range(loops):
        extent = plan.run_extent(position)
        batches, final = (steps, plan.batch_size) if extent is None else extent
        served = []
        for step in range(min(steps, batches)):
            size = final if step == batches - 1 else plan.batch_size
            start, epoch = plan.batch_start(position, epoch)
            records = range(start, start + size)
            served.append(([r % length for r in records], [epoch + r // length for r in records]))
            position, epoch = plan.advance(start, epoch, size)
        table.append(served)
    return table


class TestEpochLoopsOverAStream:
    """An epoch loop of ``N // B`` steps over a fresh iterator each epoch (F1, T5, W2b-72)."""

    _LOOPS = 4
    _BATCH = 4

    @pytest.mark.parametrize("records", [8, 10])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("num_epochs", [None, 3])
    def test_every_loop_serves_what_an_indexed_source_serves_and_the_plan_states(
        self, records: int, drop_last: bool, num_epochs: int | None
    ) -> None:
        settings = {"drop_last": drop_last, "num_epochs": num_epochs, "batch_size": self._BATCH}
        steps = records // self._BATCH
        streamed = _epoch_loops(
            _pipeline(RecordStream(_columns(records)), **settings), self._LOOPS, steps
        )
        indexed = _epoch_loops(
            _pipeline(MemorySource(MemorySourceConfig(), _columns(records)), **settings),
            self._LOOPS,
            steps,
        )
        plan = EpochPlan(
            length=records, batch_size=self._BATCH, drop_last=drop_last, num_epochs=num_epochs
        )

        assert streamed == indexed
        assert streamed == _plan_loops(plan, self._LOOPS, steps)
        assert all(len(loop) == steps for loop in streamed)


# ---------------------------------------------------------------------------
# The pipeline's key (OWN-0930-SHUFFLE-OWNER, D4, W2b-57)
# ---------------------------------------------------------------------------


class TestThePipelineKey:
    def test_the_stream_gets_the_key_only_when_the_pipeline_shuffles(self) -> None:
        sequential, shuffled = _stream(), _stream()

        list(_pipeline(sequential, num_epochs=2))
        list(_pipeline(shuffled, num_epochs=2, shuffle=True))

        # The first open is the declared spec's read of the first record, in the stream's order.
        assert [key for _, key in sequential.opened[1:]] == [None, None]
        assert all(key is not None for _, key in shuffled.opened[1:])

    def test_the_same_seed_serves_the_same_order_and_another_seed_another(self) -> None:
        def order(seed: int) -> list[tuple[list[int], list[int]]]:
            return _served(
                list(_pipeline(_stream(), num_epochs=3, shuffle=True, rngs=nnx.Rngs(seed)))
            )

        assert order(0) == order(0)
        assert order(0) != order(1)
        passes = [
            sorted(i for names, eps in order(0) for i, e in zip(names, eps) if e == p)
            for p in range(3)
        ]
        assert passes == [list(range(_N))] * 3

    def test_without_shuffle_every_pass_is_the_stream_s_order(self) -> None:
        served = [
            i for names, _ in _served(list(_pipeline(_stream(), num_epochs=2))) for i in names
        ]

        assert served == list(range(_N)) * 2


# ---------------------------------------------------------------------------
# Validation before the DAG
# ---------------------------------------------------------------------------


class _Declared(RecordStream):
    """A stream whose declared spec is given, to test the check of what it serves."""

    def __init__(self, columns: Mapping[str, np.ndarray], spec: Mapping[str, Any]) -> None:
        super().__init__(columns)
        self._declared = dict(spec)
        self.spec_calls = 0

    def element_spec(self) -> Any:
        self.spec_calls += 1
        return self._declared


_SPEC = {"x": jax.ShapeDtypeStruct((3,), jnp.float32), "y": jax.ShapeDtypeStruct((), jnp.int32)}


@pytest.mark.parametrize(
    ("columns", "fragment"),
    [
        ({"x": np.ones((4, 4), np.float32), "y": np.zeros(4, np.int32)}, "['x']"),
        ({"x": np.ones((4, 3), np.int32), "y": np.zeros(4, np.int32)}, "['x']"),
        (
            {"x": np.ones((4, 3), np.float32), "y": np.zeros(4, np.int32), "z": np.ones(4)},
            "['z']",
        ),
        ({"x": np.ones((4, 3), np.float32)}, "['y']"),
    ],
    ids=["trailing-shape", "int32-declared-float32", "undeclared-field", "missing-field"],
)
def test_a_batch_disagreeing_with_the_declaration_stops_before_the_dag(
    columns: dict[str, np.ndarray], fragment: str
) -> None:
    stage = _Counting()
    pipeline = _pipeline(_Declared(columns, _SPEC), [stage])

    with pytest.raises(SpecMismatchError, match=re.escape(fragment)):
        list(pipeline)

    assert int(stage.calls[...]) == 0


def test_a_host_int64_column_the_device_holds_as_int32_passes() -> None:
    """A stream's host column keeps its stored dtype; the check compares as the device holds it."""
    stream = RecordStream({"x": np.ones((4, 3), np.float32), "y": np.arange(4, dtype=np.int64)})

    (batch,) = list(_pipeline(stream))

    assert batch["y"].dtype == jnp.int32


def test_a_declared_dtype_the_device_would_narrow_is_refused_before_any_pull() -> None:
    stream = _Declared(
        {"x": np.ones((4, 3), np.float64)}, {"x": jax.ShapeDtypeStruct((3,), np.float64)}
    )

    with pytest.raises(SpecMismatchError, match="jax_enable_x64"):
        list(_pipeline(stream))

    assert stream.opened == []


def test_float64_streams_through_unchanged_when_x64_is_on() -> None:
    with jax.enable_x64(True):
        stream = RecordStream({"x": np.ones((4, 3), np.float64)})
        (batch,) = list(_pipeline(stream))

    assert batch["x"].dtype == np.float64


def test_the_declared_spec_is_read_once_per_source_and_precision_mode() -> None:
    stream = _Declared(_columns(), _SPEC)
    pipeline = _pipeline(stream)

    for _ in range(2):
        assert len(list(pipeline)) == 3
    assert stream.spec_calls == 1
    with jax.enable_x64(True):
        list(pipeline)
    assert stream.spec_calls == 2


def test_a_source_serving_something_other_than_a_batch_is_refused() -> None:
    class _Dicts(RecordStream):
        def get_batch(self, batch_size: int, **kwargs: Any) -> Any:  # type: ignore[override]
            return {"x": np.ones((batch_size, 3), np.float32)}

    with pytest.raises(TypeError, match="Batch"):
        list(_pipeline(_Dicts(_columns())))


# ---------------------------------------------------------------------------
# The compiled DAG over the stream's Batch (section 7 of the step's brief)
# ---------------------------------------------------------------------------


def test_the_dag_compiles_once_per_batch_shape_across_passes_and_pipelines() -> None:
    _TRACED.clear()
    for _ in range(2):
        pipeline = _pipeline(_stream(), [_Tracing()], num_epochs=3, shuffle=True)
        for _ in range(2):
            list(pipeline)

    assert sorted(set(_TRACED)) == [(2, 3), (4, 3)]
    assert len(_TRACED) == 2


def test_batches_differing_in_values_names_and_epochs_compile_nothing_new() -> None:
    pipeline = _pipeline(_stream(), [_JitteredScale()], num_epochs=None, shuffle=True)
    iterator = iter(pipeline)
    jax.block_until_ready(next(iterator)["x"])

    with expect_compiles(0):
        for _ in range(12):  # three passes: other values, ids and epochs every batch
            jax.block_until_ready(next(iterator)["x"])


def test_stochastic_outputs_equal_the_dag_through_nnx_jit_on_the_same_batch() -> None:
    streamed = _pipeline(_stream(), [_JitteredScale()], num_epochs=2, shuffle=True)
    read_stream = _stream()
    reference = _pipeline(read_stream, [_JitteredScale()], num_epochs=2, shuffle=True)
    reads = list(
        stream_batches(
            lambda size: read_stream.get_batch(
                size, key=jax.random.wrap_key_data(reference._epoch_key_base[...])
            ),
            4,
            drop_last=False,
            num_epochs=2,
        )
    )
    apply = nnx.jit(lambda dag, batch: dag(batch))

    outputs = list(streamed)

    assert len(outputs) == len(reads)
    for got, read in zip(outputs, reads, strict=True):
        np.testing.assert_array_equal(np.asarray(got.indices), read.indices)
        np.testing.assert_array_equal(
            np.asarray(got["x"]), np.asarray(apply(reference.dag, read)["x"])
        )


def test_a_crossing_batch_draws_as_the_same_records_in_batches_that_do_not_cross() -> None:
    """DD-KEY-ORDER: a record's draws follow its identity and epoch, not its batch."""

    def outputs(batch_size: int) -> dict[tuple[int, int], np.ndarray]:
        pipeline = _pipeline(_stream(), [_JitteredScale()], batch_size=batch_size, num_epochs=2)
        found = {}
        for batch in pipeline:
            names = from_words(np.asarray(batch.indices))
            for row, (name, epoch) in enumerate(zip(names, np.asarray(batch.epochs), strict=True)):
                found[(int(name), int(epoch))] = np.asarray(batch["x"][row])
        return found

    crossing, aligned = outputs(4), outputs(2)  # 4 crosses at [8, 9 | 0, 1]; 2 never crosses

    assert crossing.keys() == aligned.keys()
    for record, value in crossing.items():
        np.testing.assert_array_equal(value, aligned[record])


def test_no_transfer_to_the_host_happens_inside_the_steps() -> None:
    if not device_to_host_raises():
        pytest.skip("the device-to-host guard does not fire on this backend (host memory)")
    pipeline = _pipeline(_stream(), [_JitteredScale()], num_epochs=2)
    with jax.transfer_guard_device_to_host("disallow"):
        outputs = [jax.block_until_ready(batch["x"]) for batch in pipeline]
    assert len(outputs) == 5


@pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
def test_the_dag_over_a_stream_read_batch_differentiates(graph: bool) -> None:
    stream = _stream()
    batch = stream.get_batch(4, key=jax.random.key(0))
    pipeline = _pipeline(_stream(), [_JitteredScale()])
    dag = pipeline.dag

    def loss(module: Any) -> jax.Array:
        return jnp.sum(module(batch)["x"] ** 2)

    gradient = check_parameter_gradients(dag, loss)
    assert jax.tree.leaves(gradient)

    def of_data(module: Any, x: jax.Array) -> jax.Array:
        return jnp.sum(module(batch.replace(data={**batch.data, "x": x}))["x"] ** 2)

    check_input_gradients(dag, of_data, batch["x"])

    graphdef, params, rest = nnx.split(dag, nnx.Param, ..., graph=graph)

    @jax.jit
    def step(params: Any, rest: Any, batch: Batch) -> tuple[jax.Array, Any]:
        return jax.value_and_grad(lambda p: loss(nnx.merge(graphdef, p, rest)))(params)

    with expect_compiles(1):
        step(params, rest, batch)
    with expect_compiles(0):
        step(params, rest, stream.get_batch(4, key=jax.random.key(0)))


# ---------------------------------------------------------------------------
# Module state through the compiled step
# ---------------------------------------------------------------------------


def test_stage_state_is_current_at_every_yield_and_written_back() -> None:
    stage = _Counting()
    iterator = iter(_pipeline(_stream(), [stage]))

    next(iterator)
    assert int(stage.calls[...]) == 1
    stage.calls[...] = jnp.int32(7)
    next(iterator)
    assert int(stage.calls[...]) == 8


def test_batch_statistics_accumulate_as_through_nnx_jit() -> None:
    streamed_stage = _Statistics()
    list(_pipeline(_stream(), [streamed_stage]))

    assert float(streamed_stage.total[...]) == float(_columns()["x"].sum())


def test_a_stage_changing_the_module_structure_is_refused() -> None:
    with pytest.raises(ValueError, match="changed the module structure"):
        list(_pipeline(_stream(), [_Growing()]))


def test_the_stream_s_reads_are_a_host_chunk_path() -> None:
    """The route reads through ``get_batch``: a stream yields ``StreamChunk``s on the host."""
    chunks: Iterator[StreamChunk] = _stream()._open_pass(0, None, 4)
    first = next(chunks)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(first.columns))
    assert RecordIdentity.STREAM_IDS is _stream().record_identity
