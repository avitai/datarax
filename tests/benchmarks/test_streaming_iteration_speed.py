"""Streaming iteration adds only its named host work to reading, placing and the step.

A stream pass (``for batch in pipe``) reads each batch on the host stage's read thread, names and
checks it, places it on the device with ``jax.device_put`` and runs the stage DAG in one compiled
call, split once per pass (Flax's performance guide: split the module once and drive a plain
``jax.jit`` step, rather than traverse its graph in ``nnx.jit`` on every call). The guards compare
that pass with the same work written by hand: the stream's own reads, an explicit
``jax.device_put`` of each batch and a split-once ``jax.jit`` step writing the stages' state back.

They count work instead of timing it, so no machine moves the verdict: a timed ratio of the two
passes measured how long the consumer waited on the read thread, which the scheduler and the cores
decide, and could not see a per-batch host copy. ``substrax.testing`` counts the Python functions
each pass starts on every thread, its ``jax.device_put`` calls and its compiled dispatches, per
batch as the slope between a short and a long pass, so work done once per pass cancels.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import CodeType
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx
from flax.nnx import graphlib
from grain._src.core import traceback_util
from grain._src.python.dataset.stats import HashableWeakRef
from substrax.testing import by_package, counted_calls, per_iteration
from substrax.testing.compiles import compiled_programs
from substrax.testing.jax_calls import JAX_CALL_WEIGHTS

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch
from datarax.pipeline.pipeline import Pipeline
from tests.test_common.streams import RecordStream


_BATCHES = 200
_SHORT_PASS = 100
_BATCH_SIZE = 64
_FEATURES = 32

# The datarax functions a stream pass starts per batch, by qualified name, and how often: the
# layers of the path in host_stage, the source and dag_call. Equality, so a new layer or a second
# call of one names itself in the failure.
_STREAM_PATH_PER_BATCH = {
    # Serving and placement: the run's iterator, the placed-ahead buffer, one device_put.
    "_Served.__next__": 1,
    "HostStage.iterator": 1,
    "_PlacedAheadIterator.__next__": 1,
    "_PlacedAheadIterator._fill": 1,
    "_place": 1,
    # The read unit and the cursor, on the read thread.
    "_StreamIterator.__next__": 1,
    "_StreamIterator._fill": 1,
    "_StreamIterator._next_batch": 1,
    "_StreamIterator._unit": 1,
    "_StreamIterator._unit.<locals>.<genexpr>": 1,
    "_StreamIterator._pull": 1,
    "_StreamIterator.get_state": 1,
    "chunk_size": 3,
    # The source's read and its records' names.
    "StreamingSourceBase._pull": 1,
    "StreamingSourceBase.read_from": 1,
    "StreamingSourceBase._named": 1,
    "to_words": 1,
    "from_arrays": 1,
    "_join": 1,
    "_joined": 1,
    "_namespace": 1,
    "_namespace.<locals>.<genexpr>": 1,
    "_record_axis": 1,
    "Batch.batch_size": 6,
    # The spec check, once per batch.
    "_checker.<locals>.check": 1,
    "validate_batch": 1,
    "_batch_matches": 1,
    "_batch_matches.<locals>.<genexpr>": 1,
    "_record_count": 2,
    "_device_dtype": 2,
    # The compiled DAG call and the write-back of the stages' state.
    "compile_dag.<locals>.apply": 1,
    "apply_writes": 1,
}

# Python functions a stream pass may start per batch in each library beyond the by-hand pass's,
# each the work of a named step of the stream path:
# - grain: the read thread's ThreadPrefetchDatasetIterator.__next__, with its timer and stats
#   (14 functions);
# - jax: the write-back's tree map and the placement's sharding checks, net of the by-hand
#   step's state rebuild (13);
# - numpy: the cursor's np.sum over the batch's epochs (sum, _wrapreduction, _sum_dispatcher);
# - other: grain's record_self_time context manager (4 contextlib frames) and its deep copy of
#   the read iterator's state, {} (3), the HostElement named tuple's constructor (1) and the
#   cast in _Served.__next__ (1);
# - flax: none. Both passes split the DAG once and run a step that closes over its graph
#   definition, so neither touches flax beyond the same per-batch state handling.
_LIBRARY_ALLOWANCE = {"flax": 0, "grain": 14, "jax": 13, "numpy": 3, "other": 9}

# Starts whose number the process's history decides, not the code under test:
# - grain's stats registry keys iterators by id(), so HashableWeakRef.__eq__ runs only when a new
#   iterator reuses a dead one's address, which the allocator decides (google/grain#1427: the
#   registry never drops a dead iterator's ref; leave it in once grain drops them);
# - grain wraps an output iterator class's __next__ in a traceback filter, permanently and for
#   every later iterator of that class, so whether a read goes through one more wrapper frame
#   depends on which pipelines the process ran before. The wrapper calls nothing on its way.
_HISTORY_DEPENDENT = frozenset(
    {
        HashableWeakRef.__eq__.__code__,
        traceback_util.run_with_traceback_filter(lambda: None).__code__,
    }
)
_BY_PACKAGE = by_package(
    "datarax",
    "grain",
    "jax",
    "flax",
    "numpy",
    named=["datarax"],
    excluded=("threading", "queue", "_weakrefset"),
)


def _keys(code: CodeType) -> tuple[str, ...]:
    return () if code in _HISTORY_DEPENDENT else _BY_PACKAGE(code)


class _Noise(nnx.Module):
    def __init__(self) -> None:
        self.rngs = nnx.Rngs(noise=0)

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(
            data={
                **batch.data,
                "x": batch["x"] + jax.random.normal(self.rngs.noise(), batch["x"].shape),
            }
        )


def _stream(batches: int = _BATCHES) -> RecordStream:
    records = batches * _BATCH_SIZE
    return RecordStream(
        {
            "x": np.ones((records, _FEATURES), np.float32),
            "y": np.zeros((records,), np.int32),
        },
        kind=RecordIdentity.ARRIVAL,
        chunk=_BATCH_SIZE,
    )


def _pipeline(source: RecordStream) -> Pipeline:
    return Pipeline(source=source, stages=[_Noise()], batch_size=_BATCH_SIZE, rngs=nnx.Rngs(0))


def _stream_pass(batches: int = _BATCHES) -> None:
    jax.block_until_ready(list(_pipeline(_stream(batches))))


@functools.cache
def _step(graphdef: Any) -> Callable[[Any, Batch], tuple[Batch, Any]]:
    """The guide's split-once step over one placed batch, its state written back.

    It closes over the graph definition, as the stream's step does, so no call hashes it.
    """

    @jax.jit
    def step(state: Any, batch: Batch) -> tuple[Batch, Any]:
        dag = nnx.merge(graphdef, state)
        return dag(batch), nnx.state(dag)

    return step


def _by_hand_pass(batches: int = _BATCHES) -> None:
    """The stream pass's work by hand: the stream's reads, ``jax.device_put`` and ``_step``."""
    source = _stream(batches)
    graphdef, state = nnx.split(_pipeline(source).dag)
    step = _step(graphdef)
    outputs = []
    while (batch := source.get_batch(_BATCH_SIZE)).batch_size:
        output, state = step(state, jax.device_put(batch))
        outputs.append(output)
    jax.block_until_ready(outputs)


_apply_per_batch = nnx.jit(lambda dag, batch: dag(batch))


def _nnx_jit_pass() -> None:
    """The DAG through ``nnx.jit`` on every batch: the graph traversed per call."""
    source = _stream()
    pipeline = _pipeline(source)
    outputs = []
    while (batch := source.get_batch(_BATCH_SIZE)).batch_size:
        outputs.append(_apply_per_batch(pipeline.dag, batch))
    jax.block_until_ready(outputs)


def _per_batch(run: Callable[[int], None]) -> dict[str, int]:
    """The work ``run`` does per batch, counted over a short and a long pass of it."""

    def count(batches: int) -> dict[str, int]:
        run(batches)  # warm, so nothing compiles or fills a cache inside the counted pass
        with (
            compiled_programs() as compiled,
            counted_calls(_keys, weights=JAX_CALL_WEIGHTS) as counts,
        ):
            run(batches)
        assert list(compiled) == []
        return dict(counts)

    return per_iteration(count, short=_SHORT_PASS, long=_BATCHES)


@pytest.fixture(scope="module")
def per_batch_work() -> tuple[dict[str, int], dict[str, int]]:
    """Per batch: a stream pass's work and the by-hand pass's."""
    return _per_batch(_stream_pass), _per_batch(_by_hand_pass)


def test_a_stream_pass_places_each_batch_once_and_dispatches_once(
    per_batch_work: tuple[dict[str, int], dict[str, int]],
) -> None:
    """One ``jax.device_put`` of the same leaves and one compiled call per batch, as by hand."""
    stream, by_hand = per_batch_work

    assert stream["device_put"] == by_hand["device_put"] == 1
    assert stream["device_put_leaves"] == by_hand["device_put_leaves"]
    assert stream["jit_dispatch"] == by_hand["jit_dispatch"] == 1


def test_a_stream_pass_starts_only_its_named_datarax_functions_per_batch(
    per_batch_work: tuple[dict[str, int], dict[str, int]],
) -> None:
    """Every datarax function a batch starts is a named layer of the stream path."""
    stream, _ = per_batch_work
    started = {
        key.removeprefix("datarax:"): calls
        for key, calls in stream.items()
        if key.startswith("datarax:")
    }

    assert started == _STREAM_PATH_PER_BATCH, {
        name: (started.get(name, 0), _STREAM_PATH_PER_BATCH.get(name, 0))
        for name in started.keys() | _STREAM_PATH_PER_BATCH.keys()
        if started.get(name, 0) != _STREAM_PATH_PER_BATCH.get(name, 0)
    }


@pytest.mark.parametrize("library", sorted(_LIBRARY_ALLOWANCE))
def test_a_stream_pass_drives_no_more_library_work_than_by_hand(
    per_batch_work: tuple[dict[str, int], dict[str, int]], library: str
) -> None:
    """Per batch, each library runs at most its itemized allowance beyond the by-hand pass."""
    stream, by_hand = per_batch_work

    extra = stream.get(library, 0) - by_hand.get(library, 0)

    assert extra <= _LIBRARY_ALLOWANCE[library], (
        library,
        stream.get(library),
        by_hand.get(library),
    )


def _graph_flattens(monkeypatch: pytest.MonkeyPatch, run: Callable[[], object]) -> int:
    """The module-graph flattens ``run`` makes: NNX's per-call traversal goes through one each."""
    calls = 0
    flatten = graphlib.flatten

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return flatten(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(graphlib, "flatten", counted)
        jax.block_until_ready(run())
    return calls


def test_a_streaming_pass_traverses_the_module_graph_once_not_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The traversal counted rather than timed, so no machine moves it."""
    _stream_pass()

    streamed = _graph_flattens(monkeypatch, lambda: list(_pipeline(_stream())))
    per_batch = _graph_flattens(monkeypatch, _nnx_jit_pass)

    assert streamed == 1
    assert per_batch >= _BATCHES  # the control: nnx.jit traverses on every call
