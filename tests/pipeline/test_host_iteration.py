"""``for batch in pipe`` runs on the host stage, and the pipeline's state is its versioned cursor.

Every source kind is read by the host stage (Grain threads, the order named on the CPU device,
placement as each batch is taken), and each batch runs through the DAG in one compiled call.
Iteration serves the compiled session's records, bit for bit, holds no dataset on the device, and
stands where the last batch taken ended: ``get_state()`` is that cursor, version 3, and
``set_state`` resumes exactly, a resumed run ending where the uninterrupted one would, whatever the
read threads had read ahead. Any other state layout is refused, naming both versions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore
from substrax.testing.compiles import expect_compiles

from datarax.checkpoint import IteratorCheckpoint
from datarax.core import batch_ops
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import from_words
from datarax.core.operator import OperatorModule, require_key
from datarax.pipeline import Pipeline
from datarax.pipeline.host_stage import HostStage
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.streams import RecordStream
from tests.test_common.transfers import implicit_upload_raises


def _columns(length: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(length)
    return {
        "image": rng.integers(0, 256, (length, 4, 4, 3), dtype=np.uint8),
        "label": np.arange(length, dtype=np.int32),
    }


def _memory(length: int = 23) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _columns(length))


def _pipeline(
    source: DataSourceModule,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    num_epochs: int | None = 3,
    stages: list[nnx.Module] | None = None,
    seed: int = 3,
) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[] if stages is None else stages,
        batch_size=batch_size,
        rngs=nnx.Rngs(seed),
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
    )


def _rows(batch: Batch) -> tuple[list[int], list[int]]:
    return (
        [int(v) for v in from_words(np.asarray(batch.indices))],
        [int(e) for e in np.asarray(batch.epochs)],
    )


class _Jitter(OperatorModule):
    """A stochastic operator adding a per-record draw to ``image`` (as float)."""

    def __init__(self) -> None:
        super().__init__(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=11)
        )

    def apply(self, element: Element, key: jax.Array | None = None, stats: Any = None) -> Element:
        del stats
        noise = jax.random.uniform(require_key(key, self), ())
        image = element.data["image"].astype(jnp.float32)
        return element.update_data({"image": image + noise})


class TestOrder:
    """``for batch in pipe`` serves the session's records, bit for bit, for every rule."""

    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 3, 4, 8])
    @pytest.mark.parametrize("length", [7, 10, 23])
    def test_iteration_equals_the_session(
        self, shuffle: bool, drop_last: bool, batch_size: int, length: int
    ) -> None:
        if drop_last and batch_size > length:
            return  # the plan refuses it
        session = list(
            _pipeline(
                _memory(length), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
            ).session()
        )
        served = list(
            _pipeline(_memory(length), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        )
        assert len(served) == len(session)
        for got, expected in zip(served, session, strict=True):
            assert _rows(got) == _rows(expected)
            np.testing.assert_array_equal(got["image"], expected["image"])

    def test_iteration_is_not_a_session(self) -> None:
        from datarax.pipeline import PipelineIterator  # noqa: PLC0415

        assert not isinstance(iter(_pipeline(_memory())), PipelineIterator)


class TestTierA:
    """Each batch runs through the DAG in one compiled call, its writes reaching the pipeline."""

    def test_stochastic_operators_draw_as_the_session_draws(self) -> None:
        session = list(_pipeline(_memory(), stages=[_Jitter()]).session())
        served = list(_pipeline(_memory(), stages=[_Jitter()]))
        for got, expected in zip(served, session, strict=True):
            np.testing.assert_array_equal(got["image"], expected["image"])

    def test_the_dag_compiles_once_per_batch_shape(self) -> None:
        pipe = _pipeline(_memory(22), stages=[_Jitter()], num_epochs=1)  # 5 full, one of 2
        batches = iter(pipe)
        jax.clear_caches()
        next(batches)  # naming and the DAG compile on first use
        with expect_compiles(0):
            for _ in range(4):
                next(batches)
        with expect_compiles(2):  # the short final batch: its naming and its DAG call
            next(batches)

    def test_a_pipeline_without_stages_serves_its_placed_batches_with_no_dag_call(self) -> None:
        """An empty DAG is the identity: no compile and no copy of each batch on the device."""
        pipe = _pipeline(_memory(24), num_epochs=1)  # six full batches: one naming shape
        reference = [_rows(b) for b in _pipeline(_memory(24), num_epochs=1).raw_batches()]
        jax.clear_caches()
        batches = iter(pipe)
        with expect_compiles(1):  # the naming; no DAG step
            first = next(batches)
        with expect_compiles(0):
            rest = list(batches)
        assert [_rows(b) for b in [first, *rest]] == reference

    def test_batch_norm_statistics_are_written_back(self) -> None:
        class Normalize(nnx.Module):
            def __init__(self) -> None:
                self.norm = nnx.BatchNorm(48, rngs=nnx.Rngs(0))

            def __call__(self, batch: Batch) -> Batch:
                image = batch["image"].astype(jnp.float32).reshape(batch.batch_size, -1)
                return batch.replace(data={**batch.data, "image": self.norm(image)})

        stage = Normalize()
        pipe = _pipeline(_memory(), stages=[stage], num_epochs=1)
        before = np.asarray(stage.norm.mean[...]).copy()
        list(pipe)
        assert not np.allclose(np.asarray(stage.norm.mean[...]), before)

    @pytest.mark.parametrize("graph", [True, False])
    def test_a_users_jitted_step_compiles_once(self, graph: bool) -> None:
        model = nnx.Linear(48, 2, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

        @nnx.jit(graph=graph)
        def step(model: nnx.Linear, optimizer: nnx.Optimizer, image: jax.Array) -> jax.Array:
            def loss(model: nnx.Linear) -> jax.Array:
                return jnp.mean(model(image.reshape(image.shape[0], -1) / 255.0) ** 2)

            value, grads = nnx.value_and_grad(loss, graph=graph)(model)
            optimizer.update(model, grads)
            return value

        batches = iter(_pipeline(_memory(), num_epochs=None, drop_last=True))
        step(model, optimizer, next(batches)["image"].astype(jnp.float32))
        with expect_compiles(0):
            for _ in range(5):
                step(model, optimizer, next(batches)["image"].astype(jnp.float32))

    def test_a_pipeline_passed_into_a_users_transform_uploads_no_column(self) -> None:
        pipe = _pipeline(_memory(), stages=[_Jitter()])
        source = pipe.source
        assert isinstance(source, MemorySource)

        @nnx.jit
        def apply(dag: nnx.Module, batch: Batch) -> Batch:
            return dag(batch)

        batch = next(iter(pipe.raw_batches()))
        apply(pipe.dag, batch)
        assert all(isinstance(column, np.ndarray) for column in source.data.values())
        assert isinstance(source.get_batch(batch.indices)["image"], np.ndarray)


class TestPlacement:
    def test_nothing_transfers_implicitly(self) -> None:
        assert implicit_upload_raises(), "the guard must fire on an implicit upload"
        for source in (_memory(), RecordStream(_columns(10))):
            pipe = _pipeline(source, stages=[_Jitter()])
            with jax.transfer_guard("disallow"):
                served = list(pipe)
            assert served

    def test_the_caller_s_precision_mode_reaches_the_read_threads(self) -> None:
        """``jax.enable_x64`` is thread-local: a run reads in the mode it was started in."""
        with jax.enable_x64(True):
            source = MemorySource(MemorySourceConfig(), {"x": np.ones((8, 3), np.float64)})
            stream = RecordStream({"x": np.ones((8, 3), np.float64)})
            served = [
                list(_pipeline(s, batch_size=8, num_epochs=1, shuffle=False))
                for s in (source, stream)
            ]
        assert [b["x"].dtype for run in served for b in run] == [np.float64, np.float64]

    def test_no_dataset_shaped_array_is_made(self) -> None:
        pipe = _pipeline(_memory(50), stages=[_Jitter()])
        before = jax.live_arrays()
        known = {id(a) for a in before}  # ``before`` holds them, so no id is reused
        for _ in pipe:
            made = [a for a in jax.live_arrays() if id(a) not in known and a.ndim]
            assert all(a.shape[0] <= 4 for a in made), [a.shape for a in made]


class TestIdentity:
    """Each kind's identity reaches the operators unchanged."""

    def test_an_arrival_stream_names_records_by_ordinals_never_repeated(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL)
        served = list(_pipeline(stream, num_epochs=3, shuffle=False))
        names = [n for b in served for n in _rows(b)[0]]
        assert names == list(range(30))

    def test_a_union_mix_iterates_unshuffled_and_shuffled(self) -> None:
        def mix() -> MixDataSourcesNode:
            other = MemorySource(MemorySourceConfig(), {"label": np.arange(9, dtype=np.int32)})
            return MixDataSourcesNode(MixDataSourcesConfig(weights=(1.0, 1.0)), [_memory(9), other])

        for shuffle in (False, True):
            served = list(_pipeline(mix(), shuffle=shuffle, num_epochs=2, batch_size=6))
            # The mix covers its children, so each epoch serves every record; the order differs.
            epochs = [
                [n for b in served for n, e in zip(*_rows(b), strict=True) if e == epoch]
                for epoch in (0, 1)
            ]
            assert sorted(epochs[0]) == sorted(epochs[1]) == list(range(18))
            assert (epochs[0] == epochs[1]) == (not shuffle)
            assert all("image" in b.data for b in served)


class TestState:
    """``get_state()`` is the host cursor, version 3; another layout is refused, naming both."""

    def test_the_layout(self) -> None:
        pipe = _pipeline(_memory())
        it = iter(pipe)
        next(it)
        state = pipe.get_state()
        assert state["version"] == 3
        assert state["kind"] == "indexed"
        assert (state["epoch"], state["position"], state["run_end_epoch"]) == (0, 4, 3)
        assert state["stream"] is None
        fingerprint = state["fingerprint"]
        assert fingerprint["order"] == {"kind": "global"}
        assert fingerprint["batch_size"] == 4 and fingerprint["length"] == 23
        assert all(isinstance(word, int) for word in fingerprint["seed"])

    def test_a_stream_s_layout(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL)
        pipe = _pipeline(stream, shuffle=False, num_epochs=None)
        it = iter(pipe)
        for _ in range(4):
            next(it)
        state = pipe.get_state()
        assert state["kind"] == "arrival"
        assert state["stream"] == {"pass": 1, "records": 6, "arrived": 16, "passes_left": None}
        assert state["epoch"] is None and state["position"] is None

    @pytest.mark.parametrize("threads", [1, 4, 8])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_resume_after_any_batch_with_threads_ahead_is_exact(
        self, threads: int, drop_last: bool
    ) -> None:
        def build() -> Pipeline:
            pipe = _pipeline(_memory(), stages=[_Jitter()], drop_last=drop_last, num_epochs=3)
            pipe.host_stage.read_threads = threads
            return pipe

        whole = [(_rows(b), np.asarray(b["image"])) for b in build()]
        for done in range(len(whole)):
            pipe = build()
            served = []
            it = iter(pipe)
            for _ in range(done):
                served.append(next(it))
            threading.Event().wait(0.02)  # let the reads run ahead
            state = pipe.get_state()
            pipe.close()
            resumed = build()
            resumed.set_state(state)
            rest = list(resumed)
            got = [(_rows(b), np.asarray(b["image"])) for b in served + rest]
            assert len(got) == len(whole)
            for (rows, image), (expected_rows, expected_image) in zip(got, whole, strict=True):
                assert rows == expected_rows
                np.testing.assert_array_equal(image, expected_image)

    def test_a_stream_resumes_exactly(self) -> None:
        def build() -> Pipeline:
            stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL, chunk=3)
            return _pipeline(stream, num_epochs=3, shuffle=True)

        whole = [_rows(b) for b in build()]
        for done in range(len(whole)):
            pipe = build()
            it = iter(pipe)
            served = [_rows(next(it)) for _ in range(done)]
            state = pipe.get_state()
            pipe.close()
            resumed = build()
            resumed.set_state(state)
            assert served + [_rows(b) for b in resumed] == whole

    def test_the_state_round_trips_through_the_checkpoint_store(self, tmp_path: Path) -> None:
        pipe = _pipeline(_memory(), num_epochs=None)
        it = iter(pipe)
        next(it)
        checkpoint = IteratorCheckpoint(
            tmp_path / "ckpt", store=OrbaxCheckpointStore(tmp_path / "ckpt")
        )
        checkpoint.save(pipe, 1)
        rest = [_rows(next(it)) for _ in range(5)]
        fresh = _pipeline(_memory(), num_epochs=None)
        checkpoint.restore(fresh)
        it = iter(fresh)
        assert [_rows(next(it)) for _ in range(5)] == rest
        checkpoint.close()

    def test_no_rng_count_moves_while_iterating(self) -> None:
        pipe = _pipeline(_memory(), stages=[_Jitter()])
        before = jax.tree.leaves(nnx.state(pipe, nnx.RngCount))
        list(pipe)
        after = jax.tree.leaves(nnx.state(pipe, nnx.RngCount))
        assert [int(a) for a in after] == [int(b) for b in before]

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda s: s.pop("version"), "without a version"),
            (lambda s: s.update(version=2), "version 2"),
            (lambda s: s.update(version=4), "version 4"),
            (lambda s: s.update(kind="arrival"), "kind"),
            (lambda s: s["fingerprint"].update(seed=[1, 2]), "seed"),
            (lambda s: s["fingerprint"].update(order={"kind": "block", "b": 4, "n": 1}), "order"),
            (lambda s: s["fingerprint"].update(batch_size=8), "batch_size"),
        ],
    )
    def test_another_layout_is_refused_naming_it(self, change: Any, message: str) -> None:
        pipe = _pipeline(_memory())
        state = pipe.get_state()
        change(state)
        with pytest.raises(ValueError, match=message):
            _pipeline(_memory()).set_state(state)


class TestRun:
    """``reset()`` starts the next epoch's run and ``batches_left()`` counts the rest of a run."""

    def test_batches_left_counts_what_is_then_served(self) -> None:
        for source in (_memory(), RecordStream(_columns(10))):
            pipe = _pipeline(source, num_epochs=2)
            it = iter(pipe)
            next(it)
            left = pipe.batches_left()
            assert left == len(list(pipe))
        unsized = _Unsized(_columns(10), kind=RecordIdentity.ARRIVAL)
        assert _pipeline(unsized, num_epochs=2).batches_left() is None

    def test_a_run_ends_and_reset_starts_the_next_epoch(self) -> None:
        pipe = _pipeline(_memory(8), batch_size=4, num_epochs=1, shuffle=False)
        assert len(list(pipe)) == 2
        assert list(pipe) == []
        pipe.reset()
        served = list(pipe)
        assert [e for b in served for e in _rows(b)[1]] == [1] * 8


class _Unsized(RecordStream):
    """A stream of unknown length, as a HuggingFace stream is."""

    def __len__(self) -> int:
        raise NotImplementedError("a stream of unknown length")


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Wide(DataSourceModule):
    """An indexed source of ``length`` records whose host read serves each record's index."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self, length: int) -> None:
        super().__init__(_Config())
        self.rows = length

    def __len__(self) -> int:
        return self.rows

    def get_batch(self, indices: Any, *, epochs: Any = 0, contiguous: bool = False) -> Batch:
        del contiguous
        words = np.asarray(indices, np.uint32)
        return batch_ops.from_arrays(
            {"lo": words[:, 1].copy()},
            indices=words,
            epochs=np.ascontiguousarray(np.broadcast_to(np.asarray(epochs, np.int32), len(words))),
        )


class TestPast2To31:
    @pytest.mark.parametrize("length", [(1 << 31) + 5, 1 << 40])
    def test_iteration_names_distinct_records_past_int32_and_resumes(self, length: int) -> None:
        pipe = _pipeline(_Wide(length), batch_size=256, num_epochs=None)
        start = min(length - 300, (1 << 32) - 512)  # positions past 2**31, and past 2**32
        state = pipe.get_state()
        state["position"] = start
        pipe.set_state(state)
        it = iter(pipe)
        first = [next(it) for _ in range(4)]
        names = [n for b in first for n in _rows(b)[0]]
        assert len(set(names)) == len(names) and max(names) < length
        position, epoch = start, 0
        for _ in range(4):
            position, epoch = pipe.epoch_plan.advance(position, epoch, 256)
        after = pipe.get_state()
        assert (after["position"], after["epoch"]) == (position, epoch)
        resumed = _pipeline(_Wide(length), batch_size=256, num_epochs=None)
        resumed.set_state(after)
        assert _rows(next(iter(resumed)))[0] == _rows(next(it))[0]

    def test_an_arrival_stream_continues_its_ordinals_past_int32(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL)
        pipe = _pipeline(stream, shuffle=False, num_epochs=None)
        state = pipe.get_state()
        state["stream"]["arrived"] = (1 << 31) + 5
        pipe.set_state(state)
        it = iter(pipe)
        assert _rows(next(it))[0] == list(range((1 << 31) + 5, (1 << 31) + 9))
        after = pipe.get_state()
        resumed = _pipeline(
            RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL), shuffle=False, num_epochs=None
        )
        resumed.set_state(after)
        assert _rows(next(iter(resumed)))[0] == list(range((1 << 31) + 9, (1 << 31) + 13))


def test_the_session_still_serves_step_and_scan() -> None:
    pipe = _pipeline(_memory())
    assert pipe.step().batch_size == 4
    assert HostStage is not None
