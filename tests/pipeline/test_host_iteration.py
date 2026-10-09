"""``for batch in pipe`` runs on the host stage, and the pipeline's state is its versioned cursor.

Every source kind is read by the host stage (Grain threads, the order named on the CPU device,
placement as each batch is taken), and each batch runs through the DAG in one compiled call.
Iteration serves the compiled session's records, bit for bit, holds no dataset on the device, and
stands where the last batch taken ended: ``get_state()`` is that cursor, version 3, and
``set_state`` resumes exactly, a resumed run ending where the uninterrupted one would, whatever the
read threads had read ahead. Any other state layout is refused, naming both versions.
"""

from __future__ import annotations

import logging
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
from datarax.core.data_source import DataSourceModule, known_length, RecordIdentity
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import from_words
from datarax.core.operator import OperatorModule, require_key
from datarax.pipeline import Pipeline
from datarax.pipeline.host_stage import HostStage
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.device_arrays import arrays_made_since
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
    @pytest.mark.parametrize("num_epochs", [1, 3])
    @pytest.mark.parametrize(
        ("length", "batch_size"),
        [
            *((length, size) for length in (7, 10, 23, 64) for size in (1, 3, 4, 8)),
            (64, 256),
            (50_000, 256),
        ],
    )
    def test_iteration_equals_the_session(
        self, shuffle: bool, drop_last: bool, num_epochs: int, length: int, batch_size: int
    ) -> None:
        if drop_last and batch_size > length:
            return  # the plan refuses it
        session = list(
            _pipeline(
                _memory(length),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_epochs=num_epochs,
            ).session()
        )
        served = list(
            _pipeline(
                _memory(length),
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_epochs=num_epochs,
            )
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
        for _ in pipe:
            made = [a for a in arrays_made_since(before) if a.ndim]
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

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_resume_after_any_batch_with_threads_ahead_is_exact(
        self, drop_last: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """1, 4 and 8 read threads with 8 units read ahead: Grain caps the threads at the buffer.

        After every batch of a 3-epoch run the state is taken while reads are ahead of it, and
        a rebuilt pipeline resumes it: the rest, the stochastic draws included, equals the
        uninterrupted run, and the states are equal across thread counts.
        """
        reads = threading.Condition()
        count = [0]
        original = MemorySource.get_batch

        def counted(self: MemorySource, indices: Any, **kwargs: Any) -> Batch:
            batch = original(self, indices, **kwargs)
            with reads:
                count[0] += 1
                reads.notify_all()
            return batch

        monkeypatch.setattr(MemorySource, "get_batch", counted)

        def build(threads: int) -> Pipeline:
            pipe = _pipeline(_memory(), stages=[_Jitter()], drop_last=drop_last, num_epochs=3)
            pipe.host_stage._read_threads = threads
            pipe.host_stage._read_buffer = 8
            return pipe

        whole = [(_rows(b), np.asarray(b["image"])) for b in build(1)]
        states: dict[int, list[Any]] = {}
        for threads in (1, 4, 8):
            states[threads] = []
            ahead = 0
            for done in range(len(whole)):
                with reads:
                    count[0] = 0
                pipe = build(threads)
                it = iter(pipe)
                served = [next(it) for _ in range(done)]
                if done == 1:  # the reads run ahead of the batch taken
                    with reads:
                        reads.wait_for(lambda: count[0] > 1, timeout=30)
                        ahead = count[0] - 1
                state = pipe.get_state()
                states[threads].append(state)
                pipe.close()
                resumed = build(threads)
                resumed.set_state(state)
                rest = list(resumed)
                got = [(_rows(b), np.asarray(b["image"])) for b in served + rest]
                assert len(got) == len(whole)
                for (rows, image), (expected_rows, expected_image) in zip(got, whole, strict=True):
                    assert rows == expected_rows
                    np.testing.assert_array_equal(image, expected_image)
            assert ahead > 0, threads
        assert states[1] == states[4] == states[8]

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

    def test_a_stream_resumed_mid_pass_logs_its_replay(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def build() -> Pipeline:
            stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL, chunk=3)
            return _pipeline(stream, num_epochs=2, shuffle=True)

        pipe = build()
        it = iter(pipe)
        next(it)
        next(it)
        state = pipe.get_state()
        pipe.close()
        assert state["stream"]["records"] == 8
        resumed = build()
        resumed.set_state(state)
        with caplog.at_level(logging.INFO, logger="datarax.pipeline.host_stage"):
            next(iter(resumed))
        replays = [r.getMessage() for r in caplog.records if "replayed" in r.getMessage()]
        assert len(replays) == 1
        assert "RecordStream" in replays[0] and "replayed 8 records of pass 0" in replays[0]

    def test_a_stream_resumed_at_a_pass_start_replays_nothing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL, chunk=3)
        pipe = _pipeline(stream, num_epochs=2, shuffle=True)
        pipe.set_state(pipe.get_state())
        with caplog.at_level(logging.INFO, logger="datarax.pipeline.host_stage"):
            next(iter(pipe))
        assert not [r for r in caplog.records if "replayed" in r.getMessage()]

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

    def test_a_state_without_a_fingerprint_is_refused(self) -> None:
        state = _pipeline(_memory()).get_state()
        del state["fingerprint"]
        with pytest.raises(ValueError, match="fingerprint"):
            _pipeline(_memory()).set_state(state)

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


def _shard(index: int, count: int, length: int = 10) -> MemorySource:
    return MemorySource(MemorySourceConfig(shard_id=index, num_workers=count), _columns(length))


class TestAShardsState:
    """A state names the shard of the records it was saved on; another shard's is refused.

    A worker's shard (``MemorySourceConfig(shard_id, num_workers)``) serves positions
    ``[shard_id::num_workers]`` of the global order, so two shards of one dataset have the same
    length and a state's position means other records on each.
    """

    def test_a_state_resumes_its_own_shard(self) -> None:
        pipe = _pipeline(_shard(1, 2), batch_size=2, shuffle=False, num_epochs=1)
        it = iter(pipe)
        next(it)
        state = pipe.get_state()
        rest = [_rows(batch) for batch in it]

        fresh = _pipeline(_shard(1, 2), batch_size=2, shuffle=False, num_epochs=1)
        fresh.set_state(state)

        assert [_rows(batch) for batch in fresh] == rest

    @pytest.mark.parametrize(
        ("restored", "length"),
        [((1, 2), 10), ((0, 3), 15), ((0, 2), 9)],
        ids=["another-shard", "another-shard-count", "another-dataset-of-equal-shard-length"],
    )
    def test_a_state_of_another_partition_is_refused_naming_it(
        self, restored: tuple[int, int], length: int
    ) -> None:
        pipe = _pipeline(_shard(0, 2, length=10), batch_size=2, shuffle=False, num_epochs=1)
        next(iter(pipe))
        state = pipe.get_state()
        other = _pipeline(
            _shard(*restored, length=length), batch_size=2, shuffle=False, num_epochs=1
        )
        # the length cannot tell them apart
        assert known_length(other.source) == known_length(pipe.source) == 5

        with pytest.raises(ValueError, match="shard"):
            other.set_state(state)

    def test_a_state_saved_before_shards_were_named_resumes_only_a_whole_source(self) -> None:
        """A version-3 state without ``shard`` was saved on a whole source or a shard unknown."""
        whole = _pipeline(_memory(10), batch_size=2, shuffle=False, num_epochs=1)
        next(iter(whole))
        state = whole.get_state()
        del state["fingerprint"]["shard"]

        _pipeline(_memory(10), batch_size=2, shuffle=False, num_epochs=1).set_state(state)
        with pytest.raises(ValueError, match="shard"):
            _pipeline(_shard(0, 2, length=20), batch_size=2, shuffle=False, num_epochs=1).set_state(
                state
            )

    def test_a_whole_source_s_state_is_refused_by_a_shard_and_back(self) -> None:
        whole = _pipeline(_memory(10), batch_size=2, shuffle=False, num_epochs=1)
        half = _pipeline(_shard(0, 2, length=20), batch_size=2, shuffle=False, num_epochs=1)
        next(iter(whole))
        next(iter(half))
        assert known_length(whole.source) == known_length(half.source)

        with pytest.raises(ValueError, match="shard"):
            half.set_state(whole.get_state())
        with pytest.raises(ValueError, match="shard"):
            whole.set_state(half.get_state())


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


class TestUnderNnxTransforms:
    """The pipeline's own graph under NNX's split, merge and clone (brief section 7)."""

    @pytest.mark.parametrize("graph", [True, False])
    def test_the_graph_definition_does_not_move_across_a_pass(self, graph: bool) -> None:
        pipe = _pipeline(_memory(), stages=[_Jitter()], num_epochs=1)
        before = nnx.graphdef(pipe, graph=graph)
        batches = iter(pipe)
        next(batches)
        during = nnx.graphdef(pipe, graph=graph)
        list(batches)
        assert before == during == nnx.graphdef(pipe, graph=graph)

    @pytest.mark.parametrize("copy", ["merge", "tree merge", "clone"])
    def test_a_copy_continues_the_cursor_after_the_original(self, copy: str) -> None:
        """A copy shares the host stage: it continues where the original stopped, in turn."""
        whole = [_rows(b) for b in _pipeline(_memory(), stages=[_Jitter()])]
        pipe = _pipeline(_memory(), stages=[_Jitter()])
        batches = iter(pipe)
        served = [_rows(next(batches)) for _ in range(3)]
        if copy == "clone":
            other = nnx.clone(pipe)
        else:
            other = nnx.merge(*nnx.split(pipe, graph=copy == "merge"))
        assert other.get_state() == pipe.get_state()
        assert other.host_stage is pipe.host_stage
        served += [_rows(b) for b in other]
        assert served == whole

    def test_tree_mode_round_trips_with_no_python_value_in_a_variable(self) -> None:
        pipe = _pipeline(_memory(), stages=[_Jitter()])
        graphdef, state = nnx.split(pipe, graph=False)
        leaves = jax.tree.leaves(state)
        assert leaves and all(isinstance(leaf, jax.Array | np.ndarray) for leaf in leaves)
        merged = nnx.merge(graphdef, state)
        reference = [_rows(b) for b in _pipeline(_memory(), stages=[_Jitter()])]
        assert [_rows(b) for b in merged] == reference
        # The pipeline's own attributes, not its stages': an operator's ``deterministic`` is
        # Flax's mode flag, which ``train()``/``eval()`` are meant to set.
        assert not {"deterministic", "use_running_average"} & set(vars(pipe))

    def test_batch_norm_statistics_written_back_equal_a_jitted_dag_s(self) -> None:
        """The Tier-A call writes back what ``nnx.jit(pipe.dag)`` would leave in the module."""

        class Normalize(nnx.Module):
            def __init__(self) -> None:
                self.norm = nnx.BatchNorm(48, rngs=nnx.Rngs(0))

            def __call__(self, batch: Batch) -> Batch:
                image = batch["image"].astype(jnp.float32).reshape(batch.batch_size, -1)
                return batch.replace(data={**batch.data, "image": self.norm(image)})

        stage, twin = Normalize(), Normalize()
        pipe = _pipeline(_memory(24), stages=[stage], num_epochs=1, shuffle=False)
        reference = _pipeline(_memory(24), stages=[twin], num_epochs=1, shuffle=False)
        apply = nnx.jit(lambda dag, batch: dag(batch))
        expected = [apply(reference.dag, batch) for batch in reference.raw_batches()]
        served = list(pipe)
        for got, want in zip(served, expected, strict=True):
            np.testing.assert_array_equal(got["image"], want["image"])
        assert len(served) == len(expected) == 6
        for name in ("mean", "var"):
            np.testing.assert_array_equal(
                np.asarray(getattr(stage.norm, name)[...]),
                np.asarray(getattr(twin.norm, name)[...]),
            )


class TestTheCompiledPath:
    @pytest.mark.parametrize("length", [(1 << 31) + 5, 1 << 40])
    @pytest.mark.parametrize("call", ["step", "session"])
    def test_it_still_refuses_a_source_past_int32(self, length: int, call: str) -> None:
        """Positions are int32 on the compiled path; the host stage serves these lengths."""
        pipe = _pipeline(_Wide(length), batch_size=256, num_epochs=None)
        with pytest.raises(OverflowError):
            pipe.step() if call == "step" else next(iter(pipe.session()))
        assert len(_rows(next(iter(pipe)))[0]) == 256


class TestATracedReadIsNamed:
    """``step()`` and ``session()`` over an INDEXED source without a traced read: the host path."""

    @pytest.mark.parametrize("call", ["step", "session"])
    def test_the_refusal_names_iteration_and_raw_batches(self, call: str) -> None:
        pipe = _pipeline(_Wide(64), batch_size=8)
        with pytest.raises(
            NotImplementedError, match=r"get_records.*for batch in pipe.*raw_batches\(\)"
        ):
            pipe.step() if call == "step" else next(iter(pipe.session()))


class _ForwardRead(_Wide):
    """An INDEXED source whose ``get_batch`` is a stream's forward read."""

    def get_batch(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, batch_size: int, *, key: Any = None, read_size: int | None = None
    ) -> Batch:
        raise AssertionError("a run must refuse this source before reading it")


def test_an_indexed_source_with_a_stream_s_read_is_refused_when_the_run_starts() -> None:
    pipe = _pipeline(_ForwardRead(16), batch_size=4)
    with pytest.raises(TypeError, match=r"get_batch\(indices, \*, epochs, contiguous\)"):
        pipe.raw_batches()
