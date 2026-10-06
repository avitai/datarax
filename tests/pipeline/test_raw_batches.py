"""``pipe.raw_batches()``: host-read batches placed on the device, read ahead by Grain threads.

The host stage names each batch's records on the CPU device, reads them with the source's
stateless host read ahead of the consumer and places each batch as it is taken, uncommitted.
Nothing of the source moves to the device but the batches, every transfer is explicit, and where
iteration stands advances as the consumer takes batches. The DAG is not applied: ``pipe.dag`` runs
inside the caller's differentiated step (OWN-0926-E2E). Chunks of ``K`` batches are one host read
and one transfer, ``(K, B, ...)``. With ``with_provenance=True`` each batch comes with its
records' provenance. One Grain iterator serves a run across calls and leaves no thread behind.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import ElementOperatorConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.core.prng import key_words
from datarax.operators import ElementOperator
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.streams import RecordStream
from tests.test_common.transfers import implicit_upload_raises


_N = 50


def _columns(length: int = _N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(length)
    return {
        "image": rng.integers(0, 256, (length, 4, 4, 3), dtype=np.uint8),
        "label": np.arange(length, dtype=np.int32),
    }


def _memory(length: int = _N, *, from_device: bool = False) -> MemorySource:
    columns: dict[str, Any] = _columns(length)
    if from_device:
        columns = {name: jnp.asarray(column) for name, column in columns.items()}
    return MemorySource(MemorySourceConfig(), columns)


def _mix() -> MixDataSourcesNode:
    return MixDataSourcesNode(MixDataSourcesConfig(weights=(1.0, 1.0)), [_memory(20), _memory(30)])


def _disk(tmp_path: Path, length: int = _N) -> StreamingDiskSource:
    path = tmp_path / "rows.npy"
    np.save(path, np.arange(length * 3, dtype=np.float32).reshape(length, 3))
    return StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))


def _pipeline(
    source: DataSourceModule,
    *,
    batch_size: int = 8,
    shuffle: bool = True,
    drop_last: bool = False,
    num_epochs: int | None = 2,
    stages: list[nnx.Module] | None = None,
) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[] if stages is None else stages,
        batch_size=batch_size,
        rngs=nnx.Rngs(3),
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
    )


def _names(batch: Batch) -> list[int]:
    return [int(v) for v in from_words(np.asarray(batch.indices))]


def _session_batches(pipe: Pipeline) -> list[Batch]:
    return list(pipe.session())


@nnx.jit
def _user_step(pipeline: Pipeline) -> jax.Array:
    return jnp.sum(pipeline.step()["image"])


def _grain_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if "grain" in t.name.lower()]


class TestOrder:
    """``raw_batches()`` serves the session's records, in its order, without the DAG."""

    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    def test_the_records_are_the_session_s(
        self, shuffle: bool, drop_last: bool, batch_size: int
    ) -> None:
        reference = _session_batches(
            _pipeline(_memory(), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        )
        served = list(
            _pipeline(
                _memory(), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
            ).raw_batches()
        )
        assert len(served) == len(reference)
        for raw, session in zip(served, reference, strict=True):
            np.testing.assert_array_equal(raw.indices, session.indices)
            np.testing.assert_array_equal(raw.epochs, session.epochs)
            np.testing.assert_array_equal(raw["image"], session["image"])
            np.testing.assert_array_equal(raw["label"], session["label"])

    def test_the_dag_is_not_applied(self) -> None:
        def double(element: Any, key: Any = None) -> Any:
            del key
            return element.update_data({"label": element.data["label"] * 2})

        stage = ElementOperator(
            ElementOperatorConfig(stochastic=False), fn=double, rngs=nnx.Rngs(0)
        )
        pipe = _pipeline(_memory(), shuffle=False, stages=[stage])
        first = next(iter(pipe.raw_batches()))
        assert _names(first) == list(range(8))
        np.testing.assert_array_equal(first["label"], np.arange(8))

    @pytest.mark.parametrize("make", [_mix, lambda: _memory(from_device=True)])
    def test_a_mix_and_a_source_built_from_device_arrays_serve_the_session_s_records(
        self, make: Any
    ) -> None:
        reference = _session_batches(_pipeline(make()))
        served = list(_pipeline(make()).raw_batches())
        assert [_names(b) for b in served] == [_names(b) for b in reference]

    def test_a_disk_source_serves_the_session_s_records(self, tmp_path: Path) -> None:
        reference = _session_batches(_pipeline(_disk(tmp_path)))
        served = list(_pipeline(_disk(tmp_path)).raw_batches())
        assert [_names(b) for b in served] == [_names(b) for b in reference]
        for raw, session in zip(served, reference, strict=True):
            np.testing.assert_array_equal(raw["x"], session["x"])

    @pytest.mark.parametrize("threads", [1, 4, 8])
    def test_any_number_of_read_threads_serves_the_same_batches(self, threads: int) -> None:
        reference = [_names(b) for b in _pipeline(_memory()).raw_batches()]
        pipe = _pipeline(_memory())
        pipe.host_stage.read_threads = threads
        assert [_names(b) for b in pipe.raw_batches()] == reference

    @pytest.mark.parametrize("depth", [1, 4])
    def test_reads_run_ahead_by_the_read_buffer(
        self, monkeypatch: pytest.MonkeyPatch, depth: int
    ) -> None:
        calls: list[int] = []
        original = MemorySource.get_batch

        def counted(self: MemorySource, indices: Any, **kwargs: Any) -> Batch:
            calls.append(1)
            return original(self, indices, **kwargs)

        monkeypatch.setattr(MemorySource, "get_batch", counted)
        pipe = _pipeline(_memory(), num_epochs=None)
        pipe.host_stage.read_buffer = depth
        batches = iter(pipe.raw_batches())
        next(batches)
        threading.Event().wait(0.3)  # time for the reads to run ahead
        assert len(calls) - 1 == depth
        pipe.close()

    def test_one_host_read_per_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []
        original = MemorySource.get_batch

        def counted(self: MemorySource, indices: Any, **kwargs: Any) -> Batch:
            calls.append(len(np.asarray(indices)))
            return original(self, indices, **kwargs)

        monkeypatch.setattr(MemorySource, "get_batch", counted)
        served = list(_pipeline(_memory()).raw_batches())
        assert calls == [b.batch_size for b in served]


class TestChunks:
    """``raw_batches(chunk=K)``: chunk ``k`` holds batches ``kK .. kK + K - 1``, read once."""

    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("chunk", [1, 2, 3])
    def test_a_chunk_is_its_batches_stacked(self, drop_last: bool, chunk: int) -> None:
        batches = list(_pipeline(_memory(), drop_last=drop_last).raw_batches())
        served = list(_pipeline(_memory(), drop_last=drop_last).raw_batches(chunk=chunk))
        full = len(batches) // chunk
        if not drop_last and batches[-1].batch_size != 8:
            full = (len(batches) - 1) // chunk  # the short final batch comes singly, last
        flat: list[Batch] = []
        for position, unit in enumerate(served):
            if position < full:
                assert np.asarray(unit["image"]).shape[:2] == (chunk, 8)
                flat.extend(jax.tree.map(lambda leaf, k=k: leaf[k], unit) for k in range(chunk))
            else:
                assert np.asarray(unit["image"]).ndim == 4  # a single batch
                flat.append(unit)
        assert len(flat) == len(batches)
        for got, expected in zip(flat, batches, strict=True):
            np.testing.assert_array_equal(got.indices, expected.indices)
            np.testing.assert_array_equal(got.epochs, expected.epochs)
            np.testing.assert_array_equal(got["image"], expected["image"])

    def test_one_host_read_and_one_transfer_per_chunk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[int] = []
        original = MemorySource.get_batch

        def counted(self: MemorySource, indices: Any, **kwargs: Any) -> Batch:
            reads.append(len(np.asarray(indices)))
            return original(self, indices, **kwargs)

        monkeypatch.setattr(MemorySource, "get_batch", counted)
        pipe = _pipeline(_memory(48), num_epochs=1, shuffle=False)
        placed: list[Any] = []
        original_put = jax.device_put

        def watched(value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, Batch):
                placed.append(value)
            return original_put(value, *args, **kwargs)

        monkeypatch.setattr(jax, "device_put", watched)
        served = list(pipe.raw_batches(chunk=3))
        assert len(served) == 2
        assert reads == [24, 24]
        assert len(placed) == 2

    def test_a_chunk_over_its_byte_bound_is_refused_naming_both(self) -> None:
        pipe = _pipeline(_memory())
        batch_bytes = 8 * (4 * 4 * 3 + 4)
        with pytest.raises(ValueError, match=rf"{3 * batch_bytes}.*{batch_bytes}"):
            next(iter(pipe.raw_batches(chunk=3, max_chunk_bytes=batch_bytes)))


class TestPlacement:
    """Batches reach the device whole, as the device holds them, and nothing else does."""

    @staticmethod
    def _new_dataset_shaped(before: list[jax.Array]) -> list[tuple[int, ...]]:
        known = {id(a) for a in before}  # ``before`` holds them, so no id is reused
        return [
            a.shape for a in jax.live_arrays() if id(a) not in known and a.ndim and a.shape[0] == _N
        ]

    @pytest.mark.parametrize("from_device", [False, True])
    def test_no_dataset_shaped_array_is_made_and_the_columns_stay_on_the_host(
        self, from_device: bool
    ) -> None:
        """On a CPU lane a source built from device arrays may view their buffers (host memory),
        so only arrays made while iterating are counted. Control: the session uploads columns."""
        source = _memory(from_device=from_device)
        columns = dict(source.data)
        gc.collect()
        before = jax.live_arrays()
        control = _pipeline(_memory(), num_epochs=1)
        list(control.session())
        assert self._new_dataset_shaped(before), "the control must upload the dataset"
        before = jax.live_arrays()
        pipe = _pipeline(source, num_epochs=3)
        for batch in pipe.raw_batches():
            assert self._new_dataset_shaped(before) == []
            assert batch.batch_size <= 8
        assert all(isinstance(column, np.ndarray) for column in source.data.values())
        assert all(source.data[name] is column for name, column in columns.items())

    def test_the_bytes_placed_per_batch_are_the_batch_s(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        placed: list[int] = []
        original_put = jax.device_put

        def watched(value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, Batch):
                placed.append(sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves(value)))
            return original_put(value, *args, **kwargs)

        monkeypatch.setattr(jax, "device_put", watched)
        served = list(_pipeline(_memory(), num_epochs=1).raw_batches())
        expected = [sum(leaf.nbytes for leaf in jax.tree.leaves(b)) for b in served]
        assert placed == expected
        assert all(b["image"].dtype == jnp.uint8 for b in served)

    @pytest.mark.parametrize("kind", ["memory", "stream"])
    def test_nothing_transfers_implicitly(self, kind: str) -> None:
        assert implicit_upload_raises(), "the guard must fire on an implicit upload"
        if kind == "memory":
            pipe = _pipeline(_memory())
        else:
            pipe = _pipeline(
                RecordStream(_columns(20), kind=RecordIdentity.STREAM_IDS), shuffle=True
            )
        with jax.transfer_guard("disallow"):
            served = list(pipe.raw_batches())
        assert served

    def test_a_batch_is_placed_when_it_is_taken_never_ahead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads run ahead on a thread; placement waits for the consumer (P3's peak RSS bound).

        A placement thread running ahead holds more device copies and, measured on the P3 shape,
        raises peak RSS past the 75 MiB bound; an accelerator's ``device_put`` is asynchronous, so
        a batch placed when taken still transfers while the previous step computes.
        """
        placed: list[int] = []
        original_put = jax.device_put

        def watched(value: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(value, Batch):
                placed.append(1)
            return original_put(value, *args, **kwargs)

        monkeypatch.setattr(jax, "device_put", watched)
        batches = iter(_pipeline(_memory(), num_epochs=None).raw_batches())
        for taken in range(1, 4):
            next(batches)
            threading.Event().wait(0.3)  # time for any thread to run ahead
            assert len(placed) == taken

    def test_batches_are_uncommitted_so_a_jitted_step_compiles_once(self) -> None:
        step = jax.jit(lambda x, image: x + jnp.sum(image.astype(jnp.float32)))
        state = jnp.zeros(())
        batches = iter(_pipeline(_memory(), num_epochs=None, drop_last=True).raw_batches())
        state = step(state, next(batches)["image"])
        with expect_compiles(0):
            for _ in range(5):
                state = step(state, next(batches)["image"])
        assert not next(batches)["image"].committed


class TestIdentity:
    """Each kind's identity reaches the batch: the source's indices and epochs."""

    def test_an_arrival_stream_names_records_by_ordinals_never_repeated(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.ARRIVAL)
        served = list(_pipeline(stream, batch_size=4, num_epochs=3).raw_batches())
        names = [n for b in served for n in _names(b)]
        assert names == list(range(30))
        assert [int(e) for b in served for e in np.asarray(b.epochs)] == [
            p for p in range(3) for _ in range(10)
        ]

    def test_a_stream_naming_records_by_id_serves_its_pass_order(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.STREAM_IDS)
        served = list(_pipeline(stream, batch_size=4, num_epochs=2, shuffle=False).raw_batches())
        assert [n for b in served for n in _names(b)] == [*range(10), *range(10)]

    def test_the_stream_s_own_position_is_not_moved(self) -> None:
        stream = RecordStream(_columns(10), kind=RecordIdentity.STREAM_IDS)
        list(_pipeline(stream, batch_size=4).raw_batches())
        assert stream.pass_index == 0
        assert _names(stream.get_batch(3)) == [0, 1, 2]


class TestProvenance:
    def test_an_indexed_source_s_provenance_is_looked_up_by_index(self) -> None:
        source = MemorySource(
            MemorySourceConfig(),
            [{"x": np.float32(i), "name": f"r{i}"} for i in range(12)],
        )
        pipe = _pipeline(source, batch_size=5)
        for batch, provenance in pipe.raw_batches(with_provenance=True):
            assert [p["name"] for p in provenance] == [f"r{n}" for n in _names(batch)]

    def test_a_stream_s_provenance_comes_beside_its_batch(self) -> None:
        stream = RecordStream(
            _columns(10), kind=RecordIdentity.ARRIVAL, texts=[f"t{i}" for i in range(10)]
        )
        for batch, provenance in _pipeline(stream, batch_size=4, shuffle=False).raw_batches(
            with_provenance=True
        ):
            assert [p["text"] for p in provenance] == [
                f"t{int(v)}" for v in np.asarray(batch["label"])
            ]


class TestAnIteratorServesItsOwnRun:
    """Each ``raw_batches()`` iterator serves the run it was created for, and no other.

    Iterators of one run continue one cursor in turn. Once its run reached its end an iterator
    keeps raising ``StopIteration``; once something else ended its run (``reset()``,
    ``set_state()``, ``close()``, a later call reading with other options, a clone's call) its
    next ``next()`` is refused, naming that call, so no iterator serves a run it was not made for.
    """

    def test_two_iterators_of_one_run_continue_one_cursor(self) -> None:
        pipe = _pipeline(_memory(32), shuffle=False, num_epochs=1)
        first, second = pipe.raw_batches(), pipe.raw_batches()
        taken = [_names(next(iterator))[0] for iterator in (first, second, first, second)]
        assert taken == [0, 8, 16, 24]
        for iterator in (first, second):
            with pytest.raises(StopIteration):
                next(iterator)

    def test_a_new_loop_after_a_break_continues_the_run(self) -> None:
        pipe = _pipeline(_memory(32), shuffle=False, num_epochs=1)
        for batch in pipe.raw_batches():
            assert _names(batch)[0] == 0
            break
        assert [_names(b)[0] for b in pipe.raw_batches()] == [8, 16, 24]

    def test_an_iterator_whose_run_ended_stays_ended_after_reset(self) -> None:
        pipe = _pipeline(_memory(16), shuffle=False, num_epochs=1)
        ended = pipe.raw_batches()
        assert len(list(ended)) == 2
        pipe.reset()
        current = pipe.raw_batches()
        assert _names(next(current))[0] == 0
        with pytest.raises(StopIteration):
            next(ended)
        assert [_names(b)[0] for b in current] == [8]

    @pytest.mark.parametrize(
        ("end_it", "cause"),
        [
            (lambda pipe, state: pipe.reset(), r"reset\(\)"),
            (lambda pipe, state: pipe.set_state(state), r"set_state\(\)"),
            (lambda pipe, state: pipe.close(), r"close\(\)"),
            (lambda pipe, state: next(pipe.raw_batches(chunk=2)), "chunk=2"),
            (
                lambda pipe, state: next(pipe.raw_batches(with_provenance=True)),
                "with_provenance=True",
            ),
        ],
    )
    def test_a_live_iterator_whose_run_was_ended_is_refused_naming_the_call(
        self, end_it: Any, cause: str
    ) -> None:
        pipe = _pipeline(_memory(), num_epochs=None)
        state = pipe.get_state()
        live = pipe.raw_batches()
        next(live)
        end_it(pipe, state)
        with pytest.raises(RuntimeError, match=cause):
            next(live)
        with pytest.raises(RuntimeError, match=cause):  # and stays refused
            next(live)

    def test_a_processing_loop_held_while_raw_batches_reads_other_units_is_refused(self) -> None:
        def double(element: Any, key: Any = None) -> Any:
            del key
            return element.update_data({"label": element.data["label"] * 2})

        stage = ElementOperator(
            ElementOperatorConfig(stochastic=False), fn=double, rngs=nnx.Rngs(0)
        )
        pipe = _pipeline(_memory(), stages=[stage], num_epochs=None)
        processed = iter(pipe)
        next(processed)
        next(pipe.raw_batches(chunk=2))
        with pytest.raises(RuntimeError, match="chunk=2"):
            next(processed)

    def test_a_clone_continues_the_cursor_in_turn_but_not_interleaved(self) -> None:
        reference = [_names(b) for b in _pipeline(_memory(), num_epochs=1).raw_batches()]
        pipe = _pipeline(_memory(), num_epochs=1)
        original = pipe.raw_batches()
        taken = [_names(next(original)) for _ in range(2)]
        clone = nnx.clone(pipe)
        assert clone.get_state() == pipe.get_state()
        copied = clone.raw_batches()
        taken.append(_names(next(copied)))
        with pytest.raises(RuntimeError, match="another source"):
            next(original)
        taken.extend(_names(b) for b in copied)
        assert taken == reference

    def test_every_served_pair_holds_one_provenance_mapping_per_row(self) -> None:
        source = MemorySource(
            MemorySourceConfig(), [{"x": np.float32(i), "name": f"r{i}"} for i in range(13)]
        )
        pipe = _pipeline(source, batch_size=4, num_epochs=2)
        pairs = list(pipe.raw_batches(with_provenance=True))
        assert pairs
        for batch, provenance in pairs:
            assert len(provenance) == batch.batch_size
            assert [p["name"] for p in provenance] == [f"r{n}" for n in _names(batch)]


class _ReadFailed(Exception):
    """A read that fails once, as a flaky file system or decode does."""


class TestAReadError:
    """A read error reaches the consumer unchanged and ends the run at the batches it delivered.

    Iterating the pipeline again resumes at the batch whose read failed: nothing is lost and
    nothing served twice, and the cursor never counts the failed batch as served.
    """

    @pytest.mark.parametrize("kind", ["indexed", "arrival stream"])
    def test_iterating_again_resumes_at_the_failed_batch(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def build() -> Pipeline:
            if kind == "indexed":
                return _pipeline(_memory(40), batch_size=4, shuffle=False, num_epochs=1)
            stream = RecordStream(_columns(40), kind=RecordIdentity.ARRIVAL, chunk=4)
            return _pipeline(stream, batch_size=4, shuffle=False, num_epochs=1)

        whole = [_names(b) for b in build().raw_batches()]
        owner: Any = MemorySource if kind == "indexed" else RecordStream
        name = "get_batch" if kind == "indexed" else "read_from"
        original = getattr(owner, name)
        reads: list[int] = []

        def flaky(self: Any, *args: Any, **kwargs: Any) -> Any:
            reads.append(1)
            if len(reads) == 4:
                raise _ReadFailed("read 4 failed")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(owner, name, flaky)
        pipe = build()
        batches = pipe.raw_batches()
        served: list[list[int]] = []
        with pytest.raises(_ReadFailed, match="read 4 failed"):
            for batch in batches:
                served.append(_names(batch))
        assert served and served == whole[: len(served)]
        state = pipe.get_state()
        delivered = 4 * len(served)
        if kind == "indexed":
            assert state["position"] == delivered
        else:
            assert (state["stream"]["records"], state["stream"]["arrived"]) == (delivered,) * 2
        assert pipe.host_stage.iterator is None
        with pytest.raises(RuntimeError, match="_ReadFailed"):
            next(batches)
        served += [_names(b) for b in pipe.raw_batches()]
        assert served == whole


class TestEndToEnd:
    """``raw_batches()`` with ``pipe.dag`` inside the caller's differentiated step."""

    def _scale_pipeline(self) -> tuple[Pipeline, Any]:
        class Scale(nnx.Module):
            def __init__(self) -> None:
                self.weight = nnx.Param(jnp.asarray(0.5))

            def __call__(self, batch: Batch) -> Batch:
                image = batch["image"].astype(jnp.float32) / 255.0 * self.weight[...]
                return batch.replace(data={**batch.data, "image": image})

        stage = Scale()
        return _pipeline(_memory(), stages=[stage], num_epochs=None, drop_last=True), stage

    @pytest.mark.parametrize("graph", [True, False])
    def test_operator_parameters_get_gradients_and_steps_compile_once(self, graph: bool) -> None:
        pipe, stage = self._scale_pipeline()
        dag = pipe.dag

        def loss(dag: nnx.Module, batch: Batch) -> jax.Array:
            return jnp.mean(dag(batch)["image"] ** 2)

        @nnx.jit(graph=graph)
        def step(dag: nnx.Module, batch: Batch) -> tuple[jax.Array, Any]:
            return nnx.value_and_grad(loss, graph=graph)(dag, batch)

        batches = iter(pipe.raw_batches())
        batch = next(batches)
        value, grads = step(dag, batch)
        image = np.asarray(batch["image"], np.float64) / 255.0
        expected = 2 * 0.5 * np.mean(image**2)  # d/dw mean((x w)^2) = 2 w mean(x^2)
        weight_grad = jax.tree.leaves(grads)[0]
        np.testing.assert_allclose(float(weight_grad), expected, rtol=1e-5, atol=1e-7)
        assert float(value) == pytest.approx(0.25 * np.mean(image**2), rel=1e-5)
        with expect_compiles(0):
            for _ in range(4):
                step(dag, next(batches))
        del stage

    def test_a_scan_over_a_chunk_equals_its_steps(self) -> None:
        pipe = _pipeline(_memory(48), shuffle=True, num_epochs=1)
        chunk = next(iter(pipe.raw_batches(chunk=3)))
        singles = list(_pipeline(_memory(48), shuffle=True, num_epochs=1).raw_batches())[:3]

        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
            value = jnp.sum(batch["image"].astype(jnp.float32))
            return total + value, value

        _, scanned = jax.lax.scan(body, jnp.zeros(()), chunk)
        stepped = [float(jnp.sum(b["image"].astype(jnp.float32))) for b in singles]
        np.testing.assert_array_equal(np.asarray(scanned), np.asarray(stepped))

    def test_an_optimizer_step_over_raw_batches_trains_the_operator(self) -> None:
        pipe, stage = self._scale_pipeline()
        optimizer = nnx.Optimizer(pipe.dag, optax.sgd(0.1), wrt=nnx.Param)

        @nnx.jit
        def step(dag: nnx.Module, optimizer: nnx.Optimizer, batch: Batch) -> jax.Array:
            def loss(dag: nnx.Module) -> jax.Array:
                return jnp.mean(dag(batch)["image"] ** 2)

            value, grads = nnx.value_and_grad(loss)(dag)
            optimizer.update(dag, grads)
            return value

        before = float(stage.weight[...])
        for batch, _ in zip(pipe.raw_batches(), range(3), strict=False):
            step(pipe.dag, optimizer, batch)
        assert float(stage.weight[...]) < before


class TestLifetime:
    """One Grain iterator per run, continued across calls, and no thread left behind."""

    def test_one_iterator_serves_a_run_across_two_loops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datarax.pipeline import host_stage  # noqa: PLC0415

        reference = [_names(b) for b in _pipeline(_memory(), num_epochs=1).raw_batches()]
        built: list[int] = []
        original = host_stage.HostStage._open

        def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
            built.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(host_stage.HostStage, "_open", counted)
        pipe = _pipeline(_memory(), num_epochs=1)
        first = []
        for batch in pipe.raw_batches():
            first.append(_names(batch))
            if len(first) == 3:
                break
        rest = [_names(batch) for batch in pipe.raw_batches()]
        assert first + rest == reference
        assert built == [1]

    @pytest.mark.parametrize("ending", ["exhausted", "close", "break"])
    def test_no_grain_thread_is_left(self, ending: str) -> None:
        pipe = _pipeline(_memory(), num_epochs=1)
        batches = pipe.raw_batches()
        if ending == "exhausted":
            list(batches)
        else:
            next(iter(batches))
            if ending == "close":
                pipe.close()
            else:
                del batches, pipe
                gc.collect()
        threads = []
        for _ in range(100):
            threads = _grain_threads()
            if not threads:
                break
            threading.Event().wait(0.05)
        assert threads == []

    @pytest.mark.parametrize("cached_by", ["step", "a user's nnx.jit"])
    def test_a_dropped_pipeline_closes_its_run_though_a_cache_holds_its_graph(
        self, cached_by: str
    ) -> None:
        """A compiled-step cache keyed by the pipeline's graph keeps its host stage, not its run."""
        pipe = _pipeline(_memory(), num_epochs=None)
        next(iter(pipe.raw_batches()))
        if cached_by == "step":
            pipe.step()
        else:
            _user_step(pipe)
        stage = pipe.host_stage
        assert stage.iterator is not None
        del pipe
        gc.collect()
        assert stage.iterator is None

    def test_an_iterator_keeps_its_run_when_its_pipeline_is_dropped(self) -> None:
        reference = [_names(b) for b in _pipeline(_memory(), num_epochs=1).raw_batches()]
        batches = iter(_pipeline(_memory(), num_epochs=1).raw_batches())
        first = _names(next(batches))
        gc.collect()
        assert [first, *(_names(b) for b in batches)] == reference

    def test_no_exception_is_ignored_at_exit(self) -> None:
        code = (
            "import numpy as np\n"
            "from flax import nnx\n"
            "from datarax.pipeline import Pipeline\n"
            "from datarax.sources.memory_source import MemorySource, MemorySourceConfig\n"
            "source = MemorySource(MemorySourceConfig(), {'x': np.arange(40.0)})\n"
            "pipe = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0),"
            " shuffle=True, num_epochs=None)\n"
            "batches = pipe.raw_batches()\n"
            "next(iter(batches))\n"
        )
        result = subprocess.run(  # noqa: S603 - this interpreter, no shell
            [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert "Exception ignored" not in result.stderr


class TestReadsPickle:
    """The read a worker runs pickles with what it holds, and a copy reads the same batches."""

    @pytest.mark.parametrize("kind", ["memory", "mix", "disk"])
    def test_a_copy_of_the_read_serves_the_same_batches(self, kind: str, tmp_path: Path) -> None:
        source = {"memory": _memory, "mix": _mix, "disk": lambda: _disk(tmp_path)}[kind]()
        pipe = _pipeline(source)
        read = pipe.host_stage.read_for_workers(pipe)
        copy = cloudpickle.loads(cloudpickle.dumps(read))
        for unit in range(5):
            original, copied = read(unit), copy(unit)
            for a, b in zip(
                jax.tree.leaves(original.batch), jax.tree.leaves(copied.batch), strict=True
            ):
                np.testing.assert_array_equal(a, b)
        size = len(cloudpickle.dumps(read))
        bound = {"memory": 64 * 1024, "mix": 64 * 1024, "disk": 32 * 1024}[kind]
        assert size < bound, size
        assert read.key is None or read.key.dtype == np.uint32  # the key travels as words

    def test_a_disk_source_pickles_as_its_path(self, tmp_path: Path) -> None:
        source = _disk(tmp_path, length=100_000)
        assert len(cloudpickle.dumps(source)) < 32 * 1024

    def test_eight_threads_on_a_fresh_disk_source_read_what_one_reads(self, tmp_path: Path) -> None:
        reference = [_names(b) for b in _pipeline(_disk(tmp_path)).raw_batches()]
        source = cloudpickle.loads(cloudpickle.dumps(_disk(tmp_path)))  # opens lazily
        pipe = _pipeline(source)
        pipe.host_stage.read_threads = 8
        served = list(pipe.raw_batches())
        assert [_names(b) for b in served] == reference


def test_the_first_raw_batch_compiles_only_the_host_naming() -> None:
    pipe = _pipeline(_memory())
    with expect_first_call_compiles("jit(_names)"):
        batches = iter(pipe.raw_batches())
        next(batches)
    pipe.close()


@pytest.mark.accelerator(kind="gpu")
def test_on_a_gpu_only_the_placed_batches_reach_it() -> None:
    """The order runs on the CPU device: iterating makes no GPU array but the batches."""
    gpu = jax.devices()[0]
    assert gpu.platform == "gpu"
    pipe = _pipeline(_memory(), num_epochs=2)  # its own state (key base, counters) is made here
    # jax's first host read of a device array leaves an array beside it (jax 0.11.1, measured);
    # the run reads its key base once, so that read happens here, before the snapshot.
    key_words(pipe._epoch_key_base.get_value())  # noqa: SLF001
    gc.collect()  # other tests' runs close here, so no other thread places batches meanwhile
    before = jax.live_arrays()
    known = {id(a) for a in before}  # ``before`` holds them, so no id is reused
    held = list(pipe.raw_batches())
    placed = {id(leaf) for b in held for leaf in jax.tree.leaves(b)}
    made_on_gpu = [a for a in jax.live_arrays() if id(a) not in known and gpu in a.devices()]
    assert made_on_gpu
    assert [a.shape for a in made_on_gpu if id(a) not in placed] == []
