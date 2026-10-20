"""``Pipeline(packing=...)``: planned rows of packed records are the indexed unit a run serves.

A source holding ragged fields is planned once into rows of a static length (first fit, a window
of the planning order at a time), and the rows are what the pipeline orders, batches and resumes:
a batch holds ``batch_size`` rows, each packed field as its values, segment ids from 1 (0 in
unused cells), positions restarting in every record and the records' identities in slot order.
Every record is served once per epoch, the rows in the cipher's order, a new order each epoch;
``len``, ``batches_left``, the final-batch rule and resume count rows. A row is named
``(2**31, row)``, no record's index, and a row's provenance is its slots' mappings. The plan is
computed once: in a decoded copy it is kept in the copy, for an in-memory source in a directory
that lives as long as the pipeline.
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.data_source import NO_PROVENANCE
from datarax.core.element_batch import PADDING_INDEX
from datarax.core.host_resources import available_cpus
from datarax.core.index_words import from_words, to_words
from datarax.pipeline import materialize, Packing, Pipeline
from datarax.pipeline.packed_rows import PackedRows
from datarax.sources import MemorySource, MemorySourceConfig
from tests.test_common.ragged_sources import IdentifiedRaggedSource
from tests.test_common.worker_reads import png_source, resources, write_png_records


_RECORDS = 53
_ROW = 16
_SLOTS = 4
_WINDOW = 16
_TOP = 2**31


def _records(count: int = _RECORDS, *, seed: int = 0) -> list[dict[str, Any]]:
    """Records whose tokens spell their identity: record ``i`` holds ``100 i, 100 i + 1, ...``."""
    lengths = np.random.default_rng(seed).integers(1, 13, count)
    return [
        {
            "tokens": (100 * index + np.arange(length)).astype(np.int32),
            "label": np.int32(index),
            "name": f"record-{index}",
        }
        for index, length in enumerate(lengths)
    ]


_PACKING = Packing(lengths={"tokens": _ROW, "label": _SLOTS}, max_segments=_SLOTS, window=_WINDOW)


def _pipeline(  # noqa: PLR0913 - the options the tests vary, each with the one they keep
    *,
    batch_size: int = 3,
    num_epochs: int | None = 1,
    packing: Packing = _PACKING,
    stages: list[nnx.Module] | None = None,
    **options: Any,
) -> Pipeline[Packing]:
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), _records()),
        stages=[] if stages is None else stages,
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
        shuffle=True,
        num_epochs=num_epochs,
        packing=packing,
        **options,
    )


def _rows(batch: Any) -> list[int]:
    """The rows a batch holds, from their ``(2**31, row)`` names."""
    words = np.asarray(jax.device_get(batch.indices))
    assert (words[:, 0] == _TOP).all(), "a row is named (2**31, row)"
    return words[:, 1].astype(np.int64).tolist()


def _segments(batch: Any) -> list[tuple[int, list[int], list[int]]]:
    """Every record a batch holds: its identity, its tokens and its label, read off the rows."""
    data = jax.device_get(batch.data)
    found = []
    for row in range(len(data["tokens"])):
        for slot, words in enumerate(data["tokens_segment_indices"][row]):
            if (words == PADDING_INDEX).all():
                continue
            tokens = data["tokens"][row][data["tokens_segment_ids"][row] == slot + 1]
            labels = data["label"][row][data["label_segment_ids"][row] == slot + 1]
            found.append((int(from_words(words[None])[0]), tokens.tolist(), labels.tolist()))
    return found


class TestABatchIsRowsOfPackedRecords:
    def test_each_packed_field_is_a_row_of_static_length_with_its_segments(self) -> None:
        batch = next(iter(_pipeline().raw_batches()))
        data = batch.data

        for field, length in (("tokens", _ROW), ("label", _SLOTS)):
            assert data[field].shape == (3, length)
            assert data[f"{field}_segment_ids"].shape == (3, length)
            assert data[f"{field}_segment_ids"].dtype == jnp.int32
            assert data[f"{field}_positions"].dtype == jnp.int32
            assert data[f"{field}_segment_indices"].shape == (3, _SLOTS, 2)
            assert data[f"{field}_segment_indices"].dtype == jnp.uint32

    def test_every_record_is_served_once_an_epoch_as_its_own_segment(self) -> None:
        records = _records()
        served = [segment for batch in _pipeline().raw_batches() for segment in _segments(batch)]

        assert sorted(record for record, _, _ in served) == list(range(_RECORDS))
        for record, tokens, labels in served:
            assert tokens == records[record]["tokens"].tolist()
            assert labels == [record]

    def test_positions_restart_in_every_record(self) -> None:
        data = jax.device_get(next(iter(_pipeline().raw_batches())).data)
        for ids, positions in zip(
            data["tokens_segment_ids"], data["tokens_positions"], strict=True
        ):
            for segment in set(ids.tolist()) - {0}:
                assert positions[ids == segment].tolist() == list(range((ids == segment).sum()))

    def test_rows_come_in_the_cipher_s_order_a_new_one_each_epoch(self) -> None:
        packed = _pipeline(num_epochs=2)
        count = packed.epoch_plan.length
        assert count is not None
        rows = [row for batch in packed.raw_batches() for row in _rows(batch)]
        reference = Pipeline.from_arrays(
            {"row": np.arange(count)}, batch_size=3, seed=0, shuffle=True, num_epochs=2
        )
        expected = [int(r) for batch in reference.raw_batches() for r in batch["row"]]

        assert rows == expected
        assert sorted(rows[:count]) == sorted(rows[count:]) == list(range(count))
        assert rows[:count] != rows[count:]

    def test_the_rows_order_and_composition_are_the_same_in_every_pipeline_built_alike(
        self,
    ) -> None:
        first = [_segments(batch) for batch in _pipeline().raw_batches()]
        again = [_segments(batch) for batch in _pipeline().raw_batches()]
        assert first == again


class TestTheBatchRuleCountsRows:
    @staticmethod
    def _batch_size_leaving_a_short_batch() -> int:
        rows = _pipeline().epoch_plan.length
        assert rows is not None
        return next(size for size in (3, 4, 5, 7) if rows % size)

    def test_a_run_compiles_a_step_twice_the_full_batches_and_the_final_short_one(self) -> None:
        size = self._batch_size_leaving_a_short_batch()
        pipe = _pipeline(batch_size=size)

        @jax.jit
        def step(tokens: jax.Array) -> jax.Array:
            return tokens.sum()

        jax.clear_caches()
        batches = list(pipe.raw_batches())
        with compiled_programs() as programs:
            for batch in batches:
                step(batch["tokens"]).block_until_ready()

        assert programs.count("jit(step)") == 2, programs
        assert {b["tokens"].shape[0] for b in batches} == {size, len(_rows(batches[-1]))}

    def test_drop_last_serves_full_batches_of_rows_only(self) -> None:
        size = self._batch_size_leaving_a_short_batch()
        pipe = _pipeline(batch_size=size, drop_last=True)
        rows = pipe.epoch_plan.length
        assert rows is not None

        batches = list(pipe.raw_batches())

        assert len(batches) == rows // size == len(pipe)
        assert {b["tokens"].shape[0] for b in batches} == {size}

    def test_len_and_batches_left_count_rows(self) -> None:
        pipe = _pipeline(num_epochs=2)
        rows = pipe.epoch_plan.length
        assert rows is not None
        assert len(pipe) == math.ceil(2 * rows / 3)

        served = pipe.raw_batches()
        next(served)
        next(served)

        assert pipe.batches_left() == len(pipe) - 2


class TestResume:
    def test_every_row_position_resumes_into_the_same_following_batches(self) -> None:
        uninterrupted = [_segments(batch) for batch in _pipeline(num_epochs=2).raw_batches()]

        for taken in range(len(uninterrupted)):
            first = _pipeline(num_epochs=2)
            served = first.raw_batches()
            for _ in range(taken):
                next(served)
            state = first.get_state()
            resumed = _pipeline(num_epochs=2)
            resumed.set_state(state)

            assert [_segments(b) for b in resumed.raw_batches()] == uninterrupted[taken:], taken

    def test_the_state_is_version_4_and_fingerprints_the_packing_and_the_lengths(self) -> None:
        state = _pipeline().get_state()

        assert state["version"] == 4
        fingerprint = state["fingerprint"]
        assert fingerprint["packing"] == {
            "lengths": {"label": _SLOTS, "tokens": _ROW},
            "max_segments": _SLOTS,
            "window": _WINDOW,
        }
        assert len(fingerprint["packing_lengths"]) == 64  # a SHA-256 of the lengths planned

    def test_a_state_of_another_packing_is_refused_naming_it(self) -> None:
        state = _pipeline().get_state()
        other = Packing(
            lengths={"tokens": 24, "label": _SLOTS}, max_segments=_SLOTS, window=_WINDOW
        )

        with pytest.raises(ValueError, match="packing"):
            _pipeline(packing=other).set_state(state)

    def test_a_state_of_version_3_is_refused_naming_both(self) -> None:
        state = dict(_pipeline().get_state(), version=3)

        with pytest.raises(ValueError, match=r"version 3.*version 4"):
            _pipeline().set_state(state)


class TestARowIsNoRecord:
    def test_a_row_is_named_by_its_ordinal_with_the_top_bit_set(self) -> None:
        pipe = _pipeline()
        rows = [row for batch in pipe.raw_batches() for row in _rows(batch)]
        assert sorted(rows) == list(range(pipe.epoch_plan.length or 0))

    def test_a_row_s_name_reads_no_record_s_provenance(self) -> None:
        pipe = _pipeline()
        batch = next(iter(pipe.raw_batches()))

        with pytest.raises(IndexError):
            pipe.source.provenance(np.asarray(jax.device_get(batch.indices)))

    def test_a_row_names_no_record_so_its_provenance_by_index_is_refused(self) -> None:
        pipe = _pipeline()
        batch = next(iter(pipe.raw_batches()))
        rows = pipe.read_source

        with pytest.raises(TypeError, match="row_provenance"):
            rows.provenance(np.asarray(jax.device_get(batch.indices)))

    def test_row_provenance_serves_each_row_s_slots_by_its_ordinal(self) -> None:
        pipe = Pipeline(
            source=MemorySource(MemorySourceConfig(), _records()),
            stages=[],
            batch_size=3,
            rngs=nnx.Rngs(0),
            packing=_PACKING,
        )
        served = tuple(
            slots
            for _, provenance in pipe.raw_batches(with_provenance=True)
            for slots in provenance
        )
        rows = pipe.read_source
        assert isinstance(rows, PackedRows)

        ordinals = to_words(np.arange(len(rows), dtype=np.uint64))
        assert rows.row_provenance(ordinals) == served

    def test_a_row_s_provenance_is_one_mapping_per_slot(self) -> None:
        records = _records()
        for batch, provenance in _pipeline().raw_batches(with_provenance=True):
            assert len(provenance) == batch["tokens"].shape[0]
            data = jax.device_get(batch.data)
            for row, slots in enumerate(provenance):
                assert len(slots) == _SLOTS
                for words, record in zip(data["tokens_segment_indices"][row], slots, strict=True):
                    if (words == PADDING_INDEX).all():
                        assert record == NO_PROVENANCE
                    else:
                        index = int(from_words(words[None])[0])
                        assert record == {"name": records[index]["name"]}


class TestThePlanIsComputedOnce:
    def test_close_keeps_an_in_memory_source_s_plan_so_iteration_resumes(self) -> None:
        whole = [_segments(batch) for batch in _pipeline().raw_batches()]
        pipe = _pipeline()
        first = _segments(next(iter(pipe.raw_batches())))

        pipe.close()

        assert pipe.pack_plan.directory.is_dir()
        assert [first, *(_segments(batch) for batch in pipe.raw_batches())] == whole

    def test_an_in_memory_source_s_plan_lives_as_long_as_its_pipeline(self) -> None:
        pipe = _pipeline()
        list(pipe.raw_batches())
        directory = pipe.pack_plan.directory
        assert directory.is_dir()

        del pipe
        gc.collect()

        assert not directory.exists()

    def test_a_decoded_copy_keeps_its_plan_for_every_pipeline_built_alike(
        self, tmp_path: Path
    ) -> None:
        """A second pipeline built alike reads the kept plan: its files are the first one's."""
        copy = materialize(
            IdentifiedRaggedSource(_records()), directory=tmp_path / "copy", unit_records=8
        )

        def over(packing: Packing) -> Pipeline[Packing]:
            return Pipeline(
                source=copy,
                stages=[],
                batch_size=3,
                rngs=nnx.Rngs(0),
                shuffle=True,
                packing=packing,
            )

        first = over(_PACKING)
        kept = first.pack_plan.directory / "plan.json"
        # Held open, so a plan written again would be a new inode, never this one's number.
        with kept.open("rb") as held:
            planned = (os.fstat(held.fileno()).st_ino, kept.stat().st_mtime_ns)
            again = over(_PACKING)
            assert (kept.stat().st_ino, kept.stat().st_mtime_ns) == planned
        other = over(
            Packing(lengths={"tokens": 24, "label": _SLOTS}, max_segments=_SLOTS, window=8)
        )

        assert again.pack_plan.directory == first.pack_plan.directory
        assert first.pack_plan.directory.parent == tmp_path / "copy"
        assert other.pack_plan.directory != first.pack_plan.directory
        assert len(list((tmp_path / "copy").glob("plan-*"))) == 2
        assert [_segments(b) for b in first.raw_batches()] == [
            _segments(b) for b in _pipeline().raw_batches()
        ]


class TestWorkerProcesses:
    @pytest.mark.skipif(available_cpus() < 2, reason="fewer than two CPUs: no worker processes")
    def test_workers_serve_the_rows_the_calling_thread_serves(self) -> None:
        """A GIL-bound read goes to worker processes, which reopen the plan by its directory."""

        def over(host_resources: Any) -> Pipeline[Packing]:
            return Pipeline(
                source=IdentifiedRaggedSource(_records()),
                stages=[],
                batch_size=3,
                rngs=nnx.Rngs(0),
                shuffle=True,
                packing=_PACKING,
                host_resources=host_resources,
            )

        alone = [_segments(batch) for batch in over(None).raw_batches()]
        workers = over(resources(2))

        assert workers.host_plan.workers == 2
        assert [_segments(batch) for batch in workers.raw_batches()] == alone

    def test_the_plan_pickles_by_its_directory_not_its_rows(self) -> None:
        import pickle  # noqa: PLC0415 - one test uses it

        pipe = Pipeline(
            source=MemorySource(MemorySourceConfig(), _records(5000)),
            stages=[],
            batch_size=3,
            rngs=nnx.Rngs(0),
            packing=_PACKING,
        )
        plan = pipe.pack_plan
        payload = pickle.dumps(plan)

        assert plan.records.nbytes + plan.row_starts.nbytes > 40_000  # the premise: a real plan
        assert len(payload) < 1024
        again = pickle.loads(payload)  # noqa: S301 - the test's own payload
        np.testing.assert_array_equal(again.records, plan.records)
        np.testing.assert_array_equal(again.row_starts, plan.row_starts)


class TestRefusals:
    def test_a_field_the_packing_does_not_name_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError, match=r"'label'"):
            _pipeline(
                packing=Packing(lengths={"tokens": _ROW}, max_segments=_SLOTS, window=_WINDOW)
            )

    def test_the_window_and_the_segment_cap_have_no_default(self) -> None:
        with pytest.raises(TypeError):
            Packing(lengths={"tokens": _ROW})  # type: ignore[call-arg]

    def test_a_source_without_the_ragged_read_is_refused(self, tmp_path: Path) -> None:
        source = png_source(write_png_records(tmp_path, 8, shards=1))

        with pytest.raises(NotImplementedError, match="does not read ragged records"):
            Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0), packing=_PACKING)

    @pytest.mark.parametrize("call", ["step", "scan", "session"])
    def test_the_compiled_step_is_refused_naming_iteration(self, call: str) -> None:
        pipe = _pipeline()
        calls = {
            "step": pipe.step,
            "scan": lambda: pipe.scan(lambda batch: batch, length=1),
            "session": pipe.session,
        }

        with pytest.raises(ValueError, match="iteration"):
            calls[call]()

    def test_stages_are_refused_naming_why(self) -> None:
        with pytest.raises(ValueError, match="packed"):
            _pipeline(stages=[nnx.Linear(1, 1, rngs=nnx.Rngs(0))])

    def test_a_block_order_is_refused_naming_why(self) -> None:
        from datarax.core.block_order import BlockOrder  # noqa: PLC0415 - one test uses it

        with pytest.raises(ValueError, match="block_order"):
            _pipeline(block_order=BlockOrder(block_size=4, blocks_per_buffer=2))


class TestUnderTransforms:
    """A packed batch is a static-shape batch: a train step over it compiles once per shape."""

    @pytest.mark.parametrize("graph", [True, False])
    def test_a_train_step_over_packed_rows_compiles_once_and_its_gradient_is_finite(
        self, graph: bool
    ) -> None:
        model = nnx.Embed(100 * _RECORDS, 4, rngs=nnx.Rngs(1))
        optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)

        @nnx.jit(graph=graph)
        def train(model: nnx.Embed, optimizer: nnx.Optimizer, tokens: Any, ids: Any) -> Any:
            def loss(module: nnx.Embed) -> jax.Array:
                cells = (ids > 0).astype(jnp.float32)
                return (module(tokens).sum(-1) ** 2 * cells).sum() / cells.sum()

            value, grads = nnx.value_and_grad(loss)(model)
            optimizer.update(model, grads)
            return value

        batches = list(_pipeline(drop_last=True).raw_batches())[:3]
        with compiled_programs() as programs:
            values = [
                train(model, optimizer, b["tokens"], b["tokens_segment_ids"]) for b in batches
            ]

        assert all(bool(jnp.isfinite(v)) for v in values)
        assert programs == ["jit(train)"]  # one compile for every batch of one shape
