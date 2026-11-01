"""A pipeline places its batches on a data mesh spanning processes, one process's batch exactly.

Two real processes, two CPU devices each, run one pipeline with a sharding over all four
devices. Each process reads the rows of every global batch its devices hold, so the batches the
processes assemble are the single-process batches row for row: records, values, a per-record
draw and MixUp's batch draws (lambda and the partner permutation, drawn from the first record's
key). The short final batch is read whole by every process and placed replicated, even when its
rows equal one process's share. Each process compiles the DAG once per placement, as one process
does, and a second pass compiles nothing. A chunk stacks each process's rows of its batches, and
a unit's provenance names the records this process read. A state saved on both processes
restores both.
Refused across processes: a stream source, a worker's shard of a source, a run whose
configuration differs between processes (in every process, not one), and the worker-process read
without a ``/dev/shm`` share in ``HostResources``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from datarax.core.index_words import from_words
from tests.test_common import multi_process_pipelines as built
from tests.test_common.multi_process_scenarios import host_rows, merged_rows, PROGRAM
from tests.test_common.processes import run_processes


class UnitReport(TypedDict):
    shape: list[int]
    spec: str
    rows: dict[str, list[int]]


class PlacementReport(TypedDict):
    units: list[UnitReport]


class ChunkUnitReport(TypedDict):
    shape: list[int]
    spec: str
    rows: dict[str, list[Any]]
    names: list[str]


class ChunksReport(TypedDict):
    units: list[ChunkUnitReport]


class BatchReport(TypedDict):
    rows: dict[str, list[int]]
    value: dict[str, list[float]]
    image: dict[str, list[list[float]]]
    partner: dict[str, list[int]]
    lam: float


class DrawsReport(TypedDict):
    batches: list[BatchReport]
    rng_counts: list[int]


class CompilesReport(TypedDict):
    first: list[str]
    again: list[str]


class RowsReport(TypedDict):
    rows: dict[str, list[int]]


class RefusalsReport(TypedDict):
    stream: str
    worker_shard: str


class ErrorReport(TypedDict):
    error: str


class ShmReport(TypedDict):
    missing: str
    tiny: str


def _records(indices: Any) -> list[int]:
    return [int(i) for i in from_words(np.asarray(indices))]


def _reference_units() -> list[list[int]]:
    return [_records(unit.indices) for unit in built.pipeline().raw_batches()]


class TestEveryGlobalBatchIsOneProcesssBatch:
    def test_units_hold_the_single_process_records_row_for_row(self) -> None:
        reports = run_processes(PROGRAM, PlacementReport, "placement")

        reference = _reference_units()
        first, second = (report["units"] for report in reports)
        assert [unit["shape"] for unit in first] == [[8], [8], [8], [4]]
        for k, expected in enumerate(reference):
            assert merged_rows([first[k]["rows"], second[k]["rows"]]) == expected

    def test_full_batches_are_split_and_the_short_final_batch_is_whole_on_every_process(
        self,
    ) -> None:
        first, second = (
            report["units"] for report in run_processes(PROGRAM, PlacementReport, "placement")
        )

        assert [unit["spec"] for unit in first] == ["P('data',)"] * 3 + ["P()"]
        final = _reference_units()[-1]
        for process_units in (first, second):
            assert process_units[-1]["rows"] == {"0": final}  # all 4 rows, as one process's share
            assert all(len(rows) == 2 for rows in process_units[0]["rows"].values())

    def test_draws_and_mixup_equal_one_processs_row_for_row(self) -> None:
        first, second = run_processes(PROGRAM, DrawsReport, "draws")

        reference_pipe = built.pipeline(stages=built.draw_stages())
        reference = list(reference_pipe)
        assert len(first["batches"]) == len(reference) == 4
        for k, batch in enumerate(reference):
            ours = (first["batches"][k], second["batches"][k])
            assert batch.mix is not None
            assert merged_rows([b["rows"] for b in ours]) == _records(batch.indices)
            np.testing.assert_array_equal(
                merged_rows([b["value"] for b in ours]), host_rows(batch["value"])
            )
            np.testing.assert_array_equal(
                merged_rows([b["image"] for b in ours]), host_rows(batch["image"])
            )
            assert merged_rows([b["partner"] for b in ours]) == host_rows(batch.mix.partners)
            # Every row holds the batch's one coefficient, on both processes.
            assert ours[0]["lam"] == ours[1]["lam"]
            assert set(np.asarray(batch.mix.lams).tolist()) == {ours[0]["lam"]}

    def test_the_streams_counts_written_back_equal_one_processs(self) -> None:
        first, second = run_processes(PROGRAM, DrawsReport, "draws")

        reference_pipe = built.pipeline(stages=built.draw_stages())
        for _ in reference_pipe:
            pass
        assert first["rng_counts"] == second["rng_counts"] == built.rng_counts(reference_pipe)


class TestCompilesAcrossProcesses:
    def test_each_process_compiles_the_dag_once_per_placement_and_a_second_pass_nothing(
        self,
    ) -> None:
        reports = run_processes(PROGRAM, CompilesReport, "compiles")

        first, second = reports
        assert first["first"].count("jit(step)") == 2  # the sharded full batches, the tail
        assert first["first"] == second["first"]
        assert first["again"] == second["again"] == []


class TestChunksAcrossProcesses:
    def test_each_process_stacks_its_rows_of_every_batch_and_names_their_provenance(self) -> None:
        reports = run_processes(PROGRAM, ChunksReport, "chunks")

        reference = list(built.pipeline().raw_batches(chunk=2))
        first, second = (report["units"] for report in reports)
        assert [unit["shape"] for unit in first] == [list(u["value"].shape) for u in reference]
        assert [unit["shape"] for unit in first] == [[2, 8], [8], [4]]
        assert [unit["spec"] for unit in first] == ["P(None, 'data')", "P('data',)", "P()"]
        chunk = reference[0].indices
        for k in range(2):
            batch = [{s: v[k] for s, v in units[0]["rows"].items()} for units in (first, second)]
            assert merged_rows(batch) == _records(chunk[k])
        for k in (1, 2):
            assert merged_rows([first[k]["rows"], second[k]["rows"]]) == _records(
                reference[k].indices
            )
        for units in (first, second):
            for unit in units:
                assert unit["names"] == [built.name(r) for r in _held_rows(unit["rows"])]


def _held_rows(shards: dict[str, list[Any]]) -> list[int]:
    """A process's records of one unit in the order it read them: batch by batch, row by row."""
    ordered = [shards[start] for start in sorted(shards, key=int)]
    if ordered and isinstance(ordered[0][0], list):  # a chunk: (K, rows) per shard
        return [r for k in range(len(ordered[0])) for rows in ordered for r in rows[k]]
    return [r for rows in ordered for r in rows]


class TestAStateSavedOnEveryProcess:
    def test_restores_every_process_at_the_next_global_batch(self, tmp_path: Path) -> None:
        reports = run_processes(PROGRAM, RowsReport, "checkpoint", str(tmp_path / "state"))

        third = _reference_units()[2]
        assert merged_rows([report["rows"] for report in reports]) == third


class TestRefusedAcrossProcesses:
    def test_a_stream_and_a_workers_shard_are_refused_in_every_process(self) -> None:
        reports = run_processes(PROGRAM, RefusalsReport, "refusals")

        for report in reports:
            assert "stream" in report["stream"]
            assert "INDEXED" in report["stream"]
            assert "shard_id" in report["worker_shard"]

    def test_a_configuration_differing_between_processes_is_refused_in_every_process(
        self,
    ) -> None:
        reports = run_processes(PROGRAM, ErrorReport, "mismatch")

        for report in reports:
            assert "differs between processes" in report["error"]

    def test_the_worker_processes_need_a_dev_shm_share(self) -> None:
        reports = run_processes(PROGRAM, ShmReport, "shm")

        for report in reports:
            assert "shm_budget_bytes" in report["missing"]
            assert "shm_budget_bytes=1" in report["tiny"]
