"""What each process of a multi-process test does, and the report it sends.

A test runs ``PROGRAM`` as every process of one ``jax.distributed`` run
(:func:`tests.test_common.processes.run_processes`); the program joins the run, then calls the
scenario named by its third argument with the process index and any further arguments, and sends
the returned report with ``substrax.testing.report``. Pipelines come from
:mod:`tests.test_common.multi_process_pipelines`, so the single-process reference is built alike.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.checkpoint import IteratorCheckpoint
from datarax.core.element_batch import Batch
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from tests.test_common import multi_process_pipelines as built
from tests.test_common.streams import RecordStream
from tests.test_common.worker_reads import ProcessReadMemorySource, resources


PROGRAM = """
import sys
import jax
from substrax.testing import report

port, process = int(sys.argv[1]), int(sys.argv[2])
jax.distributed.initialize(f"localhost:{port}", num_processes=2, process_id=process)

from tests.test_common.multi_process_scenarios import SCENARIOS

report(SCENARIOS[sys.argv[3]](process, sys.argv[4:]))
jax.distributed.shutdown()
"""


def _placed_mix(batch: Batch) -> tuple[jax.Array, jax.Array]:
    """A placed batch's partners and coefficients: the draw stages end in a mixer."""
    mix = batch.mix
    assert mix is not None
    assert isinstance(mix.partners, jax.Array) and isinstance(mix.lams, jax.Array)
    return mix.partners, mix.lams


def _units(pipe: Pipeline) -> list[dict[str, Any]]:
    """Each unit's global shape, its sharding and this process's rows of its records."""
    return [
        {
            "shape": list(unit["value"].shape),
            "spec": str(unit["value"].sharding.spec),
            "rows": built.record_shards(unit.indices),
        }
        for unit in pipe.raw_batches()
    ]


def placement(process: int, args: Sequence[str]) -> dict[str, Any]:
    """Every unit of one epoch, unprocessed, as this process holds it."""
    del process, args
    return {"units": _units(built.pipeline(sharding=built.rows_on_every_device()))}


def draws(process: int, args: Sequence[str]) -> dict[str, Any]:
    """Every processed batch's draws as this process holds them, then the streams' counts."""
    del process, args
    pipe = built.pipeline(stages=built.draw_stages(), sharding=built.rows_on_every_device())
    batches = [
        {
            "rows": built.record_shards(batch.indices),
            "value": built.shards(batch["value"]),
            "image": built.shards(batch["image"]),
            "partner": built.shards(_placed_mix(batch)[0]),
            # The batch's one coefficient, written to every row: a process reads its rows' value.
            "lam": float(np.asarray(_placed_mix(batch)[1].addressable_shards[0].data)[0]),
        }
        for batch in pipe
    ]
    return {"batches": batches, "rng_counts": built.rng_counts(pipe)}


def chunks(process: int, args: Sequence[str]) -> dict[str, Any]:
    """Every unit of chunks of two batches, with its provenance, as this process holds it."""
    del process, args
    pipe = built.pipeline(sharding=built.rows_on_every_device())
    return {
        "units": [
            {
                "shape": list(unit["value"].shape),
                "spec": str(unit["value"].sharding.spec),
                "rows": built.record_shards(unit.indices),
                "names": [record["name"] for record in provenance],
            }
            for unit, provenance in pipe.raw_batches(chunk=2, with_provenance=True)
        ]
    }


def compiles(process: int, args: Sequence[str]) -> dict[str, Any]:
    """The programs one epoch of the draws pipeline compiles from cleared caches, then a second."""
    del process, args
    jax.clear_caches()
    with compiled_programs() as first:
        list(built.pipeline(stages=built.draw_stages(), sharding=built.rows_on_every_device()))
    with compiled_programs() as again:
        list(built.pipeline(stages=built.draw_stages(), sharding=built.rows_on_every_device()))
    return {"first": list(first), "again": list(again)}


def checkpoint(process: int, args: Sequence[str]) -> dict[str, Any]:
    """Two units read, the state saved, a fresh pipeline restored: its next unit's rows."""
    del process
    directory = args[0]
    pipe = built.pipeline(sharding=built.rows_on_every_device())
    units = pipe.raw_batches()
    next(units)
    next(units)
    with IteratorCheckpoint(directory) as store:
        store.save(pipe, 2)
    fresh = built.pipeline(sharding=built.rows_on_every_device())
    with IteratorCheckpoint(directory) as store:
        store.restore(fresh, step=2)
    return {"rows": built.record_shards(next(fresh.raw_batches()).indices)}


def _refusal(build: Callable[[], Any]) -> str:
    """The message of the ``ValueError`` ``build`` raises, or ``""`` when it raises none."""
    try:
        build()
    except ValueError as error:
        return str(error)
    return ""


def refusals(process: int, args: Sequence[str]) -> dict[str, Any]:
    """A stream source and a worker's shard of a source, each under a mesh spanning processes."""
    del args
    sharding = built.rows_on_every_device()

    def over(source: Any) -> Pipeline:
        return Pipeline(
            source=source,
            stages=[],
            batch_size=built.BATCH_SIZE,
            rngs=nnx.Rngs(0),
            sharding=sharding,
        )

    shard = MemorySourceConfig(shard_id=process, num_workers=2)
    records = built.records()
    stream = RecordStream(
        {"value": records["value"], "image": records["image"]}, texts=records["name"]
    )
    return {
        "stream": _refusal(lambda: over(stream)),
        "worker_shard": _refusal(lambda: over(MemorySource(shard, built.records()))),
    }


def mismatch(process: int, args: Sequence[str]) -> dict[str, Any]:
    """Each process builds the pipeline with its own seed and asks for the first unit."""
    del args
    pipe = built.pipeline(sharding=built.rows_on_every_device(), seed=process)
    return {"error": _refusal(lambda: next(pipe.raw_batches()))}


def shm(process: int, args: Sequence[str]) -> dict[str, Any]:
    """The worker-process path under a mesh spanning processes, without and with a tiny share."""
    del process, args
    sharding = built.rows_on_every_device()

    def first(budget: Any) -> Any:
        source = ProcessReadMemorySource(MemorySourceConfig(), built.records())
        pipe = Pipeline(
            source=source,
            stages=[],
            batch_size=built.BATCH_SIZE,
            rngs=nnx.Rngs(0),
            host_resources=budget,
            sharding=sharding,
        )
        return next(pipe.raw_batches())

    return {
        "missing": _refusal(lambda: first(resources(1))),
        "tiny": _refusal(lambda: first(dataclasses.replace(resources(1), shm_budget_bytes=1))),
    }


SCENARIOS: dict[str, Callable[[int, Sequence[str]], dict[str, Any]]] = {
    "placement": placement,
    "draws": draws,
    "compiles": compiles,
    "chunks": chunks,
    "checkpoint": checkpoint,
    "refusals": refusals,
    "mismatch": mismatch,
    "shm": shm,
}


def merged_rows(per_process: Sequence[dict[str, list[Any]]]) -> list[Any]:
    """One global batch from every process's rows keyed by their first global row."""
    rows: dict[int, list[Any]] = {}
    for shards in per_process:
        for start, values in shards.items():
            rows[int(start)] = values
    return [value for start in sorted(rows) for value in rows[start]]


def host_rows(array: Any) -> list[Any]:
    """A single-process array's rows as plain values."""
    return np.asarray(jax.device_get(array)).tolist()


__all__ = ["PROGRAM", "SCENARIOS", "host_rows", "merged_rows"]
