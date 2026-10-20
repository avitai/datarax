"""A packed pipeline's host memory is O(batch), never O(dataset): the plan is a memory-mapped file.

The plan of ``N`` records in ``R`` rows is two files, ``8 N + 8 (R + 1)`` bytes, read through
memory maps: planning holds a window (``tests/core/test_pack_plan.py``, the planner's own test),
and iterating holds the batches in flight. The allocations iteration makes are counted with
``tracemalloc``, which sees every NumPy allocation whether or not the allocator reuses memory
already resident, and none of the plan's file pages: a process's resident peak (``VmHWM``) misses
both ways here, since a 1.6 MiB array filled in a process that freed as much before raises it by
nothing, while the plan's pages a read touches raise it. The peak allocated while iterating ten
times the records is within one packed batch of the peak at ``N``, and a table of every record
held while iterating (the positive control) exceeds that bound.
"""

from __future__ import annotations

import collections
import gc
import tracemalloc

import numpy as np
import pytest
from flax import nnx

from datarax.pipeline import Packing, Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


_RECORDS = 20_000
_STEPS = 64
_PACKING = Packing(lengths={"tokens": 256, "label": 8}, max_segments=8, window=1024)


def _pipeline(records: int) -> Pipeline[Packing]:
    lengths = np.random.default_rng(0).integers(1, 65, records)
    source = MemorySource(
        MemorySourceConfig(),
        [
            {"tokens": np.full(length, index, np.int32), "label": np.int32(index)}
            for index, length in enumerate(lengths)
        ],
    )
    return Pipeline(
        source=source,
        stages=[],
        batch_size=16,
        rngs=nnx.Rngs(0),
        shuffle=True,
        num_epochs=None,
        packing=_PACKING,
    )


def _peak_while_iterating(records: int, *, table: bool = False) -> tuple[int, int]:
    """The peak bytes allocated while iterating, after the run started, and one batch's bytes."""
    pipe = _pipeline(records)
    batches = iter(pipe.raw_batches())
    held = collections.deque(maxlen=1)
    for _ in range(4):  # the run starts, its read threads fill and its naming compiles
        held.append(next(batches))
    unit = sum(np.asarray(leaf).nbytes for leaf in held[-1].data.values())
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        start = tracemalloc.get_traced_memory()[0]
        kept = np.arange(records, dtype=np.int64) if table else None  # 8 bytes a record
        for _ in range(_STEPS):
            held.append(next(batches))
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()
        pipe.close()
    del kept
    return peak, unit


def test_iteration_over_ten_times_the_records_allocates_no_more() -> None:
    small, unit = _peak_while_iterating(_RECORDS)
    large, _ = _peak_while_iterating(10 * _RECORDS)

    assert large <= small + unit, (small, large, unit)


def test_positive_control_a_table_of_every_record_exceeds_the_bound() -> None:
    small, unit = _peak_while_iterating(_RECORDS)
    leaking, _ = _peak_while_iterating(10 * _RECORDS, table=True)

    assert leaking > small + unit, (small, leaking, unit)


@pytest.mark.parametrize("records", [1_000, 10_000])
def test_the_plan_file_is_8_bytes_a_record_and_8_a_row_boundary(records: int) -> None:
    lengths = np.random.default_rng(1).integers(1, 33, records)
    source = MemorySource(
        MemorySourceConfig(),
        [{"tokens": np.zeros(length, np.int32)} for length in lengths],
    )
    pipe = Pipeline(
        source=source,
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
        packing=Packing(lengths={"tokens": 64}, max_segments=8, window=512),
    )
    plan = pipe.pack_plan

    assert pipe.epoch_plan.length == plan.rows
    assert (plan.directory / "records.bin").stat().st_size == 8 * records
    assert (plan.directory / "row_starts.bin").stat().st_size == 8 * (plan.rows + 1)
    pipe.close()
