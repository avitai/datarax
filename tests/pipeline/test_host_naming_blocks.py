"""The host stage names a block of batches per call of its naming program, as each batch alone.

Naming costs a fixed overhead per call (a dispatch to the CPU device and back) that dominates a
small batch's read, so the host stage names a block of consecutive full batches of the run in one
call of the same program, vmapped over their starts, and serves each batch its slice. The names
(indices and epochs) are those of naming each batch alone, for every rule, every source kind, every
resume position, chunked units and any number of read threads. The run's short final batch is
named alone, by the program that already serves that shape.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import Batch
from datarax.core.prng import key_words
from datarax.pipeline import Pipeline, run_units
from datarax.pipeline.epochs import HostNaming, Run
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.host_plans import reading_with


pytestmark = pytest.mark.usefixtures("still_resident")
"""Plans read with chosen threads and buffers (:func:`reading_with`): ``M_main`` held still."""


def _memory(length: int) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), {"x": np.arange(length * 2, dtype=np.float32).reshape(length, 2)}
    )


def _mix(length: int) -> MixDataSourcesNode:
    first = length // 3 + 1
    return MixDataSourcesNode(
        MixDataSourcesConfig(weights=(1.0, 2.0)), [_memory(first), _memory(length - first)]
    )


def _pipeline(
    source: DataSourceModule,
    *,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    num_epochs: int | None = 3,
    threads: int = 1,
) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(7),
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
        host_resources=None
        if threads == 1
        else reading_with(source, batch_size, threads=threads, read_buffer=threads),
    )


def _named(batch: Batch) -> tuple[bytes, bytes]:
    return np.asarray(batch.indices).tobytes(), np.asarray(batch.epochs).tobytes()


def _alone(pipe: Pipeline, start: tuple[int, int] = (0, 0)) -> list[tuple[bytes, bytes]]:
    """Each batch of the pipeline's run from ``start`` named alone, by the per-batch program."""
    plan = pipe.epoch_plan
    run = Run(plan=plan, position=start[0], epoch=start[1], end_epoch=pipe.num_epochs)
    naming = HostNaming(pipe.source, plan, shuffled=pipe.shuffle)
    key = key_words(pipe._epoch_key_base.get_value()) if pipe.shuffle else None
    names = []
    ordinal = 0
    while (batch := run.batch(ordinal)) is not None:
        indices, epochs = naming(*batch, key)
        names.append((indices.tobytes(), epochs.tobytes()))
        ordinal += 1
    return names


@pytest.fixture
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> int:
    """Blocks of 12 records, so short runs cross many blocks and resume mid-block."""
    monkeypatch.setattr(run_units, "_NAMING_BLOCK_RECORDS", 12)
    return 12


@pytest.mark.parametrize("make", [_memory, _mix], ids=["memory", "mix"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("num_epochs", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 3, 4, 8])
@pytest.mark.parametrize("length", [7, 10, 23, 64])
def test_a_batch_named_in_a_block_is_named_as_alone(
    small_blocks: int,
    make: Any,
    shuffle: bool,
    drop_last: bool,
    num_epochs: int,
    batch_size: int,
    length: int,
) -> None:
    del small_blocks
    if drop_last and batch_size > length:
        return  # the plan refuses it
    pipe = _pipeline(
        make(length),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
    )
    expected = _alone(pipe)
    assert [_named(batch) for batch in pipe.raw_batches()] == expected


@pytest.mark.parametrize("threads", [1, 4, 8])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_resumed_at_every_position_the_rest_is_named_as_alone(
    small_blocks: int, threads: int, drop_last: bool, shuffle: bool
) -> None:
    """Blocks are counted from where a run starts; a resume mid-block names the same records."""
    del small_blocks

    def build() -> Pipeline:
        return _pipeline(
            _memory(23), batch_size=4, shuffle=shuffle, drop_last=drop_last, threads=threads
        )

    whole = _alone(build())
    for done in range(len(whole) + 1):
        pipe = build()
        batches = iter(pipe.raw_batches())
        served = [_named(next(batches)) for _ in range(done)]
        state = pipe.get_state()
        pipe.close()
        resumed = build()
        resumed.set_state(state)
        assert served + [_named(batch) for batch in resumed.raw_batches()] == whole


@pytest.mark.parametrize("chunk", [2, 3, 5])
def test_a_chunk_s_batches_are_named_as_alone(small_blocks: int, chunk: int) -> None:
    del small_blocks
    pipe = _pipeline(_memory(23), batch_size=4)
    expected = _alone(pipe)
    served: list[tuple[bytes, bytes]] = []
    for unit in pipe.raw_batches(chunk=chunk):
        if np.asarray(unit.indices).ndim == 3:  # a chunk (K, B, 2)
            served.extend(
                _named(jax.tree.map(lambda leaf, k=k: leaf[k], unit))
                for k in range(np.asarray(unit.indices).shape[0])
            )
        else:
            served.append(_named(unit))
    assert served == expected


def test_full_batches_are_named_a_block_per_call(
    small_blocks: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of 12 full batches of 3 and a short one: 3 block calls and 1 call alone."""
    del small_blocks
    calls = {"alone": 0, "block": 0}
    alone, block = HostNaming.__call__, getattr(HostNaming, "block", None)

    def counted_alone(self: HostNaming, *args: Any, **kwargs: Any) -> Any:
        calls["alone"] += 1
        return alone(self, *args, **kwargs)

    def counted_block(self: HostNaming, *args: Any, **kwargs: Any) -> Any:
        calls["block"] += 1
        assert block is not None
        return block(self, *args, **kwargs)

    monkeypatch.setattr(HostNaming, "__call__", counted_alone)
    monkeypatch.setattr(HostNaming, "block", counted_block, raising=False)
    pipe = _pipeline(_memory(38), batch_size=3, shuffle=True, num_epochs=1)
    served = list(pipe.raw_batches())
    assert [batch.batch_size for batch in served] == [3] * 12 + [2]
    assert calls == {"alone": 1, "block": 3}


def test_pipelines_built_alike_compile_the_block_program_once() -> None:
    """One program per structure: the block's and the short batch's, whatever the pipeline."""

    def run() -> None:
        pipe = _pipeline(_memory(50), batch_size=8, num_epochs=2)
        list(pipe.raw_batches())
        pipe.close()

    jax.clear_caches()
    with compiled_programs() as programs:
        run()
        run()
        run()
    # 100 records in batches of 8: the block of full batches and the run's short final batch.
    assert sum(str(program).startswith("jit(_names") for program in programs) == 2


def test_the_names_a_run_holds_are_two_blocks_at_most(small_blocks: int) -> None:
    """A run's naming holds at most two blocks of names: 12 bytes a record (words and epoch)."""
    pipe = _pipeline(_memory(64), batch_size=4, num_epochs=1)
    read = pipe.host_stage.read_for_workers(pipe)
    for unit in range(16):
        read(unit)
    held = list(read.names._blocks.values())  # noqa: SLF001 - the names a run holds
    assert 0 < len(held) <= 2
    assert sum(indices.nbytes + epochs.nbytes for indices, epochs in held) <= 2 * small_blocks * 12
