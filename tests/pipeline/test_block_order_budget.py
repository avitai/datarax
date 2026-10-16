"""A block order's page-cache working set is a term of the host stage's RAM budget.

A block order keeps a buffer's records resident while a batch reads them, so a run read from
storage holds about ``blocks_per_buffer * block_size`` records' stored bytes in the page cache: the
budget's ``O`` term. Each source reports its own (``OrderWorkingSet``): a disk source from its
stored rows, ArrayRecord from its files' bytes per record, an in-memory source 0 (its records are
already the training process's resident memory), a mix the sum of its children's. A budget that
cannot hold the term is refused naming it, and a source that cannot report one is refused when a
budget is given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx

from datarax.core.block_order import BlockOrder
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.host_resources import HostResources, order_working_bytes, OrderWorkingSet
from datarax.pipeline import Pipeline
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.streaming_disk_source import (
    DiskColumn,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


_ORDER = BlockOrder(block_size=3, blocks_per_buffer=2)
_ROW_BYTES = 4 * 4 + 8  # x: 4 float32, y: one int64


def _disk(tmp_path: Path, length: int = 100) -> StreamingDiskSource:
    x, y = tmp_path / "x.npy", tmp_path / "y.npy"
    np.save(x, np.arange(length * 4, dtype=np.float32).reshape(length, 4))
    np.save(y, np.arange(length, dtype=np.int64))
    return StreamingDiskSource(
        StreamingDiskSourceConfig(
            columns=(DiskColumn.from_npy("x", x), DiskColumn.from_npy("y", y))
        )
    )


def _memory(length: int = 40) -> MemorySource:
    return MemorySource(MemorySourceConfig(), {"x": np.arange(length, dtype=np.float32)})


def _array_record_files(tmp_path: Path, counts: tuple[int, ...]) -> list[str]:
    paths = []
    for index, count in enumerate(counts):
        path = tmp_path / f"records-{index}.array_record"
        writer = ArrayRecordWriter(str(path), "group_size:1")
        for record in range(count):
            writer.write(np.full(25, record, np.float32).tobytes())
        writer.close()
        paths.append(str(path))
    return paths


@dataclass(frozen=True)
class _FileSlice:
    """ArrayRecord's ``FileInstruction`` protocol (TFDS's shape) without TFDS: part of a file."""

    filename: str
    skip: int
    take: int
    examples_in_shard: int


def _decode(records: Sequence[bytes]) -> Sequence[Mapping[str, Any]]:
    return [{"x": np.frombuffer(record, np.float32)} for record in records]


class TestEachSourceReportsItsWorkingSet:
    @pytest.mark.parametrize(
        ("order", "held"),
        [(_ORDER, 6), (BlockOrder(block_size=64, blocks_per_buffer=4), 100)],
        ids=["a-buffer", "every-record"],
    )
    def test_a_disk_source_holds_a_buffer_of_its_stored_rows(
        self, tmp_path: Path, order: BlockOrder, held: int
    ) -> None:
        source = _disk(tmp_path)
        assert isinstance(source, OrderWorkingSet)
        assert source.order_working_bytes(order) == held * _ROW_BYTES

    def test_array_record_holds_a_buffer_of_its_files_bytes_per_record(
        self, tmp_path: Path
    ) -> None:
        paths = _array_record_files(tmp_path, (30, 20))
        source = ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths, decode=_decode)
        stored = sum(Path(path).stat().st_size for path in paths)
        assert source.order_working_bytes(_ORDER) == 6 * -(-stored // 50)  # rounded up

    def test_a_slice_of_the_files_is_charged_their_bytes_per_record(self, tmp_path: Path) -> None:
        """A slice reads part of a file: the file's bytes spread over all its records."""
        (path,) = _array_record_files(tmp_path, (30,))
        stored = Path(path).stat().st_size
        for sliced in (
            _FileSlice(filename=path, skip=5, take=10, examples_in_shard=30),
            f"{path}[5:15]",
        ):
            source = ArrayRecordSourceModule(ArrayRecordSourceConfig(), [sliced], decode=_decode)
            assert len(source) == 10
            assert source.order_working_bytes(_ORDER) == 6 * -(-stored // 30)

    def test_an_in_memory_source_holds_nothing_more(self) -> None:
        assert _memory().order_working_bytes(_ORDER) == 0

    def test_a_mix_holds_each_child_s_working_set(self, tmp_path: Path) -> None:
        disk = _disk(tmp_path)
        memory = MemorySource(
            MemorySourceConfig(),
            {"x": np.zeros((40, 4), np.float32), "y": np.zeros(40, np.int64)},
        )
        mix = MixDataSourcesNode(MixDataSourcesConfig(weights=(1.0, 1.0)), [disk, memory])
        assert mix.order_working_bytes(_ORDER) == disk.order_working_bytes(_ORDER)

    def test_without_a_block_order_the_order_holds_nothing(self, tmp_path: Path) -> None:
        assert order_working_bytes(_disk(tmp_path), None) == 0


class _Unreported(DataSourceModule):
    """An indexed source that declares no working set."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(StructuralConfig())

    def __len__(self) -> int:
        return 10


def test_a_source_that_reports_no_working_set_is_refused() -> None:
    with pytest.raises(TypeError, match="OrderWorkingSet"):
        order_working_bytes(_Unreported(), _ORDER)


def _pipeline(source: DataSourceModule, *, order: BlockOrder | None, budget: int) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
        shuffle=True,
        block_order=order,
        host_resources=HostResources(ram_budget_bytes=budget, max_workers=2),
    )


class TestTheBudget:
    def test_the_plan_charges_the_order_s_working_set(self, tmp_path: Path) -> None:
        source = _disk(tmp_path)
        pipe = _pipeline(source, order=_ORDER, budget=1 << 36)
        next(iter(pipe.raw_batches()))
        terms = pipe.host_plan.terms
        assert terms is not None
        assert terms.order_bytes == source.order_working_bytes(_ORDER)
        pipe.close()

    def test_the_global_order_is_charged_nothing(self, tmp_path: Path) -> None:
        pipe = _pipeline(_disk(tmp_path), order=None, budget=1 << 36)
        next(iter(pipe.raw_batches()))
        terms = pipe.host_plan.terms
        assert terms is not None
        assert terms.order_bytes == 0
        pipe.close()

    def test_a_refused_budget_names_the_working_set_it_was_charged(self, tmp_path: Path) -> None:
        """A buffer of 2**12 records of 24 bytes: the refusal names it as the order's term."""
        order = BlockOrder(block_size=1 << 6, blocks_per_buffer=1 << 6)
        pipe = _pipeline(_disk(tmp_path, length=1 << 13), order=order, budget=1 << 20)
        with pytest.raises(ValueError, match=f"O={order.buffer_size * _ROW_BYTES} "):
            next(iter(pipe.raw_batches()))
        pipe.close()
