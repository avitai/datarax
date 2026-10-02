"""How Pipeline drives a source: an INDEXED one through get_records, a stream through get_batch.

An indexed source names the record at each position (``record_indices_at``, sequential by
default) and gathers records by those indices (``get_records``). The pipeline computes each
batch's record indices once and hands the same indices to the gather and to the stages that key
randomness on them. The source's declared kind (``record_identity``) routes it; an ``INDEXED``
source that does not implement ``get_records`` says so at its first pull.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import to_words
from datarax.pipeline import iteration, Pipeline, PipelineIterator
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.step_jaxpr import compiled_step


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


_ROWS = 8


class _IndexedOnly(DataSourceModule):
    """Source implementing get_records and nothing about access modes."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(_Config())
        self.data = nnx.data(jnp.arange(_ROWS, dtype=jnp.float32).reshape(_ROWS, 1))

    def __len__(self) -> int:
        return _ROWS

    def get_records(self, indices: jax.Array) -> dict[str, jax.Array]:
        return {"x": self.data[indices[:, 1]]}

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((1,), jnp.float32)}


class _NoAccess(DataSourceModule):
    """Source implementing neither get_records nor get_batch."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(_Config())

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((1,), jnp.float32)}


def test_a_source_without_get_records_has_no_indexed_read() -> None:
    with pytest.raises(NotImplementedError, match="get_records"):
        _NoAccess().get_records(_NoAccess().record_indices_at(0, 4))


def test_pipeline_iterates_a_get_records_source_through_the_compiled_session() -> None:
    iterated = Pipeline(source=_IndexedOnly(), stages=[], batch_size=4, rngs=nnx.Rngs(0))
    stepped = Pipeline(source=_IndexedOnly(), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    iterator = iter(iterated)
    assert isinstance(iterator, PipelineIterator)
    batches = [np.asarray(batch["x"]) for batch in iterator]
    expected = [np.asarray(stepped.step()["x"]) for _ in batches]

    assert len(batches) == 2
    for got, want in zip(batches, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_source_that_serves_records_in_order_names_them_by_position() -> None:
    """Without an override, record ids are the wrapped positions."""
    ids = _IndexedOnly().record_indices_at(start=6, size=4)

    np.testing.assert_array_equal(to_words([6, 7, 0, 1]), ids)


def test_an_indexed_source_without_get_records_is_refused_at_its_first_pull() -> None:
    pipeline = Pipeline(source=_NoAccess(), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    with pytest.raises(NotImplementedError, match=r"get_records.*INDEXED"):
        next(iter(pipeline))


_NAMED: list[int] = []


class _Counted(_IndexedOnly):
    """Counts, outside the module, how often the step computes record indices."""

    def record_indices_at(self, start: Any, size: int, key: Any = None) -> jax.Array:
        _NAMED.append(size)
        return super().record_indices_at(start, size, key)


@pytest.mark.parametrize("stochastic_stage", [False, True])
def test_the_step_names_its_records_once(stochastic_stage: bool) -> None:
    """The gather and the stages share one computation of the batch's record indices.

    A shuffled order's index computation is its most expensive part on a GPU (a cycle-walking
    loop), and XLA does not merge copies of it placed in different branches, so the step names
    its records with one ``record_indices_at`` call, vmapped over every epoch a batch can touch.
    """

    class _Keyed(nnx.Module):
        def __call__(self, batch: Batch) -> Batch:
            return batch

    source = _Counted()
    stages = [_Keyed()] if stochastic_stage else []
    pipeline = Pipeline(source=source, stages=stages, batch_size=4, rngs=nnx.Rngs(0))
    iteration._SESSION_STEPS.clear()
    _NAMED.clear()
    pipeline.step()  # traces the compiled step once

    assert len(_NAMED) == 1


def test_a_session_and_step_run_one_program_with_no_conditional() -> None:
    """Every batch, the one reaching an epoch's end included, runs one compiled program.

    Compiled for the platform it runs on, the step holds no conditional: naming records needs no
    branch, and the shuffle's platform choice is resolved by the compiler. ``step()`` and an
    iteration session share the program, so they serve bit-identical batches.
    """

    def shuffled() -> Pipeline:
        source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(8.0)})
        return Pipeline(
            source=source,
            stages=[],
            batch_size=3,
            rngs=nnx.Rngs(0),
            num_epochs=None,
            shuffle=True,
        )

    pipeline = shuffled()
    program = compiled_step(pipeline)
    assert " while(" in program  # control: the shuffle's cycle-walk is in the program
    assert " conditional(" not in program

    iteration._SESSION_STEPS.clear()
    pipeline.step()
    compiled = len(iteration._SESSION_STEPS)
    session = shuffled().session()
    for _ in range(6):  # crosses two epoch boundaries
        next(session)
    assert len(iteration._SESSION_STEPS) == compiled


def test_a_pipeline_over_record_list_data_serves_its_columns() -> None:
    """A list of records is stored as columns, so the compiled session serves it.

    The records' text is their provenance and never part of a batch.
    """
    records = [{"x": np.full((2,), i, np.float32), "name": f"r{i}"} for i in range(6)]
    source = MemorySource(MemorySourceConfig(), records)
    pipeline = Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))

    iterator = iter(pipeline)
    assert isinstance(iterator, PipelineIterator)
    served = [np.asarray(batch["x"]) for batch in iterator]
    np.testing.assert_array_equal(np.concatenate(served)[:, 0], np.arange(6, dtype=np.float32))
    assert all(
        "name" not in batch
        for batch in Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))
    )


class _ListStream(DataSourceModule):
    """A forward-only source whose batches are lists."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: ARRIVAL."""
        return RecordIdentity.ARRIVAL

    def __init__(self) -> None:
        super().__init__(_Config())

    def get_batch(
        self, batch_size: int, *, key: jax.Array | None = None, read_size: int | None = None
    ) -> list[float]:
        del key, read_size
        return [1.0] * batch_size

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((), jnp.float32)}


def test_a_stream_yielding_something_other_than_a_batch_is_refused() -> None:
    pipeline = Pipeline(source=_ListStream(), stages=[], batch_size=2, rngs=nnx.Rngs(0))

    with pytest.raises(TypeError, match="_ListStream.get_batch returned list"):
        next(iter(pipeline))
