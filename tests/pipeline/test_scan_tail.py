"""A user's scan over ``raw_batches(chunk=K)`` covers every record of an evaluation once.

``raw_batches(chunk=K)`` serves chunks of ``K`` full batches while ``K`` remain, then the remaining
batches singly, a bounded run's short final batch last. A user (or the Trainer) scans each chunk
with a ``lax.scan`` body and calls the same body once per single batch, so an evaluation whose
length the batch size does not divide covers every record exactly once, with no padding: one
program per chunk shape and one per single shape on the first pass, none on the second.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.compiles import expect_first_call_compiles


def _evaluation(length: int, batch_size: int, *, shuffle: bool) -> Pipeline:
    """One epoch over ``length`` records whose value is their index, the run's final batch short."""
    source = MemorySource(MemorySourceConfig(), {"value": np.arange(length, dtype=np.float32)})
    return Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
        shuffle=shuffle,
        num_epochs=1,
    )


def _body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
    """One evaluation step: add the batch's values to the total and return its record count."""
    return total + jnp.sum(batch["value"]), jnp.asarray(batch["value"].shape[0])


@jax.jit
def scan_chunk(total: jax.Array, chunk: Batch) -> tuple[jax.Array, jax.Array]:
    """The step scanned over a chunk's leading axis."""
    total, counts = jax.lax.scan(_body, total, chunk)
    return total, jnp.sum(counts)


@jax.jit
def step(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
    """The same step called on one batch: a single full batch or the short final one."""
    return _body(total, batch)


def _evaluate(units: list[Batch], total: jax.Array) -> tuple[jax.Array, int]:
    """Scan each chunk and call the step on each single batch, as a Trainer evaluation does."""
    records = 0
    for unit in units:
        total, count = scan_chunk(total, unit) if unit["value"].ndim == 2 else step(total, unit)
        records += int(count)
    return total, records


def _record_indices(units: list[Batch]) -> list[int]:
    words = [np.asarray(unit.indices).reshape(-1, 2) for unit in units]
    return [int(index) for block in words for index in from_words(block)]


@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
@pytest.mark.parametrize(
    ("length", "batch_size", "chunk", "shapes"),
    [
        (100, 32, 2, [(2, 32), (32,), (4,)]),
        (300, 32, 4, [(4, 32), (4, 32), (32,), (12,)]),
        (96, 32, 3, [(3, 32)]),
    ],
    ids=["a chunk, a single, a tail", "two chunks, a single, a tail", "no tail"],
)
class TestScanOverChunksAndSingles:
    def test_every_record_is_evaluated_once(
        self, length: int, batch_size: int, chunk: int, shapes: list[Any], shuffle: bool
    ) -> None:
        units = list(_evaluation(length, batch_size, shuffle=shuffle).raw_batches(chunk=chunk))

        total, records = _evaluate(units, jnp.zeros(()))

        assert [unit["value"].shape for unit in units] == shapes
        assert sorted(_record_indices(units)) == list(range(length))
        assert records == length
        assert float(total) == float(np.arange(length, dtype=np.float64).sum())

    def test_the_first_pass_builds_one_program_per_shape_and_the_second_none(
        self, length: int, batch_size: int, chunk: int, shapes: list[Any], shuffle: bool
    ) -> None:
        units = list(_evaluation(length, batch_size, shuffle=shuffle).raw_batches(chunk=chunk))
        programs = [
            "jit(scan_chunk)" if len(shape) == 2 else "jit(step)" for shape in dict.fromkeys(shapes)
        ]
        start = jnp.zeros(())

        with expect_first_call_compiles(*programs):
            _evaluate(units, start)
        with expect_compiles(0):
            _evaluate(units, start)
