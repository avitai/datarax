"""How Pipeline drives a source: indexed through get_batch_at, streaming through get_batch.

``DataSourceModule.get_batch_at`` is documented as stateless and JAX-traceable, so
a source that implements it supports indexed access without restating that, and
``Pipeline`` iterates it through the compiled session. A source that implements
neither ``get_batch_at`` nor ``get_batch`` cannot be iterated, and says so when
iteration starts rather than failing inside the loop.
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
from datarax.core.data_source import DataSourceModule
from datarax.pipeline import Pipeline, PipelineIterator


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


_ROWS = 8


class _IndexedOnly(DataSourceModule):
    """Source implementing get_batch_at and nothing about access modes."""

    def __init__(self) -> None:
        super().__init__(_Config())
        self.data = nnx.data(jnp.arange(_ROWS, dtype=jnp.float32).reshape(_ROWS, 1))

    def __len__(self) -> int:
        return _ROWS

    def get_batch_at(self, start: Any, size: int, key: Any = None) -> dict[str, jax.Array]:
        del key
        indices = (jnp.asarray(start, jnp.int32) + jnp.arange(size, dtype=jnp.int32)) % _ROWS
        return {"x": self.data[indices]}

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((1,), jnp.float32)}


class _NoAccess(DataSourceModule):
    """Source implementing neither get_batch_at nor get_batch."""

    def __init__(self) -> None:
        super().__init__(_Config())

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((1,), jnp.float32)}


def test_implementing_get_batch_at_is_what_indexed_access_means() -> None:
    assert _IndexedOnly().supports_indexed_access() is True
    assert _NoAccess().supports_indexed_access() is False


def test_pipeline_iterates_a_get_batch_at_source_through_the_compiled_session() -> None:
    iterated = Pipeline(source=_IndexedOnly(), stages=[], batch_size=4, rngs=nnx.Rngs(0))
    stepped = Pipeline(source=_IndexedOnly(), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    iterator = iter(iterated)
    assert isinstance(iterator, PipelineIterator)
    batches = [np.asarray(batch["x"]) for batch in iterator]
    expected = [np.asarray(stepped.step()["x"]) for _ in batches]  # type: ignore[call-arg]

    assert len(batches) == 2
    for got, want in zip(batches, expected, strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_source_that_serves_records_in_order_names_them_by_position() -> None:
    """Without an override, record ids are the wrapped positions ``get_batch_at`` reads."""
    ids = _IndexedOnly().record_indices_at(start=6, size=4)

    np.testing.assert_array_equal(np.asarray(ids), np.array([6, 7, 0, 1]))


def test_iterating_a_source_without_an_access_method_names_both() -> None:
    pipeline = Pipeline(source=_NoAccess(), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    with pytest.raises(TypeError, match=r"get_batch_at.*get_batch"):
        iter(pipeline)
