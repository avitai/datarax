"""A pipeline's provenance is typed by its unit: one mapping per record, or a packed row's slots.

``Pipeline`` is generic over the type of its ``packing`` argument, so
``raw_batches(with_provenance=True)`` is typed exactly for each pipeline and code over an
unpacked pipeline reads ``provenance[i]["name"]`` with no narrowing. The ``assert_type`` calls
are checked by the repository's pyright gate, which covers ``tests``; each test checks the same
shape at run time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_type

import numpy as np
from flax import nnx

from datarax.core.data_source import PackedProvenance, Provenance
from datarax.pipeline import Packing, Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


_PACKING = Packing(lengths={"tokens": 8}, max_segments=2, window=8)


def _ragged_source() -> MemorySource:
    return MemorySource(
        MemorySourceConfig(),
        [
            {"tokens": np.arange(index % 3 + 1, dtype=np.int32), "name": f"r{index}"}
            for index in range(6)
        ],
    )


def _fixed_source() -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), [{"x": np.int32(index), "name": f"r{index}"} for index in range(6)]
    )


def _unpacked() -> Pipeline[None]:
    return Pipeline(
        source=_fixed_source(),
        stages=[],
        batch_size=2,
        rngs=nnx.Rngs(0),
    )


def _packed() -> Pipeline[Packing]:
    return Pipeline(
        source=_ragged_source(), stages=[], batch_size=2, rngs=nnx.Rngs(0), packing=_PACKING
    )


def _is_record_provenance(entry: object) -> bool:
    return isinstance(entry, Mapping)


def _is_row_provenance(entry: object) -> bool:
    return isinstance(entry, tuple) and all(isinstance(slot, Mapping) for slot in entry)


def test_a_pipeline_is_typed_by_its_packing_argument() -> None:
    unpacked = Pipeline(source=_fixed_source(), stages=[], batch_size=2, rngs=nnx.Rngs(0))
    packed = Pipeline(
        source=_ragged_source(), stages=[], batch_size=2, rngs=nnx.Rngs(0), packing=_PACKING
    )

    assert_type(unpacked, Pipeline[None])
    assert_type(packed, Pipeline[Packing])
    assert unpacked.packing is None
    assert packed.packing is _PACKING


def test_an_unpacked_pipeline_serves_one_mapping_per_record() -> None:
    for batch, provenance in _unpacked().raw_batches(with_provenance=True):
        assert_type(provenance, Provenance)
        assert len(provenance) == batch.batch_size
        assert all(_is_record_provenance(entry) for entry in provenance)
        assert provenance[0]["name"].startswith("r")


def test_a_packed_pipeline_serves_each_row_s_slots() -> None:
    for batch, provenance in _packed().raw_batches(with_provenance=True):
        assert_type(provenance, PackedProvenance)
        assert len(provenance) == batch.batch_size
        assert all(_is_row_provenance(entry) for entry in provenance)
        assert all(len(slots) == _PACKING.max_segments for slots in provenance)


def _first_provenance(pipe: Pipeline[Packing | None]) -> Provenance | PackedProvenance:
    _, provenance = next(iter(pipe.raw_batches(with_provenance=True)))
    assert_type(provenance, Provenance | PackedProvenance)
    return provenance


def test_code_taking_either_pipeline_is_given_either_form() -> None:
    unpacked = _first_provenance(_unpacked())
    packed = _first_provenance(_packed())

    assert all(_is_record_provenance(entry) for entry in unpacked)
    assert all(_is_row_provenance(entry) for entry in packed)
