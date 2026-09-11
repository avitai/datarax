"""``ArrayRecordSourceModule`` streams decoded batches through ``Pipeline``.

ArrayRecord files hold ``bytes`` records, so the source takes a ``decode``
function that turns one record into a dict of arrays. ``get_batch`` decodes and
stacks up to ``batch_size`` records of the current epoch on the host and returns
an empty batch at the epoch boundary, so each ``for batch in pipeline`` pass
covers one epoch and the next pass starts the next one. ``num_epochs`` bounds
the passes, and ``shuffle_files`` draws a new record order for every epoch.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.spec import SpecMismatchError
from datarax.pipeline import Pipeline
from datarax.sources.array_record_source import (
    ArrayRecordSourceConfig,
    ArrayRecordSourceModule,
)


_RECORDS = 10


def _encode(index: int) -> bytes:
    return np.full(3, index, dtype=np.float32).tobytes()


def _decode(record: bytes) -> dict[str, Any]:
    values = np.frombuffer(record, dtype=np.float32)
    return {"x": values, "label": np.int32(values[0])}


def _source(
    config: ArrayRecordSourceConfig | None = None, *, decode: Any = _decode
) -> ArrayRecordSourceModule:
    grain_source = MagicMock()
    grain_source.__len__.return_value = _RECORDS
    grain_source.__getitem__.side_effect = _encode
    grain_source._getitems = None
    with patch("grain.sources.ArrayRecordDataSource", return_value=grain_source):
        return ArrayRecordSourceModule(
            config or ArrayRecordSourceConfig(), paths="/fake/path", decode=decode
        )


def _labels(batch: dict[str, Any]) -> list[int]:
    return [int(label) for label in batch["label"]]


def _pass(source: ArrayRecordSourceModule, batch_size: int) -> list[dict[str, Any]]:
    batches = []
    while batch := source.get_batch(batch_size):
        batches.append(batch)
    return batches


def test_get_batch_decodes_and_stacks_records() -> None:
    batch = _source().get_batch(4)

    assert batch["x"].shape == (4, 3)
    assert batch["x"].dtype == np.float32
    assert _labels(batch) == [0, 1, 2, 3]


def test_a_pass_ends_at_the_epoch_boundary_and_the_next_pass_starts_the_next_epoch() -> None:
    source = _source()

    first = _pass(source, 4)
    second = _pass(source, 4)

    assert [len(_labels(b)) for b in first] == [4, 4, 2]
    assert [len(_labels(b)) for b in second] == [4, 4, 2]
    assert int(source.current_epoch.get_value()) == 2


def test_num_epochs_bounds_the_passes() -> None:
    source = _source(ArrayRecordSourceConfig(num_epochs=1))

    assert len(_pass(source, 4)) == 3
    assert _pass(source, 4) == []


def test_shuffle_serves_a_permutation_each_epoch() -> None:
    source = _source(ArrayRecordSourceConfig(shuffle_files=True, seed=3))

    first = [label for batch in _pass(source, 4) for label in _labels(batch)]
    second = [label for batch in _pass(source, 4) for label in _labels(batch)]

    assert sorted(first) == sorted(second) == list(range(_RECORDS))
    assert first != second


def test_element_spec_describes_a_decoded_record() -> None:
    assert _source().element_spec() == {
        "x": jax.ShapeDtypeStruct((3,), np.float32),
        "label": jax.ShapeDtypeStruct((), np.int32),
    }


def test_without_a_decoder_batches_are_refused() -> None:
    source = _source(decode=None)

    with pytest.raises(TypeError, match="decode"):
        source.get_batch(4)
    with pytest.raises(TypeError, match="decode"):
        source.element_spec()


def test_array_record_is_a_streaming_source() -> None:
    assert _source().supports_indexed_access() is False


def test_pipeline_iterates_decoded_batches_one_epoch_per_pass() -> None:
    pipeline = Pipeline(source=_source(), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    first = [label for batch in pipeline for label in _labels(batch)]
    second = [label for batch in pipeline for label in _labels(batch)]

    assert first == second == list(range(_RECORDS))


def test_a_decoder_whose_records_disagree_with_the_first_is_refused() -> None:
    def ragged(record: bytes) -> dict[str, Any]:
        values = np.frombuffer(record, dtype=np.float32)
        return {"x": values if values[0] < 4 else values[:2], "label": np.int32(values[0])}

    pipeline = Pipeline(source=_source(decode=ragged), stages=[], batch_size=4, rngs=nnx.Rngs(0))

    with pytest.raises((SpecMismatchError, ValueError)):
        list(pipeline)
