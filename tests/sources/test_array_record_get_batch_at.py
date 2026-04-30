"""Contracts for ``ArrayRecordSourceModule.get_batch_at``.

ArrayRecord's records are loaded host-side via Grain, so
``get_batch_at`` is eager-only (Tier A / B) and rejects JAX tracers
with a clear error message. Future enhancement could add an
``io_callback``-wrapped scan-friendly variant.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from datarax.sources.array_record_source import (
    ArrayRecordSourceConfig,
    ArrayRecordSourceModule,
)


def _make_source() -> ArrayRecordSourceModule:
    mock_grain = MagicMock()
    mock_grain.__len__.return_value = 16
    mock_grain.__getitem__.side_effect = lambda idx: {"data": np.array([idx])}
    # Force the fallback per-index path; otherwise MagicMock auto-generates
    # an empty _getitems and the batched access returns an empty list.
    mock_grain._getitems = None
    with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain):
        return ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths="/fake/path")


def test_array_record_get_batch_at_returns_size_records() -> None:
    src = _make_source()

    records = src.get_batch_at(start=2, size=4, key=None)

    assert len(records) == 4
    # Records are dicts with "data" arrays; verify the indices match.
    indices = [int(rec["data"][0]) for rec in records]
    assert indices == [2, 3, 4, 5]


def test_array_record_get_batch_at_wraps_at_end() -> None:
    src = _make_source()

    records = src.get_batch_at(start=14, size=4, key=None)
    indices = [int(rec["data"][0]) for rec in records]
    assert indices == [14, 15, 0, 1]


def test_array_record_get_batch_at_does_not_advance_internal_state() -> None:
    src = _make_source()
    src.current_index.set_value(7)

    src.get_batch_at(start=0, size=4, key=None)

    assert int(src.current_index.get_value()) == 7


def test_array_record_get_batch_at_rejects_traced_start() -> None:
    """Tracers can't drive a host-side fetch; clear error message points to the iterator."""
    src = _make_source()

    with pytest.raises(TypeError, match="concrete Python int"):
        src.get_batch_at(start=jnp.int32(0), size=4, key=None)
