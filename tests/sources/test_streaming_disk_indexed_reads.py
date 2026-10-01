"""Contracts for ``StreamingDiskSource``'s indexed read.

The streaming-disk source backs the on-disk array via memmap and
dispatches reads through ``jax.experimental.io_callback``.
``get_records(record_indices_at(start, size, key))`` lets ``Pipeline`` drive
indexed access through the same io_callback boundary, with ``stop_gradient``
applied at the output so downstream gradient code is well-defined.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from datarax.sources.streaming_disk_source import (
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


@pytest.fixture
def synthetic_disk_array(tmp_path: Path) -> Path:
    """Write a deterministic (16, 4) float32 array as a .npy file."""
    array = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
    path = tmp_path / "data.npy"
    np.save(path, array)
    return path


def test_streaming_disk_indexed_read_returns_contiguous_slice(
    synthetic_disk_array: Path,
) -> None:
    src = StreamingDiskSource(
        StreamingDiskSourceConfig(path=str(synthetic_disk_array), feature_key="x")
    )

    batch = src.get_records(src.record_indices_at(4, 4, None))

    assert batch["x"].shape == (4, 4)
    expected = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)[4:8]
    np.testing.assert_array_equal(np.asarray(batch["x"]), expected)


def test_streaming_disk_indexed_read_wraps_at_end(synthetic_disk_array: Path) -> None:
    src = StreamingDiskSource(
        StreamingDiskSourceConfig(path=str(synthetic_disk_array), feature_key="x")
    )

    # start=14, size=4 → indices 14, 15, 0, 1
    batch = src.get_records(src.record_indices_at(14, 4, None))

    arr = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
    expected = np.stack([arr[14], arr[15], arr[0], arr[1]], axis=0)
    np.testing.assert_array_equal(np.asarray(batch["x"]), expected)


def test_streaming_disk_indexed_read_accepts_traced_start(
    synthetic_disk_array: Path,
) -> None:
    """Traced start (jax.Array) is required for nnx.scan-driven iteration."""
    src = StreamingDiskSource(
        StreamingDiskSourceConfig(path=str(synthetic_disk_array), feature_key="x")
    )

    batch = src.get_records(src.record_indices_at(jnp.int32(4), 4, None))
    assert batch["x"].shape == (4, 4)
