"""Tests for ``StreamingDiskSource`` — io_callback-based out-of-core source.

The source reads requested indices from a memory-mapped on-disk array via
``jax.experimental.io_callback``. ``jax.lax.stop_gradient`` wraps the
io_callback output so gradients cannot flow back through disk reads (which
``io_callback`` does not support anyway — JVP/VJP rules raise).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.sources.streaming_disk_source import (
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


def _write_npy(path: Path, array: np.ndarray) -> Path:
    """Write a numpy array to disk and return the path."""
    np.save(path, array)
    return path


def test_streaming_disk_source_reads_requested_indices(tmp_path: Path) -> None:
    """``read_batch(indices)`` returns the rows at those indices, ``stop_gradient``-wrapped."""
    array = np.arange(40, dtype=np.float32).reshape(10, 4)
    path = _write_npy(tmp_path / "data.npy", array)

    config = StreamingDiskSourceConfig(path=str(path), feature_key="x")
    source = StreamingDiskSource(config, rngs=nnx.Rngs(0))

    indices = jnp.asarray([0, 2, 4, 9], dtype=jnp.int32)
    out = source.read_batch(indices)

    assert isinstance(out, dict)
    assert "x" in out
    np.testing.assert_array_equal(np.asarray(out["x"]), array[[0, 2, 4, 9]])


def test_streaming_disk_source_length_matches_array(tmp_path: Path) -> None:
    """``len(source)`` returns the leading-dim length of the on-disk array."""
    array = np.zeros((25, 7), dtype=np.float32)
    path = _write_npy(tmp_path / "data.npy", array)

    config = StreamingDiskSourceConfig(path=str(path), feature_key="x")
    source = StreamingDiskSource(config, rngs=nnx.Rngs(0))

    assert len(source) == 25


def test_streaming_disk_source_element_spec_strips_leading_axis(tmp_path: Path) -> None:
    """``element_spec`` describes one element (leading axis stripped)."""
    array = np.zeros((30, 5, 3), dtype=np.float32)
    path = _write_npy(tmp_path / "data.npy", array)

    config = StreamingDiskSourceConfig(path=str(path), feature_key="img")
    source = StreamingDiskSource(config, rngs=nnx.Rngs(0))

    spec = source.element_spec()

    assert isinstance(spec, dict)
    assert "img" in spec
    assert isinstance(spec["img"], jax.ShapeDtypeStruct)
    assert spec["img"].shape == (5, 3)


def test_streaming_disk_source_usable_in_downstream_gradient(tmp_path: Path) -> None:
    """The source composes with downstream gradient computation.

    ``io_callback`` rejects JVP/VJP outright (its rules raise). The
    ``stop_gradient`` boundary on read outputs makes the disk read look like
    a constant to autograd: a downstream loss that multiplies the data by a
    learnable scalar gets a finite, correct gradient on the scalar — and
    JAX never invokes io_callback's JVP rule because there are no incoming
    cotangents on the data.
    """
    array = np.arange(20, dtype=np.float32).reshape(5, 4)
    path = _write_npy(tmp_path / "data.npy", array)

    config = StreamingDiskSourceConfig(path=str(path), feature_key="x")
    source = StreamingDiskSource(config, rngs=nnx.Rngs(0))

    indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    def loss(model_param: jax.Array) -> jax.Array:
        # Disk read with non-differentiated indices; downstream loss is
        # differentiable in model_param.
        out = source.read_batch(indices)
        return jnp.sum(out["x"]) * model_param

    grad_fn = jax.grad(loss)
    grad_param = grad_fn(jnp.asarray(2.0, dtype=jnp.float32))

    # ``loss = sum(data) * param`` ⇒ ``d(loss)/d(param) = sum(data)``.
    expected = float(np.sum(array[:3]))
    assert jnp.isclose(grad_param, expected)


def test_streaming_disk_source_rejects_missing_path(tmp_path: Path) -> None:
    """Constructor fails fast when the on-disk file does not exist."""
    missing = tmp_path / "missing.npy"
    config = StreamingDiskSourceConfig(path=str(missing), feature_key="x")

    with pytest.raises(FileNotFoundError):
        StreamingDiskSource(config, rngs=nnx.Rngs(0))
