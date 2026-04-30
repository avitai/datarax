"""Tests for SlidingWindowSampler — windowed-index sampling for time-series workloads.

The sampler emits windows of consecutive indices (shape ``(window_size,)``) at a
configurable stride, exposing both ``drop_incomplete`` and ``pad_incomplete``
modes for the final partial window. ``index_spec`` must declare the windowed
output so downstream consumers can pre-allocate buffers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.samplers.sliding_window_sampler import (
    SlidingWindowSampler,
    SlidingWindowSamplerConfig,
)


def _consume_indices(sampler: SlidingWindowSampler) -> list[np.ndarray]:
    """Iterate the sampler and collect windows as numpy arrays."""
    return [np.asarray(window) for window in sampler]


def test_sliding_window_yields_correct_index_ranges() -> None:
    """data_length=10, window=3, stride=2 → 4 full windows: [0:3], [2:5], [4:7], [6:9]."""
    config = SlidingWindowSamplerConfig(
        num_records=10, window_size=3, stride=2, drop_incomplete=True
    )
    sampler = SlidingWindowSampler(config, rngs=nnx.Rngs(0))

    windows = _consume_indices(sampler)

    assert len(windows) == 4
    np.testing.assert_array_equal(windows[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(windows[1], np.array([2, 3, 4]))
    np.testing.assert_array_equal(windows[2], np.array([4, 5, 6]))
    np.testing.assert_array_equal(windows[3], np.array([6, 7, 8]))


def test_sliding_window_drop_incomplete_short_dataset() -> None:
    """data_length=8, window=3, stride=2, drop_incomplete=True → exactly 3 full windows."""
    config = SlidingWindowSamplerConfig(
        num_records=8, window_size=3, stride=2, drop_incomplete=True
    )
    sampler = SlidingWindowSampler(config, rngs=nnx.Rngs(0))

    windows = _consume_indices(sampler)

    assert len(windows) == 3
    np.testing.assert_array_equal(windows[-1], np.array([4, 5, 6]))


def test_sliding_window_pad_incomplete_short_dataset() -> None:
    """data_length=8, window=3, stride=2, pad_incomplete=True → 4 windows; last is padded.

    The padded window starts at position 6 and covers indices 6, 7, then a
    pad index (clamped to the last valid index, 7). Per-window validity is
    surfaced via the optional ``valid_indices`` companion API on the sampler.
    """
    config = SlidingWindowSamplerConfig(
        num_records=8, window_size=3, stride=2, drop_incomplete=False
    )
    sampler = SlidingWindowSampler(config, rngs=nnx.Rngs(0))

    windows = _consume_indices(sampler)

    assert len(windows) == 4
    np.testing.assert_array_equal(windows[-1], np.array([6, 7, 7]))


def test_sliding_window_index_spec_returns_window_shaped_struct() -> None:
    """``index_spec`` returns ``ShapeDtypeStruct(shape=(window_size,), dtype=int32)``.

    Overrides the default scalar ``index_spec`` from ``SamplerModule`` because
    each emitted sample is a window of indices, not a single index.
    """
    config = SlidingWindowSamplerConfig(
        num_records=10, window_size=4, stride=2, drop_incomplete=True
    )
    sampler = SlidingWindowSampler(config, rngs=nnx.Rngs(0))

    spec = sampler.index_spec()

    assert isinstance(spec, jax.ShapeDtypeStruct)
    assert spec.shape == (4,)
    assert spec.dtype == jnp.int32


def test_sliding_window_invalid_window_size_rejected() -> None:
    """window_size must be positive."""
    with pytest.raises(ValueError, match="window_size"):
        SlidingWindowSamplerConfig(num_records=10, window_size=0, stride=2)


def test_sliding_window_invalid_stride_rejected() -> None:
    """stride must be positive."""
    with pytest.raises(ValueError, match="stride"):
        SlidingWindowSamplerConfig(num_records=10, window_size=3, stride=0)


def test_sliding_window_jit_compatible() -> None:
    """A JIT-wrapped per-window function over the sampler produces the same windows.

    This is the load-bearing contract: SlidingWindowSampler emits ``jax.Array``
    that can be consumed inside ``jax.jit``-compiled bodies (Phase 4 scan
    executor will rely on this).
    """
    config = SlidingWindowSamplerConfig(
        num_records=10, window_size=3, stride=2, drop_incomplete=True
    )
    sampler = SlidingWindowSampler(config, rngs=nnx.Rngs(0))

    @jax.jit
    def double_window(window: jax.Array) -> jax.Array:
        return window * 2

    eager_windows = [np.asarray(w * 2) for w in sampler]
    jit_windows = []
    sampler2 = SlidingWindowSampler(config, rngs=nnx.Rngs(0))
    for window in sampler2:
        jit_windows.append(np.asarray(double_window(window)))

    for a, b in zip(eager_windows, jit_windows, strict=True):
        np.testing.assert_array_equal(a, b)
