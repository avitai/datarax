"""Tests for BufferSampler — replay/reservoir-buffer semantics.

The buffer stores PyTree records (not indices); ``next(value, mask)`` writes
+ samples atomically; the entire body is JIT-traceable through
``jax.lax.cond`` so the buffer composes with ``nnx.jit``-wrapped pipelines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.samplers.buffer_sampler import BufferSampler, BufferSamplerConfig


def _scalar_spec() -> dict:
    """Element spec for a single scalar float32 record."""
    return {"x": jax.ShapeDtypeStruct(shape=(), dtype=jnp.float32)}


def _scalar(value: float) -> dict:
    return {"x": jnp.asarray(value, dtype=jnp.float32)}


def test_buffer_records_are_stored_then_replayed() -> None:
    """After prefill writes, ``next`` returns a chunk drawn from the buffered records.

    Two-phase semantics:
    1. Pre-warmup (count < prefill): ``next(value)`` returns the input value
       as a length-1 chunk (passthrough).
    2. Post-warmup (count >= prefill): each ``next(value)`` first writes value
       into the buffer (FIFO/reservoir), then samples from the *updated*
       buffer.

    For FIFO + sequential, the post-warmup buffer-write-before-read means each
    sequential read returns the slot just written (slot 0 cycle).
    """
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, read_mode="sequential", write_mode="fifo"
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    # Warmup: 4 writes to fill the buffer.
    for v in [10.0, 20.0, 30.0, 40.0]:
        sampler.next(_scalar(v))

    # Post-warmup: each next writes value at write_index then reads at read_index.
    # write_index advances 0,1,2,3,0,1,...; read_index advances 0,1,2,3,0,1,...
    # First post-warmup call with v=99 writes 99→slot 0, then reads slot 0 (=99).
    chunk, mask = sampler.next(_scalar(99.0))
    assert float(chunk["x"][0]) == 99.0
    assert bool(mask[0]) is True

    # Buffer after first post-warmup write: [99, 20, 30, 40]; read_index now 1.
    chunk, _ = sampler.next(_scalar(88.0))
    # Second write goes to slot 1 (88), then reads slot 1 (=88).
    assert float(chunk["x"][0]) == 88.0


def test_buffer_fifo_overwrites_oldest_after_capacity() -> None:
    """FIFO write mode evicts the oldest record on overflow (ring-buffer semantics)."""
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, read_mode="sequential", write_mode="fifo"
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    # 6 writes into capacity=4: positions wrap around, last 4 written are [3,4,5,6].
    for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        sampler.next(_scalar(v))

    # The buffer ring slots 0..3 should hold values [5, 6, 3, 4] after 6 writes
    # at write positions 0,1,2,3,0,1 (FIFO ring overwrites slots 0 and 1).
    contents = sorted(np.asarray(sampler.buffer["x"]).tolist())
    assert contents == [3.0, 4.0, 5.0, 6.0]


def test_buffer_index_spec_returns_chunk_shape() -> None:
    """``index_spec`` declares ``(sample_size, *leaf.shape)`` per leaf — chunk-shaped output."""
    config = BufferSamplerConfig(capacity=8, prefill=4, sample_size=4)
    sampler = BufferSampler(
        config,
        element_spec={"feature": jax.ShapeDtypeStruct(shape=(3,), dtype=jnp.float32)},
        rngs=nnx.Rngs(0),
    )

    spec = sampler.index_spec()

    assert "feature" in spec
    assert spec["feature"].shape == (4, 3)
    assert spec["feature"].dtype == jnp.float32


def test_buffer_next_is_jit_compatible() -> None:
    """The full ``next(value, mask)`` body compiles under ``jax.jit``.

    This is the load-bearing JIT contract — write/read mode branching uses
    ``jax.lax.cond`` so the entire ``next`` body is JIT-traceable.
    """
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, read_mode="sequential", write_mode="fifo"
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    @nnx.jit
    def jitted_next(buffer_sampler: BufferSampler, value: dict) -> tuple:
        return buffer_sampler.next(value, mask=jnp.array(True))

    # Should not raise during tracing.
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        chunk, mask = jitted_next(sampler, _scalar(v))

    # Final state still consistent.
    assert sampler.count[...] == 4


def test_buffer_invalid_mask_skips_write() -> None:
    """When the input mask is False, the write is skipped entirely (state unchanged)."""
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, read_mode="sequential", write_mode="fifo"
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    sampler.next(_scalar(1.0), mask=jnp.array(True))
    sampler.next(_scalar(2.0), mask=jnp.array(False))  # should NOT advance

    # Only one valid write happened; count stays at 1.
    assert int(sampler.count[...]) == 1


def test_buffer_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        BufferSamplerConfig(capacity=0, prefill=1, sample_size=1)


def test_buffer_rejects_prefill_below_sample_size() -> None:
    """Invariant: ``prefill >= sample_size`` (warmup must cover one full sample)."""
    with pytest.raises(ValueError, match="prefill"):
        BufferSamplerConfig(capacity=10, prefill=2, sample_size=5)


def test_buffer_rejects_sample_size_above_capacity() -> None:
    """Invariant: ``sample_size <= capacity`` (cannot draw more than the buffer holds)."""
    with pytest.raises(ValueError, match="sample_size"):
        BufferSamplerConfig(capacity=4, prefill=4, sample_size=8)
