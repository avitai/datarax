"""Replay buffer / reservoir sampler with JIT-traceable atomic write+sample.

Stores full PyTree records (not indices) of shape ``(capacity, *leaf.shape)``
per leaf so the buffer can replay actual sample data. ``next(value, mask)``
performs a single atomic step: write ``value`` into the buffer per
``write_mode``, then return ``sample_size`` records per ``read_mode``.

The two write modes (``fifo``, ``reservoir``) and two read modes
(``sequential``, ``shuffled``) are selected via ``jax.lax.cond`` so the entire
``next`` body is JIT-traceable. Reservoir uses the single-draw acceptance
form: ``rand_idx = randint(0, max(seen, 1))``, write iff ``rand_idx < capacity``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


logger = logging.getLogger(__name__)


_WRITE_MODES = ("fifo", "reservoir")
_READ_MODES = ("sequential", "shuffled")


@dataclass(frozen=True)
class BufferSamplerConfig(StructuralConfig):
    """Configuration for ``BufferSampler``.

    Attributes:
        capacity: Maximum samples retained in the buffer.
        prefill: Number of writes required before reads return buffered samples.
            Until this threshold, ``next`` passes the input through directly
            as a length-1 chunk (warmup mode).
        sample_size: Number of samples returned per ``next`` call.
        read_mode: ``"sequential"`` (round-robin) or ``"shuffled"`` (random).
        write_mode: ``"fifo"`` (ring buffer) or ``"reservoir"`` (uniform replacement).
        seed: Integer seed used to derive the buffer's RNG key.
    """

    capacity: int = 1
    prefill: int = 1
    sample_size: int = 1
    read_mode: Literal["sequential", "shuffled"] = "sequential"
    write_mode: Literal["fifo", "reservoir"] = "fifo"
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate buffer-sampler invariants."""
        super().__post_init__()
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}.")
        if self.prefill <= 0:
            raise ValueError(f"prefill must be positive, got {self.prefill}.")
        if self.prefill > self.capacity:
            raise ValueError("prefill cannot exceed capacity.")
        if self.sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {self.sample_size}.")
        if self.sample_size > self.capacity:
            raise ValueError("sample_size cannot exceed capacity.")
        if self.prefill < self.sample_size:
            raise ValueError("prefill must be at least sample_size.")
        if self.write_mode not in _WRITE_MODES:
            raise ValueError(f"write_mode must be one of {_WRITE_MODES}, got {self.write_mode!r}.")
        if self.read_mode not in _READ_MODES:
            raise ValueError(f"read_mode must be one of {_READ_MODES}, got {self.read_mode!r}.")


def _write_buffer(buffer: Any, value: Any, index: jax.Array) -> Any:
    """Write ``value`` into ``buffer`` at the given index along axis 0."""

    def _update(buf, val):
        val_expanded = jnp.expand_dims(val, axis=0)
        return jax.lax.dynamic_update_index_in_dim(buf, val_expanded, index, axis=0)

    return jax.tree.map(_update, buffer, value)


def _gather_many(buffer: Any, indices: jax.Array) -> Any:
    """Gather ``len(indices)`` records from ``buffer`` along axis 0."""

    def gather_leaf(buf):
        return jax.vmap(lambda idx: jax.lax.dynamic_index_in_dim(buf, idx, axis=0, keepdims=False))(
            indices
        )

    return jax.tree.map(gather_leaf, buffer)


class BufferSampler(SamplerModule):
    """Stateful replay buffer with atomic write+sample per ``next`` call."""

    config: BufferSamplerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: BufferSamplerConfig,
        *,
        element_spec: Any,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize buffer state from a per-element spec.

        Args:
            config: Sampler configuration.
            element_spec: PyTree of ``jax.ShapeDtypeStruct`` describing one
                stored record. The buffer is allocated as a PyTree of zeros
                shaped ``(capacity, *leaf.shape)``.
            rngs: Optional NNX RNGs (currently unused; the sampler derives its
                own key from ``config.seed``).
            name: Optional module name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self._capacity = nnx.static(int(config.capacity))
        self._prefill = nnx.static(int(config.prefill))
        self._sample_size = nnx.static(int(config.sample_size))
        self._write_mode_is_fifo = nnx.static(config.write_mode == "fifo")
        self._read_mode_is_sequential = nnx.static(config.read_mode == "sequential")
        self._element_spec = nnx.static(element_spec)

        # Pre-allocated buffer template (record-shaped, not index-shaped):
        # one zero-filled array per spec leaf, with leading dim ``capacity``.
        self.buffer = nnx.Variable(
            jax.tree.map(
                lambda leaf: jnp.zeros((self._capacity, *leaf.shape), dtype=leaf.dtype),
                element_spec,
            )
        )
        self.count = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.write_index = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.read_index = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.seen = nnx.Variable(jnp.array(0, dtype=jnp.int32))
        self.rng = nnx.Variable(jax.random.PRNGKey(int(config.seed)))

    # ------------------------------------------------------------------
    # Atomic next: write + sample
    # ------------------------------------------------------------------

    def _apply_write(
        self,
        *,
        buffer_in: Any,
        value: Any,
        mask_scalar: jax.Array,
        write_idx_in: jax.Array,
        count_in: jax.Array,
        new_seen: jax.Array,
        write_key: jax.Array,
        one: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array]:
        """Resolve write decision (skip / FIFO / reservoir) and return the post-write buffer.

        Extracted from :meth:`next` to keep its complexity below the
        SWE-policy threshold; semantics are unchanged.
        """

        def _fifo(_: None):
            buffer = _write_buffer(buffer_in, value, write_idx_in)
            next_write = (write_idx_in + one) % self._capacity
            return buffer, next_write, jnp.array(True)

        def _reservoir(_: None):
            def _fill(_: None):
                buffer = _write_buffer(buffer_in, value, write_idx_in)
                next_write = (write_idx_in + one) % self._capacity
                return buffer, next_write, jnp.array(True)

            def _replace(_: None):
                maxval = jnp.maximum(new_seen, one)
                rand_idx = jax.random.randint(write_key, (), minval=0, maxval=maxval)

                def _commit(_: None):
                    buffer = _write_buffer(buffer_in, value, rand_idx)
                    return buffer, write_idx_in, jnp.array(True)

                def _drop(_: None):
                    return buffer_in, write_idx_in, jnp.array(False)

                return jax.lax.cond(rand_idx < self._capacity, _commit, _drop, operand=None)

            return jax.lax.cond(count_in < self._capacity, _fill, _replace, operand=None)

        def _write(_: None):
            return jax.lax.cond(self._write_mode_is_fifo, _fifo, _reservoir, operand=None)

        def _skip(_: None):
            return buffer_in, write_idx_in, jnp.array(False)

        return jax.lax.cond(mask_scalar, _write, _skip, operand=None)

    def _read_or_passthrough(
        self,
        *,
        buffer_ready: jax.Array,
        updated_buffer: Any,
        value: Any,
        mask_scalar: jax.Array,
        new_count: jax.Array,
        read_idx_in: jax.Array,
        sample_key: jax.Array,
        one: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array]:
        """Read from the buffer once warm, otherwise return a single-slot passthrough.

        Extracted from :meth:`next` to keep its complexity below the
        SWE-policy threshold; semantics are unchanged.
        """

        def _sequential(_: None):
            idxs = (read_idx_in + jnp.arange(self._sample_size, dtype=jnp.int32)) % (
                jnp.maximum(new_count, one)
            )
            chunk = _gather_many(updated_buffer, idxs)
            mask_vec = jnp.ones(self._sample_size, dtype=jnp.bool_)
            next_read = (read_idx_in + jnp.int32(self._sample_size)) % jnp.maximum(new_count, one)
            return chunk, mask_vec, next_read

        def _shuffled(_: None):
            idxs = jax.random.randint(
                sample_key,
                (self._sample_size,),
                minval=0,
                maxval=jnp.maximum(new_count, one),
            )
            chunk = _gather_many(updated_buffer, idxs)
            mask_vec = jnp.ones(self._sample_size, dtype=jnp.bool_)
            return chunk, mask_vec, read_idx_in

        def _from_buffer(_: None):
            return jax.lax.cond(self._read_mode_is_sequential, _sequential, _shuffled, operand=None)

        def _passthrough(_: None):
            chunk = jax.tree.map(
                lambda leaf, val: jnp.zeros((self._sample_size, *leaf.shape), dtype=leaf.dtype)
                .at[0]
                .set(val),
                self._element_spec,
                value,
            )
            mask_vec = jnp.zeros(self._sample_size, dtype=jnp.bool_).at[0].set(mask_scalar)
            return chunk, mask_vec, read_idx_in

        return jax.lax.cond(buffer_ready, _from_buffer, _passthrough, operand=None)

    def next(self, value: Any, mask: jax.Array | bool = True) -> tuple[Any, jax.Array]:
        """Write ``value`` then return ``sample_size`` records from the buffer.

        Returns ``(chunk, mask)`` where ``chunk`` is a PyTree shaped
        ``(sample_size, *leaf.shape)`` per leaf and ``mask`` flags valid
        positions in the chunk (all-True post-warmup; pre-warmup is the
        passthrough of the input mask).
        """
        mask_scalar = jnp.all(jnp.asarray(mask, dtype=jnp.bool_))
        one = jnp.array(1, dtype=jnp.int32)
        zero = jnp.array(0, dtype=jnp.int32)
        increment = jnp.where(mask_scalar, one, zero)

        rng, write_key, sample_key = jax.random.split(self.rng[...], 3)
        new_seen = self.seen[...] + increment

        # buffer is a PyTree (often a dict) so use get_value() rather than
        # [...] which is reserved for Variable[Array].
        buffer_in = self.buffer.get_value()
        count_in = self.count[...]
        write_idx_in = self.write_index[...]
        read_idx_in = self.read_index[...]

        updated_buffer, new_write, wrote = self._apply_write(
            buffer_in=buffer_in,
            value=value,
            mask_scalar=mask_scalar,
            write_idx_in=write_idx_in,
            count_in=count_in,
            new_seen=new_seen,
            write_key=write_key,
            one=one,
        )

        new_count = jnp.minimum(
            count_in + jnp.where(wrote, one, zero),
            jnp.array(self._capacity, dtype=jnp.int32),
        )
        buffer_ready = count_in >= self._prefill

        chunk, mask_vec, next_read = self._read_or_passthrough(
            buffer_ready=buffer_ready,
            updated_buffer=updated_buffer,
            value=value,
            mask_scalar=mask_scalar,
            new_count=new_count,
            read_idx_in=read_idx_in,
            sample_key=sample_key,
            one=one,
        )

        # Persist updated state.
        self.buffer.set_value(updated_buffer)
        self.count.set_value(new_count)
        self.write_index.set_value(new_write)
        self.read_index.set_value(next_read)
        self.seen.set_value(new_seen)
        self.rng.set_value(rng)

        return chunk, mask_vec

    def index_spec(self) -> Any:
        """Return chunk-shaped spec — `(sample_size, *leaf.shape)` per leaf."""
        return jax.tree.map(
            lambda leaf: jax.ShapeDtypeStruct(
                shape=(self._sample_size, *leaf.shape), dtype=leaf.dtype
            ),
            self._element_spec,
        )


__all__ = ["BufferSampler", "BufferSamplerConfig"]
