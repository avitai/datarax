"""Sliding-window sampler emitting windows of consecutive indices.

For time-series workloads (Neural CDE / ODE, audio frame extraction, sensor
fusion), each sample is a contiguous window of timesteps. Rather than emitting
single indices and rebuilding windows downstream, this sampler emits the full
``(window_size,)`` index array per call so the dataset gather is a single
``jnp.take`` and the spec contract (``index_spec``) describes the windowed
shape directly.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlidingWindowSamplerConfig(StructuralConfig):
    """Configuration for ``SlidingWindowSampler``.

    Attributes:
        num_records: Total length of the underlying sequence (required).
        window_size: Number of contiguous indices emitted per window.
        stride: Step between successive window start positions.
        drop_incomplete: If True, the final partial window is dropped.
            If False, the final window is padded by clamping out-of-range
            indices to the last valid index (``num_records - 1``).
        num_epochs: Number of epochs to iterate (``-1`` for infinite).
    """

    num_records: int | None = None
    window_size: int = 1
    stride: int = 1
    drop_incomplete: bool = True
    num_epochs: int = 1

    def __post_init__(self) -> None:
        """Validate window/stride invariants."""
        super().__post_init__()
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}.")
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}.")
        if self.num_records is not None and self.num_records < self.window_size:
            raise ValueError(
                f"num_records ({self.num_records}) must be >= window_size ({self.window_size})."
            )


class SlidingWindowSampler(SamplerModule):
    """Emit windows of ``window_size`` consecutive indices at ``stride`` step.

    Each call to ``__next__`` returns a ``jax.Array`` of shape
    ``(window_size,)`` containing the indices for one window. The final
    partial window is either dropped or clamp-padded depending on
    ``config.drop_incomplete``.
    """

    config: SlidingWindowSamplerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: SlidingWindowSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize sliding-window sampler with config."""
        super().__init__(config, rngs=rngs, name=name)

        if config.num_records is None:
            raise ValueError("SlidingWindowSamplerConfig requires num_records.")

        self._num_records = nnx.static(int(config.num_records))
        self._window_size = nnx.static(int(config.window_size))
        self._stride = nnx.static(int(config.stride))
        self._drop_incomplete = nnx.static(bool(config.drop_incomplete))
        self._num_epochs = nnx.static(int(config.num_epochs))

        self._total_windows = nnx.static(self._compute_total_windows())

        # Mutable iteration state.
        self.window_index = nnx.Variable(0)
        self.current_epoch = nnx.Variable(0)

    def _compute_total_windows(self) -> int:
        """Number of windows emitted per epoch."""
        n = int(self.config.num_records or 0)
        w = int(self.config.window_size)
        s = int(self.config.stride)
        if n < w:
            return 0
        last_start_full = n - w  # inclusive
        full_windows = last_start_full // s + 1
        if self.config.drop_incomplete:
            return full_windows
        # When padding, include any window whose start is < n.
        last_start_any = (n - 1) // s
        return last_start_any + 1

    def __iter__(self) -> Iterator[jax.Array]:
        """Reset window/epoch counters and return self as an iterator."""
        self.window_index.set_value(0)
        self.current_epoch.set_value(0)
        return self

    def __next__(self) -> jax.Array:
        """Return the next ``(window_size,)`` window of indices."""
        idx = int(self.window_index.get_value())
        epoch = int(self.current_epoch.get_value())

        if idx >= self._total_windows:
            epoch += 1
            if self._num_epochs != -1 and epoch >= self._num_epochs:
                raise StopIteration
            self.current_epoch.set_value(epoch)
            self.window_index.set_value(0)
            idx = 0

        start = idx * self._stride
        # Build window indices, clamping to the last valid index when padding
        # the final partial window.
        offsets = jnp.arange(self._window_size, dtype=jnp.int32)
        last_valid = self._num_records - 1
        window = jnp.minimum(start + offsets, last_valid)

        self.window_index.set_value(idx + 1)
        return window

    def index_spec(self) -> Any:
        """Override base default to declare the windowed output shape."""
        return jax.ShapeDtypeStruct(shape=(self._window_size,), dtype=jnp.int32)

    def __len__(self) -> int:
        """Total windows per epoch (does not multiply by ``num_epochs``)."""
        return int(self._total_windows)


__all__ = ["SlidingWindowSampler", "SlidingWindowSamplerConfig"]
