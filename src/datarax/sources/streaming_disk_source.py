"""Out-of-core streaming source backed by a memory-mapped numpy array.

Reads a contiguous numpy array (``.npy``) from disk via ``np.memmap`` for
zero-copy random access, exposes a JAX-friendly ``read_batch(indices)``
that issues the disk read through ``jax.experimental.io_callback``, and
wraps the result in ``jax.lax.stop_gradient`` so downstream gradient
computations cannot attempt to backprop through the disk read.

Why ``stop_gradient`` is mandatory
----------------------------------

``jax.experimental.io_callback`` raises on both JVP and transpose rules — its
output is non-differentiable by design. Without an explicit
``stop_gradient`` boundary, any downstream ``jax.grad`` over a function that
reads from this source would fail with a JVP error. With the boundary,
gradients on data-derived computations exist but are zero (the documented
"non-differentiable input" semantics).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule


@dataclass(frozen=True)
class StreamingDiskSourceConfig(StructuralConfig):
    """Configuration for ``StreamingDiskSource``.

    Attributes:
        path: Filesystem path to a ``.npy`` file with leading dataset-size axis.
        feature_key: Key under which the read array is exposed in returned dicts.
    """

    path: str = ""
    feature_key: str = "x"

    def __post_init__(self) -> None:
        """Validate the config."""
        super().__post_init__()
        if not self.path:
            raise ValueError("StreamingDiskSourceConfig.path must be set.")
        if not self.feature_key:
            raise ValueError("StreamingDiskSourceConfig.feature_key must be non-empty.")


class StreamingDiskSource(DataSourceModule):
    """``io_callback``-backed streaming source for arrays larger than RAM."""

    config: StreamingDiskSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: StreamingDiskSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Open the on-disk array as a memory-map and pre-compute the element spec."""
        super().__init__(config, rngs=rngs, name=name)

        path = Path(config.path)
        if not path.exists():
            raise FileNotFoundError(f"StreamingDiskSource: file not found at {path}.")

        # Open as memory-map so we don't load the entire array into RAM.
        memmap = np.load(path, mmap_mode="r")
        if not hasattr(memmap, "shape") or memmap.ndim < 1:
            raise ValueError(
                f"StreamingDiskSource expects a 1-D-or-higher numpy array; "
                f"got shape {getattr(memmap, 'shape', None)!r}."
            )

        # Static metadata captured at construction.
        self._memmap = nnx.data(memmap)
        self._length = nnx.static(int(memmap.shape[0]))
        self._feature_key = nnx.static(config.feature_key)
        self._element_shape = nnx.static(tuple(int(d) for d in memmap.shape[1:]))
        self._element_dtype = nnx.static(jnp.dtype(memmap.dtype))

    def __len__(self) -> int:
        """Total number of records on disk."""
        return self._length

    def element_spec(self) -> Any:
        """Return per-element spec — leading dataset axis stripped."""
        leaf = jax.ShapeDtypeStruct(shape=self._element_shape, dtype=self._element_dtype)
        return {self._feature_key: leaf}

    def read_batch(self, indices: jax.Array) -> dict[str, jax.Array]:
        """Fetch the rows at ``indices`` from disk and return a stop_gradient'd dict.

        Args:
            indices: 1-D ``jax.Array`` of int32 indices into the on-disk array.

        Returns:
            ``{feature_key: array}`` where ``array`` has shape
            ``(len(indices), *element_shape)`` and is wrapped with
            ``jax.lax.stop_gradient``.
        """
        result_spec = jax.ShapeDtypeStruct(
            shape=(indices.shape[0], *self._element_shape),
            dtype=self._element_dtype,
        )

        def _host_read(idx_array: np.ndarray) -> np.ndarray:
            # ``idx_array`` arrives as a numpy array on the host. Index the
            # memory-map and copy to a contiguous numpy array (memmap rows are
            # already contiguous; the np.asarray ensures owned memory).
            return np.asarray(self._memmap[idx_array.astype(np.int64)])

        raw = io_callback(_host_read, result_spec, indices)
        # io_callback outputs are non-differentiable by design — make that
        # explicit so downstream gradient code returns zero through the
        # boundary instead of raising a JVP error.
        return {self._feature_key: jax.lax.stop_gradient(raw)}

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Stateless indexed batch access for ``Pipeline``-driven iteration.

        Returns ``size`` records starting at logical position ``start``,
        wrapping at the end of the on-disk array. The actual disk read
        is dispatched via :meth:`read_batch` (which uses
        ``jax.experimental.io_callback`` with a ``stop_gradient`` boundary).

        Random-order shuffling is not yet implemented for the streaming
        disk source — a future enhancement could materialize the
        permutation host-side; the ``key`` argument is currently ignored.

        Args:
            start: Starting logical index; concrete int or traced ``jax.Array``.
            size: Number of records (Python int).
            key: Reserved for future shuffled-mode support; currently unused.

        Returns:
            ``{feature_key: array}`` with leading dim ``size`` and
            ``stop_gradient`` applied at the io_callback boundary.
        """
        del key  # streaming-disk shuffling deferred to a future enhancement

        start_arr = jnp.asarray(start, dtype=jnp.int32)
        offsets = jnp.arange(size, dtype=jnp.int32)
        indices = (start_arr + offsets) % jnp.int32(self._length)
        return self.read_batch(indices)


__all__ = ["StreamingDiskSource", "StreamingDiskSourceConfig"]
