"""Out-of-core indexed source backed by a memory-mapped numpy array.

Reads a contiguous numpy array (``.npy``) from disk via ``np.memmap`` for
zero-copy random access. Two reads serve it:

- the host read, ``get_batch(indices, *, epochs=0, contiguous=False)``, the eager sources' read
  (:func:`~datarax.sources.eager_source.read_host_batch`): a NumPy gather of the named rows of the
  memory map (a contiguous run as a view of it), returned as a ``Batch`` named with the given
  words and epochs, creating no device array;
- the traced read, ``get_records(indices)``, which issues the disk read through
  ``jax.experimental.io_callback`` for the compiled pipeline session, and wraps the result in
  ``jax.lax.stop_gradient`` so downstream gradient computations cannot attempt to backprop
  through the disk read.

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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import io_callback
from jax.typing import ArrayLike

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.sources.eager_source import read_host_batch
from datarax.typing import DataDict


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


class _HostArray:
    """A host-side array handle that NNX keeps out of module state.

    A memory-map stored as ``nnx.data`` would become a traced input of every
    compiled step, and the host callback would then close over a tracer. As a
    plain object it stays static and compares equal only to itself.
    """

    __slots__ = ("array",)

    def __init__(self, array: np.ndarray) -> None:
        self.array = array


class StreamingDiskSource(DataSourceModule):
    """An indexed source over an on-disk array larger than RAM, read through a memory map.

    ``get_batch`` reads rows on the host; ``get_records`` reads them inside a compiled program
    through ``io_callback``.
    """

    config: StreamingDiskSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def record_identity(self) -> RecordIdentity:
        """A record's index is its row in the on-disk array."""
        return RecordIdentity.INDEXED

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

        # The memory-map stays on the host, outside NNX state; the rest is static metadata.
        self._host = _HostArray(memmap)
        self._length = nnx.static(int(memmap.shape[0]))
        self._feature_key = nnx.static(config.feature_key)
        self._element_shape = nnx.static(tuple(int(d) for d in memmap.shape[1:]))
        self._element_dtype = nnx.static(jnp.dtype(memmap.dtype))

    def __len__(self) -> int:
        """Total number of records on disk."""
        return self._length

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Includes the on-disk path, feature key, record count, and element
        shape/dtype so a restore can detect a source pointed at different data.
        """
        return (
            f"StreamingDiskSource(path={self.config.path!r}, "
            f"feature_key={self._feature_key!r}, "
            f"length={self._length}, "
            f"element_shape={self._element_shape}, "
            f"element_dtype={self._element_dtype!s})"
        )

    def element_spec(self) -> Any:
        """Return per-element spec — leading dataset axis stripped."""
        leaf = jax.ShapeDtypeStruct(shape=self._element_shape, dtype=self._element_dtype)
        return {self._feature_key: leaf}

    def get_batch(  # noqa: DOC502 - read_host_batch raises
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the rows ``indices`` names from the memory map, as a ``Batch``, on the host.

        The eager sources' host read (:func:`~datarax.sources.eager_source.read_host_batch`): one
        NumPy gather of the named rows, the ``Batch`` named with the given words and epochs, no
        state read or changed and no device array created. With ``contiguous=True`` the caller
        states that ``indices`` is a run of consecutive rows, read as a view of the memory map.

        Args:
            indices: uint32 ``(n, 2)`` row indices, each as its words ``(hi, lo)``.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive rows.

        Returns:
            ``{feature_key: rows}`` as a host ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, or a run declared
                contiguous is not one.
            IndexError: If an index is the padding index or outside the array.
        """
        return read_host_batch(
            {self._feature_key: self._host.array},
            self._length,
            indices,
            epochs=epochs,
            contiguous=contiguous,
        )

    def get_records(self, indices: jax.Array) -> DataDict:
        """Fetch the rows at ``indices`` from disk and return a stop_gradient'd dict.

        Args:
            indices: uint32 ``(n, 2)`` indices into the on-disk array, each as its words
                ``(hi, lo)``, so every row of an array past ``2**32`` rows is addressed.

        Returns:
            ``{feature_key: array}`` where ``array`` has shape
            ``(len(indices), *element_shape)`` and is wrapped with
            ``jax.lax.stop_gradient``.
        """
        result_spec = jax.ShapeDtypeStruct(
            shape=(indices.shape[0], *self._element_shape),
            dtype=self._element_dtype,
        )

        host = self._host

        def _host_read(idx_array: np.ndarray) -> np.ndarray:
            # ``idx_array`` arrives as a numpy array on the host. Index the
            # memory-map and copy to a contiguous numpy array (memmap rows are
            # already contiguous; the np.asarray ensures owned memory).
            return np.asarray(host.array[from_words(idx_array)])

        raw = io_callback(_host_read, result_spec, indices)
        # io_callback outputs are non-differentiable by design — make that
        # explicit so downstream gradient code returns zero through the
        # boundary instead of raising a JVP error.
        return {self._feature_key: jax.lax.stop_gradient(raw)}


__all__ = ["StreamingDiskSource", "StreamingDiskSourceConfig"]
