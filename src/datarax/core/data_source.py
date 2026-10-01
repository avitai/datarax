"""Base module for data sources in Datarax.

This module defines the base class for all Datarax data source components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

import abc
import enum
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.index_shuffle import shuffle_positions
from datarax.core.index_words import wrapped_positions
from datarax.core.structural import StructuralModule
from datarax.typing import DataDict


logger = logging.getLogger(__name__)


class LocalFilesOnlyMixin:
    """Adds a uniform ``local_files_only`` flag to data sources.

    Sources that download external archives (HuggingFace, TFDS, ArrayRecord,
    etc.) compose this mixin and call ``_check_local_cache`` before any
    network attempt. The check enforces the air-gapped contract: when
    ``local_files_only=True`` and the cache is missing, the source raises a
    ``FileNotFoundError`` whose message names the dataset and the exact paths
    the user must populate, instead of a generic "file not found".

    Subclasses must define ``self.local_files_only: bool`` (typically wired
    through their config dataclass).
    """

    local_files_only: bool

    def _check_local_cache(
        self,
        expected_paths: Sequence[Path],
        *,
        dataset_name: str,
    ) -> None:
        """Raise if ``local_files_only`` is set but the cache is missing.

        Args:
            expected_paths: Files whose presence indicates a populated cache.
            dataset_name: Human-readable name of the dataset (included in the
                error message so users know which source raised).

        Raises:
            FileNotFoundError: If ``local_files_only`` is True and any
                ``expected_paths`` entry does not exist.
        """
        if not self.local_files_only:
            return
        missing = [str(p.resolve()) for p in expected_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{dataset_name}: local_files_only=True but the local cache is "
                f"incomplete. Expected the following file(s) to exist: {missing}. "
                "Populate the cache offline or set local_files_only=False to "
                "allow the source to download."
            )


class RecordIdentity(enum.Enum):
    """What a source's record index means: the kind of identity its records carry.

    Every source declares one (:attr:`DataSourceModule.record_identity`); the pipeline serves an
    ``INDEXED`` source through its compiled session and any other through its streaming path.
    """

    INDEXED = "indexed"
    """The record's stable position in the source, after the epoch's shuffle: the pipeline turns
    (epoch, position) into it with ``record_indices_at``. Unique, stable across passes and runs,
    and global across workers; its epoch is the pipeline's pass counter."""

    STREAM_IDS = "stream_ids"
    """The id the stream reports for the record (a shard and an offset, in the two words).
    Unique, stable and global; its epoch is the source's pass counter."""

    ARRIVAL = "arrival"
    """The record's arrival ordinal in the run, never reset, so never repeated. Unique only: the
    record's provenance travels beside the batch, and a table keyed by record is refused."""


class DataSourceModule(StructuralModule):
    """Enhanced base module for all Datarax data source components.

    This class extends StructuralModule for non-parametric structural data loading.
    Concrete data sources define their own config classes extending StructuralConfig.

    A DataSourceModule is responsible for reading data from an external source
    (e.g., files, memory, network) and yielding data elements as PyTrees. Each
    data element is typically a dictionary or other PyTree structure containing
    JAX arrays or Python primitives.

    **Important**: When subclassing, if you store data containing JAX Arrays in
    an attribute (like `self.data`), wrap the assigned value with `nnx.Param`
    or assignment-time `nnx.data(value)`:

    Examples:
        ```python
        @dataclass(frozen=True)
        class MyDataSourceConfig(StructuralConfig):
            required_param: int | None = None
            def __post_init__(self):
                super().__post_init__()
                if self.required_param is None:
                    raise ValueError("required_param is required")
        class MyDataSource(DataSourceModule):
            data: list[dict]
            def __init__(self, config: MyDataSourceConfig, data: list[dict], *,
                         rngs: nnx.Rngs | None = None, name: str | None = None):
                super().__init__(config, rngs=rngs, name=name)
                self.data = nnx.data(data)  # Mark as pytree data, not parameters.
        ```

    This prevents NNX from trying to track individual JAX Arrays within the data
    structure as trainable parameters.

    Every source declares what its record index means, :attr:`record_identity`; a subclass that
    does not is refused at construction.
    """

    @property
    @abc.abstractmethod
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means (see :class:`RecordIdentity`).

        ``INDEXED`` sources implement ``get_records`` and ``record_indices_at`` and are served by
        the pipeline's compiled session; ``STREAM_IDS`` and ``ARRIVAL`` sources pull forward with
        ``get_batch(batch_size)`` and are served by its streaming path. A subclass declares it
        with a property returning its kind.
        """

    def __iter__(self) -> Iterator[PyTree]:
        """Return an iterator over individual data elements.

        Returns:
            An iterator that yields data elements as PyTrees.

        Raises:
            NotImplementedError: If a subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def __next__(self) -> PyTree:  # noqa: DOC503
        """Get the next element from this data source.

        Returns:
            The next data element as a PyTree.

        Raises:
            StopIteration: When there are no more elements to yield.
            NotImplementedError: If a subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement __next__")

    def __len__(self) -> int:
        """Return the total number of data elements.

        Implementing this method allows downstream components to know the
        dataset size in advance, which can be useful for progress tracking
        or specific sampling strategies.

        Returns:
            The total number of data elements in the source.

        Raises:
            NotImplementedError: If the source cannot determine its length.
        """
        raise NotImplementedError("This DataSourceModule does not support length determination.")

    def __getitem__(self, idx: int) -> PyTree | None:
        """Get element by index.

        This method provides subscriptable access to data elements.
        Subclasses should override this method if they support random access.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The data element at the given index, or None if not implemented.
        """
        return None

    def get_records(self, indices: jax.Array) -> DataDict:
        """Gather the records at ``indices``: indexed access for ``Pipeline``-driven iteration.

        ``indices`` are the stable record indices :meth:`record_indices_at` names. The pipeline
        computes a batch's indices once and passes the same ones here and to the stages that key
        randomness on them. Implementations must be stateless (no mutation of internal counters)
        and JAX-traceable (``indices`` may be a traced array) so the call composes with
        ``nnx.jit`` and ``nnx.scan``.

        Args:
            indices: uint32 ``(n, 2)`` record indices in ``[0, len(self))``, each as its words
                ``(hi, lo)``.

        Returns:
            One array per field, with leading dim ``n``.

        Raises:
            NotImplementedError: If the source does not implement it: an ``INDEXED`` source
                must, a stream (``STREAM_IDS`` or ``ARRIVAL``) is pulled with ``get_batch``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_records(indices), which an INDEXED "
            "source serves its records with; a stream declares STREAM_IDS or ARRIVAL and "
            "implements get_batch(batch_size)."
        )

    def record_indices_at(
        self,
        start: int | Any,
        size: int,
        key: Any | None = None,
    ) -> Any:
        """Return the stable index of each record at positions ``start .. start + size``.

        These are the indices :meth:`get_records` gathers. Stochastic operators key each
        record's randomness on them, so within an epoch a record keeps its augmentation however
        records are batched, ordered or split across workers. The order is the one ``key``
        selects, or the sequential order when ``key`` is ``None``: the pipeline owns the shuffle
        and passes its epoch key exactly when it was built with ``shuffle=True``. The default
        names records by their wrapped position ``(start + arange(size)) % len(self)``, and with
        a key by the keyed permutation of those positions
        (:func:`~datarax.core.index_shuffle.shuffle_positions`). A source that partitions
        or mixes records overrides it. Indices are 64-bit, each a uint32 ``(hi, lo)`` pair, the
        layout of ``Batch.indices``.

        Args:
            start: Starting position; a Python int of any size or a traced int32 ``jax.Array``.
            size: Number of records (Python int).
            key: The key selecting the order, or ``None`` for the sequential order.

        Returns:
            uint32 array of shape ``(size, 2)``.

        Raises:
            ValueError: If a key is given to a source without a length, which has no order to
                shuffle.
        """
        try:
            length: int | None = len(self)
        except NotImplementedError:
            length = None
        if key is None:
            return wrapped_positions(start, size, length)
        if length is None:
            raise ValueError(
                f"{type(self).__name__} has no length, so it has no order to shuffle; build its "
                "pipeline with shuffle=False"
            )
        return shuffle_positions(wrapped_positions(start, size, length), length, key)

    def element_spec(self) -> Any:
        """Return a PyTree of ``jax.ShapeDtypeStruct`` describing per-element output.

        Downstream consumers (operators, batchers, models) use this contract to
        pre-allocate buffers, auto-size learnable layers, and statically validate
        operator chains. Subclasses MUST override this method.

        Returns:
            A PyTree (typically a dict) whose leaves are ``jax.ShapeDtypeStruct``
            instances describing one emitted element.

        Raises:
            NotImplementedError: Always, on the base class. Subclasses must
                override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement element_spec(). "
            "Return a PyTree of jax.ShapeDtypeStruct describing one emitted element."
        )
