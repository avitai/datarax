"""Base module for data sources in Datarax.

This module defines the base class for all Datarax data source components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datarax.core.structural import StructuralModule
from datarax.typing import Element


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
    """

    def __iter__(self) -> Iterator[Element]:
        """Return an iterator over individual data elements.

        Returns:
            An iterator that yields data elements as PyTrees.
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def __next__(self) -> Element:
        """Get the next element from this data source.

        Returns:
            The next data element as a PyTree.

        Raises:
            StopIteration: When there are no more elements to yield.
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

    def __getitem__(self, idx: int) -> Element | None:
        """Get element by index.

        This method provides subscriptable access to data elements.
        Subclasses should override this method if they support random access.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The data element at the given index, or None if not implemented.
        """
        return None

    def get_batch_at(
        self,
        start: int | Any,
        size: int,
        key: Any | None = None,
    ) -> Any:
        """Stateless indexed batch access for ``Pipeline``-driven iteration.

        Returns ``size`` records starting at ``start``. Implementations must
        be stateless (no mutation of internal counters) and JAX-traceable
        (must accept tracer values for ``start``) so the call composes with
        ``nnx.scan``.

        Args:
            start: Starting index. Sources that support indexed access
                accept a Python int or a traced ``jax.Array``.
            size: Number of records to return (Python int — JAX shapes are
                static).
            key: Optional PRNG key for shuffled or stochastic sampling.

        Returns:
            A batch dict (or PyTree) with leading dim ``size``.

        Raises:
            NotImplementedError: If the source does not support indexed
                access (e.g. forward-only streams). Pipeline falls back to
                its ``__iter__`` debug path in that case.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support indexed batch access. "
            f"Implement get_batch_at(start, size, key) or use the Pipeline "
            f"iterator path (for batch in pipeline)."
        )

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
