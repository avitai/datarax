"""Data source for reading from ArrayRecord format files."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import flax.nnx as nnx
import grain
import grain.sources
import numpy as np

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._grain_bridge import validate_index_batch
from datarax.utils.state import build_state_with_iteration_fields, restore_iteration_fields


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArrayRecordSourceConfig(StructuralConfig):
    """Configuration for ArrayRecordSourceModule.

    Inherits from StructuralConfig for runtime immutability.

    Attributes:
        seed: Random seed for shuffling (used internally, not by Grain).
        num_epochs: Number of epochs (-1 for infinite).
        shuffle_files: Whether to shuffle file order (handled internally).
        local_files_only: If True, validate every path exists at construction
            time and raise ``FileNotFoundError`` with path context if any are
            missing. ArrayRecord sources never download, so this flag is
            primarily a UX improvement over Grain's lower-level errors.
    """

    seed: int = 42
    num_epochs: int = -1
    shuffle_files: bool = False
    local_files_only: bool = False


class ArrayRecordSourceModule(DataSourceModule):
    """Stateful wrapper for Grain's ArrayRecordDataSource.

    This module wraps Grain's ArrayRecordDataSource while maintaining
    stateful iteration through NNX Variables, following TDD principles
    and critical technical guidelines.

    Note: Grain's ArrayRecordDataSource doesn't accept a seed parameter directly.
    Shuffling is handled at the sampler level or through file ordering.
    """

    # Narrow config type for pyright (base stores via nnx.static)
    config: ArrayRecordSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: ArrayRecordSourceConfig,
        paths: str | list[str],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize ArrayRecord source with state management.

        Args:
            config: Configuration for the source.
            paths: Path pattern or list of paths to ArrayRecord files.
            rngs: NNX Rngs for additional randomness.
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)

        # When local_files_only is set, fail fast with a clear message instead
        # of letting Grain raise its lower-level error on a missing file.
        if config.local_files_only:
            from pathlib import Path  # noqa: PLC0415

            path_list = [paths] if isinstance(paths, str) else list(paths)
            missing = [p for p in path_list if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(
                    f"ArrayRecordSourceModule: local_files_only=True but the "
                    f"following path(s) do not exist: {missing}. Either populate "
                    "the paths or set local_files_only=False to defer the error "
                    "to Grain."
                )

        # Initialize Grain data source (doesn't take seed parameter)
        self.grain_source = grain.sources.ArrayRecordDataSource(paths=paths)

        # Stateful variables using nnx.Variable
        self.current_index = nnx.Variable(0)
        self.current_epoch = nnx.Variable(0)
        self.total_records = nnx.Variable(len(self.grain_source))  # type: ignore[arg-type]

        # Cache for prefetched records
        self.prefetch_cache: nnx.Variable[dict[str, Any]] = nnx.Variable({})

        # Iterator state
        self.iterator_initialized = nnx.Variable(False)
        # NOTE: Don't use nnx.Variable for iterator storage - it causes copying
        # issues with NNX modules. current_iterator was unused dead code.

        # Shuffled indices if shuffling is enabled
        self.shuffled_indices: nnx.Variable[np.ndarray | None] = nnx.Variable(None)
        if self.config.shuffle_files:
            self._initialize_shuffle()

    def _initialize_shuffle(self) -> None:
        """Initialize shuffled indices for the epoch."""
        if self.config.shuffle_files:
            # Create shuffled indices
            rng = np.random.RandomState(self.config.seed + self.current_epoch.get_value())
            indices = np.arange(self.total_records.get_value())
            rng.shuffle(indices)
            self.shuffled_indices.set_value(indices)

    def __len__(self) -> int:
        """Return total number of records."""
        return self.total_records.get_value()

    def __iter__(self) -> Self:
        """Initialize iteration with state tracking."""
        self.current_index.set_value(0)
        if self.current_epoch.get_value() == 0 or not self.iterator_initialized.get_value():
            self._initialize_iterator()
        return self

    def __next__(self) -> Any:  # type: ignore[override]
        """Get next element with state management."""
        current_epoch = self.current_epoch.get_value()
        # Check if we've completed all epochs
        if self.config.num_epochs != -1 and current_epoch >= self.config.num_epochs:
            raise StopIteration

        current_index = self.current_index.get_value()
        total_records = self.total_records.get_value()
        # Check if we need to start a new epoch
        if current_index >= total_records:
            current_epoch += 1
            self.current_epoch.set_value(current_epoch)
            current_index = 0
            self.current_index.set_value(0)

            # Check epoch limit again
            if self.config.num_epochs != -1 and current_epoch >= self.config.num_epochs:
                raise StopIteration

            # Re-shuffle for new epoch if needed
            if self.config.shuffle_files:
                self._initialize_shuffle()

        # Get the actual index (shuffled or sequential)
        shuffled_indices = self.shuffled_indices.get_value()
        if shuffled_indices is not None:
            actual_idx = shuffled_indices[current_index]
        else:
            actual_idx = current_index

        # Get from Grain source
        element = self.grain_source[int(actual_idx)]
        self.current_index.set_value(current_index + 1)

        return element

    def _initialize_iterator(self) -> None:
        """Initialize internal iterator with proper state."""
        if self.config.shuffle_files:
            self._initialize_shuffle()
        self.iterator_initialized.set_value(True)

    def get_state(self) -> dict[str, Any]:
        """Get complete state for checkpointing."""
        shuffled_indices = self.shuffled_indices.get_value()
        return build_state_with_iteration_fields(
            super().get_state(),
            current_index=self.current_index.get_value(),
            current_epoch=self.current_epoch.get_value(),
            extra_fields={
                "prefetch_cache": self.prefetch_cache.get_value(),
                "shuffled_indices": shuffled_indices.tolist()
                if shuffled_indices is not None
                else None,
            },
        )

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        super().set_state(state)
        restore_iteration_fields(
            state,
            current_index=self.current_index,
            current_epoch=self.current_epoch,
            prefetch_cache=self.prefetch_cache,
        )
        if "shuffled_indices" in state and state["shuffled_indices"] is not None:
            self.shuffled_indices.set_value(np.array(state["shuffled_indices"]))

    def __getitem__(self, idx: int) -> Any:
        """Get element by index for subscriptable access.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            Element at the given index.
        """
        total_records = self.total_records.get_value()
        # Handle negative indices
        if idx < 0:
            idx = total_records + idx

        # Check bounds
        if idx < 0 or idx >= total_records:
            raise IndexError(f"Index {idx} out of range for dataset with {total_records} elements")

        # Apply shuffling if enabled
        shuffled_indices = self.shuffled_indices.get_value()
        if shuffled_indices is not None:
            actual_idx = shuffled_indices[idx]
        else:
            actual_idx = idx

        # Get from Grain source
        return self.grain_source[int(actual_idx)]

    def get_batch_at(
        self,
        start: int,
        size: int,
        key: Any | None = None,
    ) -> list[Any]:
        """Stateless indexed batch access for ``Pipeline``-driven iteration.

        Returns ``size`` records starting at logical position ``start``,
        wrapping at the end of the dataset and applying any active
        shuffle permutation. Does not advance ``self.current_index`` or
        any other internal state.

        ArrayRecord records are loaded host-side (Grain is a Python
        library), so this method requires a concrete Python ``int`` for
        ``start``. Driving an ArrayRecord source under ``nnx.scan``
        (Tier C of the pipeline integration story) currently requires
        wrapping the host-side fetch in ``jax.experimental.io_callback``
        — left as a future enhancement. Tier A (Python iteration) and
        Tier B (single ``step()``) work today.

        Args:
            start: Concrete starting index (Python int).
            size: Number of records to return.
            key: Reserved for future shuffled-mode support; currently
                ignored (shuffle uses ``self.shuffled_indices``).

        Returns:
            List of ``size`` records as returned by the underlying
            Grain source. Records are typically Python dicts; callers
            (typically a parse / decode operator) handle structure.

        Raises:
            TypeError: If ``start`` is a JAX tracer (not host-side
                concrete). ArrayRecord cannot be traced through
                ``nnx.scan`` without an io_callback wrapper.
        """
        del key  # ArrayRecord uses self.shuffled_indices for shuffle, not key

        if hasattr(start, "shape"):
            raise TypeError(
                "ArrayRecordSourceModule.get_batch_at requires a concrete "
                "Python int for `start`. ArrayRecord records are loaded "
                "host-side and cannot be traced through nnx.scan in this "
                "form. Use the Pipeline iterator (`for batch in pipeline:`) "
                "for ArrayRecord, or wrap fetches via jax.experimental."
                "io_callback if scan compatibility is needed."
            )

        total = self.total_records.get_value()
        indices = [(int(start) + i) % total for i in range(size)]
        return self._getitems(indices)

    def _getitems(self, indices: Sequence[int]) -> list[Any]:
        """Get multiple records using Grain's batched random-access protocol."""
        total_records = self.total_records.get_value()
        resolved = validate_index_batch(indices, total_records)
        shuffled_indices = self.shuffled_indices.get_value()
        actual_indices = (
            [int(shuffled_indices[index]) for index in resolved]
            if shuffled_indices is not None
            else resolved
        )

        getitems = getattr(self.grain_source, "_getitems", None)
        if getitems is not None:
            return list(getitems(actual_indices))
        return [self.grain_source[index] for index in actual_indices]
