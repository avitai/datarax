"""Data source for reading from ArrayRecord format files."""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import grain
import grain.sources
import jax
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.spec import array_to_spec
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

    ArrayRecord records are ``bytes``. Pass ``decode`` to turn one record into a
    dict of arrays; ``get_batch`` then decodes and stacks records of the current
    epoch and returns an empty batch at the epoch boundary, so every
    ``for batch in pipeline`` pass covers one epoch.

    Resource management: the underlying ArrayRecord readers hold C++ file handles
    that are not reliably freed by garbage collection. Use the module as a context
    manager (``with ArrayRecordSourceModule(...) as source:``) or call ``close()``
    explicitly between phases to avoid "Too many open files" on long-running jobs.
    """

    # Narrow config type for pyright (base stores via nnx.static)
    config: ArrayRecordSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: ArrayRecordSourceConfig,
        paths: str | list[str],
        *,
        decode: Callable[[bytes], Mapping[str, Any]] | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize ArrayRecord source with state management.

        Args:
            config: Configuration for the source.
            paths: Path pattern or list of paths to ArrayRecord files.
            decode: Turns one ``bytes`` record into a dict of arrays; required for
                ``get_batch``, ``element_spec`` and ``Pipeline`` iteration.
            rngs: NNX Rngs for additional randomness.
            name: Optional name for the module.

        Raises:
            FileNotFoundError: If ``local_files_only`` is set and any of ``paths`` does not exist.
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
        self._decode = decode

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

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Enumerates every parameter that affects which records are read and in
        what order (paths, record count, shuffle/seed/epoch settings) so a
        checkpoint restore can detect an incompatible source configuration.
        """
        paths = getattr(self.grain_source, "paths", None)
        return (
            f"ArrayRecordSourceModule(paths={paths!r}, "
            f"num_records={self.total_records.get_value()}, "
            f"shuffle_files={self.config.shuffle_files}, "
            f"seed={self.config.seed}, "
            f"num_epochs={self.config.num_epochs})"
        )

    def __iter__(self) -> Self:
        """Initialize iteration with state tracking."""
        self.current_index.set_value(0)
        if self.current_epoch.get_value() == 0 or not self.iterator_initialized.get_value():
            self._initialize_iterator()
        return self

    def __next__(self) -> Any:  # type: ignore[override]
        """Get next element with state management."""
        if self._epochs_exhausted():
            raise StopIteration

        current_index = self.current_index.get_value()
        # Check if we need to start a new epoch
        if current_index >= self.total_records.get_value():
            self._start_next_epoch()
            if self._epochs_exhausted():
                raise StopIteration
            current_index = 0

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

        Raises:
            IndexError: If ``idx`` is outside the dataset after negative-index wrapping.
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
        actual_idx = shuffled_indices[idx] if shuffled_indices is not None else idx

        # Get from Grain source
        return self.grain_source[int(actual_idx)]

    def _epochs_exhausted(self) -> bool:
        """Whether ``num_epochs`` epochs have been served."""
        num_epochs = self.config.num_epochs
        return num_epochs != -1 and self.current_epoch.get_value() >= num_epochs

    def _start_next_epoch(self) -> None:
        """Advance to the next epoch: first record, and a new order when shuffling."""
        self.current_epoch.set_value(self.current_epoch.get_value() + 1)
        self.current_index.set_value(0)
        if self.config.shuffle_files:
            self._initialize_shuffle()

    def _require_decode(self) -> Callable[[bytes], Mapping[str, Any]]:
        """Return the record decoder, refusing to produce arrays without one.

        Returns:
            The ``decode`` function the source was built with.

        Raises:
            TypeError: If the source was built without ``decode``.
        """
        if self._decode is None:
            raise TypeError(
                "ArrayRecordSourceModule needs decode= to produce batches: ArrayRecord "
                "records are bytes, and batches are dicts of arrays."
            )
        return self._decode

    def get_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Decode and stack up to ``batch_size`` records of the current epoch.

        Returns an empty dict at the end of an epoch, and the next call starts the
        next epoch, so each ``for batch in pipeline`` pass covers one epoch. Once
        ``num_epochs`` epochs have been served every call returns an empty dict.

        Args:
            batch_size: Largest number of records to return.

        Returns:
            Arrays stacked along a leading record axis, or ``{}`` at an epoch boundary.
        """
        decode = self._require_decode()
        if self._epochs_exhausted():
            return {}
        index = self.current_index.get_value()
        total = self.total_records.get_value()
        if index >= total:
            self._start_next_epoch()
            return {}
        stop = min(index + batch_size, total)
        records = [decode(record) for record in self._getitems(list(range(index, stop)))]
        self.current_index.set_value(stop)
        return {
            key: np.stack([np.asarray(record[key]) for record in records]) for key in records[0]
        }

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Describe one decoded record exactly as ``get_batch`` stacks it.

        Returns:
            The shape and dtype of each field of the first decoded record.
        """
        record = self._require_decode()(self.grain_source[0])
        return {key: array_to_spec(np.asarray(value)) for key, value in record.items()}

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

    def close(self) -> None:
        """Release the underlying ArrayRecord C++ file handles.

        ArrayRecord readers hold C++ file handles that Python garbage collection
        does **not** reliably release; long-running pipelines that repeatedly
        create sources can exhaust the file-descriptor limit ("Too many open
        files"). Call ``close()`` — or use the source as a context manager —
        between phases that open new sources.

        Delegates to Grain's own cleanup: ``ArrayRecordDataSource.close()`` on
        grain >= 0.2.19, falling back to its context-manager ``__exit__`` on
        grain 0.2.18. Idempotent and safe to call multiple times.
        """
        source = getattr(self, "grain_source", None)
        if source is None:
            return
        closer = getattr(source, "close", None)
        if callable(closer):
            closer()
            return
        exiter = getattr(source, "__exit__", None)
        if callable(exiter):
            exiter(None, None, None)

    def __enter__(self) -> Self:
        """Enter a context that guarantees ``close()`` on exit."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release ArrayRecord file handles on context exit."""
        self.close()
