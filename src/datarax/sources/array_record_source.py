"""Data source for reading from ArrayRecord format files."""

from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import grain.python as grain
import numpy as np

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.typing import Element


@dataclass
class ArrayRecordSourceConfig(StructuralConfig):
    """Configuration for ArrayRecordSourceModule.

    Inherits from StructuralConfig for runtime immutability.

    Attributes:
        seed: Random seed for shuffling (used internally, not by Grain).
        num_epochs: Number of epochs (-1 for infinite).
        shuffle_files: Whether to shuffle file order (handled internally).
    """

    seed: int = 42
    num_epochs: int = -1
    shuffle_files: bool = False


class ArrayRecordSourceModule(DataSourceModule):
    """Stateful wrapper for Grain's ArrayRecordDataSource.

    This module wraps Grain's ArrayRecordDataSource while maintaining
    stateful iteration through NNX Variables, following TDD principles
    and critical technical guidelines.

    Note: Grain's ArrayRecordDataSource doesn't accept a seed parameter directly.
    Shuffling is handled at the sampler level or through file ordering.
    """

    def __init__(
        self,
        config: ArrayRecordSourceConfig,
        paths: str | list[str],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize ArrayRecord source with state management.

        Args:
            config: Configuration for the source.
            paths: Path pattern or list of paths to ArrayRecord files.
            rngs: NNX Rngs for additional randomness.
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Initialize Grain data source (doesn't take seed parameter)
        self.grain_source = grain.ArrayRecordDataSource(paths=paths)

        # Stateful variables using nnx.Variable
        self.current_index = nnx.Variable(0)
        self.current_epoch = nnx.Variable(0)
        self.total_records = nnx.Variable(len(self.grain_source))

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

    def _initialize_shuffle(self):
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

    def __iter__(self):
        """Initialize iteration with state tracking."""
        self.current_index.set_value(0)
        if self.current_epoch.get_value() == 0 or not self.iterator_initialized.get_value():
            self._initialize_iterator()
        return self

    def __next__(self) -> Element:
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

    def _initialize_iterator(self):
        """Initialize internal iterator with proper state."""
        if self.config.shuffle_files:
            self._initialize_shuffle()
        self.iterator_initialized.set_value(True)

    def get_state(self) -> dict[str, Any]:
        """Get complete state for checkpointing."""
        state = super().get_state()
        shuffled_indices = self.shuffled_indices.get_value()
        state.update(
            {
                "current_index": self.current_index.get_value(),
                "current_epoch": self.current_epoch.get_value(),
                "prefetch_cache": self.prefetch_cache.get_value(),
                "shuffled_indices": shuffled_indices.tolist()
                if shuffled_indices is not None
                else None,
            }
        )
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        super().set_state(state)
        if "current_index" in state:
            self.current_index.set_value(state["current_index"])
        if "current_epoch" in state:
            self.current_epoch.set_value(state["current_epoch"])
        if "prefetch_cache" in state:
            self.prefetch_cache.set_value(state["prefetch_cache"])
        if "shuffled_indices" in state and state["shuffled_indices"] is not None:
            self.shuffled_indices.set_value(np.array(state["shuffled_indices"]))

    def __getitem__(self, idx: int) -> Element:
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
