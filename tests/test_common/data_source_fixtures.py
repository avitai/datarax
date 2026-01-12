"""Data source test fixtures for Datarax testing.

This module provides test fixtures and mock data sources for testing
Datarax components, particularly for state management and pipeline testing.
"""

from typing import Any, Iterator

import flax.nnx as nnx
import jax.numpy as jnp

from datarax.core.data_source import DataSourceModule
from datarax.typing import Element


class MockDataSourceModule(DataSourceModule):
    """A test data source with controllable state for testing.

    This fixture is specifically designed for testing Pipeline's
    state management capabilities, checkpointing, and state restoration.
    It provides predictable data generation with controllable state.
    """

    size: nnx.Variable[int]
    _current_position: nnx.Variable[int]
    custom_state: nnx.Variable[dict[str, Any]]

    def __init__(self, size: int = 10, *, rngs: nnx.Rngs | None = None, name: str | None = None):
        """Initialize with a specific size.

        Args:
            size: Number of elements to generate.
            rngs: Optional Rngs object for randomness.
            name: Optional name for the module.
        """
        super().__init__(rngs=rngs, name=name)
        self.size = nnx.Variable(size)
        self._current_position = nnx.Variable(0)
        self.custom_state = nnx.Variable({"extra_data": "initial value"})

    @property
    def current_position(self) -> int:
        """Get the current position value."""
        return self._current_position.get_value()

    @current_position.setter
    def current_position(self, value: int) -> None:
        """Set the current position value."""
        self._current_position.set_value(value)

    def __iter__(self) -> Iterator[Element]:
        """Return an iterator over the data source.

        Returns:
            An iterator yielding integers starting from current_position.
        """
        for i in range(self._current_position.get_value(), self.size.get_value()):
            self._current_position.set_value(i + 1)
            yield {"value": jnp.array(i)}

    def __len__(self):
        """Return the total size of the data source.

        Returns:
            The total number of elements in the data source.
        """
        return self.size.get_value()

    def get_state(self) -> dict[str, Any]:
        """Return the current state for checkpointing.

        Returns:
            A dictionary containing the current position and custom state.
        """
        return {
            "current_position": self._current_position.get_value(),
            "custom_state": self.custom_state.get_value(),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the state from a checkpoint.

        Args:
            state: A dictionary containing the state to restore.
        """
        if "current_position" in state:
            self._current_position.set_value(state["current_position"])
        if "custom_state" in state:
            self.custom_state.set_value(state["custom_state"])

    def __getitem__(self, idx: int) -> Element:
        """Get element by index for random access.

        Args:
            idx: Index of the element to retrieve.

        Returns:
            The data element at the given index.
        """
        if 0 <= idx < self.size.get_value():
            return {"value": jnp.array(idx)}  # type: ignore
        else:
            raise IndexError(f"Index {idx} out of range [0, {self.size.get_value()})")

    def reset(self, seed: int | None = None) -> None:
        """Reset the data source to start from the beginning.

        Args:
            seed: Optional seed for randomness (not used in this implementation).
        """
        self._current_position.set_value(0)
