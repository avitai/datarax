"""Range sampler for Datarax.

This module provides a unified range sampler that generates a sequence of integers.
Supports both static method usage and NNX module instantiation.
"""

from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

import flax.nnx as nnx

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule
from datarax.typing import Element


@dataclass
class RangeSamplerConfig(StructuralConfig):
    """Configuration for RangeSampler.

    Attributes:
        start: The start of the range (inclusive, default: 0)
        stop: The end of the range (exclusive, if None uses start as stop and start=0)
        step: The step size between consecutive elements (default: 1)
    """

    start: int = 0
    stop: int | None = None
    step: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        # RangeSampler is always deterministic
        object.__setattr__(self, "stochastic", False)

        # Call parent validation
        super().__post_init__()

        # Validate step
        if self.step == 0:
            raise ValueError("Step cannot be zero")

        # Handle Python range() convention: range(10) means range(0, 10)
        if self.stop is None:
            object.__setattr__(self, "stop", self.start)
            object.__setattr__(self, "start", 0)

        # Validate range would not be empty
        range_length = (self.stop - self.start) / self.step  # type: ignore
        length = max(0, int(range_length))
        if length == 0 and self.start != self.stop:
            msg = f"Range with start={self.start}, stop={self.stop}, "
            msg += f"step={self.step} would be empty"
            raise ValueError(msg)


class RangeSampler(SamplerModule):
    """Unified range sampler implementation for Datarax.

    This class provides methods for generating a sequence of integers,
    with support for both static method usage and NNX module instantiation.
    Similar to Python's built-in range(), but implements the SamplerModule
    interface for use with Datarax pipelines.

    Attributes:
        start: The start of the range (inclusive).
        stop: The end of the range (exclusive).
        step: The step size between consecutive elements.
    """

    def __init__(
        self,
        config: RangeSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize a RangeSampler with config.

        Args:
            config: Configuration for the sampler.
            rngs: Optional Rngs object (not used for this deterministic sampler).
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Config already validated start, stop, step (including None → 0 conversion)
        self.start = config.start
        self.stop = config.stop
        self.step = config.step

        # Initialize current position for stateful iteration
        self._current_position = nnx.Variable(self.start)

        # Calculate length (config already validated non-empty)
        range_length = (self.stop - self.start) / self.step  # type: ignore
        self._length = max(0, int(range_length))

    @staticmethod
    def create_static(start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[Element]:
        """Static method to create a range iterator.

        Args:
            start: The start of the range (inclusive). If stop is None, this
                becomes the stop value, and start is set to 0.
            stop: The end of the range (exclusive).
            step: The step size between consecutive elements.

        Returns:
            An iterator that yields integers in the specified range.

        Raises:
            ValueError: If step is 0, or if the range parameters would result
                in an empty range.
        """
        if stop is None:
            start, stop = 0, start

        if step == 0:
            raise ValueError("Step cannot be zero")

        # Calculate length
        range_length = (stop - start) / step
        length = max(0, int(range_length))
        if length == 0 and start != stop:
            msg = f"Range with start={start}, stop={stop}, "
            msg += f"step={step} would be empty"
            raise ValueError(msg)

        current = start
        for _ in range(length):
            yield current  # type: ignore
            current += step

    def __iter__(self) -> Iterator[int]:
        """Generate the sequence of integers in the range.

        Returns:
            An iterator that yields integers in the specified range.
        """
        current = self.start
        for _ in range(self._length):
            yield current
            current += self.step

    def __len__(self) -> int:
        """Return the total number of elements in the range.

        Returns:
            The number of elements in the range.
        """
        return self._length

    @staticmethod
    def get_length_static(start: int = 0, stop: int | None = None, step: int = 1) -> int:
        """Static method to get the length of a range.

        Args:
            start: The start of the range (inclusive). If stop is None, this
                becomes the stop value, and start is set to 0.
            stop: The end of the range (exclusive).
            step: The step size between consecutive elements.

        Returns:
            The number of elements in the range.

        Raises:
            ValueError: If step is 0.
        """
        if stop is None:
            start, stop = 0, start

        if step == 0:
            raise ValueError("Step cannot be zero")

        # Calculate length
        range_length = (stop - start) / step
        return max(0, int(range_length))

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the sampler for checkpointing."""
        state = super().get_state()
        state.update(
            {
                "sampler_state": {
                    "start": self.start,
                    "stop": self.stop,
                    "step": self.step,
                    "current_position": self._current_position.get_value(),
                }
            }
        )
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the state from a checkpoint.

        Args:
            state: A dictionary containing the internal state to restore.
        """
        custom = self._split_state(state, {"sampler_state"})

        if "sampler_state" in custom:
            sampler_state = custom["sampler_state"]
            self.start = sampler_state.get("start", self.start)
            self.stop = sampler_state.get("stop", self.stop)
            self.step = sampler_state.get("step", self.step)

            # Restore current position if available
            if "current_position" in sampler_state:
                self._current_position.set_value(sampler_state["current_position"])

            # Recalculate length
            range_length = (self.stop - self.start) / self.step
            self._length = max(0, int(range_length))
        else:
            # Alternative state format support
            if "start" in state:
                self.start = state["start"]
            if "stop" in state:
                self.stop = state["stop"]
            if "step" in state:
                self.step = state["step"]

            # Recalculate length
            range_length = (self.stop - self.start) / self.step
            self._length = max(0, int(range_length))

    def reset(self, seed: int | None = None) -> None:
        """Reset the sampler to the beginning.

        Args:
            seed: Optional seed (unused for range sampler but kept for API consistency).
        """
        self._current_position.set_value(self.start)

    def get_current_position(self) -> int:
        """Get the current position in the range.

        Returns:
            The current position value.
        """
        return self._current_position.get_value()

    def set_current_position(self, position: int) -> None:
        """Set the current position in the range.

        Args:
            position: The position to set.
        """
        self._current_position.set_value(position)
