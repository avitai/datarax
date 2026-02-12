"""Shuffle sampler implementation for Datarax.

This module provides a unified sampler that shuffles the order of data access.
Supports both static method usage and NNX module instantiation.
"""

import random
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterator

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


@dataclass
class ShuffleSamplerConfig(StructuralConfig):
    """Configuration for ShuffleSampler.

    Attributes:
        buffer_size: The size of the shuffling buffer (required)
        dataset_size: Optional size of the dataset being sampled from
        seed: Optional integer seed for reproducible shuffling
    """

    buffer_size: int | None = None
    dataset_size: int | None = None
    seed: int | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # ShuffleSampler is always stochastic
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "shuffle")

        # Call parent validation
        super().__post_init__()

        # Validate buffer_size (required)
        if self.buffer_size is None:
            raise ValueError("buffer_size is required")
        if self.buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {self.buffer_size}")


@dataclass
class _FallbackState:
    """Python random fallback mode state."""

    buffer: list[int] = field(default_factory=list)
    next_index: int = 0
    indices_yielded: int = 0
    rng: random.Random | None = None


class ShuffleSampler(SamplerModule):
    """Unified shuffle sampler implementation for Datarax.

    This class provides methods for shuffling the order of data access,
    with support for both static method usage and NNX module instantiation.
    Maintains a buffer of indices that are shuffled randomly and from which
    indices are drawn. As the buffer is depleted, new indices are added
    and the buffer is reshuffled.

    Attributes:
        buffer_size: The size of the shuffling buffer.
        dataset_size: The size of the dataset being sampled from.
    """

    def __init__(
        self,
        config: ShuffleSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize a ShuffleSampler with config.

        Args:
            config: Configuration for the sampler.
            rngs: Optional Rngs object for randomness.
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)

        # Store config values
        self.buffer_size = config.buffer_size
        self.dataset_size = config.dataset_size
        self.seed = config.seed

        # Internal state for NNX mode
        self._buffer: list[int] | None = None
        self._buffer_pos = nnx.Variable(0)
        self._next_index = nnx.Variable(0)
        self._indices_yielded = nnx.Variable(0)
        self._buffer_count = nnx.Variable(0)
        self._is_restored = nnx.Variable(False)

        # Internal state for fallback mode (Python random)
        self._fallback = _FallbackState()

        # Initialize random state for fallback mode
        if self.seed is not None:
            self._init_fallback_random_state()

    def _init_fallback_random_state(self):
        """Initialize the random state for fallback shuffling."""
        if self.seed is not None:
            self._fallback.rng = random.Random(self.seed)  # nosec B311
        else:
            self._fallback.rng = random.Random()  # nosec B311

    @staticmethod
    def create_static_iterator(
        buffer_size: int,
        dataset_size: int,
        seed: int | None = None,
    ) -> Iterator[int]:
        """Static method to create a shuffle iterator.

        Args:
            buffer_size: The size of the shuffling buffer.
            dataset_size: The size of the dataset being sampled from.
            seed: Optional integer seed for reproducible shuffling.

        Returns:
            An iterator that yields shuffled indices.
        """
        # Initialize random state
        rng = random.Random(seed) if seed is not None else random.Random()  # nosec B311

        buffer: list[int] = []
        next_index = 0
        indices_yielded = 0

        while indices_yielded < dataset_size:
            # If the buffer is empty, fill it with new indices and shuffle
            if not buffer:
                # For the initial fill or if we've gone through the entire dataset
                if next_index >= dataset_size:
                    next_index = 0

                # Calculate how many indices to add to the buffer
                indices_remaining = dataset_size - next_index
                num_to_add = min(buffer_size, indices_remaining)

                # Add the next batch of indices to the buffer
                buffer = list(range(next_index, next_index + num_to_add))
                next_index += num_to_add

                # Shuffle the buffer
                rng.shuffle(buffer)

            # Yield the next index from the buffer
            if buffer:
                index = buffer.pop(0)
                indices_yielded += 1
                yield index

    def __len__(self) -> int:
        """Return the total number of indices that will be sampled.

        Returns:
            The total number of indices in the sampler.

        Raises:
            ValueError: If dataset_size is not provided.
        """
        if self.dataset_size is None:
            msg = "dataset_size must be provided to determine length"
            raise ValueError(msg)
        return self.dataset_size

    def __iter__(self) -> Iterator[int]:
        """Yield shuffled indices into the dataset.

        Returns:
            An iterator yielding shuffled indices.

        Raises:
            ValueError: If dataset_size is not provided and cannot be inferred.
        """
        # Use JAX-based shuffling if we have rngs, otherwise fall back to Python random
        if self.rngs is not None:
            return self._iter_jax_mode()
        else:
            return self._iter_fallback_mode()

    def _iter_jax_mode(self) -> Iterator[int]:
        """JAX-based iteration with NNX state management."""
        # Only reset state if we're not in a restored state
        if not self._is_restored.get_value():
            # Reset state if we're starting a new iteration
            self._buffer = None
            self._buffer_pos.set_value(0)
            self._next_index.set_value(0)
            self._indices_yielded.set_value(0)

        # Ensure we have a dataset size
        if self.dataset_size is None:
            msg = "dataset_size must be provided"
            raise ValueError(msg)

        while self._indices_yielded.get_value() < self.dataset_size:
            # If the buffer is exhausted, fill it with new indices and shuffle
            if self._buffer is None or self._buffer_pos.get_value() >= len(self._buffer):
                # For initial fill or if we've gone through dataset
                if self._next_index.get_value() >= self.dataset_size:
                    # We've gone through the entire dataset
                    break

                # Calculate how many indices to add to the buffer
                next_idx = self._next_index.get_value()
                indices_remaining = self.dataset_size - next_idx
                num_to_add = min(self.buffer_size, indices_remaining)

                # Create list of the next batch of indices
                indices = list(range(next_idx, next_idx + num_to_add))
                self._next_index.set_value(next_idx + num_to_add)

                # Shuffle the buffer using JAX's PRNGs for reproducibility
                if self.rngs is not None and self.stream_name in self.rngs:
                    # Get the base key from the stream
                    base_key = self.rngs[self.stream_name].key.get_value()
                    # Create a deterministic key based on buffer count
                    buffer_count = self._buffer_count.get_value()
                    key = jax.random.fold_in(base_key, buffer_count)
                    self._buffer_count.set_value(buffer_count + 1)

                    # Generate random permutation using JAX
                    perm = jax.random.permutation(key, jnp.arange(num_to_add))
                    # Apply permutation to indices list (convert to Python ints)
                    self._buffer = [indices[int(i)] for i in perm]
                else:
                    # Fallback using JAX with default key
                    key = jax.random.key(0)
                    perm = jax.random.permutation(key, jnp.arange(num_to_add))
                    self._buffer = [indices[int(i)] for i in perm]

                # Reset buffer position
                self._buffer_pos.set_value(0)

            # Yield the next index from the buffer
            buffer_pos = self._buffer_pos.get_value()
            if buffer_pos < len(self._buffer):
                # Extract single index (already a Python int)
                index = self._buffer[buffer_pos]
                self._buffer_pos.set_value(buffer_pos + 1)
                indices_yielded = self._indices_yielded.get_value()
                self._indices_yielded.set_value(indices_yielded + 1)
                yield index

        # Clear the restored flag only when iteration is complete
        if self._is_restored.get_value():
            self._is_restored.set_value(False)

    def _iter_fallback_mode(self) -> Iterator[int]:
        """Fallback Python random-based iteration."""
        # Reset state if we're starting a new iteration
        self._fallback.buffer = []
        self._fallback.next_index = 0
        self._fallback.indices_yielded = 0

        # Ensure we have a dataset size
        if self.dataset_size is None:
            msg = "dataset_size must be provided"
            raise ValueError(msg)

        while self._fallback.indices_yielded < self.dataset_size:
            # If the buffer is empty, fill it with new indices and shuffle
            if not self._fallback.buffer:
                # For the initial fill or if we've gone through the entire dataset
                if self._fallback.next_index >= self.dataset_size:
                    self._fallback.next_index = 0

                # Calculate how many indices to add to the buffer
                indices_remaining = self.dataset_size - self._fallback.next_index
                num_to_add = min(self.buffer_size, indices_remaining)

                # Add the next batch of indices to the buffer
                self._fallback.buffer = list(
                    range(self._fallback.next_index, self._fallback.next_index + num_to_add)
                )
                self._fallback.next_index += num_to_add

                # Shuffle the buffer
                if self._fallback.rng is None:
                    self._init_fallback_random_state()
                assert self._fallback.rng is not None, "RNG should be initialized"
                self._fallback.rng.shuffle(self._fallback.buffer)

            # Yield the next index from the buffer
            if self._fallback.buffer:
                index = self._fallback.buffer.pop(0)
                self._fallback.indices_yielded += 1
                yield index

    def get_state(self) -> dict[str, Any]:
        """Return the current state for checkpointing.

        Returns:
            A dictionary containing the internal state of the sampler.
        """
        # Get the base state from the module
        state = super().get_state()

        if self.rngs is not None:
            # JAX mode state
            state.update(
                {
                    "sampler_state": {
                        "buffer": self._buffer[:] if self._buffer is not None else None,
                        "buffer_pos": self._buffer_pos.get_value(),
                        "next_index": self._next_index.get_value(),
                        "indices_yielded": self._indices_yielded.get_value(),
                        "buffer_count": self._buffer_count.get_value(),
                        "buffer_size": self.buffer_size,
                        "dataset_size": self.dataset_size,
                        "is_restored": self._is_restored.get_value(),
                    }
                }
            )
        else:
            # Fallback mode state
            state.update(
                {
                    "fallback_state": {
                        "buffer": self._fallback.buffer.copy(),
                        "next_index": self._fallback.next_index,
                        "indices_yielded": self._fallback.indices_yielded,
                        "rng_state": self._fallback.rng.getstate() if self._fallback.rng else None,
                        "buffer_size": self.buffer_size,
                        "dataset_size": self.dataset_size,
                        "seed": self.seed,
                    }
                }
            )

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state: A dictionary containing the internal state to restore.
        """
        custom = self._split_state(state, {"sampler_state", "fallback_state"})

        # Restore sampler-specific state
        if "sampler_state" in custom:
            # JAX mode state
            sampler_state = custom["sampler_state"]
            self._buffer = (
                sampler_state["buffer"][:] if sampler_state["buffer"] is not None else None
            )
            self._buffer_pos.set_value(sampler_state["buffer_pos"])
            self._next_index.set_value(sampler_state["next_index"])
            self._indices_yielded.set_value(sampler_state["indices_yielded"])
            self._buffer_count.set_value(sampler_state["buffer_count"])
            self.buffer_size = sampler_state["buffer_size"]
            self.dataset_size = sampler_state["dataset_size"]
            self._is_restored.set_value(sampler_state.get("is_restored", True))
        elif "fallback_state" in custom:
            # Fallback mode state
            fallback_state = custom["fallback_state"]
            self._fallback.buffer = fallback_state["buffer"].copy()
            self._fallback.next_index = fallback_state["next_index"]
            self._fallback.indices_yielded = fallback_state["indices_yielded"]
            self.buffer_size = fallback_state["buffer_size"]
            self.dataset_size = fallback_state["dataset_size"]
            self.seed = fallback_state.get("seed")
            if fallback_state["rng_state"] and self._fallback.rng:
                self._fallback.rng.setstate(fallback_state["rng_state"])

    def reset(self, seed: int | None = None) -> None:
        """Reset the sampler state, typically used to start a new epoch.

        Args:
            seed: Optional seed to use for shuffling.
                If provided, overrides the original seed.
        """
        if self.rngs is not None:
            # JAX mode reset
            if seed is not None:
                # Create a fresh Rngs with the new seed
                self.rngs = nnx.Rngs(**{self.stream_name: seed})

            # Reset buffer state
            self._buffer = None
            self._buffer_pos.set_value(0)
            self._next_index.set_value(0)
            self._indices_yielded.set_value(0)
            self._buffer_count.set_value(0)
            self._is_restored.set_value(False)
        else:
            # Fallback mode reset
            if seed is not None:
                self.seed = seed
                self._fallback.rng = random.Random(seed)  # nosec B311
            else:
                # Re-initialize with the original seed
                self._init_fallback_random_state()

            self._fallback.buffer = []
            self._fallback.next_index = 0
            self._fallback.indices_yielded = 0
