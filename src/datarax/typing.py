"""Type definitions for Datarax.

Provides common type aliases, functional interface definitions, and the
checkpointable-iterator protocol used throughout the codebase. The checkpoint-state
protocol itself is substrax's :class:`~substrax.typing.Checkpointable`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable, TypeVar

import jax
from substrax.typing import Checkpointable


logger = logging.getLogger(__name__)


# Generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

# Common type aliases
type DataDict = dict[str, jax.Array]
# What a pipeline yields: field names to arrays, or to pytrees of arrays when a stage nests them.
# A source's batch is the flat ``DataDict``.
type PipelineBatch = dict[str, Any]

# JAX types

type ArrayShape = tuple[int, ...]
PRNGKey = jax.Array

# Function signatures
type ArrayTransform = Callable[[jax.Array], jax.Array]


@runtime_checkable
class CheckpointableIterator(Checkpointable, Protocol[T_co]):
    """Protocol for iterators that can be checkpointed.

    Combines Iterator behavior with substrax's ``Checkpointable`` state management.
    """

    def __iter__(self) -> CheckpointableIterator[T_co]:
        """Return iterator."""
        ...

    def __next__(self) -> T_co:
        """Get next item."""
        ...


# Export public API
__all__ = [
    # Type aliases
    "PipelineBatch",
    "DataDict",
    "ArrayShape",
    "PRNGKey",
    # Function types
    "ArrayTransform",
    "CheckpointableIterator",
]
