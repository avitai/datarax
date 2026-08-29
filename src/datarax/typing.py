"""Type definitions for Datarax.

Provides common type aliases, functional interface definitions, and
checkpointing protocols used throughout the codebase.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable, TypeVar

import jax

# Import concrete implementations
from datarax.core.element_batch import Batch as BatchImpl, Element as ElementImpl
from datarax.core.metadata import Metadata


logger = logging.getLogger(__name__)


# Type aliases for implementations
# Re-exports of concrete classes, not type aliases. They are called as constructors and
# carry classmethods, so they must stay plain assignments: `type X = Y` binds a lazy
# TypeAliasType, which is not callable and exposes none of the class's attributes.
Element = ElementImpl
Batch = BatchImpl

# Generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
E = TypeVar("E", bound=Element)
B = TypeVar("B", bound=Batch)

# Common type aliases
type DataDict = dict[str, jax.Array]
type StateDict = dict[str, Any]
type MetadataDict = dict[str, Any]

# JAX types

type ArrayShape = tuple[int, ...]
PRNGKey = jax.Array

# Function signatures
type ElementTransform = Callable[[Element], Element]
type BatchTransform = Callable[[Batch], Batch]
type ArrayTransform = Callable[[jax.Array], jax.Array]
type DataProcessor = Callable[[DataDict], DataDict]
type StateProcessor = Callable[[StateDict], StateDict]
type MetadataProcessor = Callable[[Metadata], Metadata]

# JAX-specific function types
type ScanFn = Callable[[Any, Element], tuple[Any, Element]]
type CondFn = Callable[[Any], bool]
type WhileBodyFn = Callable[[Any], Any]


# Checkpointing protocol
@runtime_checkable
class Checkpointable(Protocol):
    """Protocol for objects that can be checkpointed via state dictionaries.

    This protocol defines the interface for objects that support state-based
    checkpointing, where state is extracted to a dictionary and restored from
    a dictionary. This aligns with NNX state management patterns.
    """

    def get_state(self) -> dict[str, Any]:
        """Get object state for checkpointing.

        Returns:
            Dictionary containing all state needed to restore the object.
        """
        ...

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore object state from a checkpoint.

        Args:
            state: Dictionary containing state to restore.
        """
        ...


@runtime_checkable
class CheckpointableIterator(Checkpointable, Protocol[T_co]):
    """Protocol for iterators that can be checkpointed.

    Combines Iterator behavior with Checkpointable state management.
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
    "Element",
    "Batch",
    "Metadata",
    "DataDict",
    "StateDict",
    "MetadataDict",
    "ArrayShape",
    "PRNGKey",
    # Function types
    "ElementTransform",
    "BatchTransform",
    "ArrayTransform",
    "DataProcessor",
    "StateProcessor",
    "MetadataProcessor",
    "ScanFn",
    "CondFn",
    "WhileBodyFn",
    "Checkpointable",
    "CheckpointableIterator",
]
