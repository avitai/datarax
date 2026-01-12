"""Type definitions for Datarax.

Provides common type aliases, functional interface definitions, and
checkpointing protocols used throughout the codebase.
"""

from __future__ import annotations
from typing import Any, Callable, Protocol, TypeAlias, TypeVar, runtime_checkable

import jax

# Import concrete implementations
from datarax.core.element_batch import Element as ElementImpl
from datarax.core.element_batch import Batch as BatchImpl
from datarax.core.metadata import Metadata

# Type aliases for implementations
Element: TypeAlias = ElementImpl
Batch: TypeAlias = BatchImpl

# Generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
E = TypeVar("E", bound=Element)
B = TypeVar("B", bound=Batch)

# Common type aliases
DataDict: TypeAlias = dict[str, jax.Array]
StateDict: TypeAlias = dict[str, Any]
MetadataDict: TypeAlias = dict[str, Any]

# JAX types

ArrayShape: TypeAlias = tuple[int, ...]
PRNGKey: TypeAlias = jax.Array

# Function signatures
ElementTransform: TypeAlias = Callable[[Element], Element]
BatchTransform: TypeAlias = Callable[[Batch], Batch]
ArrayTransform: TypeAlias = Callable[[jax.Array], jax.Array]
DataProcessor: TypeAlias = Callable[[DataDict], DataDict]
StateProcessor: TypeAlias = Callable[[StateDict], StateDict]
MetadataProcessor: TypeAlias = Callable[[Metadata], Metadata]

# JAX-specific function types
ScanFn: TypeAlias = Callable[[Any, Element], tuple[Any, Element]]
CondFn: TypeAlias = Callable[[Any], bool]
WhileBodyFn: TypeAlias = Callable[[Any], Any]


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
