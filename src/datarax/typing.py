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
# What a pipeline yields: field names to arrays, or to pytrees of arrays when a stage nests them.
# A source's batch is the flat ``DataDict``.
type PipelineBatch = dict[str, Any]
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
    "Element",
    "Batch",
    "Metadata",
    "PipelineBatch",
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
    "CheckpointableIterator",
]
