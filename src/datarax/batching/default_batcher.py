"""Default batcher module implementation for Datarax.

This module provides a default implementation of the BatcherModule interface
that handles batching of PyTrees.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from flax import nnx

from datarax.core import batch_ops
from datarax.core.batcher import BatcherModule
from datarax.core.config import StructuralConfig
from datarax.core.element_batch import Batch, Element


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DefaultBatcherConfig(StructuralConfig):
    """Configuration for DefaultBatcher.

    DefaultBatcher is deterministic and requires no additional configuration
    beyond the base StructuralConfig.
    """

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # DefaultBatcher is deterministic (no randomness)
        object.__setattr__(self, "stochastic", False)
        super().__post_init__()


class DefaultBatcher(BatcherModule):
    """Default implementation of the BatcherModule interface.

    This batcher module accumulates records and forms batches by stacking every leaf along a
    new leading record axis (``batch_ops.from_stacked``). Each record's identity comes with it,
    so its randomness in later operators does not depend on the batch it lands in.
    """

    def __init__(
        self,
        config: DefaultBatcherConfig,
        *,
        collate_fn: Callable[[list[Element]], Batch] | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a DefaultBatcher.

        Args:
            config: Configuration for the batcher.
            collate_fn: Optional custom function to use for combining elements
                into a batch. If None, a default stacking approach is used.
            rngs: Optional Rngs object for randomness.
            name: Optional name for the module.
        """
        super().__init__(config, rngs=rngs, name=name)
        self.collate_fn: Callable[[list[Element]], Batch] | None = collate_fn

    def process(
        self,
        elements: Iterator[Element],
        *_args: Any,
        batch_size: int,
        drop_remainder: bool = False,
        **_kwargs: Any,
    ) -> Iterator[Batch]:
        """Group individual data elements into batches.

        Args:
            elements: An iterator yielding individual data elements.
            *_args: Additional positional arguments (ignored).
            batch_size: The number of elements to include in each batch.
            drop_remainder: Whether to drop the last batch if it's smaller than
                batch_size.
            **_kwargs: Additional keyword arguments (ignored).

        Yields:
            Batches of data elements.

        Raises:
            ValueError: If batch_size is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch_buffer: list[Element] = []

        for element in elements:
            batch_buffer.append(element)

            if len(batch_buffer) == batch_size:
                # We have a full batch, collate and yield it
                yield self._collate_batch(batch_buffer)
                batch_buffer = []

        # Handle any remaining elements
        if batch_buffer and not drop_remainder:
            yield self._collate_batch(batch_buffer)

    def _collate_batch(self, elements: list[Element]) -> Batch:
        """Combine records into a batch: every leaf stacked along a new record axis.

        Args:
            elements: The records, with one structure.

        Returns:
            The batch, each record keeping its own identity.
        """
        if self.collate_fn is not None:
            return self.collate_fn(elements)
        return batch_ops.from_stacked(batch_ops.stack(elements))
