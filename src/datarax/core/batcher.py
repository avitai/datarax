"""Base module for batcher components in Datarax.

This module defines the base class for all Datarax batcher components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

from typing import Any
from collections.abc import Iterator

from datarax.core.structural import StructuralModule
from datarax.typing import Batch, Element


class BatcherModule(StructuralModule):
    """Base module for all Datarax batcher components.

    A BatcherModule is responsible for grouping individual data elements
    into batches. It handles the accumulation and collation of elements,
    maintaining the PyTree structure in the batched output.

    This class extends StructuralModule for non-parametric structural processing.
    Subclasses implement the process() method for batching logic.

    Args:
        config: StructuralConfig or subclass with batcher-specific parameters
        rngs: Random number generators (required if stochastic=True)
        name: Optional name for the batcher

    Examples:
        Basic Batcher implementation:

        ```python
        from dataclasses import dataclass
        from datarax.core.config import StructuralConfig
        from datarax.core.batcher import BatcherModule
        from flax import nnx

        class DefaultBatcherConfig(StructuralConfig):
            pass

        class DefaultBatcher(BatcherModule):
            def process(self, elements, batch_size, drop_remainder=False):
                # In a real implementation, this would yield actual batches
                return []

        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        batches = list(batcher([], batch_size=32)) # Call the batcher instance
        ```
    """

    # No custom __init__ needed - StructuralModule.__init__ handles everything

    def process(
        self,
        elements: list[Element] | Iterator[Element],
        *args: Any,
        batch_size: int,
        drop_remainder: bool = False,
        **kwargs: Any,
    ) -> list[Batch] | Iterator[Batch]:
        """Group individual data elements into batches.

        This is the main interface for batching operations.
        Subclasses MUST override this method to implement their batching logic.

        Args:
            elements: An iterator or list yielding individual data elements.
            *args: Additional positional arguments (processor-specific).
            batch_size: The number of elements to include in each batch.
            drop_remainder: Whether to drop the last batch if it's smaller than
                batch_size.
            **kwargs: Additional keyword arguments (processor-specific).

        Returns:
            An iterator or list that yields batches of data elements.

        Raises:
            ValueError: If batch_size is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Subclasses implement the actual batching logic
        raise NotImplementedError(f"{self.__class__.__name__} must implement process() method")
