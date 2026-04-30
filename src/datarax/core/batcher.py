"""Base module for batcher components in Datarax.

This module defines the base class for all Datarax batcher components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

import logging
from collections.abc import Iterator
from typing import Any

from datarax.core.structural import StructuralModule
from datarax.typing import Batch, Element


logger = logging.getLogger(__name__)


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
        del elements, args, drop_remainder, kwargs
        raise NotImplementedError(f"{self.__class__.__name__} must implement process() method")

    def batch_spec(self, element_spec: Any, *, batch_size: int) -> dict[str, Any]:
        """Return the batched-output spec given a per-element spec and ``batch_size``.

        The default implementation prepends a leading ``(batch_size,)`` dimension
        to every ``jax.ShapeDtypeStruct`` leaf of ``element_spec`` and adds a
        top-level ``valid_mask`` leaf of shape ``(batch_size,)`` and dtype bool.
        The mask flags valid positions so end-of-epoch padding does not
        contribute to mask-weighted loss.

        Subclasses (e.g., ``MultiRateBatcher``) override only when the batch
        layout requires more than a simple leading-dim prepend.

        Args:
            element_spec: PyTree of ``jax.ShapeDtypeStruct`` describing one
                element (typically the output of the upstream operator's
                ``output_spec`` or the source's ``element_spec``).
            batch_size: Number of elements per emitted batch.

        Returns:
            A dict containing the batched element spec under the original keys
            plus a ``"valid_mask"`` key of shape ``(batch_size,)`` and bool dtype.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        # Imported lazily to keep core/batcher import lightweight.
        from datarax.utils.spec import batched_spec

        return batched_spec(element_spec, batch_size=batch_size)
