"""Default batcher module implementation for Datarax.

This module provides a default implementation of the BatcherModule interface
that handles batching of PyTrees.
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.batcher import BatcherModule
from datarax.core.config import StructuralConfig
from datarax.typing import Batch, Element


@dataclass
class DefaultBatcherConfig(StructuralConfig):
    """Configuration for DefaultBatcher.

    DefaultBatcher is deterministic and requires no additional configuration
    beyond the base StructuralConfig.
    """

    def __post_init__(self):
        """Validate configuration after initialization."""
        # DefaultBatcher is deterministic (no randomness)
        object.__setattr__(self, "stochastic", False)
        super().__post_init__()


class DefaultBatcher(BatcherModule):
    """Default implementation of the BatcherModule interface.

    This batcher module accumulates individual data elements and forms batches
    by stacking arrays along a new leading dimension. It handles PyTrees of
    arbitrary structure, maintaining the same structure in the batched output.
    """

    def __init__(
        self,
        config: DefaultBatcherConfig,
        *,
        collate_fn: Callable[[list[Element]], Batch] | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
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
        *args: Any,
        batch_size: int,
        drop_remainder: bool = False,
        **kwargs: Any,
    ) -> Iterator[Batch]:
        """Group individual data elements into batches.

        Args:
            elements: An iterator yielding individual data elements.
            *args: Additional positional arguments (ignored).
            batch_size: The number of elements to include in each batch.
            drop_remainder: Whether to drop the last batch if it's smaller than
                batch_size.
            **kwargs: Additional keyword arguments (ignored).

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

    def _safe_tree_map(self, f: Callable[..., Any], *trees: Any) -> Any:
        """Handle PRNGKey dtypes in tree_map operations.

        This traverses the tree and pre-processes any PRNGKey dtypes before
        applying the function f.

        Args:
            f: Function to apply to leaves
            *trees: PyTrees to process

        Returns:
            A PyTree with the same structure as the input trees
        """

        from datarax.utils.pytree_utils import is_batch_leaf

        def is_prng_key(x: Any) -> bool:
            return hasattr(x, "dtype") and str(x.dtype) == "prng_key"

        def pre_process_leaf(*xs: Any) -> Any:
            # Convert any PRNGKey values to their underlying integer data
            processed_xs: list[Any] = []
            for x in xs:
                if is_prng_key(x):
                    processed_xs.append(jax.random.key_data(x))
                else:
                    processed_xs.append(x)
            return f(*processed_xs)

        result = jax.tree.map(pre_process_leaf, *trees, is_leaf=is_batch_leaf)
        return result

    def _collate_batch(self, elements: list[Element]) -> Batch:
        """Combine a list of elements into a batch.

        Args:
            elements: A list of data elements to combine.

        Returns:
            A batch containing the combined elements.
        """
        if self.collate_fn is not None:
            return self.collate_fn(elements)

        # Use our safe tree_map implementation that handles PRNGKey dtypes
        return self._safe_tree_map(self._stack_leaves, *elements)

    @staticmethod
    def _stack_leaves(
        *leaves: jax.Array | np.ndarray | Any,
    ) -> jax.Array | list:
        """Stack leaf values from multiple elements.

        Args:
            *leaves: Leaf values from multiple elements to stack.

        Returns:
            A stacked JAX array or a list if the leaves cannot be stacked.
        """
        # If all leaves are arrays, stack them as JAX arrays
        if all(isinstance(leaf, jax.Array | np.ndarray) for leaf in leaves):
            try:
                # Always return JAX arrays for consistency
                return jnp.stack(leaves)
            except Exception:
                # If stacking fails, fall back to a list
                return list(leaves)

        # If leaves are scalars of the same type, convert to JAX array
        scalar_types = (int, float, bool, np.integer, np.floating, np.bool_)
        if all(isinstance(leaf, scalar_types) for leaf in leaves):
            return jnp.array(leaves)

        # If nothing else works, just return a list
        return list(leaves)
