"""Data source, batch, and shuffle node wrappers for the DAG."""

from __future__ import annotations
import jax

import flax.nnx as nnx
from typing import Any, Union
from collections.abc import Iterator

from datarax.dag.nodes.base import Node
from datarax.typing import Batch, Element
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.batcher import BatcherModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule


class DataSourceNode(Node):
    """Node wrapper for data sources.

    This node wraps a DataSourceModule to act as the entry point
    for a DAG pipeline. It maintains iteration state and supports
    checkpointing.
    """

    def __init__(self, source: DataSourceModule, name: str | None = None):
        """Initialize data source node.

        Args:
            source: DataSourceModule to wrap
            name: Optional name for the node
        """
        super().__init__(name=name or "DataSource")
        self.source = source
        # CRITICAL: Use object.__setattr__ to bypass NNX attribute tracking.
        # - nnx.Variable creates copies on access, breaking iterator state
        # - Plain assignment triggers NNX's static/data type checking
        # - object.__setattr__ creates a truly private Python attribute
        object.__setattr__(self, "_iterator", None)

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Element | None:
        """Get next element from source.

        Args:
            data: Ignored (source generates data)
            key: Optional RNG key

        Returns:
            Next element from source, or None if exhausted
        """
        if self._iterator is None:
            object.__setattr__(self, "_iterator", iter(self.source))

        try:
            return next(self._iterator)
        except StopIteration:
            # Try to reset for next epoch
            try:
                object.__setattr__(self, "_iterator", iter(self.source))
                return next(self._iterator)
            except StopIteration:
                # Source is truly exhausted
                return None

    def __iter__(self):
        """Iterate over source."""
        return iter(self.source)

    def get_state(self) -> dict[str, Any]:
        """Get source state."""
        state = {}
        if hasattr(self.source, "get_state"):
            state["source_state"] = self.source.get_state()
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore source state."""
        if "source_state" in state and hasattr(self.source, "set_state"):
            self.source.set_state(state["source_state"])


class BatchNode(Node):
    """Node for creating batches from elements.

    CRITICAL: Enforces Datarax's batch-first principle.
    All downstream nodes receive batched data.
    """

    def __init__(self, batch_size: int, drop_remainder: bool = False, name: str | None = None):
        """Initialize batch node.

        Args:
            batch_size: Number of elements per batch
            drop_remainder: Whether to drop incomplete final batch
            name: Optional name for the node
        """
        super().__init__(name=name or "Batch")
        self.batch_size = batch_size
        self.drop_remainder = drop_remainder
        # Use nnx.data() to mark buffer as pytree data (traced but not trainable)
        # This allows the buffer to contain JAX arrays without NNX static attribute errors
        self._buffer = nnx.data([])

    @property
    def buffer(self) -> list[Element]:
        """Get the buffer list."""
        return self._buffer

    def __call__(self, data: Element | Batch, *, key: jax.Array | None = None) -> Batch | None:
        """Add element to buffer and return batch when ready.

        Handles both Element inputs (normal case) and Batch inputs (re-batching).
        When receiving a Batch, unbatches it into individual elements first.

        Args:
            data: Single element or Batch to add to buffer
            key: Optional RNG key (unused)

        Returns:
            Batch when buffer is full, None otherwise
        """

        # Handle Batch input: unbatch into individual elements
        if isinstance(data, Batch):
            # Extract all elements from the batch and add to buffer
            for i in range(data.batch_size):
                element = data.get_element(i)
                self._buffer.append(element)
        else:
            # Normal case: add single element to buffer
            self._buffer.append(data)

        # Collect all ready batches (there could be multiple if a large Batch was added)
        results = []
        while len(self._buffer) >= self.batch_size:
            batch = self._create_batch(self._buffer[: self.batch_size])
            # Replace buffer with remaining elements (wrap with nnx.data)
            self._buffer = nnx.data(self._buffer[self.batch_size :])
            if batch is not None:
                results.append(batch)

        # Return the first ready batch (or None if not ready)
        # Note: Additional batches will be returned on subsequent calls
        if results:
            return results[0]
        return None

    def flush(self) -> Batch | None:
        """Flush remaining elements as final batch.

        Returns:
            Final batch or None if drop_remainder=True
        """
        if not self._buffer:
            return None

        if self.drop_remainder:
            self._buffer = nnx.data([])
            return None

        batch = self._create_batch(self._buffer)
        self._buffer = nnx.data([])
        return batch

    def _create_batch(self, elements: list[Element]) -> Batch:
        """Create batch from list of elements.

        Args:
            elements: List of elements to batch

        Returns:
            Batched data with batch dimension as first axis
        """
        if not elements:
            return None

        # Convert elements to proper Element objects if they aren't already

        proper_elements = []
        for element in elements:
            if isinstance(element, Element):
                proper_elements.append(element)
            elif isinstance(element, dict):
                # Check if dict is already in Element format (has 'data' key with nested structure)
                # vs being the actual data payload itself
                # Element format: {"data": {...}, "state": {...}, "metadata": ...}
                # Data payload: {"data": array, "labels": array} or {"images": array}
                is_element_format = (
                    "data" in element
                    and "state" in element
                    and isinstance(element.get("data"), dict)
                )
                if is_element_format:
                    # Element format - extract parts
                    data = element["data"]
                    state = element.get("state", {})
                    metadata = element.get("metadata", None)
                    proper_elements.append(Element(data=data, state=state, metadata=metadata))
                else:
                    # Dict IS the data payload - use whole dict as Element.data
                    proper_elements.append(Element(data=element, state={}, metadata=None))
            else:
                # Assume it's raw data
                proper_elements.append(Element(data=element, state={}, metadata=None))

        return Batch(proper_elements, validate=False)

    def get_state(self) -> dict[str, Any]:
        """Get buffer state."""
        return {"buffer": list(self._buffer)}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore buffer state."""
        if "buffer" in state:
            # Wrap with nnx.data() to mark as pytree data
            self._buffer = nnx.data(list(state["buffer"]))

    def __repr__(self) -> str:
        """String representation."""
        return f"BatchNode(batch_size={self.batch_size}, drop_remainder={self.drop_remainder})"


class OperatorNode(Node):
    """Node wrapper for OperatorModule.

    Wraps an OperatorModule to work in the DAG.
    Handles both deterministic and stochastic operators.
    Ensures batch-based processing.
    """

    def __init__(
        self,
        operator: Union[OperatorModule, BatcherModule],
        name: str | None = None,
    ):
        """Initialize operator node.

        Args:
            operator: OperatorModule to wrap
            name: Optional name for the node
        """
        # Get name from operator if available, otherwise use class name
        if name is None:
            name = getattr(operator, "name", operator.__class__.__name__)
        super().__init__(name=name)
        self.operator = operator

        # Ensure stochastic operators have RNG
        if hasattr(operator, "config") and getattr(operator.config, "stochastic", False):
            if not hasattr(operator, "rngs") or operator.rngs is None:
                operator.rngs = nnx.Rngs(operator=0)

    @property
    def is_jit_fusible(self) -> bool:
        """Operator node is always fusible — nnx.jit handles all NNX state.

        This includes stochastic operators (Rngs extracted by nnx.jit),
        operators with nnx.Param (included in grad), and operators with
        IterationCount (jnp.array, JIT-mutable via slice assignment).
        """
        return True

    def __call__(self, data: Batch, *, key: jax.Array | None = None) -> Batch:
        """Apply operator to batch.

        Args:
            data: Batch to process
            key: Optional RNG key (unused - operators manage their own RNG)

        Returns:
            Processed batch

        Raises:
            TypeError: If data is not a Batch object
        """
        # DAG nodes always work with Batch objects
        if not isinstance(data, Batch):
            raise TypeError(
                f"{self.name} expects Batch object, got {type(data).__name__}. "
                f"Ensure BatchNode is used before operators."
            )

        # Apply operator - OperatorModule handles batch processing internally
        return self.operator(data)

    def get_state(self) -> dict[str, Any]:
        """Get operator state."""
        if hasattr(self.operator, "get_state"):
            return self.operator.get_state()
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore operator state."""
        if hasattr(self.operator, "set_state"):
            self.operator.set_state(state)


class ShuffleNode(Node):
    """Node for shuffling data using a buffer.

    Implements reservoir sampling for efficient shuffling.
    """

    def __init__(self, buffer_size: int, seed: int | None = None, name: str | None = None):
        """Initialize shuffle node.

        Args:
            buffer_size: Size of shuffle buffer
            seed: Random seed for reproducibility
            name: Optional name
        """
        super().__init__(name=name or "Shuffle")
        self.buffer_size = buffer_size
        self.seed = seed  # Store seed for inspection
        # Use nnx.data() to mark buffer as pytree data (traced but not trainable)
        # This allows the buffer to contain JAX arrays without NNX static attribute errors
        self._buffer = nnx.data([])
        self.rng_key = jax.random.key(seed if seed is not None else 0)
        self.iteration = nnx.Variable(0)

    @property
    def buffer(self) -> list[Any]:
        """Get the buffer list."""
        return self._buffer

    @property
    def buffer_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.buffer_size

    def __call__(self, data: Batch, *, key: jax.Array | None = None) -> Batch | None:
        """Add to buffer and return random element if full.

        Args:
            data: Batch to add to buffer
            key: Optional RNG key (overrides internal)

        Returns:
            Random batch from buffer if full, None otherwise
        """
        # Use provided key or generate new one
        if key is None:
            current_iteration = self.iteration.get_value() + 1
            self.iteration.set_value(current_iteration)
            key = jax.random.fold_in(self.rng_key, current_iteration)

        # Fill buffer initially
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(data)
            return None

        # Buffer is full - swap and return
        idx = jax.random.randint(key, (), 0, self.buffer_size)
        idx_int = int(idx)  # Convert to Python int for indexing

        output = self._buffer[idx_int]
        self._buffer[idx_int] = data

        return output

    def flush(self) -> Batch | None:
        """Flush one random element from buffer.

        Returns:
            Random batch from buffer or None if empty
        """
        if not self._buffer:
            return None

        current_iteration = self.iteration.get_value() + 1
        self.iteration.set_value(current_iteration)
        key = jax.random.fold_in(self.rng_key, current_iteration)
        idx = jax.random.randint(key, (), 0, len(self._buffer))
        idx_int = int(idx)  # Convert to Python int for indexing

        output = self._buffer[idx_int]
        self._buffer.pop(idx_int)

        return output

    def get_state(self) -> dict[str, Any]:
        """Get shuffle state."""
        return {"buffer": list(self._buffer), "iteration": self.iteration.get_value()}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore shuffle state."""
        if "buffer" in state:
            # Wrap with nnx.data() to mark as pytree data
            self._buffer = nnx.data(list(state["buffer"]))
        if "iteration" in state:
            self.iteration.set_value(state["iteration"])

    def __repr__(self) -> str:
        """String representation."""
        return f"ShuffleNode(buffer_size={self.buffer_size}, seed={self.seed})"


class PrefetchNode(Node):
    """Node that prefetches data in the background using threading.

    This node wraps an iterator to load data asynchronously in a background
    thread, allowing computation to overlap with data loading for better
    performance.
    """

    def __init__(self, buffer_size: int = 2, name: str | None = None):
        """Initialize prefetch node.

        Args:
            buffer_size: Number of batches to prefetch (default: 2)
            name: Optional name for the node
        """
        super().__init__(name=name or f"Prefetch(buffer_size={buffer_size})")
        self.buffer_size = buffer_size
        self._prefetcher = None

    def __call__(self, data: Any, *, key: jax.Array | None = None) -> Any:
        """Pass through data without modification.

        Note: Prefetching is handled at the iterator level, not per-batch.

        Args:
            data: Input data
            key: Optional RNG key (unused)

        Returns:
            Input data unchanged
        """
        return data

    def wrap_iterator(self, iterator: Iterator[Any]) -> Iterator[Any]:
        """Wrap an iterator with prefetching.

        Args:
            iterator: Iterator to wrap

        Returns:
            Prefetching iterator
        """
        from datarax.control.prefetcher import Prefetcher

        if self._prefetcher is None:
            self._prefetcher = Prefetcher(buffer_size=self.buffer_size)

        return self._prefetcher.prefetch(iterator)

    def reset(self) -> None:
        """Reset the prefetcher state."""
        self._prefetcher = None

    def get_state(self) -> dict[str, Any]:
        """Get prefetch state."""
        return {"buffer_size": self.buffer_size}

    def set_state(self, state: dict[str, Any]) -> None:
        """Set prefetch state."""
        if "buffer_size" in state:
            self.buffer_size = state["buffer_size"]


class SamplerNode(Node):
    """Node wrapper for samplers.

    Wraps a SamplerModule to work in the DAG.
    Samplers control the iteration order of data elements.
    """

    def __init__(self, sampler: SamplerModule, name: str | None = None):
        """Initialize sampler node.

        Args:
            sampler: SamplerModule to wrap
            name: Optional name for the node
        """
        # Get name from sampler if available, otherwise use class name
        if name is None:
            name = getattr(sampler, "name", sampler.__class__.__name__)
        super().__init__(name=name)
        self.sampler = sampler

    def __call__(self, data: Element, *, key: jax.Array | None = None) -> Element:
        """Apply sampling to data.

        Args:
            data: Element to sample
            key: Optional RNG key for stochastic sampling

        Returns:
            Sampled element
        """
        # Samplers typically work on indices or elements directly
        # They control iteration order, not transformation
        return self.sampler(data, key=key) if key is not None else self.sampler(data)

    def __iter__(self):
        """Create iterator from sampler.

        Returns:
            Iterator over sampled elements
        """
        return iter(self.sampler)

    def __next__(self):
        """Get next sampled element.

        Returns:
            Next element according to sampling strategy
        """
        return next(self.sampler)

    def get_state(self) -> dict[str, Any]:
        """Get sampler state."""
        if hasattr(self.sampler, "get_state"):
            return self.sampler.get_state()
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore sampler state."""
        if hasattr(self.sampler, "set_state"):
            self.sampler.set_state(state)


class SharderNode(Node):
    """Node wrapper for sharders.

    Wraps a SharderModule to work in the DAG.
    Sharders distribute data across devices/processes.
    """

    def __init__(self, sharder: SharderModule, name: str | None = None):
        """Initialize sharder node.

        Args:
            sharder: SharderModule to wrap
            name: Optional name for the node
        """
        # Get name from sharder if available, otherwise use class name
        if name is None:
            name = getattr(sharder, "name", sharder.__class__.__name__)
        super().__init__(name=name)
        self.sharder = sharder

    def __call__(self, data: Batch, *, key: jax.Array | None = None) -> Batch:
        """Apply sharding to batch.

        Args:
            data: Batch to shard across devices
            key: Optional RNG key (usually not needed for sharding)

        Returns:
            Sharded batch
        """
        # Sharders distribute data across devices
        # They typically work on batched data
        return self.sharder(data)

    def get_state(self) -> dict[str, Any]:
        """Get sharder state."""
        if hasattr(self.sharder, "get_state"):
            return self.sharder.get_state()
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore sharder state."""
        if hasattr(self.sharder, "set_state"):
            self.sharder.set_state(state)
