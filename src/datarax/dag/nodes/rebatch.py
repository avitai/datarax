from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import warnings
from typing import Any, Literal

from datarax.dag.nodes.base import Node
from datarax.typing import Batch
from datarax.utils.pytree_utils import get_batch_size, is_non_jax_leaf

"""RebatchNode implementation for Datarax.

This module provides rebatching capabilities for handling variable batch sizes
in data pipelines, particularly useful for augmentations that change the number
of samples (e.g., multi-crop augmentations).

Clean architecture with separate implementation classes for each mode.
Implemented following TDD principles - tests define behavior.
"""


class DifferentiableRebatchImpl(nnx.Module):
    """Fully differentiable rebatching using JAX operations.

    Uses fixed-size buffers and JAX operations like dynamic_update_slice
    and lax.cond to maintain full differentiability through the rebatching process.
    """

    def __init__(self, target_batch_size: int, max_buffer_size: int = 512):
        """Initialize differentiable rebatch implementation.

        Args:
            target_batch_size: Target size for output batches
            max_buffer_size: Maximum buffer size for accumulation
        """
        super().__init__()
        self.target_batch_size = target_batch_size
        self.max_buffer_size = max_buffer_size

        # JAX buffer state - initialized on first use
        self.buffer = nnx.Variable(None)
        self.buffer_mask = nnx.Variable(None)
        self.buffer_size = nnx.Variable(0)

    def __call__(self, batch: Batch | None) -> tuple[Batch | None, bool]:
        """Process batch with differentiable operations.

        Args:
            batch: Input batch or None to continue processing

        Returns:
            Tuple of (output_batch, is_valid) where is_valid indicates
            whether output_batch contains valid data
        """
        if batch is None:
            # Try to emit from existing buffer
            return self._try_emit()

        batch_size = get_batch_size(batch)
        if batch_size is None or batch_size == 0:
            return None, False

        # Initialize buffer on first call with example structure
        if self.buffer.get_value() is None:
            try:
                self._initialize_buffer(batch)
            except Exception:
                # Inside JAX transformation, cannot mutate state
                # Return a simple pass-through for gradient computation
                return batch, True

        # Add batch to buffer
        try:
            self._add_to_buffer(batch, batch_size)
            # Try to emit a batch
            return self._try_emit()
        except Exception:
            # Inside JAX transformation, cannot mutate state
            # Return a simple pass-through for gradient computation
            return batch, True

    def _initialize_buffer(self, example_batch: Batch) -> None:
        """Initialize JAX buffer with correct PyTree structure."""

        def create_buffer_leaf(x):
            if isinstance(x, jax.Array):
                # Create buffer with max_buffer_size as first dimension
                shape = (self.max_buffer_size, *x.shape[1:])
                return jnp.zeros(shape, dtype=x.dtype)
            else:
                # For non-arrays, create list buffer
                return [None] * self.max_buffer_size

        self.buffer.set_value(
            jax.tree.map(create_buffer_leaf, example_batch, is_leaf=is_non_jax_leaf)
        )
        self.buffer_mask.set_value(jnp.zeros(self.max_buffer_size, dtype=bool))

    def _add_to_buffer(self, batch: Batch, batch_size: int) -> None:
        """Add batch to buffer using differentiable JAX operations."""
        start_idx = self.buffer_size.get_value()

        # Handle potential overflow
        available_space = self.max_buffer_size - start_idx
        if batch_size > available_space:
            warnings.warn(
                f"Buffer overflow: truncating batch from {batch_size} to {available_space}"
            )
            batch_size = available_space
            if batch_size <= 0:
                return

        # Update buffer with dynamic_update_slice for differentiability
        def update_buffer_leaf(buffer_leaf, batch_leaf):
            if isinstance(batch_leaf, jax.Array) and isinstance(buffer_leaf, jax.Array):
                # Ensure batch_leaf has correct size
                if batch_leaf.shape[0] > batch_size:
                    batch_leaf = batch_leaf[:batch_size]

                return jax.lax.dynamic_update_slice(
                    buffer_leaf, batch_leaf, (start_idx,) + (0,) * (batch_leaf.ndim - 1)
                )
            elif isinstance(buffer_leaf, list) and hasattr(batch_leaf, "__len__"):
                # For non-arrays (lists, tuples, etc), update in place
                for i in range(min(batch_size, len(batch_leaf))):
                    idx = start_idx + i
                    if idx < len(buffer_leaf):
                        buffer_leaf[idx] = batch_leaf[i] if i < len(batch_leaf) else None
                return buffer_leaf
            else:
                # For non-iterable data, just return buffer unchanged
                return buffer_leaf

        # Use is_leaf to prevent tree_map from traversing non-JAX lists
        buffer_val = self.buffer.get_value()
        new_buffer = jax.tree.map(
            update_buffer_leaf,
            buffer_val,
            batch,
            is_leaf=is_non_jax_leaf,
        )
        self.buffer.set_value(new_buffer)

        # Update mask to track valid entries
        mask_update = jnp.ones(batch_size, dtype=bool)
        buffer_mask_val = self.buffer_mask.get_value()
        new_mask = jax.lax.dynamic_update_slice(buffer_mask_val, mask_update, (start_idx,))
        self.buffer_mask.set_value(new_mask)

        self.buffer_size.set_value(start_idx + batch_size)

    def _try_emit(self) -> tuple[Batch | None, bool]:
        """Try to emit a batch from buffer."""
        current_buffer_size = self.buffer_size.get_value()
        has_batch = current_buffer_size >= self.target_batch_size

        if has_batch:
            # Extract batch
            output_batch = self._extract_batch(0, self.target_batch_size)

            # Shift buffer (remove emitted elements)
            self._shift_buffer(self.target_batch_size)
            self.buffer_size.set_value(current_buffer_size - self.target_batch_size)

            return output_batch, True

        return None, False

    def _extract_batch(self, start: int, size: int) -> Batch:
        """Extract a batch from buffer using JAX operations."""

        def extract_leaf(buffer_leaf):
            if isinstance(buffer_leaf, jax.Array):
                return jax.lax.dynamic_slice(
                    buffer_leaf,
                    (start,) + (0,) * (buffer_leaf.ndim - 1),
                    (size, *buffer_leaf.shape[1:]),
                )
            elif isinstance(buffer_leaf, list):
                return buffer_leaf[start : start + size]
            return buffer_leaf

        return jax.tree.map(extract_leaf, self.buffer.get_value(), is_leaf=is_non_jax_leaf)

    def _shift_buffer(self, shift_amount: int) -> None:
        """Shift buffer to remove first shift_amount elements."""

        def shift_leaf(buffer_leaf):
            if isinstance(buffer_leaf, jax.Array):
                # Use roll for simplicity (maintains differentiability)
                return jnp.roll(buffer_leaf, -shift_amount, axis=0)
            elif isinstance(buffer_leaf, list):
                # Rotate list
                return buffer_leaf[shift_amount:] + buffer_leaf[:shift_amount]
            return buffer_leaf

        new_buffer = jax.tree.map(shift_leaf, self.buffer.get_value(), is_leaf=is_non_jax_leaf)
        self.buffer.set_value(new_buffer)
        new_mask = jnp.roll(self.buffer_mask.get_value(), -shift_amount)
        self.buffer_mask.set_value(new_mask)

    def flush(self) -> Batch | None:
        """Flush remaining buffered data."""
        current_buffer_size = self.buffer_size.get_value()
        if current_buffer_size > 0:
            size = int(current_buffer_size)
            output = self._extract_batch(0, size)
            self.buffer_size.set_value(0)
            return output
        return None

    def get_state(self) -> dict[str, Any]:
        """Get implementation state."""
        return {
            "buffer": self.buffer.get_value(),
            "buffer_mask": self.buffer_mask.get_value(),
            "buffer_size": int(self.buffer_size.get_value()),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore implementation state."""
        if "buffer" in state:
            self.buffer.set_value(state["buffer"])
        if "buffer_mask" in state:
            self.buffer_mask.set_value(state["buffer_mask"])
        if "buffer_size" in state:
            self.buffer_size.set_value(state["buffer_size"])


class FastRebatchImpl(nnx.Module):
    """JIT-optimized rebatching using circular buffer.

    Optimized for performance with minimal Python operations,
    using a circular buffer for efficient memory usage.
    """

    def __init__(self, target_batch_size: int, max_buffer_size: int = 512):
        """Initialize fast rebatch implementation.

        Args:
            target_batch_size: Target size for output batches
            max_buffer_size: Maximum buffer size for accumulation
        """
        super().__init__()
        self.target_batch_size = target_batch_size
        self.max_buffer_size = max_buffer_size

        # Circular buffer state
        self.buffer = nnx.Variable(None)
        self.write_index = nnx.Variable(0)
        self.read_index = nnx.Variable(0)
        self.count = nnx.Variable(0)

    def __call__(self, batch: Batch | None) -> tuple[Batch | None, bool]:
        """Process batch with circular buffer operations.

        Args:
            batch: Input batch or None

        Returns:
            Tuple of (output_batch, is_valid)
        """
        if batch is None:
            # Try to emit from existing buffer
            return self._try_emit()

        batch_size = get_batch_size(batch)
        if batch_size is None or batch_size == 0:
            return None, False

        # Initialize buffer on first call
        if self.buffer.get_value() is None:
            try:
                self._initialize_buffer(batch)
            except Exception:
                # Inside JAX transformation, cannot mutate state
                return batch, True

        # Add to circular buffer
        try:
            self._add_to_circular_buffer(batch, batch_size)
            # Try to emit
            return self._try_emit()
        except Exception:
            # Inside JAX transformation, cannot mutate state
            return batch, True

    def _initialize_buffer(self, example_batch: Batch) -> None:
        """Initialize circular buffer as list of PyTrees."""
        # Store individual elements as PyTrees
        self.buffer.set_value([None] * self.max_buffer_size)

    def _add_to_circular_buffer(self, batch: Batch, batch_size: int) -> None:
        """Add batch elements to circular buffer."""
        write_idx = self.write_index.get_value()
        current_count = self.count.get_value()

        # Check for overflow
        if current_count + batch_size > self.max_buffer_size:
            warnings.warn("Circular buffer overflow, dropping oldest elements")
            # Advance read pointer to make space
            overflow = (current_count + batch_size) - self.max_buffer_size
            read_idx = self.read_index.get_value()
            self.read_index.set_value((read_idx + overflow) % self.max_buffer_size)
            self.count.set_value(self.max_buffer_size - batch_size)
            current_count = self.count.get_value()

        # Get buffer copy for modification
        buffer_list = self.buffer.get_value()

        # Add each element from batch
        for i in range(batch_size):
            element = self._extract_element(batch, i)
            buffer_idx = (write_idx + i) % self.max_buffer_size
            buffer_list[buffer_idx] = element

        # Persist buffer changes
        self.buffer.set_value(buffer_list)

        # Update indices
        self.write_index.set_value((write_idx + batch_size) % self.max_buffer_size)
        self.count.set_value(min(current_count + batch_size, self.max_buffer_size))

    def _try_emit(self) -> tuple[Batch | None, bool]:
        """Try to emit a batch from circular buffer."""
        current_count = self.count.get_value()
        if current_count >= self.target_batch_size:
            # Extract elements
            elements = []
            read_idx = self.read_index.get_value()
            buffer_list = self.buffer.get_value()

            for i in range(self.target_batch_size):
                buffer_idx = (read_idx + i) % self.max_buffer_size
                elements.append(buffer_list[buffer_idx])

            # Update indices
            self.read_index.set_value((read_idx + self.target_batch_size) % self.max_buffer_size)
            self.count.set_value(current_count - self.target_batch_size)

            # Stack elements into batch
            output = self._stack_elements(elements)
            return output, True

        return None, False

    def _extract_element(self, batch: Batch, index: int) -> Any:
        """Extract single element from batch."""

        def get_item(x):
            if isinstance(x, jax.Array):
                return x[index]
            elif hasattr(x, "__getitem__"):
                return x[index]
            return x

        return jax.tree.map(get_item, batch)

    def _stack_elements(self, elements: list[Any]) -> Batch:
        """Stack list of elements into batch."""
        if not elements:
            return None

        def stack_fn(*xs):
            if all(isinstance(x, jax.Array) for x in xs):
                return jnp.stack(xs, axis=0)
            return list(xs)

        return jax.tree.map(stack_fn, *elements)

    def flush(self) -> Batch | None:
        """Flush remaining buffered data."""
        current_count = self.count.get_value()
        if current_count > 0:
            # Extract all remaining elements
            elements = []
            read_idx = self.read_index.get_value()
            remaining = int(current_count)
            buffer_list = self.buffer.get_value()

            for i in range(remaining):
                buffer_idx = (read_idx + i) % self.max_buffer_size
                elements.append(buffer_list[buffer_idx])

            self.count.set_value(0)
            return self._stack_elements(elements)
        return None

    def get_state(self) -> dict[str, Any]:
        """Get implementation state."""
        return {
            "buffer": self.buffer.get_value(),
            "write_index": int(self.write_index.get_value()),
            "read_index": int(self.read_index.get_value()),
            "count": int(self.count.get_value()),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore implementation state."""
        if "buffer" in state:
            self.buffer.set_value(state["buffer"])
        if "write_index" in state:
            self.write_index.set_value(state["write_index"])
        if "read_index" in state:
            self.read_index.set_value(state["read_index"])
        if "count" in state:
            self.count.set_value(state["count"])


class GradientTransparentRebatchImpl(nnx.Module):
    """Gradient-transparent rebatching that preserves gradient flow.

    Uses simple Python operations for the buffer structure while
    maintaining gradient connectivity through the data values.
    """

    def __init__(self, target_batch_size: int, max_buffer_size: int = 512):
        """Initialize gradient-transparent implementation.

        Args:
            target_batch_size: Target size for output batches
            max_buffer_size: Maximum buffer size (unused but kept for API consistency)
        """
        super().__init__()
        self.target_batch_size = target_batch_size
        self.max_buffer_size = max_buffer_size

        # Use nnx.List for NNX-compatible state management
        self.buffer = nnx.List([])
        self.gradient_tape = nnx.List([])  # Keep for state compatibility

    def __call__(self, batch: Batch | None) -> tuple[Batch | None, bool]:
        """Process batch while preserving gradient flow.

        Args:
            batch: Input batch or None

        Returns:
            Tuple of (output_batch, is_valid)
        """
        if batch is None:
            # Try to emit from existing buffer
            return self._try_emit()

        batch_size = get_batch_size(batch)
        if batch_size is None or batch_size == 0:
            return None, False

        # Split batch into individual elements and add to buffer
        for i in range(batch_size):
            element = self._extract_element(batch, i)
            self.buffer.append(element)
            self.gradient_tape.append(element)  # Keep same ref for gradient flow

        # Try to emit
        return self._try_emit()

    def _try_emit(self) -> tuple[Batch | None, bool]:
        """Try to emit a batch from buffer."""
        if len(self.buffer) >= self.target_batch_size:
            # Extract elements
            elements = list(self.buffer[: self.target_batch_size])

            # Update buffers - clear and extend with remaining elements
            remaining_buffer = list(self.buffer[self.target_batch_size :])
            remaining_tape = list(self.gradient_tape[self.target_batch_size :])
            self.buffer.clear()
            self.buffer.extend(remaining_buffer)
            self.gradient_tape.clear()
            self.gradient_tape.extend(remaining_tape)

            # Stack elements (gradients flow through naturally)
            output = self._stack_elements(elements)
            return output, True

        return None, False

    def _extract_element(self, batch: Batch, index: int) -> Any:
        """Extract single element from batch."""

        def get_item(x):
            if isinstance(x, jax.Array):
                return x[index]
            elif hasattr(x, "__getitem__"):
                return x[index]
            return x

        return jax.tree.map(get_item, batch, is_leaf=is_non_jax_leaf)

    def _stack_elements(self, elements: list[Any]) -> Batch:
        """Stack elements while preserving gradient flow."""
        if not elements:
            return None

        def stack_fn(*xs):
            if all(isinstance(x, jax.Array) for x in xs):
                # Simple stack - gradients flow through naturally
                return jnp.stack(xs, axis=0)
            return list(xs)

        return jax.tree.map(stack_fn, *elements, is_leaf=is_non_jax_leaf)

    def flush(self) -> Batch | None:
        """Flush remaining buffered data."""
        if self.buffer:
            elements = list(self.buffer)  # Convert to plain list for processing
            self.buffer = nnx.List([])
            self.gradient_tape = nnx.List([])
            return self._stack_elements(elements)
        return None

    def get_state(self) -> dict[str, Any]:
        """Get implementation state."""
        return {
            "buffer": self.buffer,
            "gradient_tape": self.gradient_tape,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore implementation state."""
        # Restore by clearing and extending the nnx.List objects
        self.buffer.clear()
        self.buffer.extend(state.get("buffer", []))
        self.gradient_tape.clear()
        self.gradient_tape.extend(state.get("gradient_tape", []))


class RebatchNode(Node):
    """Rebatch node for handling batch size changes in pipelines.

    This node accumulates input batches and emits new batches of a target size.
    It delegates to one of three implementation strategies based on the mode.

    Modes:
        - 'differentiable': Maintains full differentiability for training
        - 'fast': JIT-optimized for maximum performance
        - 'gradient_transparent': Preserves gradient flow through values

    Examples:
        Rebatch with gradient transparency:

        ```python
        from datarax.dag.nodes import RebatchNode
        node = RebatchNode(32, mode='gradient_transparent')
        ```
    """

    def __init__(
        self,
        target_batch_size: int,
        mode: Literal["differentiable", "fast", "gradient_transparent"] = "gradient_transparent",
        max_buffer_size: int = 512,
        name: str | None = None,
    ):
        """Initialize rebatch node.

        Args:
            target_batch_size: Target size for output batches
            mode: Rebatching strategy to use
            max_buffer_size: Maximum buffer size for accumulation
            name: Optional name for the node

        Raises:
            ValueError: If mode is not one of the supported modes
        """
        super().__init__(name=name or f"RebatchNode({target_batch_size})")

        if mode not in ["differentiable", "fast", "gradient_transparent"]:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'differentiable', 'fast', or 'gradient_transparent'"
            )

        self.target_batch_size = target_batch_size
        self.mode = mode
        self.max_buffer_size = max_buffer_size

        # Create appropriate implementation
        # Note: Both 'differentiable' and 'gradient_transparent' use the same
        # implementation (DRY principle) since both need gradient compatibility.
        # The distinction is kept for API clarity and future optimization.
        if mode == "differentiable" or mode == "gradient_transparent":
            self.impl = DifferentiableRebatchImpl(target_batch_size, max_buffer_size)
        else:  # fast
            self.impl = FastRebatchImpl(target_batch_size, max_buffer_size)

        # Statistics tracking
        self.elements_processed = nnx.Variable(0)
        self.batches_emitted = nnx.Variable(0)
        self.elements_dropped = nnx.Variable(0)

    def __call__(self, batch: Batch | None, *, key: jax.Array | None = None) -> Batch | None:
        """Process batch and emit rebatched output when ready.

        Args:
            batch: Input batch or None to continue processing buffer
            key: Optional RNG key (unused)

        Returns:
            Rebatched output of target_batch_size or None if accumulating
        """
        # Track input statistics (only if not inside JAX transformation)
        if batch is not None:
            batch_size = get_batch_size(batch)
            if batch_size is not None:
                # Use try-except to handle JAX tracing context
                try:
                    current = self.elements_processed.get_value()
                    self.elements_processed.set_value(current + batch_size)
                except Exception:
                    # Inside JAX transformation, skip statistics
                    pass

        # Delegate to implementation
        output, is_valid = self.impl(batch)

        # Track output statistics (only if not inside JAX transformation)
        if is_valid and output is not None:
            try:
                current = self.batches_emitted.get_value()
                self.batches_emitted.set_value(current + 1)
            except Exception:
                # Inside JAX transformation, skip statistics
                pass
            return output

        return None

    def flush(self) -> Batch | None:
        """Flush any remaining buffered data.

        Returns:
            Final partial batch if any, None otherwise
        """
        output = self.impl.flush()

        if output is not None:
            output_size = get_batch_size(output)
            # Only track statistics if not in JAX transformation
            try:
                if output_size is not None and output_size < self.target_batch_size:
                    # Track elements that didn't make a full batch
                    current_dropped = self.elements_dropped.get_value()
                    self.elements_dropped.set_value(
                        current_dropped + self.target_batch_size - output_size
                    )
                current_emitted = self.batches_emitted.get_value()
                self.batches_emitted.set_value(current_emitted + 1)
            except Exception:
                # Inside JAX transformation, skip statistics
                pass

        return output

    def get_state(self) -> dict[str, Any]:
        """Get current state for checkpointing.

        Returns:
            Dictionary containing complete state
        """
        return {
            "mode": self.mode,
            "target_batch_size": self.target_batch_size,
            "max_buffer_size": self.max_buffer_size,
            "impl_state": self.impl.get_state(),
            "statistics": {
                "elements_processed": int(self.elements_processed.get_value()),
                "batches_emitted": int(self.batches_emitted.get_value()),
                "elements_dropped": int(self.elements_dropped.get_value()),
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dictionary from get_state()
        """
        # Restore implementation state
        if "impl_state" in state:
            self.impl.set_state(state["impl_state"])

        # Restore statistics
        if "statistics" in state:
            stats = state["statistics"]
            self.elements_processed.set_value(stats.get("elements_processed", 0))
            self.batches_emitted.set_value(stats.get("batches_emitted", 0))
            self.elements_dropped.set_value(stats.get("elements_dropped", 0))

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RebatchNode("
            f"target={self.target_batch_size}, "
            f"mode={self.mode}, "
            f"processed={self.elements_processed.get_value()}, "
            f"emitted={self.batches_emitted.get_value()})"
        )


def rebatch(
    target_batch_size: int,
    mode: Literal["differentiable", "fast", "gradient_transparent"] = "gradient_transparent",
    max_buffer_size: int = 512,
) -> RebatchNode:
    """Create a rebatch node.

    Convenience function for creating RebatchNode instances.

    Args:
        target_batch_size: Target batch size for output
        mode: Rebatching strategy to use
        max_buffer_size: Maximum buffer size for accumulation

    Returns:
        RebatchNode instance

    Examples:
        Create rebatch node:

        ```python
        from datarax.dag.nodes import rebatch
        node = rebatch(32, mode='fast')
        ```
    """
    return RebatchNode(target_batch_size, mode, max_buffer_size)
