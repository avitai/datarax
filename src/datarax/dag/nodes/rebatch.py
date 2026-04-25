from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import flax.nnx as nnx
import grain
import jax
import jax.numpy as jnp
import numpy as np

from datarax.dag.nodes.base import Node
from datarax.sources._grain_streaming import ensure_iter_dataset
from datarax.typing import Batch
from datarax.utils.pytree_utils import get_batch_size, is_non_jax_leaf


logger = logging.getLogger(__name__)


"""RebatchNode implementation for Datarax.

This module keeps differentiable in-DAG rebatching for JAX/NNX operator
pipelines. Ordinary iterator rebatching belongs to Grain and is exposed through
``rebatch_iterable``.
"""


class DifferentiableRebatchImpl(nnx.Module):
    """Fully differentiable rebatching using JAX operations.

    Uses fixed-size buffers and JAX operations like dynamic_update_slice
    and lax.cond to maintain full differentiability through the rebatching process.
    """

    def __init__(self, target_batch_size: int, max_buffer_size: int = 512) -> None:
        """Initialize differentiable rebatch implementation.

        Args:
            target_batch_size: Target size for output batches
            max_buffer_size: Maximum buffer size for accumulation
        """
        super().__init__()
        self.target_batch_size = target_batch_size
        self.max_buffer_size = max_buffer_size

        # JAX buffer state - initialized on first use
        self.buffer: nnx.Variable[Any] = nnx.Variable(None)
        self.buffer_mask: nnx.Variable[jax.Array | None] = nnx.Variable(None)
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
            except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
                # Inside JAX transformation, cannot mutate state
                # Return a simple pass-through for gradient computation
                return batch, True

        # Add batch to buffer
        try:
            self._add_to_buffer(batch, batch_size)
            # Try to emit a batch
            return self._try_emit()
        except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
            # Inside JAX transformation, cannot mutate state
            # Return a simple pass-through for gradient computation
            return batch, True

    def _initialize_buffer(self, example_batch: Batch) -> None:
        """Initialize JAX buffer with correct PyTree structure."""

        def create_buffer_leaf(x) -> Any:
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
        def write_leaf_to_buffer(buffer_leaf: Any, batch_leaf: Any) -> Any:
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
            write_leaf_to_buffer,
            buffer_val,
            batch,
            is_leaf=is_non_jax_leaf,
        )
        self.buffer.set_value(new_buffer)

        # Update mask to track valid entries
        mask_update = jnp.ones(batch_size, dtype=bool)
        buffer_mask_val = self.buffer_mask.get_value()
        assert buffer_mask_val is not None
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

        def extract_leaf(buffer_leaf: Any) -> Any:
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

        def shift_leaf(buffer_leaf: Any) -> Any:
            if isinstance(buffer_leaf, jax.Array):
                # Use roll for simplicity (maintains differentiability)
                return jnp.roll(buffer_leaf, -shift_amount, axis=0)
            elif isinstance(buffer_leaf, list):
                # Rotate list
                return buffer_leaf[shift_amount:] + buffer_leaf[:shift_amount]
            return buffer_leaf

        new_buffer = jax.tree.map(shift_leaf, self.buffer.get_value(), is_leaf=is_non_jax_leaf)
        self.buffer.set_value(new_buffer)
        mask_val = self.buffer_mask.get_value()
        assert mask_val is not None
        new_mask = jnp.roll(mask_val, -shift_amount)
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


class GradientTransparentRebatchImpl(nnx.Module):
    """Gradient-transparent rebatching that preserves gradient flow.

    Uses simple Python operations for the buffer structure while
    maintaining gradient connectivity through the data values.
    """

    buffer: list[Any]

    def __init__(self, target_batch_size: int, max_buffer_size: int = 512) -> None:
        """Initialize gradient-transparent implementation.

        Args:
            target_batch_size: Target size for output batches
            max_buffer_size: Maximum buffer size for validation symmetry
        """
        super().__init__()
        self.target_batch_size = target_batch_size
        self.max_buffer_size = max_buffer_size

        self.buffer = nnx.data([])

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

        try:
            for i in range(batch_size):
                self.buffer.append(self._extract_element(batch, i))
            return self._try_emit()
        except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
            return batch, True

    def _try_emit(self) -> tuple[Batch | None, bool]:
        """Try to emit a batch from buffer."""
        if len(self.buffer) >= self.target_batch_size:
            # Extract elements
            elements = list(self.buffer[: self.target_batch_size])

            # Update buffers - clear and extend with remaining elements
            remaining_buffer = list(self.buffer[self.target_batch_size :])
            self.buffer.clear()
            self.buffer.extend(remaining_buffer)

            output = self._stack_elements(elements)
            return output, True

        return None, False

    def _extract_element(self, batch: Batch, index: int) -> Any:
        """Extract single element from batch."""

        def get_item(x: Any) -> Any:
            if isinstance(x, jax.Array):
                return x[index]
            elif hasattr(x, "__getitem__"):
                return x[index]
            return x

        return jax.tree.map(get_item, batch, is_leaf=is_non_jax_leaf)

    def _stack_elements(self, elements: list[Any]) -> Batch | None:
        """Stack elements while preserving gradient flow."""
        if not elements:
            return None

        def stack_fn(*xs: Any) -> Any:
            if all(isinstance(x, jax.Array) for x in xs):
                # Simple stack - gradients flow through naturally
                return jnp.stack(xs, axis=0)
            return list(xs)

        return jax.tree.map(stack_fn, *elements, is_leaf=is_non_jax_leaf)

    def flush(self) -> Batch | None:
        """Flush remaining buffered data."""
        if self.buffer:
            elements = list(self.buffer)  # Convert to plain list for processing
            self.buffer = []
            return self._stack_elements(elements)
        return None

    def get_state(self) -> dict[str, Any]:
        """Get implementation state."""
        return {"buffer": self.buffer}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore implementation state."""
        self.buffer = list(state.get("buffer", []))


class RebatchNode(Node):
    """Rebatch node for handling batch size changes in pipelines.

    This node accumulates input batches and emits new batches of a target size.
    It delegates to one of two differentiable implementation strategies.

    Modes:

        - 'differentiable': Maintains full differentiability for training
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
        mode: Literal["differentiable", "gradient_transparent"] = "gradient_transparent",
        max_buffer_size: int = 512,
        name: str | None = None,
    ) -> None:
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

        if mode == "fast":
            raise ValueError(
                "fast rebatching moved outside RebatchNode; use rebatch_iterable() "
                "for ordinary iterator rebatching"
            )
        if mode not in ["differentiable", "gradient_transparent"]:
            raise ValueError(
                f"Unknown mode: {mode}. Must be 'differentiable' or 'gradient_transparent'"
            )

        self.target_batch_size = target_batch_size
        self.mode = mode
        self.max_buffer_size = max_buffer_size

        if mode == "differentiable":
            self.impl = DifferentiableRebatchImpl(target_batch_size, max_buffer_size)
        else:
            self.impl = GradientTransparentRebatchImpl(target_batch_size, max_buffer_size)

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
        del key
        # Track input statistics (only if not inside JAX transformation)
        if batch is not None:
            batch_size = get_batch_size(batch)
            if batch_size is not None:
                # Use try-except to handle JAX tracing context
                try:
                    current = self.elements_processed.get_value()
                    self.elements_processed.set_value(current + batch_size)
                except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
                    # Inside JAX transformation, skip statistics
                    pass

        # Delegate to implementation
        output, is_valid = self.impl(batch)

        # Track output statistics (only if not inside JAX transformation)
        if is_valid and output is not None:
            try:
                current = self.batches_emitted.get_value()
                self.batches_emitted.set_value(current + 1)
            except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
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
            except Exception:  # noqa: BLE001 - JAX tracing can raise multiple runtime errors
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
    mode: Literal["differentiable", "gradient_transparent"] = "gradient_transparent",
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
        node = rebatch(32, mode='gradient_transparent')
        ```
    """
    return RebatchNode(target_batch_size, mode, max_buffer_size)


def rebatch_iterable(
    iterator: Any,
    target_batch_size: int,
    *,
    drop_remainder: bool = False,
    pad: bool = False,
    pad_value: Any = 0,
) -> grain.IterDataset:
    """Rebatch an explicit Grain/sequence iterator with Grain primitives."""
    if target_batch_size <= 0:
        raise ValueError("target_batch_size must be positive")
    if drop_remainder and pad:
        raise ValueError("drop_remainder and pad are mutually exclusive")

    dataset = ensure_iter_dataset(iterator)
    rebatched = grain.experimental.RebatchIterDataset(
        dataset,
        batch_size=target_batch_size,
        drop_remainder=drop_remainder,
    )
    if not pad:
        return rebatched
    return rebatched.map(lambda batch: _pad_rebatched_payload(batch, target_batch_size, pad_value))


def _pad_rebatched_payload(batch: Any, target_batch_size: int, pad_value: Any) -> Any:
    batch_size = get_batch_size(batch)
    if batch_size is None or batch_size >= target_batch_size:
        return batch
    return grain.experimental.batch_and_pad(
        _batch_to_records(batch, batch_size),
        batch_size=target_batch_size,
        pad_value=pad_value,
    )


def _batch_to_records(batch: Any, batch_size: int) -> Sequence[Any]:
    return [
        jax.tree.map(
            lambda leaf, index=index: _slice_leaf_from_batch(leaf, index, batch_size),
            batch,
            is_leaf=is_non_jax_leaf,
        )
        for index in range(batch_size)
    ]


def _slice_leaf_from_batch(leaf: Any, index: int, batch_size: int) -> Any:
    if isinstance(leaf, jax.Array | np.ndarray):
        return leaf[index]
    if (
        isinstance(leaf, Sequence)
        and not isinstance(leaf, str | bytes | bytearray)
        and len(leaf) == batch_size
    ):
        return leaf[index]
    return leaf
