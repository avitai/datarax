"""Element and Batch modules following JAX and Flax NNX best practices.

Key design decisions:

- Element uses flax.struct for immutability and automatic pytree registration
- Batch uses Flax NNX Module pattern for state management
- No object dtype arrays (JAX limitation)
- Proper handling of static arguments in JIT compilation
- Efficient vectorized operations without Python loops
"""

from typing import Any
from collections.abc import Callable

import flax.nnx as nnx
import flax.struct as struct
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .metadata import Metadata


@struct.dataclass
class Element:
    """Immutable data element with JAX-compatible operations.

    Element represents a single data point with:

    - data: PyTree structure containing JAX arrays (supports nested dicts)
    - state: Dictionary of arbitrary Python values
    - metadata: Optional Metadata instance

    All operations return new instances (immutable design).
    """

    data: PyTree = struct.field(default_factory=dict)
    state: dict[str, Any] = struct.field(default_factory=dict)
    metadata: Metadata | None = struct.field(default=None)

    def update_state(self, updates: dict[str, Any]) -> "Element":
        """Update state with partial updates (merge behavior)."""
        new_state = dict(self.state)
        new_state.update(updates)
        return self.replace(state=new_state)

    def update_data(self, updates: dict[str, jax.Array]) -> "Element":
        """Update data with partial updates (merge behavior)."""
        new_data = dict(self.data)
        new_data.update(updates)
        return self.replace(data=new_data)

    def transform(self, fn: Callable[[jax.Array], jax.Array]) -> "Element":
        """Transform all data arrays with a function.

        Note: Cannot be directly JIT compiled as fn must be static.
        Use transform_element_jit with static function IDs instead.
        """
        new_data = jax.tree.map(fn, self.data)
        return self.replace(data=new_data)

    def with_metadata(self, metadata: Metadata) -> "Element":
        """Return new Element with updated metadata."""
        return self.replace(metadata=metadata)

    def apply_to_data(self, fn: Callable[[jax.Array], jax.Array]) -> "Element":
        """Apply differentiable transformation to all data arrays.

        This method preserves gradients through JAX transformations by applying
        the function directly to each array in the data dictionary using jax.tree.map.

        Args:
            fn: Differentiable function to apply to each array

        Returns:
            New Element with transformed data, preserving state and metadata

        Examples:
            element = Element(data={"x": jnp.array([1.0, 2.0])})
            scaled = element.apply_to_data(lambda x: x * 2.0)
            # Gradients flow through the scaling operation
        """
        transformed_data = jax.tree.map(fn, self.data)
        return self.replace(data=transformed_data)


class Batch(nnx.Module):
    """Batch container using Flax NNX patterns.

    Design rationale:

    - Stores data as stacked JAX PyTrees for efficiency
    - Stores states as stacked JAX PyTrees (enables vmap)
    - Stores metadata as Python list (immutable, not vmapped)
    - Uses NNX Variables for mutable state management
    - All operations are JAX-compatible for JIT compilation
    """

    def __init__(self, elements: list[Element], validate: bool = True):
        """Initialize batch from list of Elements.

        Args:
            elements: List of Element instances
            validate: Whether to validate consistency
        """
        super().__init__()

        self.batch_size = len(elements)

        if not elements:
            # Empty batch initialization
            self.data = nnx.Variable({})
            self.states = nnx.Variable([])
            # Use nnx.Variable for metadata to ensure it's treated as state (contains JAX arrays)
            self._metadata_list = nnx.Variable([])
            self._batch_metadata = nnx.Variable(None)
            self.batch_state = nnx.Variable({})
            return

        # Stack data using jax.tree.map to handle nested PyTree structures
        # This works for both flat dicts and nested structures
        element_data_list = [elem.data for elem in elements]
        batched_data = jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *element_data_list)

        # Stack states using jax.tree.map (same as data)
        # States must be JAX-compatible PyTrees for efficient vmap operations
        element_states_list = [elem.state for elem in elements]
        batched_states = jax.tree.map(
            lambda *arrays: jnp.stack(arrays, axis=0), *element_states_list
        )

        # Store metadata as list (immutable, not vmapped)
        metadata_list = [elem.metadata for elem in elements]

        # Initialize NNX Variables for mutable state
        self.data = nnx.Variable(batched_data)
        self.states = nnx.Variable(batched_states)
        self.batch_state = nnx.Variable({})

        # Store metadata in Variables because they contain JAX arrays (_encoded_key)
        # and must be part of State (not GraphDef) to avoid recompilation.
        self._metadata_list = nnx.Variable(metadata_list)
        self._batch_metadata = nnx.Variable(None)

        if validate:
            self._validate()

    @classmethod
    def from_parts(
        cls,
        data: PyTree,
        states: PyTree,
        metadata_list: list[Any] | None = None,
        batch_metadata: Any | None = None,
        batch_state: PyTree | None = None,
        *,
        validate: bool = True,
    ) -> "Batch":
        """Create Batch directly from pre-built parts with validation.

        This is the recommended way to construct batches from transformed data
        in operators, as it avoids Python loops and validates structure.

        Args:
            data: PyTree with arrays having batch dimension as axis 0
            states: PyTree with arrays having batch dimension as axis 0
                   (same structure as data, all leaves must have matching batch size)
            metadata_list: Optional list of metadata, length must match batch size
            batch_metadata: Optional batch-level metadata (immutable)
            batch_state: Optional batch-level state PyTree (no batch dimension)
            validate: If True, validates batch axis consistency and lengths

        Returns:
            New Batch instance

        Raises:
            ValueError: If validation fails (mismatched batch sizes, inconsistent shapes)

        Examples:
            # Simple flat PyTrees
            data = {"image": jnp.ones((32, 224, 224, 3))}
            states = {"count": jnp.zeros((32,)), "flag": jnp.ones((32,), dtype=bool)}
            batch = Batch.from_parts(data, states)

            # Nested PyTrees
            data = {
                "vision": {"image": jnp.ones((32, 224, 224, 3))},
                "text": jnp.ones((32, 512))
            }
            states = {
                "counters": {"augment": jnp.zeros((32,)), "transform": jnp.zeros((32,))},
                "score": jnp.ones((32,))
            }
            batch = Batch.from_parts(data, states)
        """
        if validate:
            # Validate batch dimension consistency across all arrays
            batch_sizes = set()

            def check_batch_dim(x):
                if isinstance(x, jax.Array):
                    batch_sizes.add(x.shape[0])
                return x

            jax.tree.map(check_batch_dim, data)

            if len(batch_sizes) == 0:
                raise ValueError("Data PyTree contains no arrays")
            if len(batch_sizes) > 1:
                raise ValueError(
                    f"Inconsistent batch dimensions: {batch_sizes}. "
                    f"All arrays must have same size for axis 0."
                )

            batch_size = batch_sizes.pop()

            # Validate states PyTree has matching batch dimension
            states_batch_sizes = set()
            jax.tree.map(check_batch_dim, states)
            jax.tree.map(
                lambda x: states_batch_sizes.add(x.shape[0]) if isinstance(x, jax.Array) else None,
                states,
            )

            if states_batch_sizes and len(states_batch_sizes) > 1:
                raise ValueError(
                    f"Inconsistent batch dimensions in states: {states_batch_sizes}. "
                    f"All state arrays must have same size for axis 0."
                )
            if states_batch_sizes and states_batch_sizes.pop() != batch_size:
                raise ValueError(
                    f"states batch dimension doesn't match data batch size ({batch_size})"
                )

            # Validate metadata_list length if provided
            if metadata_list is not None and len(metadata_list) != batch_size:
                raise ValueError(
                    f"metadata_list length ({len(metadata_list)}) doesn't match "
                    f"batch size ({batch_size})"
                )
        else:
            # Non-validated path: extract batch size from first array
            first_array = jax.tree.leaves(data)[0]
            batch_size = first_array.shape[0]

        # Create batch instance using empty init to set up NNX Module properly
        # Then replace the fields with the provided data
        batch = cls([], validate=False)  # Initialize with empty list to set up NNX state

        # Replace fields with provided data
        batch.batch_size = batch_size
        batch.data.set_value(data)
        batch.states.set_value(states)
        batch._metadata_list = nnx.Variable(
            metadata_list if metadata_list is not None else [None] * batch_size
        )
        batch._batch_metadata = nnx.Variable(batch_metadata)
        if batch_state is not None:
            batch.batch_state.set_value(batch_state)

        return batch

    def _validate(self):
        """Validate batch consistency."""
        if self.batch_size == 0:
            return

        # Check batch dimensions using tree.map to handle nested PyTrees
        def check_batch_size(x):
            if isinstance(x, jax.Array) and x.shape[0] != self.batch_size:
                raise ValueError(f"Array has batch size {x.shape[0]}, expected {self.batch_size}")
            return x

        jax.tree.map(check_batch_size, self.data.get_value())
        jax.tree.map(check_batch_size, self.states.get_value())

    def get_element(self, index: int) -> Element:
        """Extract single element at index."""
        if not (0 <= index < self.batch_size):
            raise IndexError(f"Index {index} out of range [0, {self.batch_size})")

        # Extract data for this index using tree.map (handles nested PyTrees)
        data_val = self.data.get_value()
        elem_data = jax.tree.map(lambda x: x[index], data_val)

        # Extract state for this index using tree.map (states are PyTrees like data)
        states_val = self.states.get_value()
        elem_state = jax.tree.map(lambda x: x[index], states_val) if states_val else {}

        # Get metadata (stored in nnx.Variable)
        metadata_list = self._metadata_list.get_value()
        elem_metadata = (
            metadata_list[index] if metadata_list and index < len(metadata_list) else None
        )

        return Element(data=elem_data, state=elem_state, metadata=elem_metadata)

    def get_elements(self, indices: slice | list[int]) -> list[Element]:
        """Get multiple elements by indices or slice."""
        if isinstance(indices, slice):
            indices = range(*indices.indices(self.batch_size))
        return [self.get_element(i) for i in indices]

    def slice(self, start: int, end: int) -> "Batch":
        """Create new batch from slice of elements (O(1) view)."""
        # Ensure indices are within bounds
        start = max(0, min(start, self.batch_size))
        end = max(0, min(end, self.batch_size))

        # Optimize: Slice data and states arrays directly
        sliced_data = jax.tree.map(lambda x: x[start:end], self.data.get_value())
        sliced_states = jax.tree.map(lambda x: x[start:end], self.states.get_value())

        # Slice metadata list
        metadata_list = self._metadata_list.get_value()
        sliced_metadata = metadata_list[start:end] if metadata_list else []

        # Batch metadata and state preserved
        return Batch.from_parts(
            data=sliced_data,
            states=sliced_states,
            metadata_list=sliced_metadata,
            batch_metadata=self._batch_metadata.get_value(),
            batch_state=self.batch_state.get_value(),
            validate=False,
        )

    def split_for_devices(self, num_devices: int) -> list["Batch"]:
        """Split batch evenly across devices."""
        if self.batch_size % num_devices != 0:
            raise ValueError(f"Batch size {self.batch_size} not divisible by {num_devices}")

        split_size = self.batch_size // num_devices
        # Note: slice() is now optimized, so this is efficient
        return [self.slice(i * split_size, (i + 1) * split_size) for i in range(num_devices)]

    def compute_stats(self) -> dict[str, jax.Array]:
        """Compute statistics over batch dimension."""
        stats = {}

        data_val = self.data.get_value()
        for key, array in data_val.items():
            stats[f"{key}_mean"] = jnp.mean(array, axis=0)
            stats[f"{key}_std"] = jnp.std(array, axis=0)
            stats[f"{key}_min"] = jnp.min(array, axis=0)
            stats[f"{key}_max"] = jnp.max(array, axis=0)

        return stats

    def get_data(self) -> dict[str, jax.Array]:
        """Get batched data dictionary."""
        return self.data.get_value()

    def __getitem__(self, key: str) -> jax.Array:
        """Get data array by key for dict-like access."""
        return self.data.get_value()[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in data for dict-like containment check."""
        return key in self.data.get_value()

    def __iter__(self):
        """Iterate over data keys for dict-like iteration."""
        return iter(self.data.get_value())

    def get_states(self) -> list[dict[str, Any]]:
        """Get list of all states."""
        return self.states.get_value()

    def get_batch_state(self) -> dict[str, Any]:
        """Get batch-level state."""
        return self.batch_state.get_value()

    def get_batch_metadata(self) -> Metadata | None:
        """Get batch-level metadata."""
        return self._batch_metadata.get_value()

    def set_batch_metadata(self, metadata: Metadata):
        """Set batch-level metadata."""
        self._batch_metadata.set_value(metadata)

    def update_batch_state(self, updates: dict[str, Any]):
        """Update batch-level state (merge behavior)."""
        current = dict(self.batch_state.get_value())
        current.update(updates)
        self.batch_state.set_value(current)


def _scan_batch_elements(batch: Batch, fn: Callable[[Element], Element]) -> Batch:
    """Scan-based batch element processing for stateful operations.

    Uses JAX scan for sequential processing when state dependencies prevent
    full vectorization. More efficient than Python loops while preserving
    state semantics.

    Args:
        batch: Input batch to transform
        fn: Function to apply to each element (may modify state)

    Returns:
        New batch with transformed elements
    """
    if batch.batch_size == 0:
        return Batch([])

    def scan_fn(carry, elem_idx):
        # Get element at index
        elem = batch.get_element(elem_idx)
        # Apply transformation
        transformed = fn(elem)
        return carry, transformed

    # Use scan for sequential processing
    indices = jnp.arange(batch.batch_size)
    _, transformed_elements = jax.lax.scan(scan_fn, None, indices)

    # Convert scan results back to batch
    # Note: This still requires some Python processing for complex state handling
    elements = []
    for i in range(batch.batch_size):
        # Extract transformed element data
        if hasattr(transformed_elements, "data"):
            # If scan preserved structure
            elem_data = jax.tree.map(lambda x: x[i], transformed_elements.data)
            elem = Element(
                data=elem_data,
                state=transformed_elements.state[i]
                if hasattr(transformed_elements, "state")
                else {},
                metadata=transformed_elements.metadata[i]
                if hasattr(transformed_elements, "metadata")
                else None,
            )
            elements.append(elem)
        else:
            # Fallback to element-wise processing
            elem = batch.get_element(i)
            elements.append(fn(elem))

    new_batch = Batch(elements, validate=False)
    new_batch.batch_state.set_value(batch.batch_state.get_value())
    new_batch._batch_metadata = nnx.Variable(batch._batch_metadata.get_value())

    return new_batch


# Batch operations
class BatchOps:
    """Utility operations for batches."""

    @staticmethod
    def filter_batch(batch: Batch, mask: jax.Array) -> Batch:
        """Filter batch using boolean mask.

        Uses JAX indexing on PyTree structures for efficiency.
        """
        if mask.shape[0] != batch.batch_size:
            raise ValueError(
                f"Mask shape {mask.shape} incompatible with batch size {batch.batch_size}"
            )

        # Filter data PyTree using mask (handles nested structures)
        filtered_data = jax.tree.map(lambda x: x[mask], batch.data.get_value())

        # Filter states PyTree using mask (handles nested structures)
        filtered_states = jax.tree.map(lambda x: x[mask], batch.states.get_value())

        # Filter metadata list (convert JAX indices to Python list for indexing)
        indices = jnp.where(mask)[0].tolist()  # Convert to Python list of ints
        metadata_list = batch._metadata_list.get_value()
        filtered_metadata = [metadata_list[i] for i in indices] if metadata_list else []

        # Use Batch.from_parts to reconstruct (handles PyTree stacking)
        return Batch.from_parts(
            data=filtered_data,
            states=filtered_states,
            metadata_list=filtered_metadata,
            batch_metadata=batch._batch_metadata.get_value(),
            batch_state=batch.batch_state.get_value(),
            validate=False,
        )

    @staticmethod
    def concatenate_batches(batches: list[Batch]) -> Batch:
        """Concatenate multiple batches."""
        if not batches:
            return Batch([])

        if len(batches) == 1:
            return batches[0]

        # Use JAX concatenation for data and states (much faster than Python loops)
        first_batch = batches[0]

        # Concatenate data PyTrees
        # We assume consistent structure across batches (standard assumption)
        all_data_vals = [b.data.get_value() for b in batches]
        concatenated_data = jax.tree.map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *all_data_vals
        )

        # Concatenate states PyTrees
        all_states_vals = [b.states.get_value() for b in batches]
        concatenated_states = jax.tree.map(
            lambda *arrays: jnp.concatenate(arrays, axis=0), *all_states_vals
        )

        # Concatenate metadata lists
        concatenated_metadata = []
        for b in batches:
            meta_list = b._metadata_list.get_value()
            if meta_list:
                concatenated_metadata.extend(meta_list)

        # Note: Discards batch-level metadata/state from subsequent batches
        return Batch.from_parts(
            data=concatenated_data,
            states=concatenated_states,
            metadata_list=concatenated_metadata,
            batch_metadata=first_batch._batch_metadata.get_value(),
            batch_state=first_batch.batch_state.get_value(),
            validate=False,
        )

    @staticmethod
    def update_batch_inplace(batch: Batch, data_updates: dict[str, jax.Array]) -> Batch:
        """Update batch data in place."""
        current = dict(batch.data.get_value())
        current.update(data_updates)
        batch.data.set_value(current)
        return batch


# JAX control flow operations


def conditional_transform(
    batch: Batch,
    true_fn: Callable[[Batch], Batch],
    false_fn: Callable[[Batch], Batch],
    condition: jax.Array,
) -> Batch:
    """Conditional transformation with dynamic condition.

    Uses jax.lax.cond for traced boolean conditions.
    """
    return jax.lax.cond(condition, true_fn, false_fn, batch)


def iterative_transform(
    batch: Batch, fn: Callable[[Batch, int], Batch], num_iterations: int
) -> Batch:
    """Apply iterative transformation."""
    result = batch
    for i in range(num_iterations):
        result = fn(result, i)
    return result


def while_transform(
    batch: Batch,
    cond_fn: Callable[[Batch], bool],
    body_fn: Callable[[Batch], Batch],
    max_iterations: int = 100,
) -> Batch:
    """Apply while loop transformation using nnx.while_loop."""

    def loop_cond(state):
        batch_state, iteration = state
        return cond_fn(batch_state) & (iteration < max_iterations)

    def loop_body(state):
        batch_state, iteration = state
        new_batch = body_fn(batch_state)
        return (new_batch, iteration + 1)

    initial_state = (batch, 0)
    final_batch, _ = nnx.while_loop(loop_cond, loop_body, initial_state)
    return final_batch


# Factory functions


def create_element(
    data: dict[str, jax.Array] | None = None,
    state: dict[str, Any] | None = None,
    metadata: Metadata | None = None,
) -> Element:
    """Create element with defaults."""
    return Element(data=data or {}, state=state or {}, metadata=metadata)


def create_batch_from_arrays(
    data: dict[str, jax.Array],
    states: list[dict[str, Any]] | None = None,
    metadata_list: list[Metadata] | None = None,
) -> Batch:
    """Create batch directly from pre-stacked arrays."""
    if not data:
        return Batch([])

    # If states is provided as list of dicts, we must stack it. This is the slow part.
    batched_states = {}
    if states:
        # Check if states is actually already a dict of arrays (optimization used by some callers?)
        # No, type hint says list[dict].

        # We have to stack.
        # Check keys from first state
        first_state = states[0]
        keys = first_state.keys()

        # Stack each key
        for k in keys:
            # We assume all states have same keys and values are arrays or stackable
            # This might fail if states are heterogeneous.
            # Fallback to naive creation if complex, but here we optimize for common case.
            try:
                batched_states[k] = jnp.stack([s[k] for s in states], axis=0)
            except Exception:
                # Fallback to slow path if stacking fails (e.g. different structures)
                return _create_batch_naive(data, states, metadata_list)

    return Batch.from_parts(
        data=data,
        states=batched_states,
        metadata_list=metadata_list,
        validate=False,  # We trust arrays match if they came from reliable source
    )


def _create_batch_naive(
    data: dict[str, jax.Array],
    states: list[dict[str, Any]] | None,
    metadata_list: list[Metadata] | None,
) -> Batch:
    """Fallback naive creation."""
    batch_size = next(iter(data.values())).shape[0]
    elements = []
    for i in range(batch_size):
        elem_data = {key: arr[i] for key, arr in data.items()}
        elem_state = states[i] if states and i < len(states) else {}
        elem_metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
        elements.append(Element(data=elem_data, state=elem_state, metadata=elem_metadata))
    return Batch(elements, validate=False)


class BatchView:
    """Lightweight batch container for the hot iteration path.

    A plain Python object (NOT an nnx.Module) that provides the same dict-like
    interface as Batch but without NNX Variable overhead. Uses __slots__ for
    minimal memory footprint.

    Used in the fused operator chain where we need:
    - get_data() for adapter materialization
    - __getitem__, __contains__, __iter__ for dict-like access
    - batch_size for consistency checks
    - to_batch() for conversion when full NNX Batch features are needed

    Creating a BatchView is essentially free (~0μs) compared to Batch
    which creates 5 nnx.Variable instances (~50-100μs).
    """

    __slots__ = ("_data", "_states", "batch_size")

    def __init__(self, data: dict, states: dict, batch_size: int):
        self._data = data
        self._states = states
        self.batch_size = batch_size

    def get_data(self) -> dict:
        """Get batched data dictionary (same interface as Batch)."""
        return self._data

    def to_batch(self) -> "Batch":
        """Convert to full NNX Batch when needed."""
        return Batch.from_parts(data=self._data, states=self._states, validate=False)

    def __getitem__(self, key: str) -> jax.Array:
        """Dict-like access to data arrays."""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """Dict-like containment check."""
        return key in self._data

    def __iter__(self):
        """Iterate over data keys."""
        return iter(self._data)


# Export public API
__all__ = [
    "Element",
    "Batch",
    "BatchView",
    "BatchOps",
    "conditional_transform",
    "iterative_transform",
    "while_transform",
    "create_element",
    "create_batch_from_arrays",
]
