"""Base module class for all Datarax modules.

This module provides DataraxModule - the base class that all Datarax modules inherit from.
It provides common functionality like:

- Statistics computation and management
- Caching system
- Iteration tracking
- Module copying
- NNX compliance

Also provides CheckpointableIteratorModule for data sources that need iteration
state tracking (position, epoch) for resumable training.
"""

import logging
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
from flax import nnx

from datarax.core.config import DataraxModuleConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Custom Variable Types for Datarax
# ============================================================================


class IterationCount(nnx.Variable):
    """Variable type for iteration counters.

    This custom Variable type wraps JAX arrays for iteration counters.
    Using jnp.array(0) instead of plain Python int is critical because:

    1. Python ints are classified as "static" by NNX (not data)
    2. Static values cannot be mutated inside JAX transforms
    3. JAX arrays are classified as "data" and CAN be mutated in transforms
    4. This avoids TraceContextError when mutating inside nnx.jit/nnx.vmap

    The custom type also enables StateAxes control:

    - Broadcast: `nnx.StateAxes({IterationCount: None})`
    - Carry: `nnx.StateAxes({IterationCount: nnx.Carry})`


    """

    pass


# Type variable for iterator return type
T_co = TypeVar("T_co", covariant=True)


class DataraxModule(nnx.Module):
    """Base class for all Datarax modules.

    Provides common functionality shared by all Datarax modules including
    statistics management, caching, iteration tracking, and module copying.

    All modules use config-based initialization with typed, validated config dataclasses.

    Args:
        config: DataraxModuleConfig (already validated via __post_init__)
        rngs: Random number generators (optional)
        name: Optional name for the module

    Attributes:
        config: Module configuration
        rngs: Random number generators
        name: Module name
        _cache: Cache storage (plain dict if cacheable, None otherwise)
        _computed_stats: Computed statistics (nnx.Variable)
        _applied_count: Applied operation counter (IterationCount)
        _skipped_count: Skipped operation counter (IterationCount)
    """

    def __init__(
        self,
        config: DataraxModuleConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize DataraxModule with config.

        Args:
            config: Module configuration (already validated)
            rngs: Random number generators
            name: Optional module name
        """
        super().__init__()  # ALWAYS call super().__init__() for NNX compliance

        # Store configuration and basic attributes
        # Mark as static using nnx.static() - configs contain strings/non-JAX types
        # that Orbax checkpointing cannot serialize
        self.config = nnx.static(config)
        self.rngs = rngs
        self.name = nnx.static(name)

        # Initialize caching system
        # Use plain dict (not nnx.Dict) - cache is internal state, not parameters
        # Mark as static to exclude from checkpoints
        self._cache: dict[int, Any] | None = nnx.static({} if config.cacheable else None)

        # Initialize statistics system
        # Use nnx.Variable to make it trackable by NNX
        self._computed_stats: nnx.Variable[dict[str, Any] | None] = nnx.Variable(None)
        # Flag to override precomputed_stats (for reset_statistics)
        # Mark as static for proper serialization
        self._is_stats_reset: bool = nnx.static(False)

        # Initialize operation counters for tracking applied/skipped operations
        # Uses IterationCount for consistency and transform compatibility
        self._applied_count: IterationCount = IterationCount(jnp.array(0, dtype=jnp.int32))
        self._skipped_count: IterationCount = IterationCount(jnp.array(0, dtype=jnp.int32))

    # ========================================================================
    # Operation Statistics
    # ========================================================================

    def get_operation_stats(self) -> dict[str, int]:
        """Get operation statistics.

        Note: This method converts JAX arrays to Python ints for introspection.
        It is intended for use outside of JIT-compiled functions.

        Returns:
            Dictionary with 'applied_count' and 'skipped_count'
        """
        return {
            "applied_count": int(self._applied_count[...]),
            "skipped_count": int(self._skipped_count[...]),
        }

    def _increment_applied_count(self) -> None:
        """Increment the applied operation counter.

        This method works inside JIT transforms because it operates on
        JAX arrays wrapped in IterationCount Variable.
        """
        self._applied_count[...] += 1

    def _increment_skipped_count(self) -> None:
        """Increment the skipped operation counter.

        This method works inside JIT transforms because it operates on
        JAX arrays wrapped in IterationCount Variable.
        """
        self._skipped_count[...] += 1

    def reset_operation_stats(self) -> None:
        """Reset operation statistics to zero.

        Note: Creates new JAX arrays to reset the counters.
        """
        self._applied_count[...] = 0
        self._skipped_count[...] = 0

    # ========================================================================
    # Statistics System
    # ========================================================================

    def compute_statistics(self, data: Any) -> dict[str, Any] | None:
        """Compute statistics from data using batch_stats_fn.

        If batch_stats_fn is not configured, returns None.
        Computed statistics are cached in _computed_stats.

        Args:
            data: Input data to compute statistics from

        Returns:
            Dictionary of statistics, or None if no batch_stats_fn configured
        """
        if self.config.batch_stats_fn is None:
            return None

        # Compute statistics using the configured function/module
        # Both Callable and nnx.Module support __call__, but need type narrowing for pyright
        batch_stats_fn: Callable[[Any], dict[str, Any]] = self.config.batch_stats_fn  # type: ignore[assignment]
        stats = batch_stats_fn(data)

        # Cache the computed statistics
        self._computed_stats.set_value(stats)

        return stats

    def get_statistics(self) -> dict[str, Any] | None:
        """Get current statistics.

        Returns precomputed_stats if configured (unless reset was called),
        otherwise returns cached computed statistics, or None if no statistics available.

        Returns:
            Dictionary of statistics, or None if no statistics available
        """
        # If reset was called, return None regardless of precomputed_stats
        # _is_stats_reset is a plain boolean (static), safe for JIT control flow
        if self._is_stats_reset:
            return None

        # Priority 1: Precomputed stats (static)
        if self.config.precomputed_stats is not None:
            return self.config.precomputed_stats

        # Priority 2: Computed stats (dynamic, cached)
        return self._computed_stats.get_value()

    def set_statistics(self, stats: dict[str, Any]) -> None:
        """Manually set statistics.

        This overwrites any previously computed statistics and clears reset flag.

        Args:
            stats: Dictionary of statistics to set
        """
        self._computed_stats.set_value(stats)
        self._is_stats_reset = False  # Clear reset flag when setting new stats

    def reset_statistics(self) -> None:
        """Reset all statistics to None.

        This clears both computed statistics and marks that precomputed_stats
        should be ignored (via internal flag). After reset, get_statistics()
        will return None until new statistics are set or computed.
        """
        self._computed_stats.set_value(None)
        self._is_stats_reset = True  # Mark that stats have been reset

    # ========================================================================
    # Caching System
    # ========================================================================

    def _compute_cache_key(self, input_data: Any) -> int:
        """Compute cache key from input data using content-based hashing.

        For JAX arrays, computes a hash based on array content (shape, dtype, and values).
        For PyTrees, recursively hashes all leaves.
        Subclasses can override for custom keys.

        Args:
            input_data: Input to compute cache key from (scalars, arrays, or PyTrees)

        Returns:
            Integer cache key based on content
        """
        import jax

        leaves_hash = self._hash_pytree_leaves(input_data, jax)
        if leaves_hash is not None:
            return leaves_hash
        return self._hash_fallback(input_data)

    def _hash_pytree_leaves(self, input_data: Any, jax_module: Any) -> int | None:
        """Hash all pytree leaves, returning None when pytree traversal fails."""
        try:
            leaves = jax_module.tree.leaves(input_data)
        except (TypeError, ValueError, AttributeError, RuntimeError):
            return None
        if not leaves:
            return hash(())
        return hash(tuple(self._hash_value(leaf, jax_module) for leaf in leaves))

    def _hash_value(self, value: Any, jax_module: Any) -> int:
        """Hash a value with special handling for arrays and nested containers."""
        if isinstance(value, jax_module.Array):
            return self._hash_jax_array(value)
        if isinstance(value, (int, float, str, bool, type(None), tuple, frozenset)):
            return hash(value)
        if isinstance(value, list):
            return hash(tuple(self._hash_value(v, jax_module) for v in value))
        if isinstance(value, dict):
            hashed_items = tuple(
                sorted((k, self._hash_value(v, jax_module)) for k, v in value.items())
            )
            return hash(hashed_items)
        return self._hash_fallback(value)

    def _hash_jax_array(self, value: Any) -> int:
        """Hash a JAX array using shape/dtype and identity.

        Uses only host-side metadata — no device-to-host transfers.
        Previous implementation sampled values via .tolist() which forced
        a blocking D2H sync on GPU/TPU.
        """
        return hash((value.shape, str(value.dtype), id(value)))

    def _hash_fallback(self, value: Any) -> int:
        """Hash a value using Python hash, falling back to object identity."""
        try:
            return hash(value)
        except TypeError:
            return id(value)

    def reset_cache(self) -> None:
        """Clear the cache.

        Only has effect if cacheable=True in config.
        """
        if self._cache is not None:
            # Clear all entries from the cache
            self._cache.clear()

    # ========================================================================
    # Module Copying
    # ========================================================================

    def copy(
        self,
        *,
        config: DataraxModuleConfig | None = None,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> "DataraxModule":
        """Create a copy of this module with optional config/parameter changes.

        This allows creating a new module instance with modified configuration
        while preserving other attributes. Useful for hyperparameter tuning.

        Args:
            config: New config (if None, uses current config)
            rngs: New RNG state (if None, uses current rngs)
            name: New name (if None, uses current name)

        Returns:
            New module instance with updated parameters

        Examples:
            # Change configuration
            new_config = DataraxModuleConfig(cacheable=True)
            new_module = module.copy(config=new_config)

            # Change name only
            renamed = module.copy(name="new_name")

        Note:
            Subclasses can override this method to provide more fine-grained
            control over copying, such as allowing individual config field
            updates without requiring dataclass replace().
        """
        return type(self)(
            config=config if config is not None else self.config,
            rngs=rngs if rngs is not None else self.rngs,
            name=name if name is not None else self.name,
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    # ========================================================================
    # State Management (Checkpointable Protocol)
    # ========================================================================

    def get_state(self) -> dict[str, Any]:
        """Get module state for checkpointing.

        This method implements the Checkpointable protocol using NNX state
        management. It extracts all state variables from the module and
        converts them to a serializable format.

        Returns:
            A dictionary containing the internal state of the component.
        """
        # Get the NNX state (all Variables and submodule states)
        state = nnx.state(self)

        # Convert to pure dict for serialization
        return nnx.to_pure_dict(state)

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore module state from a checkpoint.

        This method implements the Checkpointable protocol using NNX state
        management. It restores the module state from a serialized format.
        Restoration is strict: checkpoint structure must match module state.

        Args:
            state: A dictionary containing the internal state to restore.

        Raises:
            TypeError: If state is not a dictionary.
            ValueError: If checkpoint structure does not match module state.
        """
        if not isinstance(state, dict):
            raise TypeError(f"State must be a dict, got {type(state).__name__}")

        current_state = nnx.state(self)
        current_dict = nnx.to_pure_dict(current_state)

        try:
            self._validate_state_compatibility(self, current_dict, state)
        except ValueError as exc:
            raise ValueError(
                "Checkpoint state is structurally incompatible with module state. "
                "Regenerate checkpoints after architecture/config changes."
            ) from exc

        self._restore_state_tree(self, state)

    @staticmethod
    def _is_array_like(value: Any) -> bool:
        """Return True when value behaves like an ndarray/jax.Array leaf."""
        return hasattr(value, "shape") and hasattr(value, "dtype")

    def _validate_state_compatibility(
        self, target: Any, current: Any, saved: Any, path: tuple[str | int, ...] = ()
    ) -> None:
        """Validate strict checkpoint compatibility.

        Rules:
        - Dict/list/tuple container structure must match exactly.
        - Array leaves must match shape and dtype.
        - Scalar/object leaves may change value/type (for flexible Variable payloads).
        """
        location = self._format_state_location(path)
        if isinstance(target, nnx.Variable):
            self._validate_array_leaf(current, saved, location)
            return

        if isinstance(target, nnx.Module):
            self._validate_module_state(target, current, saved, path, location)
            return

        if isinstance(current, dict):
            self._validate_mapping_state(current, saved, location)
            return

        self._validate_array_leaf(current, saved, location)

    @staticmethod
    def _format_state_location(path: tuple[str | int, ...]) -> str:
        """Format state path tuple for readable error messages."""
        return ".".join(str(part) for part in path) if path else "<root>"

    def _validate_array_leaf(self, current: Any, saved: Any, location: str) -> None:
        """Validate shape/dtype compatibility for array-like leaves."""
        if not (self._is_array_like(current) and self._is_array_like(saved)):
            return
        if current.shape != saved.shape:
            raise ValueError(
                f"Array shape mismatch at {location}: expected {current.shape}, got {saved.shape}"
            )
        if current.dtype != saved.dtype:
            raise ValueError(
                f"Array dtype mismatch at {location}: expected {current.dtype}, got {saved.dtype}"
            )

    def _validate_mapping_state(self, current: dict[Any, Any], saved: Any, location: str) -> None:
        """Validate mapping type and exact key-set equality."""
        if not isinstance(saved, dict):
            raise ValueError(
                f"State node type mismatch at {location}: expected dict, got {type(saved).__name__}"
            )
        current_keys = set(current)
        saved_keys = set(saved)
        if current_keys == saved_keys:
            return
        missing = sorted(current_keys - saved_keys)
        extra = sorted(saved_keys - current_keys)
        raise ValueError(f"State keys mismatch at {location}: missing={missing}, extra={extra}")

    def _validate_module_state(
        self,
        target: nnx.Module,
        current: Any,
        saved: Any,
        path: tuple[str | int, ...],
        location: str,
    ) -> None:
        """Validate module subtree recursively against a saved subtree."""
        if not isinstance(current, dict):
            raise ValueError(
                f"Current state mismatch at {location}: expected dict, got {type(current).__name__}"
            )
        self._validate_mapping_state(current, saved, location)
        saved_dict = saved
        for key in current:
            child = self._resolve_state_child(target, key, path)
            self._validate_state_compatibility(child, current[key], saved_dict[key], (*path, key))

    def _resolve_state_child(
        self, target: nnx.Module, key: str | int, path: tuple[str | int, ...]
    ) -> Any:
        """Resolve a child object referenced by a state key."""
        if isinstance(key, str) and hasattr(target, key):
            return getattr(target, key)
        if hasattr(target, "__getitem__"):
            try:
                return target[key]  # type: ignore[index]
            except (KeyError, IndexError, TypeError):
                pass
        location = ".".join(str(p) for p in path) if path else "<root>"
        raise ValueError(f"State path {location} has no child for key {key!r}")

    def _restore_state_tree(
        self, target: Any, saved: Any, path: tuple[str | int, ...] = ()
    ) -> None:
        """Restore validated state into an NNX object tree."""
        if isinstance(target, nnx.Variable):
            target.set_value(saved)
            return

        if isinstance(target, nnx.Module):
            if not isinstance(saved, dict):
                location = ".".join(str(p) for p in path) if path else "<root>"
                raise ValueError(
                    f"Invalid state node at {location}: expected dict, got {type(saved).__name__}"
                )
            for key, value in saved.items():
                child = self._resolve_state_child(target, key, path)
                self._restore_state_tree(child, value, (*path, key))
            return

        if isinstance(target, list | tuple):
            if not isinstance(saved, list | tuple):
                location = ".".join(str(p) for p in path) if path else "<root>"
                raise ValueError(
                    f"Invalid sequence state at {location}: got {type(saved).__name__}"
                )
            for index, (child, value) in enumerate(zip(target, saved, strict=True)):
                self._restore_state_tree(child, value, (*path, index))

    def clone(self) -> "DataraxModule":
        """Create a new instance with the same state as this module.

        Uses NNX's clone function for proper deep cloning of all state.

        Returns:
            A new module instance with the same state.
        """
        return nnx.clone(self)

    # ========================================================================
    # RNG Stream Validation
    # ========================================================================

    def requires_rng_streams(self) -> list[str] | None:
        """Get the list of RNG streams required by this module.

        Returns:
            A list of required RNG stream names, or None if no RNG streams
            are required.
        """
        return None

    def ensure_rng_streams(self, stream_names: list[str]) -> None:
        """Ensure that the required RNG streams are available.

        Args:
            stream_names: A list of available RNG stream names.

        Raises:
            ValueError: If a required RNG stream is not available.
        """
        required_streams = self.requires_rng_streams()
        if required_streams is not None:
            for stream in required_streams:
                if stream not in stream_names:
                    msg = f"RNG stream '{stream}' is required but not "
                    msg += f"available. Available streams: {stream_names}"
                    raise ValueError(msg)

    # ========================================================================
    # Utilities
    # ========================================================================

    def __repr__(self) -> str:
        """Return string representation of module.

        Returns:
            String representation including class name, name, and key config info
        """
        class_name = self.__class__.__name__
        parts = []

        if self.name:
            parts.append(f"name='{self.name}'")

        # Add key config info
        if self.config.cacheable:
            parts.append("cacheable=True")

        if parts:
            return f"{class_name}({', '.join(parts)})"
        return f"{class_name}()"


class CheckpointableIteratorModule(DataraxModule, Generic[T_co]):
    """Base class for iterator modules that can be checkpointed.

    This class extends DataraxModule to implement the CheckpointableIterator
    protocol, providing unified state management for iterators that need to
    save and restore their position and internal state for resumable training.

    Useful for data sources, data loaders, and any module that iterates
    through data and needs checkpoint/restore capability.

    Args:
        config: DataraxModuleConfig for the module
        rngs: Optional Rngs object for randomness
        name: Optional name for the module

    Attributes:
        epoch: Current epoch (nnx.Variable)
        position: Current position in iteration (nnx.Variable)
        idx: Current index (nnx.Variable)
        current: Current item being processed (nnx.Variable)
    """

    def __init__(
        self,
        config: DataraxModuleConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the CheckpointableIteratorModule.

        Args:
            config: DataraxModuleConfig for the module
            rngs: Optional Rngs object for randomness
            name: Optional name for the module
        """
        super().__init__(config, rngs=rngs, name=name)

        # Initialize iterator state variables as NNX Variables
        self.epoch: nnx.Variable[int | None] = nnx.Variable(None)
        self.position: nnx.Variable[int | None] = nnx.Variable(None)
        self.idx: nnx.Variable[int | None] = nnx.Variable(None)
        self.current: nnx.Variable[Any | None] = nnx.Variable(None)

    def __iter__(self) -> Iterator[T_co]:
        """Return the iterator object.

        Returns:
            The iterator object (usually self).
        """
        return self  # type: ignore[return-value]

    def __next__(self) -> T_co:
        """Get the next item from the iterator.

        This method should be implemented by subclasses.

        Returns:
            The next item.

        Raises:
            StopIteration: When the iterator is exhausted.
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement __next__")

    def __len__(self) -> int:
        """Return the number of items in the iterator.

        This method should be implemented by subclasses.

        Returns:
            The total number of items.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement __len__")

    def reset(self) -> None:
        """Reset the iterator to its initial state.

        Subclasses should override this to add additional reset logic.
        """
        self.epoch.set_value(None)
        self.position.set_value(None)
        self.idx.set_value(None)
        self.current.set_value(None)
