"""Base module class for all Datarax modules.

This module provides DataraxModule - the base class that all Datarax modules inherit from.
It provides common functionality like:

- Config, RNG state and naming
- Iteration tracking
- Checkpointing through get_state/set_state
- NNX compliance

Also provides CheckpointableIteratorModule for data sources that need iteration
state tracking (position, epoch) for resumable training.
"""

import logging
from collections.abc import Iterator
from typing import Any

from flax import nnx

from datarax.core.config import DataraxModuleConfig


logger = logging.getLogger(__name__)


def _copy_state_nodes(value: Any) -> Any:
    """Copy a saved state's containers while sharing its leaves.

    An upgrade rewrites container entries, so the containers are copied and the array leaves
    are not; the caller's dictionary is left as it was.

    Args:
        value: A node of a saved state tree.

    Returns:
        A copy of the node's containers, sharing every leaf.
    """
    if isinstance(value, dict):
        return {key: _copy_state_nodes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_state_nodes(item) for item in value]
    return value


class DataraxModule(nnx.Module):
    """Base class for all Datarax modules.

    Provides the configuration, RNG state and checkpointing every Datarax module shares.
    Statistics belong to operators, which hold them in their own store
    (``OperatorModule.set_statistics``), not to every module.

    All modules use config-based initialization with typed, validated config dataclasses.

    Attributes:
        config: Module configuration
        rngs: Random number generators
        name: Module name
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

    # ========================================================================
    # Utilities
    # ========================================================================

    # ========================================================================
    # State Management (Checkpointable Protocol)
    # ========================================================================

    def get_state(self) -> dict[str, Any]:
        """Get module state for checkpointing.

        This method implements the Checkpointable protocol using NNX state
        management. It captures the module's ``nnx.Variable`` state -- positions, epochs, RNG
        keys and counts, parameters -- in a serializable format. Plain data leaves, such as the
        records a source was built with, belong to the module's construction and are not
        captured: a checkpoint of a source over N records does not hold the records.

        Returns:
            A dictionary containing the internal state of the component.
        """
        state = nnx.state(self, nnx.Variable, graph=True)

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

        state = self._upgrade_state_tree(state)

        current_state = nnx.state(self, nnx.Variable, graph=True)
        current_dict = nnx.to_pure_dict(current_state)

        try:
            self._validate_state_compatibility(self, current_dict, state)
        except ValueError as exc:
            raise ValueError(
                "Checkpoint state is structurally incompatible with module state. "
                "Regenerate checkpoints after architecture/config changes."
            ) from exc

        self._restore_state_tree(self, state)

    def _upgrade_state_tree(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return ``state`` with each module's earlier layout rewritten to the current one.

        Every module in the graph is offered its own subtree, so a checkpoint written before a
        layout change restores without its reader knowing which module changed. A module whose
        subtree the saved state does not carry is skipped, and the validation below reports that
        difference as it would have anyway.

        Args:
            state: The saved state dictionary, as ``get_state`` produced it.

        Returns:
            The upgraded state. The argument is not modified.
        """
        upgraded = _copy_state_nodes(state)
        for path, node in nnx.iter_graph(self, graph=True):
            if not isinstance(node, DataraxModule):
                continue
            subtree: Any = upgraded
            for key in path:
                if not isinstance(subtree, dict) or key not in subtree:
                    subtree = None
                    break
                subtree = subtree[key]
            if isinstance(subtree, dict):
                node._upgrade_saved_state(subtree)
        return upgraded

    def _upgrade_saved_state(self, saved: dict[str, Any]) -> None:
        """Rewrite this module's saved subtree from an earlier layout to the current one.

        The base module's state layout has not changed, so this does nothing. A subclass whose
        layout changed overrides it and edits ``saved`` in place.

        Args:
            saved: This module's own saved subtree.
        """

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
        """Restore validated state into an NNX object tree.

        A saved dict is applied through the target's attributes, which covers
        ``nnx.Module`` as well as ``nnx.Rngs`` and its streams (neither is a
        Module, and both carry Variables that must resume with the module).

        Args:
            target: The node receiving the state: a Variable, an object with
                children, or a sequence.
            saved: The saved state for that node.
            path: Path to ``target`` from the tree root, for error messages.

        Raises:
            ValueError: If ``saved`` has a shape no target node can receive.
        """
        if isinstance(target, nnx.Variable):
            target.set_value(saved)
            return

        if isinstance(saved, dict):
            self._restore_module_state(target, saved, path)
            return

        if isinstance(target, list | tuple):
            self._restore_sequence_state(target, saved, path)
            return

        raise ValueError(
            f"Cannot restore state at {self._format_path(path)}: "
            f"{type(saved).__name__} into {type(target).__name__}"
        )

    @staticmethod
    def _format_path(path: tuple[str | int, ...]) -> str:
        """Render a state-tree path tuple as a dotted location string for errors."""
        return ".".join(str(p) for p in path) if path else "<root>"

    def _restore_module_state(self, target: Any, saved: Any, path: tuple[str | int, ...]) -> None:
        """Restore a saved dict of child state through the target's attributes.

        Args:
            target: Module, ``nnx.Rngs`` or stream whose children are being restored.
            saved: Saved state, a dict keyed by child name.
            path: Path to ``target`` from the tree root, for error messages.
        """
        for key, value in saved.items():
            child = self._resolve_state_child(target, key, path)
            self._restore_state_tree(child, value, (*path, key))

    def _restore_sequence_state(self, target: Any, saved: Any, path: tuple[str | int, ...]) -> None:
        """Restore a saved sequence of child state into a list/tuple target.

        Args:
            target: List or tuple whose elements are being restored.
            saved: Saved state; must be a list or tuple of matching length.
            path: Path to ``target`` from the tree root, for error messages.

        Raises:
            ValueError: If ``saved`` is not a list or tuple.
        """
        if not isinstance(saved, list | tuple):
            raise ValueError(
                f"Invalid sequence state at {self._format_path(path)}: got {type(saved).__name__}"
            )
        for index, (child, value) in enumerate(zip(target, saved, strict=True)):
            self._restore_state_tree(child, value, (*path, index))

    def clone(self) -> "DataraxModule":
        """Create a new instance with the same state as this module.

        Uses NNX's clone function for proper deep cloning of all state.

        Returns:
            A new module instance with the same state.
        """
        return nnx.clone(self, graph=True)

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

        if parts:
            return f"{class_name}({', '.join(parts)})"
        return f"{class_name}()"


class CheckpointableIteratorModule[T_co](DataraxModule):
    """Base class for iterator modules that can be checkpointed.

    This class extends DataraxModule to implement the CheckpointableIterator
    protocol, providing unified state management for iterators that need to
    save and restore their position and internal state for resumable training.

    Useful for data sources, data loaders, and any module that iterates
    through data and needs checkpoint/restore capability.

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

    def __next__(self) -> T_co:  # noqa: DOC503
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
