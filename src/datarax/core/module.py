"""Base module class for all Datarax modules.

This module provides DataraxModule - the base class that all Datarax modules inherit from.
It provides common functionality like:

- Config, RNG state and naming
- Checkpointing through get_state/set_state
- NNX compliance

The checkpoint state logic is the module-level ``module_state`` / ``restore_module_state``, so a
module that is not a DataraxModule (a ``Pipeline``) implements the same protocol with it.
"""

import logging
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
        """Get module state for checkpointing: :func:`module_state` of this module.

        Returns:
            A dictionary containing the internal state of the component.
        """
        return module_state(self)

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore module state from a checkpoint: :func:`restore_module_state` into it.

        Args:
            state: A dictionary containing the internal state to restore.
        """
        restore_module_state(self, state)

    def _upgrade_saved_state(self, saved: dict[str, Any]) -> None:
        """Rewrite this module's saved subtree from an earlier layout to the current one.

        The base module's state layout has not changed, so this does nothing. A subclass whose
        layout changed overrides it and edits ``saved`` in place.

        Args:
            saved: This module's own saved subtree.
        """

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


# ============================================================================
# Checkpoint state of an NNX module (the Checkpointable protocol's implementation)
# ============================================================================


def module_state(module: nnx.Module) -> dict[str, Any]:
    """The checkpoint state of ``module``: its ``nnx.Variable`` state as a pure dictionary.

    It captures positions, epochs, RNG keys and counts, and parameters, including every submodule
    reachable from ``module`` (a pipeline's stages and source). Plain data leaves, such as the
    records a source was built with, belong to the module's construction and are not captured:
    a checkpoint of a source over N records does not hold the records.

    Args:
        module: The module whose state is captured.

    Returns:
        A dictionary of the module's Variable values, nested as the module graph is.
    """
    return nnx.to_pure_dict(nnx.state(module, nnx.Variable, graph=True))


def restore_module_state(module: nnx.Module, state: dict[str, Any]) -> None:
    """Restore ``state``, as :func:`module_state` produced it, into ``module``.

    Each ``DataraxModule`` in the graph first rewrites its own subtree from an earlier layout
    (:meth:`DataraxModule._upgrade_saved_state`). Restoration is strict: the saved structure
    must match the module's, and array leaves their shapes and dtypes.

    Args:
        module: The module receiving the state, built the way the saved one was.
        state: The saved state dictionary.

    Raises:
        TypeError: If ``state`` is not a dictionary.
        ValueError: If the saved structure does not match the module's.
    """
    if not isinstance(state, dict):
        raise TypeError(f"State must be a dict, got {type(state).__name__}")

    upgraded = _upgraded_state(module, state)
    try:
        _validate_state(module, module_state(module), upgraded)
    except ValueError as exc:
        raise ValueError(
            "Checkpoint state is structurally incompatible with module state. "
            "Regenerate checkpoints after architecture/config changes."
        ) from exc
    _restore_state(module, upgraded)


def _upgraded_state(module: nnx.Module, state: dict[str, Any]) -> dict[str, Any]:
    """Return ``state`` with each module's earlier layout rewritten to the current one.

    Every ``DataraxModule`` in the graph is offered its own subtree, so a checkpoint written
    before a layout change restores without its reader knowing which module changed. A module
    whose subtree the saved state does not carry is skipped, and validation reports that
    difference as it would have anyway.

    Args:
        module: The module the state is restored into.
        state: The saved state dictionary.

    Returns:
        The upgraded state. The argument is not modified.
    """
    upgraded = _copy_state_nodes(state)
    for path, node in nnx.iter_graph(module, graph=True):
        if not isinstance(node, DataraxModule):
            continue
        subtree: Any = upgraded
        for key in path:
            if not isinstance(subtree, dict) or key not in subtree:
                subtree = None
                break
            subtree = subtree[key]
        if isinstance(subtree, dict):
            node._upgrade_saved_state(subtree)  # noqa: SLF001 - the hook each module overrides
    return upgraded


def _format_path(path: tuple[str | int, ...]) -> str:
    """Render a state-tree path as a dotted location for errors."""
    return ".".join(str(part) for part in path) if path else "<root>"


def _is_array_like(value: Any) -> bool:
    """Return True when value behaves like an ndarray/jax.Array leaf."""
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _validate_state(
    target: Any, current: Any, saved: Any, path: tuple[str | int, ...] = ()
) -> None:
    """Validate strict checkpoint compatibility.

    Rules:
    - Dict/list/tuple container structure must match exactly.
    - Array leaves must match shape and dtype.
    - Scalar/object leaves may change value/type (for flexible Variable payloads).
    """
    location = _format_path(path)
    if isinstance(target, nnx.Variable):
        _validate_array_leaf(current, saved, location)
        return
    if isinstance(target, nnx.Module):
        _validate_module_state(target, current, saved, path, location)
        return
    if isinstance(current, dict):
        _validate_mapping_state(current, saved, location)
        return
    _validate_array_leaf(current, saved, location)


def _validate_array_leaf(current: Any, saved: Any, location: str) -> None:
    """Validate shape/dtype compatibility for array-like leaves."""
    if not (_is_array_like(current) and _is_array_like(saved)):
        return
    if current.shape != saved.shape:
        raise ValueError(
            f"Array shape mismatch at {location}: expected {current.shape}, got {saved.shape}"
        )
    if current.dtype != saved.dtype:
        raise ValueError(
            f"Array dtype mismatch at {location}: expected {current.dtype}, got {saved.dtype}"
        )


def _validate_mapping_state(current: dict[Any, Any], saved: Any, location: str) -> None:
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
    target: nnx.Module, current: Any, saved: Any, path: tuple[str | int, ...], location: str
) -> None:
    """Validate module subtree recursively against a saved subtree."""
    if not isinstance(current, dict):
        raise ValueError(
            f"Current state mismatch at {location}: expected dict, got {type(current).__name__}"
        )
    _validate_mapping_state(current, saved, location)
    for key in current:
        child = _resolve_state_child(target, key, path)
        _validate_state(child, current[key], saved[key], (*path, key))


def _resolve_state_child(target: Any, key: str | int, path: tuple[str | int, ...]) -> Any:
    """Resolve a child object referenced by a state key."""
    if isinstance(key, str) and hasattr(target, key):
        return getattr(target, key)
    if hasattr(target, "__getitem__"):
        try:
            return target[key]
        except (KeyError, IndexError, TypeError):
            pass
    raise ValueError(f"State path {_format_path(path)} has no child for key {key!r}")


def _restore_state(target: Any, saved: Any, path: tuple[str | int, ...] = ()) -> None:
    """Restore validated state into an NNX object tree.

    A saved dict is applied through the target's attributes, which covers ``nnx.Module`` as well
    as ``nnx.Rngs`` and its streams (neither is a Module, and both carry Variables that must
    resume with the module).

    Args:
        target: The node receiving the state: a Variable, an object with children, or a sequence.
        saved: The saved state for that node.
        path: Path to ``target`` from the tree root, for error messages.

    Raises:
        ValueError: If ``saved`` has a shape no target node can receive.
    """
    if isinstance(target, nnx.Variable):
        target.set_value(saved)
        return
    if isinstance(saved, dict):
        for key, value in saved.items():
            _restore_state(_resolve_state_child(target, key, path), value, (*path, key))
        return
    if isinstance(target, list | tuple):
        if not isinstance(saved, list | tuple):
            raise ValueError(
                f"Invalid sequence state at {_format_path(path)}: got {type(saved).__name__}"
            )
        for index, (child, value) in enumerate(zip(target, saved, strict=True)):
            _restore_state(child, value, (*path, index))
        return
    raise ValueError(
        f"Cannot restore state at {_format_path(path)}: "
        f"{type(saved).__name__} into {type(target).__name__}"
    )
