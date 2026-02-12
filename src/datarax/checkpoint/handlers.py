"""Checkpoint handlers for Datarax.

This module provides checkpoint handlers for Datarax components using Orbax.
Follows Orbax patterns - leverages StandardCheckpointHandler and PyTreeCheckpointer
rather than reimplementing serialization logic.

References:
- orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py
- orbax/checkpoint/_src/handlers/random_key_checkpoint_handler.py
"""

from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from datarax.typing import Checkpointable


# Import after others to avoid circular imports
try:
    from datarax.core.module import DataraxModule
except ImportError:
    DataraxModule = Any  # type: ignore[misc]


class OrbaxCheckpointHandler:
    """Checkpoint handler for Datarax components using Orbax.

    This handler provides a high-level interface to Orbax checkpoint capabilities,
    leveraging StandardCheckpointer for PyTree serialization rather than
    reimplementing serialization logic.

    Following Orbax patterns from:

    - standard_checkpoint_handler.py for PyTree checkpointing
    - random_key_checkpoint_handler.py for PRNG key handling
    """

    def __init__(self, async_checkpointing: bool = False):
        """Initialize the handler.

        Args:
            async_checkpointing: If True, save() returns immediately and
                serialization happens in the background. Call
                wait_until_finished() before restore() or before the next
                save() if you need ordering guarantees.
        """
        self.async_checkpointing = async_checkpointing
        if async_checkpointing:
            self.checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        else:
            self.checkpointer = ocp.StandardCheckpointer()

    def wait_until_finished(self) -> None:
        """Block until any outstanding async save completes."""
        self.checkpointer.wait_until_finished()

    def _preprocess(self, target: Any) -> Any:
        """Single-pass preprocessing for checkpoint serialization.

        Converts strings to char-code marker dicts and PRNG keys to
        key-data marker dicts, in one tree walk.

        Args:
            target: The state tree to preprocess.

        Returns:
            Preprocessed tree ready for Orbax serialization.
        """
        # Strings must be checked before dicts (strings are iterable)
        if isinstance(target, str):
            return {"__string__": True, "char_codes": [ord(c) for c in target]}
        if isinstance(target, dict):
            return {k: self._preprocess(v) for k, v in target.items()}
        if isinstance(target, list):
            return [self._preprocess(item) for item in target]
        if isinstance(target, tuple):
            return tuple(self._preprocess(item) for item in target)
        if isinstance(target, jax.Array):
            # jax.dtypes.issubdtype exists at runtime (used by Orbax) but missing from stubs
            if jax.dtypes.issubdtype(target.dtype, jax.dtypes.prng_key):  # type: ignore[attr-defined]
                return {
                    "__prng_key__": True,
                    "key_data": jax.random.key_data(target).tolist(),
                }
        return target

    def _restore(self, target: Any) -> Any:
        """Single-pass restoration from checkpoint serialization.

        Restores strings and PRNG keys from their marker-dict format,
        in one tree walk.

        Args:
            target: The serialized tree to restore.

        Returns:
            Restored tree with original types.
        """
        if isinstance(target, dict):
            if target.get("__string__"):
                return "".join(chr(c) for c in target["char_codes"])
            if target.get("__prng_key__"):
                key_data = jnp.array(target["key_data"], dtype=jnp.uint32)
                return jax.random.wrap_key_data(key_data)
            return {k: self._restore(v) for k, v in target.items()}
        if isinstance(target, list):
            return [self._restore(item) for item in target]
        if isinstance(target, tuple):
            return tuple(self._restore(item) for item in target)
        return target

    def save(
        self,
        directory: str | Path,
        target: Any,
        step: int | None = None,
        keep: int | None = 1,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a checkpoint.

        Args:
            directory: Directory to save to.
            target: Object to checkpoint (Checkpointable, dict, or PyTree).
            step: Optional step number for versioned checkpoints.
            keep: Number of checkpoints to keep (for versioned checkpoints).
            overwrite: Whether to overwrite existing checkpoints.
            metadata: Optional metadata to save with the checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Extract state from Checkpointable objects
        if isinstance(target, Checkpointable):
            state = target.get_state()
        else:
            state = target

        # Add metadata if provided
        if metadata:
            if isinstance(state, dict):
                state = {**state, "__metadata__": metadata}
            else:
                state = {"__state__": state, "__metadata__": metadata}

        # Single-pass preprocessing: strings + PRNG keys in one tree walk
        state = self._preprocess(state)

        if step is not None:
            # Versioned checkpoints with CheckpointManager
            options = ocp.CheckpointManagerOptions(max_to_keep=keep, create=True)
            with ocp.CheckpointManager(str(directory), options=options) as manager:
                save_args = ocp.args.StandardSave(state)  # type: ignore[call-arg]
                manager.save(step, args=save_args)
                manager.wait_until_finished()

            saved_path = str(directory / str(step))
            if not Path(saved_path).exists():
                ckpt_path = str(directory / f"ckpt-{step}")
                if Path(ckpt_path).exists():
                    saved_path = ckpt_path
        else:
            # Single checkpoint
            self.checkpointer.save(str(directory / "checkpoint"), state, force=overwrite)
            if not self.async_checkpointing:
                self.checkpointer.wait_until_finished()
            saved_path = str(directory / "checkpoint")

        return saved_path

    def restore(
        self,
        directory: str | Path,
        target: Any | None = None,
        step: int | None = None,
        metadata_only: bool = False,
        restore_args: Any | None = None,  # noqa: ARG002
    ) -> Any:
        """Restore from a checkpoint.

        Args:
            directory: Directory to restore from.
            target: Optional target to restore into.
            step: Optional step to restore from (None = latest).
            metadata_only: If True, only return metadata.
            restore_args: Reserved for Orbax interoperability.

        Returns:
            The restored object, state, or metadata.

        Raises:
            ValueError: If directory doesn't exist or no checkpoints found.
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Checkpoint directory not found: {directory}")

        checkpoint_path = self._find_checkpoint_path(directory, step)
        default_subdir = checkpoint_path / "default"
        actual_path = default_subdir if default_subdir.exists() else checkpoint_path

        # Ensure any async save is complete before restoring
        if self.async_checkpointing:
            self.checkpointer.wait_until_finished()

        # Restore state
        state = self.checkpointer.restore(str(actual_path))

        # Single-pass restoration: strings + PRNG keys in one tree walk
        state = self._restore(state)

        # Extract metadata
        metadata = None
        if isinstance(state, dict) and "__metadata__" in state:
            metadata = state.pop("__metadata__")
        if isinstance(state, dict) and "__state__" in state:
            state = state["__state__"]

        if metadata_only:
            return metadata

        # Apply state to target if provided
        if target is not None:
            if isinstance(target, Checkpointable):
                target.set_state(state)
                return target
            elif isinstance(target, nnx.Module):
                nnx.update(target, state)
                return target

        return state

    def _find_checkpoint_path(self, directory: Path, step: int | None) -> Path:
        """Find checkpoint path for a given step."""
        if step is not None:
            for fmt in [f"ckpt-{step}", str(step)]:
                path = directory / fmt
                if path.exists():
                    return path
            if (directory / "checkpoint").exists():
                return directory / "checkpoint"
            raise ValueError(f"Checkpoint not found for step {step}")

        steps = self.get_checkpoint_steps(directory)
        if steps:
            latest = max(steps)
            for fmt in [f"ckpt-{latest}", str(latest)]:
                path = directory / fmt
                if path.exists():
                    return path

        checkpoint_path = directory / "checkpoint"
        if checkpoint_path.exists():
            return checkpoint_path
        raise ValueError(f"No checkpoints found in {directory}")

    def get_checkpoint_steps(self, directory: str | Path) -> list[int]:
        """Get all checkpoint steps in a directory."""
        directory = Path(directory)
        if not directory.exists():
            return []

        steps = []
        for item in directory.iterdir():
            if item.is_dir():
                name = item.name
                if name.startswith("ckpt-"):
                    try:
                        steps.append(int(name.split("-")[1]))
                    except (IndexError, ValueError):
                        continue
                else:
                    try:
                        steps.append(int(name))
                    except ValueError:
                        continue
        return sorted(steps)

    def latest_step(self, directory: str | Path) -> int | None:
        """Get the latest checkpoint step."""
        steps = self.get_checkpoint_steps(directory)
        return max(steps) if steps else None

    def list_checkpoints(self, directory: str | Path) -> dict[int, str]:
        """List all checkpoints in a directory."""
        directory = Path(directory)
        return {
            step: str(directory / f"ckpt-{step}") for step in self.get_checkpoint_steps(directory)
        }

    def close(self) -> None:
        """Close the checkpoint handler and release resources.

        This method waits for any outstanding async operations to finish
        and properly cleans up the underlying checkpointer. Should be called
        when done with checkpointing, or use the context manager protocol.
        """
        self.checkpointer.close()

    def __enter__(self) -> "OrbaxCheckpointHandler":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close handler."""
        self.close()
