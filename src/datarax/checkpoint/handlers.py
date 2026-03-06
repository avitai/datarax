"""Checkpoint handlers for Datarax.

This module provides checkpoint handlers for Datarax components using Orbax.
Follows Orbax patterns - leverages StandardCheckpointHandler and PyTreeCheckpointer
rather than reimplementing serialization logic.

References:
- orbax/checkpoint/_src/handlers/standard_checkpoint_handler.py
- orbax/checkpoint/_src/handlers/random_key_checkpoint_handler.py
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from datarax.typing import Checkpointable


logger = logging.getLogger(__name__)

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

    def __init__(self, async_checkpointing: bool = False) -> None:
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

    def _orbax_restore(self, checkpoint_path: str, item: Any | None = None) -> Any:
        """Restore checkpoint data with cross-version Orbax compatibility."""
        if item is None:
            return self.checkpointer.restore(checkpoint_path)
        try:
            # Orbax accepts item= at runtime but stubs may not declare it
            return self.checkpointer.restore(checkpoint_path, item=item)  # type: ignore[reportCallIssue]
        except TypeError:
            # Older Orbax accepted positional target only.
            return self.checkpointer.restore(checkpoint_path, item)

    def _state_for_checkpoint(self, target: Any) -> Any:
        """Normalize a checkpoint target into a serializable state tree."""
        if isinstance(target, Checkpointable):
            return target.get_state()
        if isinstance(target, nnx.Module):
            return nnx.to_pure_dict(nnx.state(target))
        return target

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

        state = self._state_for_checkpoint(target)

        # Add metadata if provided
        if metadata:
            if isinstance(state, dict):
                state = {**state, "__metadata__": metadata}
            else:
                state = {"__state__": state, "__metadata__": metadata}

        # Single-pass preprocessing: strings + PRNG keys in one tree walk
        state = self._preprocess(state)

        if step is not None:
            # Versioned checkpoints: save each step into its own checkpoint directory.
            # Using direct checkpointers here avoids CheckpointManager-specific stalls
            # observed in some environments while preserving restore-by-step behavior.
            step_dir = directory / f"ckpt-{step}"
            self.checkpointer.save(str(step_dir), state, force=overwrite)
            if not self.async_checkpointing:
                self.checkpointer.wait_until_finished()
            saved_path = str(step_dir)
            self._prune_old_checkpoints(directory, keep)
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
        actual_path = self._resolve_restore_path(directory, step)

        if self.async_checkpointing:
            self.checkpointer.wait_until_finished()

        restore_target = self._build_restore_target(target)
        state = self._orbax_restore(str(actual_path), item=restore_target)
        state = self._decode_restored_state(state, restore_target)
        state, metadata = self._extract_state_and_metadata(state)

        if metadata_only:
            return metadata

        return self._apply_restored_state(target, state)

    def _resolve_restore_path(self, directory: str | Path, step: int | None) -> Path:
        """Resolve concrete checkpoint path, including Orbax default subdir."""
        base_dir = Path(directory)
        if not base_dir.exists():
            raise ValueError(f"Checkpoint directory not found: {base_dir}")
        checkpoint_path = self._find_checkpoint_path(base_dir, step)
        default_subdir = checkpoint_path / "default"
        return default_subdir if default_subdir.exists() else checkpoint_path

    def _build_restore_target(self, target: Any | None) -> Any | None:
        """Build preprocessed restore target tree for Orbax validation."""
        if target is None:
            return None
        # Restore target must mirror the serialized structure saved to disk.
        # This keeps Orbax metadata validation consistent for transformed
        # leaves such as PRNG keys and string markers.
        return self._preprocess(self._state_for_checkpoint(target))

    def _decode_restored_state(self, state: Any, restore_target: Any | None) -> Any:
        """Decode serialized tree and recover integer paths when needed."""
        decoded_state = self._restore(state)
        # Orbax can stringify integer path segments when restoring pure dicts
        # without a target tree. Recover integer paths for NNX container states.
        if restore_target is None and isinstance(decoded_state, dict):
            return nnx.restore_int_paths(decoded_state)
        return decoded_state

    def _extract_state_and_metadata(self, state: Any) -> tuple[Any, Any | None]:
        """Split payload into state and optional metadata envelope."""
        metadata: Any | None = None
        if isinstance(state, dict) and "__metadata__" in state:
            metadata = state.pop("__metadata__")
        if isinstance(state, dict) and "__state__" in state:
            return state["__state__"], metadata
        return state, metadata

    def _apply_restored_state(self, target: Any | None, state: Any) -> Any:
        """Apply restored state to target when one is provided."""
        if target is None:
            return state
        if isinstance(target, Checkpointable):
            target.set_state(state)
            return target
        if isinstance(target, nnx.Module):
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

    def _prune_old_checkpoints(self, directory: Path, keep: int | None) -> None:
        """Delete oldest versioned checkpoints beyond the keep count."""
        if keep is None or keep <= 0:
            return
        steps = self.get_checkpoint_steps(directory)
        if len(steps) <= keep:
            return
        for step in steps[: len(steps) - keep]:
            for name in (f"ckpt-{step}", str(step)):
                checkpoint_path = directory / name
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path, ignore_errors=True)

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
