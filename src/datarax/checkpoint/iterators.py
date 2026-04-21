"""Iterator checkpoint functionality for Datarax.

This module provides checkpoint handlers for data iterators and streams,
supporting save and restore of iterator state for resumable iteration.
"""

import logging
from pathlib import Path
from typing import TypeVar

from datarax.checkpoint.handlers import OrbaxCheckpointHandler
from datarax.typing import CheckpointableIterator


logger = logging.getLogger(__name__)


# Define covariant type parameter for iterators
T_co = TypeVar("T_co", covariant=True)


class IteratorCheckpoint:
    """Handler for checkpointing iterators.

    This class provides methods for saving and restoring the state of iterators
    that implement the CheckpointableIterator interface.
    """

    def __init__(
        self,
        base_dir: str | Path,
        handler: OrbaxCheckpointHandler | None = None,
        async_checkpointing: bool = False,
    ) -> None:
        """Initialize the iterator checkpoint handler.

        Args:
            base_dir: Base directory to store checkpoints in.
            handler: Optional custom checkpoint handler to use. If None,
                a default OrbaxCheckpointHandler will be created.
            async_checkpointing: Whether to enable asynchronous checkpointing.
        """
        self.base_dir = Path(base_dir)
        self.handler = handler or OrbaxCheckpointHandler(async_checkpointing=async_checkpointing)

    def save_to_directory(
        self,
        iterator: CheckpointableIterator[T_co],
        step: int | None = None,
        keep: int = 1,
        overwrite: bool = False,
        metadata: dict[str, int | str | float | bool] | None = None,
    ) -> str:
        """Save the state of an iterator to the checkpoint directory.

        Args:
            iterator: The iterator to checkpoint.
            step: Optional step number for versioning.
            keep: Number of checkpoints to keep.
            overwrite: Whether to overwrite existing checkpoints.
            metadata: Optional metadata to save with checkpoint.

        Returns:
            Path to the saved checkpoint.

        Raises:
            ValueError: If the iterator does not implement get_state.
        """
        if not hasattr(iterator, "get_state") or not callable(iterator.get_state):
            raise ValueError("Iterator does not implement get_state method")
        state_dict = iterator.get_state()
        if not isinstance(state_dict, dict):
            raise ValueError(
                f"Iterator get_state() must return dict, got {type(state_dict).__name__}"
            )

        return self.handler.save_to_directory(
            self.base_dir,
            state_dict,
            step=step,
            keep=keep,
            overwrite=overwrite,
            metadata=metadata,
        )

    def restore(
        self,
        iterator: CheckpointableIterator[T_co],
        step: int | None = None,
    ) -> CheckpointableIterator[T_co]:
        """Restore the state of an iterator from a checkpoint.

        Args:
            iterator: The iterator to restore state into.
            step: Optional step number to restore from. If None, uses latest.

        Returns:
            The restored iterator.

        Raises:
            ValueError: If the iterator does not implement set_state or if the
                checkpoint cannot be restored.
        """
        if not hasattr(iterator, "set_state") or not callable(iterator.set_state):
            msg = f"Iterator of type {type(iterator)} does not implement "
            msg += "set_state"
            raise ValueError(msg)

        if not hasattr(iterator, "get_state") or not callable(iterator.get_state):
            raise ValueError("Iterator does not implement get_state method")

        restored = self.handler.restore(self.base_dir, step=step, target=iterator)
        if restored is not iterator:
            raise ValueError("Iterator restore failed: handler did not return the target iterator")
        return restored

    def restore_latest(
        self,
        iterator: CheckpointableIterator[T_co],
    ) -> CheckpointableIterator[T_co]:
        """Restore from the latest available checkpoint.

        Args:
            iterator: The iterator to restore state into.

        Returns:
            The restored iterator.

        Raises:
            ValueError: If no checkpoints are found or restoration fails.
        """
        latest_step = self.get_latest_step()
        if latest_step is None:
            raise ValueError(f"No checkpoints found in {self.base_dir}")
        return self.restore(iterator, step=latest_step)

    def get_latest_step(self) -> int | None:
        """Get the latest checkpoint step.

        Returns:
            The latest checkpoint step, or None if no checkpoints are found.
        """
        return self.handler.latest_step(self.base_dir)

    def list_checkpoints(self) -> dict[int, str]:
        """List all available checkpoints.

        Returns:
            A dictionary mapping step numbers to checkpoint paths.
        """
        return self.handler.list_checkpoints(self.base_dir)

    def has_checkpoint(self) -> bool:
        """Check if any checkpoints exist.

        Returns:
            True if at least one checkpoint exists, False otherwise.
        """
        return self.get_latest_step() is not None


class PipelineCheckpoint(IteratorCheckpoint):
    """Specialized checkpoint handler for Pipeline objects.

    This class extends IteratorCheckpoint with Pipeline-specific
    functionality like conditional saving based on step intervals.
    """

    def save_to_step(
        self,
        data_stream: CheckpointableIterator[T_co],
        step: int,
        interval: int = 1000,
        keep: int = 5,
        overwrite: bool = False,
        metadata: dict[str, int | str | float | bool] | None = None,
    ) -> str | None:
        """Save a checkpoint conditionally based on the step.

        Only saves when the step is a multiple of the interval.

        Args:
            data_stream: The data stream to checkpoint.
            step: Current step number.
            interval: Step interval for saving.
            keep: Number of checkpoints to keep.
            overwrite: Whether to overwrite existing checkpoints.
            metadata: Optional metadata to store with the checkpoint.

        Returns:
            Path to the saved checkpoint or None if not saved.
        """
        # Only save at specified intervals
        if step % interval != 0:
            return None

        # Save checkpoint using the inherited method
        return self.save_to_directory(
            data_stream,
            step=step,
            keep=keep,
            overwrite=overwrite,
            metadata=metadata,
        )
