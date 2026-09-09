"""Step-addressed checkpoints for objects that expose ``get_state`` / ``set_state``.

Data iterators, pipelines and Datarax modules all implement the
:class:`~datarax.typing.Checkpointable` protocol. :class:`IteratorCheckpoint`
persists such a state dictionary under an integer step with substrax's
:class:`~substrax.checkpoint.OrbaxCheckpointStore`, which carries arrays, typed
PRNG keys and plain-Python leaves (positions, seeds, sampler reprs) alike, and
restores it back into a freshly built object after checking that the object was
built the same way as the one that was saved.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from substrax.checkpoint import CheckpointStore, OrbaxCheckpointStore

from datarax.typing import Checkpointable


logger = logging.getLogger(__name__)

# Grain-style identity fields that must stay compatible across a checkpoint
# restore. Following Grain's checkpoint validation: sampler / data-source
# representations must match exactly, ``shard_count`` must match (``shard_index``
# may differ), and ``worker_count`` must match. Only fields present in BOTH the
# saved checkpoint and the current state are compared, so objects that do not
# expose identity fields are simply not validated.
_RESTORE_IDENTITY_FIELDS: tuple[str, ...] = (
    "sampler_repr",
    "data_source_repr",
    "shard_count",
    "worker_count",
)


def validate_restore_compatibility(
    current_state: Mapping[str, Any],
    saved_state: Mapping[str, Any],
) -> None:
    """Raise if a checkpoint's identity fields are incompatible with the live object.

    Compares the Grain-style identity fields in ``_RESTORE_IDENTITY_FIELDS`` that
    appear in both ``current_state`` (from the live object) and ``saved_state``
    (from the checkpoint). A mismatch means the checkpoint was produced with a
    different sampler / data-source configuration or a different shard/worker
    topology, which would silently corrupt resumed iteration order or the
    per-host data distribution.

    Args:
        current_state: ``get_state()`` of the object being restored into.
        saved_state: The state dict loaded from the checkpoint.

    Raises:
        ValueError: If any shared identity field differs between the two states.
    """
    mismatches = [
        (field, saved_state[field], current_state[field])
        for field in _RESTORE_IDENTITY_FIELDS
        if field in current_state
        and field in saved_state
        and current_state[field] != saved_state[field]
    ]
    if mismatches:
        details = "; ".join(
            f"{field}: checkpoint={saved!r} but current={current!r}"
            for field, saved, current in mismatches
        )
        raise ValueError(
            "Checkpoint is incompatible with the current iterator configuration "
            f"({details}). Restoring would corrupt iteration order or the per-host "
            "data distribution. Rebuild the iterator with the checkpointed "
            "configuration, or discard this checkpoint."
        )


def _state_of(target: Checkpointable) -> dict[str, Any]:
    """Return ``target``'s state dictionary, rejecting anything that is not a dict.

    Args:
        target: The object being checkpointed.

    Returns:
        The state ``target.get_state()`` produced.

    Raises:
        TypeError: If ``get_state`` returned something other than a dict.
        ValueError: If the dict is empty; Orbax cannot write an empty tree.
    """
    state = target.get_state()
    # The protocol promises a dict; untyped callers can still hand over anything.
    if not isinstance(state, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"{type(target).__name__}.get_state() must return a dict, got {type(state).__name__}"
        )
    if not state:
        raise ValueError(f"{type(target).__name__}.get_state() returned nothing to checkpoint")
    return state


class IteratorCheckpoint:
    """Save and restore ``Checkpointable`` state under a directory, addressed by step.

    Each ``save`` writes one checkpoint for an integer step; Orbax keeps the most
    recent ``max_to_keep`` of them. ``restore`` reads a step (the latest by
    default) back into an object built the same way as the one that was saved.
    The store is injectable so the same class runs over any
    :class:`~substrax.checkpoint.CheckpointStore`.

    Example:
        ```python
        checkpoint = IteratorCheckpoint(run_dir / "pipeline", max_to_keep=3)
        for step, batch in enumerate(pipeline):
            train_step(batch)
            checkpoint.save_if_due(pipeline, step, interval=1000)

        fresh = build_pipeline()
        checkpoint.restore(fresh)  # the latest step
        ```
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        max_to_keep: int = 5,
        store: CheckpointStore | None = None,
    ) -> None:
        """Open (creating if needed) the checkpoint directory.

        Args:
            base_dir: Directory the checkpoints live under.
            max_to_keep: How many of the most recent steps the default store retains.
            store: A store to use instead of an Orbax store over ``base_dir``.
        """
        self.base_dir = Path(base_dir)
        self.store: CheckpointStore = (
            OrbaxCheckpointStore(self.base_dir, max_to_keep=max_to_keep) if store is None else store
        )

    def __enter__(self) -> Self:
        """Enter a context that closes the store on exit.

        Returns:
            This checkpoint.
        """
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close the store."""
        self.close()

    def close(self) -> None:
        """Release the store's resources."""
        self.store.close()

    def save(
        self,
        target: Checkpointable,
        step: int,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Save ``target``'s state under ``step``.

        Args:
            target: The object whose ``get_state()`` is written.
            step: Non-negative step the checkpoint is addressed by.
            metadata: JSON-serialisable values recorded beside the state.

        Returns:
            The path of the saved checkpoint.
        """
        state = _state_of(target)
        path = self.store.save(
            state, step, additional_metadata=dict(metadata) if metadata else None
        )
        logger.info("Saved %s state at step %d to %s", type(target).__name__, step, path)
        return path

    def save_if_due(
        self,
        target: Checkpointable,
        step: int,
        *,
        interval: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Save when ``step`` is a multiple of ``interval``.

        Args:
            target: The object whose ``get_state()`` is written.
            step: The current step.
            interval: Steps between checkpoints; must be positive.
            metadata: JSON-serialisable values recorded beside the state.

        Returns:
            The saved checkpoint's path, or ``None`` when the step is not due.

        Raises:
            ValueError: If ``interval`` is not positive.
        """
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        if step % interval != 0:
            return None
        return self.save(target, step, metadata=metadata)

    def restore(self, target: Checkpointable, *, step: int | None = None) -> None:
        """Restore a saved state into ``target``.

        The checkpoint describes its own tree and is read back as saved; the
        identity fields of ``target`` (sampler and data-source reprs, shard and
        worker counts) must match the checkpoint's, and ``target.set_state``
        decides whether the structure fits.

        Args:
            target: The object whose ``set_state`` receives the saved state.
            step: The step to restore; the latest when ``None``.

        Raises:
            ValueError: If the directory holds no checkpoint at ``step`` (or none at
                all), or the checkpoint's identity fields differ from ``target``'s.
        """
        if step is None:
            step = self.latest_step()
            if step is None:
                raise ValueError(f"No checkpoints found in {self.base_dir}")
        restored, _ = self.store.restore(step=step, return_original_on_missing=False)
        if not isinstance(restored, dict):
            raise ValueError(f"No checkpoint at step {step} in {self.base_dir}")
        validate_restore_compatibility(target.get_state(), restored)
        target.set_state(restored)
        logger.info("Restored %s state from step %d", type(target).__name__, step)

    def latest_step(self) -> int | None:
        """Return the most recent saved step, or ``None`` without checkpoints."""
        return self.store.latest_step()

    def all_steps(self) -> list[int]:
        """Return every saved step, oldest first."""
        return sorted(self.store.list_steps())

    def has_checkpoint(self) -> bool:
        """Return whether at least one step has been saved."""
        return self.latest_step() is not None
