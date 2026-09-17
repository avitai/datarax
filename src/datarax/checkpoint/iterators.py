"""Step-addressed checkpoints for objects that expose ``get_state`` / ``set_state``.

Data iterators, pipelines and Datarax modules all implement the
:class:`~datarax.typing.Checkpointable` protocol. :class:`IteratorCheckpoint`
persists such a state dictionary under an integer step as the ``data_iterator``
item of substrax's :class:`~substrax.checkpoint.OrbaxCheckpointStore`, which
carries arrays, typed PRNG keys and plain-Python leaves (positions, seeds, sampler
reprs) alike, and restores it back into a freshly built object after checking
that the object was built the same way as the one that was saved. A root written
by datarax 0.1.11 or earlier (substrax's format 2, the state as the one payload)
is read through :data:`ITERATOR_STATE_FORMAT2`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

from substrax.checkpoint import CheckpointStore, LegacyLayout, OrbaxCheckpointStore

from datarax.typing import Checkpointable


logger = logging.getLogger(__name__)

ITEM = "data_iterator"

ITERATOR_STATE_FORMAT2 = LegacyLayout(
    name="datarax-iterator",
    items_of=lambda payload: {ITEM: payload},
    template_of=lambda templates: templates[ITEM],
)
"""The format-2 layout datarax 0.1.11 wrote: the state dictionary as the one payload.

Pass it to ``substrax.checkpoint.upgrade_checkpoints`` to rewrite such a root in the
current format; :meth:`IteratorCheckpoint.restore` reads one in place.
"""

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
        epoch: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Save ``target``'s state under ``step`` as the ``data_iterator`` item.

        Args:
            target: The object whose ``get_state()`` is written.
            step: Non-negative step the checkpoint is addressed by.
            epoch: The epoch ``step`` belongs to, recorded as the record's ``epoch``.
            metadata: JSON-serialisable values recorded beside the state, in the
                record's ``extra``; a key that names a field of the record is refused.

        Returns:
            The directory of the saved checkpoint.
        """
        state = _state_of(target)
        path = self.store.save(step, {ITEM: state}, epoch=epoch, extra=metadata)
        logger.info("Saved %s state at step %d to %s", type(target).__name__, step, path)
        return path

    def save_if_due(
        self,
        target: Checkpointable,
        step: int,
        *,
        interval: int,
        epoch: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Save when ``step`` is a multiple of ``interval``.

        Args:
            target: The object whose ``get_state()`` is written.
            step: The current step.
            interval: Steps between checkpoints; must be positive.
            epoch: The epoch ``step`` belongs to, recorded as the record's ``epoch``.
            metadata: JSON-serialisable values recorded beside the state, in the
                record's ``extra``.

        Returns:
            The saved checkpoint's directory, or ``None`` when the step is not due.

        Raises:
            ValueError: If ``interval`` is not positive.
        """
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        if step % interval != 0:
            return None
        return self.save(target, step, epoch=epoch, metadata=metadata)

    def restore(self, target: Checkpointable, *, step: int | None = None) -> None:
        """Restore a saved state into ``target``.

        The checkpoint describes its own tree and is read back as saved; the
        identity fields of ``target`` (sampler and data-source reprs, shard and
        worker counts) must match the checkpoint's, and ``target.set_state``
        decides whether the structure fits. A ``step`` the directory holds no
        checkpoint at propagates the store's ``CheckpointNotFoundError``.

        Args:
            target: The object whose ``set_state`` receives the saved state.
            step: The step to restore; the latest when ``None``.

        Raises:
            ValueError: If the directory holds no checkpoint at all, or the
                checkpoint's identity fields differ from ``target``'s.
        """
        if step is None:
            step = self.latest_step()
            if step is None:
                raise ValueError(f"No checkpoints found in {self.base_dir}")
        checkpoint = self.store.restore(step, legacy_layout=ITERATOR_STATE_FORMAT2)
        restored = checkpoint.items[ITEM]
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
