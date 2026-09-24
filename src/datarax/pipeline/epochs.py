"""How a pipeline's records divide into batches and epochs.

:class:`EpochPlan` is the one place the rule lives. No batch holds padding. Under
``drop_last`` the records short of a full batch are skipped and the next batch starts the
next epoch, tf.data's and Grain's ``batch(drop_remainder=True).repeat()``. Otherwise a batch
reaching the epoch's end is completed from the head of the next epoch's order, their
``repeat().batch()``, so every epoch serves each record once. A session serving a number of
epochs stops after exactly those records, so its final batch may be short.
:class:`~datarax.pipeline.pipeline.Pipeline` builds a plan from its source's current length,
and its length, ``batches_left``, iteration sessions and compiled step all read it.
"""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpochPlan:
    """How records divide into batches and epochs.

    Attributes:
        length: Records per epoch, or ``None`` for a source without a length, whose
            epochs never end.
        batch_size: Records per batch.
        drop_last: Whether an epoch's records short of a full batch are skipped, rather
            than completed from the next epoch.
        num_epochs: Epochs a session serves, or ``None`` for a session that never stops.
    """

    length: int | None
    batch_size: int
    drop_last: bool
    num_epochs: int | None

    def __post_init__(self) -> None:
        """Refuse a plan no pipeline can serve.

        Raises:
            ValueError: If ``batch_size`` is not positive, ``num_epochs`` is neither
                ``None`` nor at least 1, the source has no records, or ``drop_last``
                skips every record because a batch is larger than an epoch.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        if self.num_epochs is not None and self.num_epochs < 1:
            raise ValueError(
                f"num_epochs must be at least 1, or None for a stream; got {self.num_epochs}"
            )
        if self.length == 0:
            raise ValueError("the source has no records, so no batch can be served")
        if self.drop_last and self.length is not None and self.batch_size > self.length:
            raise ValueError(
                f"drop_last needs batch_size <= len(source), or every record is skipped; got "
                f"batch_size {self.batch_size} over {self.length} records"
            )

    @property
    def crosses(self) -> bool:
        """Whether a batch reaching an epoch's end continues in the next epoch."""
        return not self.drop_last and self.length is not None

    def exhausted[T: (int, jax.Array)](self, position: T) -> bool | jax.Array:
        """Whether the epoch cannot start another batch at ``position``.

        It can while at least one record is left, or under ``drop_last`` a full batch.

        Args:
            position: Records of the epoch already served; a Python int or a traced array.

        Returns:
            A bool for an int position, a bool array for an array; never exhausted for a
            source without a length.
        """
        if self.length is None:
            return False
        needed = self.batch_size if self.drop_last else 1
        return position + needed > self.length

    def batch_start[T: (int, jax.Array)](self, position: T, epoch: T) -> tuple[T, T]:
        """Where the next batch starts: here, or at the next epoch's start if this one is done.

        Args:
            position: Records of the epoch already served.
            epoch: The epoch counter.

        Returns:
            ``(position, epoch)`` the batch starts at.
        """
        done = self.exhausted(position)
        first = position * 0  # position 0, as an int or an array of the position's dtype
        return _select(done, first, position), _select(done, epoch + 1, epoch)

    def advance[T: (int, jax.Array)](self, start: T, epoch: T, size: int) -> tuple[T, T]:
        """The position and epoch after a batch of ``size`` records starting at ``start``.

        A batch crossing the epoch's end leaves the position in the epoch it ends in (a
        batch larger than an epoch spans several); one ending exactly at an epoch's end
        leaves that epoch exhausted, so the next batch starts the next epoch.

        Args:
            start: Where the batch started, as :meth:`batch_start` gives it.
            epoch: The epoch the batch started in.
            size: Records in the batch.

        Returns:
            ``(position, epoch)`` after the batch.
        """
        end = start + size
        if self.length is None:
            return end, epoch
        crossed = (end - 1) // self.length  # epoch ends the batch passed, not reached
        return end - crossed * self.length, epoch + crossed

    def epochs_touched(self, size: int) -> int:
        """The most epochs a batch of ``size`` records starting inside an epoch holds.

        Args:
            size: Records in the batch.

        Returns:
            The count; 1 for a source without a length.
        """
        if self.length is None:
            return 1
        return (self.length + size - 2) // self.length + 1

    @staticmethod
    def next_epoch[T: (int, jax.Array)](epoch: T) -> tuple[int, T]:
        """Where the epoch after ``epoch`` starts: position 0.

        Args:
            epoch: The epoch just finished.

        Returns:
            ``(0, epoch + 1)``.
        """
        return 0, epoch + 1

    def run_extent(self, position: int) -> tuple[int, int] | None:
        """The batches a session starting at ``position`` serves, and its final batch's size.

        The session finishes the current epoch, an exhausted one counting as served, then
        serves the rest of ``num_epochs``.

        Args:
            position: Records of the current epoch already served.

        Returns:
            ``(batches, final_batch_size)``, or ``None`` for a session that never stops: a
            stream or a source without a length.
        """
        if self.num_epochs is None or self.length is None:
            return None
        left = 0 if self.exhausted(position) else self.length - position
        later = self.num_epochs - 1
        if self.drop_last:
            return left // self.batch_size + later * (self.length // self.batch_size), (
                self.batch_size
            )
        records = left + later * self.length
        batches = math.ceil(records / self.batch_size)
        if batches == 0:
            return 0, 0
        return batches, records - (batches - 1) * self.batch_size


def _select[T: (int, jax.Array)](condition: bool | jax.Array, if_true: T, if_false: T) -> T:
    """``if_true`` where ``condition`` holds, else ``if_false``: on the host or traced alike."""
    if isinstance(condition, bool):
        return if_true if condition else if_false
    return cast(T, jnp.where(condition, if_true, if_false))
