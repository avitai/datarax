"""How a pipeline's records divide into batches and epochs.

:class:`EpochPlan` is the one place the rule lives: batches per epoch (floor under
``drop_last``, ceil otherwise), batches left from a position, when an epoch is exhausted, how
a position advances -- wrapping into the next epoch for a continuous stream -- and which rows
of a batch are records of the epoch. :class:`~datarax.pipeline.pipeline.Pipeline` builds a
plan from its source's current length, and its length, ``batches_left``, iteration sessions
and compiled step all read it.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpochPlan:
    """How records divide into batches and epochs.

    Attributes:
        length: Records per epoch, or ``None`` for a source without a length, whose
            epochs never end.
        batch_size: Records per batch.
        drop_last: Whether an epoch's final partial batch is skipped rather than served
            padded, with its missing rows marked invalid.
        num_epochs: Epochs a session serves, or ``None`` for a continuous stream whose
            batches cross epoch boundaries without padding.
    """

    length: int | None
    batch_size: int
    drop_last: bool
    num_epochs: int | None

    def __post_init__(self) -> None:
        """Refuse a plan no pipeline can serve.

        Raises:
            ValueError: If ``batch_size`` is not positive, ``num_epochs`` is neither
                ``None`` nor at least 1, or a continuous stream's batch is larger than an
                epoch, which no single boundary batch can hold.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        if self.num_epochs is not None and self.num_epochs < 1:
            raise ValueError(
                f"num_epochs must be at least 1, or None for a stream; got {self.num_epochs}"
            )
        if self.num_epochs is None and self.length is not None and self.batch_size > self.length:
            raise ValueError(
                f"a continuous stream needs batch_size <= len(source); got batch_size "
                f"{self.batch_size} over {self.length} records"
            )

    @property
    def continuous(self) -> bool:
        """Whether batches cross epoch boundaries: a stream over a source with a length."""
        return self.num_epochs is None and self.length is not None

    def batches(self, records: int) -> int:
        """Batches ``records`` records make: floor under ``drop_last``, ceil otherwise.

        Args:
            records: A record count.

        Returns:
            The batch count.
        """
        if self.drop_last:
            return records // self.batch_size
        return math.ceil(records / self.batch_size)

    def batches_per_epoch(self) -> int:
        """Batches one epoch serves.

        Returns:
            The batch count.

        Raises:
            TypeError: If the source has no length.
        """
        if self.length is None:
            raise TypeError("the source has no length, so an epoch has no batch count")
        return self.batches(self.length)

    def batches_left(self, position: int) -> int | None:
        """Batches the epoch still holds from ``position``.

        Args:
            position: Records of the epoch already served.

        Returns:
            The count, or ``None`` when it is not known: a source without a length, or a
            continuous stream, which never runs out.
        """
        if self.length is None or self.num_epochs is None:
            return None
        return self.batches(max(self.length - position, 0))

    def exhausted(self, position: int) -> bool:
        """Whether the epoch has served its last batch at ``position``.

        Args:
            position: Records of the epoch already served.

        Returns:
            ``True`` when no batch is left; never for a stream or a source without a length.
        """
        return self.batches_left(position) == 0

    def position_after[T: (int, jax.Array)](self, position: T) -> T:
        """The position after one batch: wrapped into the next epoch for a continuous stream.

        Written with operators Python integers and JAX arrays share, so the compiled step
        and a session's host-side count advance by one rule.

        Args:
            position: Records of the epoch served before the batch.

        Returns:
            The position after the batch.
        """
        consumed = position + self.batch_size
        if not self.continuous:
            return consumed
        assert self.length is not None  # noqa: S101 - continuous implies a length
        return consumed % self.length

    def advance[T: (int, jax.Array)](self, position: T, epoch: T) -> tuple[T, T]:
        """The position and epoch after one batch.

        Args:
            position: Records of the epoch served before the batch.
            epoch: The epoch the batch starts in.

        Returns:
            ``(position, epoch)`` after the batch: a continuous stream carries into the next
            epoch when the batch crosses its end; otherwise the epoch is unchanged.
        """
        if not self.continuous:
            return self.position_after(position), epoch
        assert self.length is not None  # noqa: S101 - continuous implies a length
        return self.position_after(position), epoch + (position + self.batch_size) // self.length

    @staticmethod
    def next_epoch[T: (int, jax.Array)](epoch: T) -> tuple[int, T]:
        """Where the epoch after ``epoch`` starts: position 0.

        Args:
            epoch: The epoch just finished.

        Returns:
            ``(0, epoch + 1)``.
        """
        return 0, epoch + 1

    def valid_rows(self, position: ArrayLike) -> jax.Array:
        """Which rows of the batch starting at ``position`` are records of the epoch.

        Args:
            position: Where the batch starts; a Python int or a traced array.

        Returns:
            Bool array of shape ``(batch_size,)``: all ``True`` for a source without a
            length, otherwise ``True`` for the rows before the epoch ends.
        """
        if self.length is None:
            return jnp.ones((self.batch_size,), dtype=jnp.bool_)
        rows = jnp.asarray(position, dtype=jnp.int32) + jnp.arange(self.batch_size, dtype=jnp.int32)
        return rows < jnp.int32(self.length)
