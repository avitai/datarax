"""Shared iteration helpers for sampler modules."""

from collections.abc import Callable
from dataclasses import dataclass


def require_record_count(num_records: int | None) -> int:
    """Return a validated record count."""
    if num_records is None:
        raise ValueError("num_records must be set")
    return num_records


def has_reached_epoch_limit(current_epoch: int, num_epochs: int) -> bool:
    """Return whether iteration has reached a finite epoch limit."""
    return num_epochs != -1 and current_epoch >= num_epochs


def total_epoch_length(num_records: int | None, num_epochs: int) -> int:
    """Return total sample count for finite epoch iteration."""
    if num_epochs == -1:
        raise ValueError("Cannot determine length for infinite epochs")
    return require_record_count(num_records) * num_epochs


@dataclass(frozen=True)
class EpochStep:
    """Normalized sampler position after checking epoch boundaries."""

    current_epoch: int
    current_index: int
    started_new_epoch: bool
    exhausted: bool


def normalize_epoch_step(
    *,
    current_epoch: int,
    num_epochs: int,
    current_index: int,
    num_records: int,
) -> EpochStep:
    """Return the next valid epoch step without mutating sampler state."""
    if has_reached_epoch_limit(current_epoch, num_epochs):
        raise StopIteration

    if current_index < num_records:
        return EpochStep(
            current_epoch=current_epoch,
            current_index=current_index,
            started_new_epoch=False,
            exhausted=False,
        )

    current_epoch += 1
    return EpochStep(
        current_epoch=current_epoch,
        current_index=0,
        started_new_epoch=True,
        exhausted=has_reached_epoch_limit(current_epoch, num_epochs),
    )


def read_epoch_step(
    *,
    current_epoch: Callable[[], int],
    num_epochs: Callable[[], int],
    current_index: Callable[[], int],
    num_records: Callable[[], int | None],
) -> EpochStep:
    """Read sampler state variables and normalize the epoch position."""
    return normalize_epoch_step(
        current_epoch=current_epoch(),
        num_epochs=num_epochs(),
        current_index=current_index(),
        num_records=require_record_count(num_records()),
    )


def consume_epoch_step_index(
    *,
    epoch_step: EpochStep,
    current_epoch: Callable[[int], None],
    current_index: Callable[[int], None],
) -> int:
    """Apply a normalized epoch step and return the consumed index."""
    if epoch_step.started_new_epoch:
        current_epoch(epoch_step.current_epoch)
        current_index(epoch_step.current_index)
        if epoch_step.exhausted:
            raise StopIteration

    idx = epoch_step.current_index
    current_index(epoch_step.current_index + 1)
    return idx
