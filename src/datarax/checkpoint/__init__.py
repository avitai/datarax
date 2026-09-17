"""Checkpointing of Datarax pipelines, iterators and modules.

Any object that implements the :class:`~datarax.typing.Checkpointable` protocol
(``get_state`` / ``set_state``) is saved and restored through
:class:`IteratorCheckpoint`, which stores each state under an integer step with
substrax's Orbax-backed checkpoint store as its ``data_iterator`` item; a root written by
datarax 0.1.11 or earlier reads through :data:`ITERATOR_STATE_FORMAT2`.
"""

from datarax.checkpoint.iterators import (
    ITERATOR_STATE_FORMAT2,
    IteratorCheckpoint,
    validate_restore_compatibility,
)


__all__ = [
    "ITERATOR_STATE_FORMAT2",
    "IteratorCheckpoint",
    "validate_restore_compatibility",
]
