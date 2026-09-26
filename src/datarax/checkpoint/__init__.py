"""Checkpointing of Datarax pipelines, iterators and modules.

Any object that implements the :class:`~substrax.typing.Checkpointable` protocol
(``get_state`` / ``set_state``) is saved and restored through
:class:`IteratorCheckpoint`, which stores each state under an integer step with
substrax's Orbax-backed checkpoint store as its ``data_iterator`` item.
"""

from datarax.checkpoint.iterators import IteratorCheckpoint, validate_restore_compatibility


__all__ = [
    "IteratorCheckpoint",
    "validate_restore_compatibility",
]
