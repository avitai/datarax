"""The configuration a pipeline's saved state is valid for, and the refusal of any other.

``Pipeline.get_state()`` records, under ``"fingerprint"``, the configuration that produced it:
the batch rule, the length, the epochs, the key and the order (:func:`run_configuration`).
``set_state`` refuses a state of another configuration, naming the first field that differs,
and one holding none (:func:`refuse_another_configuration`). A checkpoint store may hand a state
back with NumPy leaves, read as plain values first (:func:`plain_state`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from datarax.core.prng import key_words
from datarax.pipeline.epochs import EpochPlan


SESSION_VERSION = 2
"""The layout of ``PipelineIterator.get_state()``, named when a pipeline refuses it."""


def run_configuration(pipeline: Any) -> dict[str, Any]:
    """The configuration a state is only valid for: batch rule, length, epochs, key and order.

    Args:
        pipeline: The pipeline whose state it is.

    Returns:
        The configuration, as plain values.
    """
    plan: EpochPlan = pipeline.epoch_plan
    return {
        "batch_size": plan.batch_size,
        "length": plan.length,
        "drop_last": plan.drop_last,
        "num_epochs": plan.num_epochs,
        "shuffled": bool(pipeline.shuffle),
        "seed": [int(word) for word in key_words(pipeline._epoch_key_base.get_value())],  # noqa: SLF001
        "order": {"kind": "global"},
    }


def refuse_another_configuration(pipeline: Any, saved: Any) -> None:
    """Refuse a state's configuration unless it is the pipeline's, naming the first field differing.

    Args:
        pipeline: The pipeline the state is restored into.
        saved: The state's configuration (its ``"fingerprint"``).

    Raises:
        ValueError: If the state holds no configuration, or a field differs from the pipeline's.
    """
    if not isinstance(saved, Mapping):
        raise ValueError(
            "the state holds no fingerprint of the configuration it was produced under, so it "
            "cannot be checked against this pipeline: save it again from this pipeline"
        )
    for field, value in run_configuration(pipeline).items():
        if saved.get(field) != value:
            raise ValueError(
                f"the state was produced with {field}={saved.get(field)!r} "
                f"but this pipeline has {field}={value!r}; a state is only valid for the "
                "configuration that produced it"
            )


def plain_state(value: Any) -> Any:
    """A saved state as plain Python values: a checkpoint store may return NumPy leaves.

    Args:
        value: The state, or any part of it.

    Returns:
        The same structure with mappings as dicts, sequences and arrays as lists and NumPy
        scalars as Python scalars.
    """
    if isinstance(value, Mapping):
        return {key: plain_state(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain_state(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain_state(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "SESSION_VERSION",
    "plain_state",
    "refuse_another_configuration",
    "run_configuration",
]
