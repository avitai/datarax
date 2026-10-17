"""The configuration a pipeline's saved state is valid for, and the refusal of any other.

``Pipeline.get_state()`` records, under ``"fingerprint"``, the configuration that produced it:
the batch rule, the length, the epochs, the key and the order (:func:`run_configuration`).
``set_state`` refuses a state of another configuration, naming the first field that differs,
and one holding none (:func:`refuse_another_configuration`). A checkpoint store may hand a state
back with NumPy leaves, read as plain values first (:func:`plain_state`). Under a mesh spanning
processes every process runs one configuration from one cursor, and a run whose configuration
differs between them is refused in every process (:func:`refuse_differing_processes`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
from jax.experimental import multihost_utils

from datarax.core.data_source import shard_identity
from datarax.core.prng import key_words
from datarax.pipeline.epochs import EpochPlan


SESSION_VERSION = 2
"""The layout of ``PipelineIterator.get_state()``, named when a pipeline refuses it."""


def run_configuration(pipeline: Any) -> dict[str, Any]:
    """The configuration a state is only valid for: batch rule, length, epochs, key, order, shard.

    ``shard`` is :func:`~datarax.core.data_source.shard_identity`: a worker's shard serves other
    records at the same position, so a state saved on one shard is refused by every other.

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
        "order": _order(plan),
        "shard": shard_identity(pipeline.source),
    }


def _order(plan: EpochPlan) -> dict[str, Any]:
    """The order a plan names its epochs in, as plain values: global, or a block order."""
    if plan.block_order is None:
        return {"kind": "global"}
    return {
        "kind": "block",
        "block_size": plan.block_order.block_size,
        "blocks_per_buffer": plan.block_order.blocks_per_buffer,
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


def refuse_differing_processes(
    pipeline: Any, *, chunk: int | None, epoch: int, position: int
) -> None:
    """Refuse a run whose configuration differs between the processes of a mesh spanning them.

    Every process names the same global batches and reads its rows of each, so every process
    runs one configuration (:func:`run_configuration`, the batches per unit and the mesh) from
    one cursor; a process running another would place rows of other records into the same
    global batch. Each process sends a digest of its run to one all-gather
    (``multihost_utils.assert_equal``) and compares all of them, so each refuses alike. Called
    where a run opens, which every process does at the same call.

    Args:
        pipeline: The pipeline whose run opens, with a sharding spanning processes.
        chunk: Batches per chunk, or ``None`` for single batches.
        epoch: The cursor's epoch.
        position: The cursor's position in it.

    Raises:
        ValueError: If any process's run differs from this one's.
    """
    sharding = pipeline.sharding
    run = {
        "configuration": run_configuration(pipeline),
        "chunk": chunk,
        "epoch": epoch,
        "position": position,
        "spec": str(sharding.spec),
        "mesh": dict(sharding.mesh.shape),
        "devices": sharding.mesh.device_ids.ravel().tolist(),
    }
    digest = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).digest()
    try:
        multihost_utils.assert_equal(np.frombuffer(digest, np.uint8), "run digests")
    except AssertionError as error:
        raise ValueError(
            "the run's configuration differs between processes (batch rule, length, epochs, "
            "key, order, chunk, cursor or mesh): every process of a mesh spanning processes "
            "reads its rows of the same global batches, so every process builds the same "
            f"pipeline; this process runs {run}"
        ) from error


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
    "refuse_differing_processes",
    "run_configuration",
]
