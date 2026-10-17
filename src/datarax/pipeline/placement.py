"""Where the host stage places each unit: the default device, or a data mesh.

Without a sharding a unit goes to the default device, uncommitted. With a ``NamedSharding`` every
unit goes through substrax's ``place_batch_on_shards`` with the ``batch_ops.shardings`` prefix:

- a batch ``(B, ...)`` has its rows on the sharding and its batch-level state replicated;
- a chunk ``(K, B, ...)`` has its rows on the sharding's axes after the chunk axis;
- a bounded run's short final batch, whose rows the sharding cannot split, is placed replicated,
  so an evaluation covers every record once without padding (design D15, the final-batch rule).

A placer is told each unit's rows per batch before any process's share (``batch_rows``), so a
short final batch is told from a full one by its global size, never by the rows a process holds.

A sharding may span the devices of several processes (``jax.distributed``, one process per GPU).
Every process then names the same global batches; each reads the rows its own devices hold
(:func:`process_rows`, in global row order, as ``jax.make_array_from_process_local_data`` maps
them) and the placement assembles the global batch, which is the one-process batch row for row.
The short final batch is read whole by every process and placed replicated. Only ``INDEXED``
sources are read this way; a stream, and a source serving a worker's shard, are refused.

A ``batch_size`` the sharding cannot split is refused when the pipeline is built, through JAX's
own ``NamedSharding.shard_shape``: a full batch that does not divide is a wrong batch size, not a
final batch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import numpy as np
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec
from substrax.spmd import place_batch_on_shards

from datarax.core import batch_ops
from datarax.core.data_source import DataSourceModule, RecordIdentity, shard_identity
from datarax.core.element_batch import Batch


@dataclasses.dataclass(frozen=True, slots=True)
class Placer:
    """How a run's units are placed: a full unit, a batch or a chunk, and the short final batch.

    The host stage takes ``full`` for a unit whose batches hold ``batch_size`` rows before any
    process's share (``HostElement.batch_rows``) and ``final`` for the short final batch. Each is
    a plain callable, ``jax.device_put`` itself on the default device, so choosing one adds no
    call to a unit's placement.

    Attributes:
        batch_size: The pipeline's batch size.
        full: Places a unit of full batches.
        final: Places the short final batch.
    """

    batch_size: int
    full: Callable[[Batch], Batch]
    final: Callable[[Batch], Batch]


def check_sharding(
    sharding: NamedSharding | None, batch_size: int, source: DataSourceModule
) -> None:
    """Refuse a sharding the host stage cannot place this pipeline's batches on.

    Args:
        sharding: The rows' sharding, or ``None`` for the default device.
        batch_size: The pipeline's batch size.
        source: The pipeline's source.

    Raises:
        TypeError: If ``sharding`` is not a ``NamedSharding``.
        ValueError: If it names more than the rows, is on a mesh with Explicit axes, is over an
            ``AbstractMesh``, or cannot split ``batch_size`` rows; or, when it spans other
            processes' devices, if ``source`` is a stream or serves a worker's shard.
    """
    if sharding is None:
        return
    if not isinstance(sharding, NamedSharding):
        raise TypeError(
            f"sharding is a jax.sharding.NamedSharding over the batch's rows, got "
            f"{type(sharding).__name__}"
        )
    if len(sharding.spec) > 1:
        raise ValueError(
            f"sharding places a batch's rows: its spec has one entry, the row axis (one mesh axis "
            f"or a tuple of them), got {sharding.spec}"
        )
    if AxisType.Explicit in sharding.mesh.axis_types:
        raise ValueError(
            "sharding is on a mesh with Explicit axes, where an operator reading other records "
            "(MixUp, CutMix) must name the sharding of what it gathers, which datarax's operators "
            "do not: build the mesh with axis_types=(jax.sharding.AxisType.Auto, ...)"
        )
    if isinstance(sharding.mesh, AbstractMesh):
        raise ValueError(
            "sharding needs a mesh of devices to place batches on, got an AbstractMesh: build "
            "the mesh from devices, e.g. jax.make_mesh"
        )
    try:
        sharding.shard_shape((batch_size,))
    except ValueError as error:  # jax's IndivisibleError, a ValueError
        raise ValueError(
            f"batch_size {batch_size} does not split over {sharding.spec} on mesh "
            f"{dict(sharding.mesh.shape)}: use a multiple of the devices on the row axes"
        ) from error
    if spans_processes(sharding):
        _check_source_across_processes(source)


def _check_source_across_processes(source: DataSourceModule) -> None:
    """Refuse a source whose records a process cannot read by global position."""
    kind = source.record_identity
    if kind is not RecordIdentity.INDEXED:
        raise ValueError(
            f"{type(source).__name__} is a {kind.name} stream, read forward, and the sharding "
            "spans other processes' devices: each process reads its rows of every global batch "
            "by position, which only an INDEXED source has; read a stream on a mesh of this "
            "process's devices"
        )
    shard = shard_identity(source)
    if shard is not None:
        raise ValueError(
            f"{type(source).__name__} serves a worker's shard (shard_id={shard['index']}, "
            f"num_workers={shard['count']}) and the sharding spans other processes' devices, "
            "which share every global batch between the processes themselves: build the source "
            "over all its records"
        )


def spans_processes(sharding: NamedSharding | None) -> bool:
    """Whether a sharding places on devices of other processes as well as this one's.

    Args:
        sharding: The rows' sharding, or ``None``.

    Returns:
        ``True`` when some of its devices belong to another process.
    """
    return sharding is not None and not sharding.is_fully_addressable


def process_rows(sharding: NamedSharding | None, batch_size: int) -> np.ndarray | None:
    """The rows of a full batch this process's devices hold, in global row order.

    Args:
        sharding: The rows' sharding, or ``None``.
        batch_size: The pipeline's batch size.

    Returns:
        The rows, or ``None`` when the process holds every row (no sharding, or a mesh of this
        process's devices alone).
    """
    if sharding is None or sharding.is_fully_addressable:
        return None
    held = sharding.devices_indices_map((batch_size,))
    rows = {
        row
        for device in sharding.addressable_devices
        for row in range(*held[device][0].indices(batch_size))
    }
    return np.asarray(sorted(rows), dtype=np.int64)


def placement_platform(sharding: NamedSharding | None) -> str | None:
    """The platform a sharding places on, or ``None`` for the default device's.

    Args:
        sharding: The rows' sharding, or ``None``.

    Returns:
        The platform of the sharding's devices, or ``None``.
    """
    if sharding is None:
        return None
    return next(iter(sharding.device_set)).platform


def placer(sharding: NamedSharding | None, batch_size: int) -> Placer:
    """How each unit of a run is placed.

    Args:
        sharding: The rows' sharding, or ``None`` for the default device, uncommitted.
        batch_size: The pipeline's batch size: a unit whose batches hold fewer rows is the short
            final one.

    Returns:
        The placer.
    """
    if sharding is None:
        return Placer(batch_size, full=jax.device_put, final=jax.device_put)
    replicated = NamedSharding(sharding.mesh, PartitionSpec())
    chunk_rows = NamedSharding(sharding.mesh, PartitionSpec(None, *sharding.spec))

    def full(unit: Batch) -> Batch:
        rows = chunk_rows if unit.indices.ndim == 3 else sharding
        return place_batch_on_shards(unit, batch_ops.shardings(unit, rows, replicated))

    def final(unit: Batch) -> Batch:
        return place_batch_on_shards(unit, batch_ops.shardings(unit, replicated, replicated))

    return Placer(batch_size, full=full, final=final)


__all__ = [
    "Placer",
    "check_sharding",
    "placement_platform",
    "placer",
    "process_rows",
    "spans_processes",
]
