"""Where the host stage places each unit: the default device, or a data mesh.

Without a sharding a unit goes to the default device, uncommitted. With a ``NamedSharding`` every
unit goes through substrax's ``place_batch_on_shards`` with the ``batch_ops.shardings`` prefix:

- a batch ``(B, ...)`` has its rows on the sharding and its batch-level state replicated;
- a chunk ``(K, B, ...)`` has its rows on the sharding's axes after the chunk axis;
- a bounded run's short final batch, whose rows the sharding cannot split, is placed replicated,
  so an evaluation covers every record once without padding (design D15, the final-batch rule).

A ``batch_size`` the sharding cannot split is refused when the pipeline is built, through JAX's
own ``NamedSharding.shard_shape``: a full batch that does not divide is a wrong batch size, not a
final batch.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec
from substrax.spmd import place_batch_on_shards

from datarax.core import batch_ops
from datarax.core.element_batch import Batch


type Placer = Callable[[Batch], Batch]
"""Places one unit, a batch or a chunk, read on the host."""


def check_sharding(sharding: NamedSharding | None, batch_size: int) -> None:
    """Refuse a sharding the host stage cannot place this pipeline's batches on.

    Args:
        sharding: The rows' sharding, or ``None`` for the default device.
        batch_size: The pipeline's batch size.

    Raises:
        TypeError: If ``sharding`` is not a ``NamedSharding``.
        ValueError: If it names more than the rows, is on a mesh with Explicit axes, is over an
            ``AbstractMesh``, spans devices another process holds, or cannot split
            ``batch_size`` rows.
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
    if not sharding.is_fully_addressable:
        raise ValueError(
            "sharding spans devices of other processes; placing each process's slice of a global "
            "batch needs a per-process order, which datarax does not provide yet: use a mesh of "
            "this process's devices"
        )
    try:
        sharding.shard_shape((batch_size,))
    except ValueError as error:  # jax's IndivisibleError, a ValueError
        raise ValueError(
            f"batch_size {batch_size} does not split over {sharding.spec} on mesh "
            f"{dict(sharding.mesh.shape)}: use a multiple of the devices on the row axes"
        ) from error


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
    """The function placing each unit of a run.

    Args:
        sharding: The rows' sharding, or ``None`` for the default device, uncommitted.
        batch_size: The pipeline's batch size: a batch with fewer rows is the short final one.

    Returns:
        A function placing one unit.
    """
    if sharding is None:
        return jax.device_put
    replicated = NamedSharding(sharding.mesh, PartitionSpec())
    chunk_rows = NamedSharding(sharding.mesh, PartitionSpec(None, *sharding.spec))

    def place(unit: Batch) -> Batch:
        if unit.indices.ndim == 3:
            rows = chunk_rows
        elif unit.indices.shape[0] == batch_size:
            rows = sharding
        else:
            rows = replicated
        return place_batch_on_shards(unit, batch_ops.shardings(unit, rows, replicated))

    return place


__all__ = ["Placer", "check_sharding", "placement_platform", "placer"]
