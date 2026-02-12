"""Device mesh helpers for distributed benchmarking.

Design ref: Section 6.4.2 of the benchmark report.
"""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def create_device_mesh(
    mesh_shape: tuple[int, ...] | None = None,
    axis_names: tuple[str, ...] = ("data",),
) -> Mesh:
    """Create a JAX device mesh for benchmark sharding.

    Args:
        mesh_shape: Shape of the mesh. Defaults to (device_count,).
        axis_names: Names for each mesh axis.

    Returns:
        A JAX Mesh object.
    """
    devices = jax.devices()
    if mesh_shape is None:
        mesh_shape = (len(devices),)
    device_array = np.array(devices).reshape(mesh_shape)
    return Mesh(device_array, axis_names)


def shard_batch(mesh: Mesh, axis_name: str = "data") -> NamedSharding:
    """Create a NamedSharding that shards along the batch dimension.

    Args:
        mesh: The device mesh to shard across.
        axis_name: The mesh axis to shard the batch dim across.

    Returns:
        A NamedSharding for batch-parallel distribution.
    """
    return NamedSharding(mesh, PartitionSpec(axis_name))
