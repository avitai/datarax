"""Device detection utilities for Datarax tests."""

import os
from typing import Any

import jax
import jax.numpy as jnp


def get_available_devices() -> dict[str, list[Any]]:
    """Get available JAX devices by type.

    Returns:
        A dictionary mapping device types to lists of devices.
    """
    devices = jax.devices()
    device_types: dict[str, list[Any]] = {}

    for device in devices:
        device_kind = device.device_kind
        if device_kind not in device_types:
            device_types[device_kind] = []
        device_types[device_kind].append(device)

    return device_types


def get_device_count(device_type: str | None = None) -> int:
    """Get the number of available devices of a specific type.

    Args:
        device_type: The device type to count ('cpu', 'gpu', 'tpu').
                     If None, returns the total number of devices.

    Returns:
        The number of available devices.
    """
    if device_type is None:
        return len(jax.devices())

    return jax.local_device_count(device_type)


def has_gpu() -> bool:
    """Check if GPU is available.

    Returns:
        True if GPU is available, False otherwise.
    """
    return jax.local_device_count("gpu") > 0


def has_tpu() -> bool:
    """Check if TPU is available.

    Returns:
        True if TPU is available, False otherwise.
    """
    return jax.local_device_count("tpu") > 0


def has_multiple_devices() -> bool:
    """Check if multiple devices are available.

    Returns:
        True if multiple devices are available, False otherwise.
    """
    return len(jax.devices()) > 1


def is_distributed_env() -> bool:
    """Check if the current environment is distributed.

    Returns:
        True if the environment is distributed, False otherwise.
    """
    # Check for JAX distribution environment variables
    process_count = int(os.environ.get("JAX_PROCESS_COUNT", "1"))
    # process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))

    # Check if we're in a multi-process environment
    return process_count > 1


def get_device_info() -> dict[str, dict[str, int]]:
    """Get detailed device information.

    Returns:
        A dictionary with device information.
    """
    device_info: dict[str, dict[str, int]] = {
        "total": len(jax.devices()),
        "by_type": {
            "cpu": jax.local_device_count("cpu"),
            "gpu": jax.local_device_count("gpu"),
            "tpu": jax.local_device_count("tpu"),
        },
        "distributed": {
            "process_count": int(os.environ.get("JAX_PROCESS_COUNT", "1")),
            "process_id": int(os.environ.get("JAX_PROCESS_ID", "0")),
        },
    }

    return device_info


def create_device_mesh(
    mesh_shape: tuple[int, ...] | None = None, mesh_axes: tuple[str, ...] | None = None
) -> jax.sharding.Mesh:
    """Create a device mesh for testing.

    Args:
        mesh_shape: The shape of the mesh. If None, a default shape based on
                    available devices will be used.
        mesh_axes: The names of the mesh axes. If None, default names will be used.

    Returns:
        A device mesh.
    """
    devices = jax.devices()
    num_devices = len(devices)

    # Default mesh shape and axes if not provided
    if mesh_shape is None:
        if num_devices >= 4:
            mesh_shape = (2, 2)
        elif num_devices >= 2:
            mesh_shape = (2, 1)
        else:
            mesh_shape = (1, 1)

    if mesh_axes is None:
        mesh_axes = ("data", "model")

    # Ensure the product of mesh_shape matches the number of devices
    total_devices_in_mesh = 1
    for dim in mesh_shape:
        total_devices_in_mesh *= dim

    if total_devices_in_mesh > num_devices:
        raise ValueError(
            f"Mesh shape {mesh_shape} requires {total_devices_in_mesh} devices, "
            f"but only {num_devices} devices are available"
        )

    # Create the device array and reshape it to the desired mesh shape
    device_array: jax.Array = jnp.array(devices[:total_devices_in_mesh])
    device_array = jnp.reshape(device_array, mesh_shape)

    return jax.sharding.Mesh(device_array, mesh_axes)
