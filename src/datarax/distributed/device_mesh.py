"""Device mesh management for JAX distributed training.

This module provides utilities for creating and managing JAX device meshes
for coordinating distributed computations across multiple devices.
"""

import numpy as np
from typing import Any

import jax
from jax.sharding import Mesh


class DeviceMeshManager:
    """Manager for creating and configuring JAX device meshes.

    This class provides utilities for creating device meshes for different
    distributed training configurations, including data-parallel, model-parallel,
    and hybrid approaches.
    """

    def create_device_mesh(
        self,
        mesh_shape: dict[str, int] | list[tuple[str, int]],
        devices: list[Any] | None = None,
    ) -> Mesh:
        """Create a JAX device mesh with the specified shape.

        Args:
            mesh_shape: The shape of the mesh, specified either as a dictionary
                mapping axis names to sizes, or as a list of (name, size) tuples.
            devices: Optional list of devices to use. If None, uses all available
                devices.

        Returns:
            A JAX device mesh.

        Raises:
            ValueError: If the mesh shape is incompatible with the number of devices.
        """
        # Convert mesh_shape to a list of (name, size) tuples if it's a dict
        if isinstance(mesh_shape, dict):
            mesh_shape_list = [(name, size) for name, size in mesh_shape.items()]
        else:
            mesh_shape_list = mesh_shape

        # Get devices if not provided
        if devices is None:
            devices = jax.devices()

        # Calculate the total number of devices needed
        total_devices = 1
        for _, size in mesh_shape_list:
            total_devices *= size

        # Verify we have enough devices
        if len(devices) < total_devices:
            raise ValueError(
                f"Not enough devices. Mesh requires {total_devices} devices, "
                f"but only {len(devices)} are available."
            )

        # Create the mesh
        mesh_dims = tuple(size for _, size in mesh_shape_list)
        mesh_devices = devices[:total_devices]

        # Reshape devices list to match mesh dimensions
        mesh_devices_array = np.array(mesh_devices).reshape(mesh_dims)

        # Create and return the Mesh object
        axis_names = tuple(name for name, _ in mesh_shape_list)
        return jax.sharding.Mesh(mesh_devices_array, axis_names=axis_names)

    def create_data_parallel_mesh(self, num_devices: int | None = None) -> Mesh:
        """Create a data-parallel device mesh.

        Args:
            num_devices: Optional number of devices to use. If None, uses all
                available devices.

        Returns:
            A JAX device mesh configured for data-parallel training.
        """
        devices = jax.devices()
        if num_devices is not None:
            devices = devices[:num_devices]

        return self.create_device_mesh([("data", len(devices))], devices)

    def create_model_parallel_mesh(self, num_devices: int) -> Mesh:
        """Create a model-parallel device mesh.

        Args:
            num_devices: Number of devices to use for model parallelism.

        Returns:
            A JAX device mesh configured for model-parallel training.
        """
        devices = jax.devices()
        if len(devices) < num_devices:
            raise ValueError(
                f"Not enough devices. Model parallelism requires {num_devices} "
                f"devices, but only {len(devices)} are available."
            )

        return self.create_device_mesh([("model", num_devices)], devices[:num_devices])

    def create_hybrid_mesh(self, data_parallel_size: int, model_parallel_size: int) -> Mesh:
        """Create a hybrid data-parallel and model-parallel device mesh.

        Args:
            data_parallel_size: Number of devices to use for data parallelism.
            model_parallel_size: Number of devices to use for model parallelism.

        Returns:
            A JAX device mesh configured for hybrid parallel training.

        Raises:
            ValueError: If there aren't enough devices available.
        """
        total_devices = data_parallel_size * model_parallel_size
        devices = jax.devices()

        if len(devices) < total_devices:
            raise ValueError(
                f"Not enough devices. Hybrid parallelism requires {total_devices} "
                f"devices, but only {len(devices)} are available."
            )

        mesh_shape = [
            ("data", data_parallel_size),
            ("model", model_parallel_size),
        ]
        return self.create_device_mesh(mesh_shape, devices[:total_devices])

    def get_mesh_info(self, mesh: Mesh) -> dict[str, int | dict[str, int]]:
        """Get information about a device mesh.

        Args:
            mesh: The device mesh to inspect.

        Returns:
            A dictionary containing information about the mesh, including the
            number of devices and the size of each axis.
        """
        info: dict[str, int | dict[str, int]] = {
            "total_devices": mesh.devices.size,
            "axes": {},
        }

        axes_info = info["axes"]
        if isinstance(axes_info, dict):
            for i, axis_name in enumerate(mesh.axis_names):
                axes_info[axis_name] = mesh.devices.shape[i]

        return info
