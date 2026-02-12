"""Tests for device mesh helpers used in distributed benchmarking.

Validates mesh creation and batch sharding on the available JAX devices
(CPU by default in CI).
"""

from __future__ import annotations

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from benchmarks.core.sharding import create_device_mesh, shard_batch


class TestCreateDeviceMesh:
    """Tests for create_device_mesh()."""

    def test_default_mesh_uses_all_devices(self):
        """Default mesh has shape (num_devices,) with axis 'data'."""
        mesh = create_device_mesh()
        assert isinstance(mesh, Mesh)
        num_devices = jax.device_count()
        assert mesh.devices.shape == (num_devices,)
        assert mesh.axis_names == ("data",)

    def test_custom_shape_1d(self):
        """A 1D mesh with explicit shape works."""
        num_devices = jax.device_count()
        mesh = create_device_mesh(mesh_shape=(num_devices,))
        assert mesh.devices.shape == (num_devices,)

    def test_custom_axis_names(self):
        """Custom axis names are applied to the mesh."""
        num_devices = jax.device_count()
        mesh = create_device_mesh(
            mesh_shape=(num_devices,),
            axis_names=("batch",),
        )
        assert mesh.axis_names == ("batch",)

    def test_2d_mesh(self):
        """A 2D mesh works when device count allows it."""
        num_devices = jax.device_count()
        if num_devices < 2:
            # Can't create a 2D mesh with 1 device, test (1, 1) shape
            mesh = create_device_mesh(
                mesh_shape=(1, 1),
                axis_names=("data", "model"),
            )
            assert mesh.devices.shape == (1, 1)
        else:
            mesh = create_device_mesh(
                mesh_shape=(num_devices // 2, 2),
                axis_names=("data", "model"),
            )
            assert mesh.devices.shape == (num_devices // 2, 2)
            assert mesh.axis_names == ("data", "model")


class TestShardBatch:
    """Tests for shard_batch()."""

    def test_returns_named_sharding(self):
        """shard_batch returns a NamedSharding instance."""
        mesh = create_device_mesh()
        sharding = shard_batch(mesh)
        assert isinstance(sharding, NamedSharding)

    def test_sharding_spec_partitions_first_axis(self):
        """The partition spec shards along the first (batch) dimension."""
        mesh = create_device_mesh()
        sharding = shard_batch(mesh, axis_name="data")
        assert sharding.spec == PartitionSpec("data")

    def test_custom_axis_name(self):
        """Custom axis name is used in the partition spec."""
        num_devices = jax.device_count()
        mesh = create_device_mesh(
            mesh_shape=(num_devices,),
            axis_names=("batch",),
        )
        sharding = shard_batch(mesh, axis_name="batch")
        assert sharding.spec == PartitionSpec("batch")
