"""Tests for mesh.py - Device mesh management for JAX distributed training."""

import unittest

import jax

from datarax.distributed.device_mesh import DeviceMeshManager


class TestDeviceMeshManagerStaticMethods(unittest.TestCase):
    """Tests for the DeviceMeshManager class with static methods."""

    def test_create_device_mesh_dict_specification(self):
        """Test creating mesh with dict specification."""
        mesh = DeviceMeshManager.create_device_mesh({"data": 1})
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("data",))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_device_mesh_list_specification(self):
        """Test creating mesh with list of tuples specification."""
        mesh = DeviceMeshManager.create_device_mesh([("batch", 1)])
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("batch",))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_device_mesh_explicit_devices(self):
        """Test creating mesh with explicitly provided devices."""
        devices = jax.devices()
        mesh = DeviceMeshManager.create_device_mesh([("data", 1)], devices=devices)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.devices[0], devices[0])

    def test_create_device_mesh_multiple_axes(self):
        """Test creating mesh with multiple axes."""
        mesh = DeviceMeshManager.create_device_mesh([("data", 1), ("model", 1)])
        self.assertEqual(mesh.devices.shape, (1, 1))
        self.assertEqual(mesh.axis_names, ("data", "model"))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_device_mesh_insufficient_devices_error(self):
        """Test that creating mesh with too few devices raises ValueError."""
        with self.assertRaises(ValueError):
            DeviceMeshManager.create_device_mesh([("data", 100)])

    def test_create_data_parallel_mesh_default(self):
        """Test creating data-parallel mesh with all available devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh()
        available_devices = len(jax.devices())
        self.assertEqual(mesh.devices.shape, (available_devices,))
        self.assertEqual(mesh.axis_names, ("data",))

    def test_create_data_parallel_mesh_single_device(self):
        """Test creating data-parallel mesh with single device."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("data",))

    def test_create_data_parallel_mesh_subset_devices(self):
        """Test creating data-parallel mesh with subset of available devices."""
        available_devices = len(jax.devices())
        if available_devices > 1:
            mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
            self.assertEqual(mesh.devices.shape, (1,))
        else:
            # Single device system - just test with 1
            mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
            self.assertEqual(mesh.devices.shape, (1,))

    def test_create_model_parallel_mesh_single_device(self):
        """Test creating model-parallel mesh with single device."""
        mesh = DeviceMeshManager.create_model_parallel_mesh(num_devices=1)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("model",))

    def test_create_model_parallel_mesh_insufficient_devices_error(self):
        """Test that model-parallel mesh with too many devices raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            DeviceMeshManager.create_model_parallel_mesh(num_devices=100)
        self.assertIn("Not enough devices", str(ctx.exception))
        self.assertIn("100", str(ctx.exception))

    def test_create_hybrid_mesh_single_device(self):
        """Test creating hybrid mesh with 1x1 configuration."""
        mesh = DeviceMeshManager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        self.assertEqual(mesh.devices.shape, (1, 1))
        self.assertEqual(mesh.axis_names, ("data", "model"))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_hybrid_mesh_insufficient_devices_error(self):
        """Test that hybrid mesh with too many devices raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            DeviceMeshManager.create_hybrid_mesh(data_parallel_size=10, model_parallel_size=10)
        self.assertIn("Not enough devices", str(ctx.exception))
        self.assertIn("100", str(ctx.exception))

    def test_get_mesh_info_single_axis(self):
        """Test getting info from a single-axis mesh."""
        mesh = DeviceMeshManager.create_device_mesh([("data", 1)])
        info = DeviceMeshManager.get_mesh_info(mesh)

        self.assertEqual(info["total_devices"], 1)
        self.assertIn("axes", info)
        axes = info["axes"]
        assert isinstance(axes, dict)  # Type narrowing for pyright
        self.assertEqual(axes["data"], 1)

    def test_get_mesh_info_multiple_axes(self):
        """Test getting info from a multi-axis mesh."""
        mesh = DeviceMeshManager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        info = DeviceMeshManager.get_mesh_info(mesh)

        self.assertEqual(info["total_devices"], 1)
        self.assertIn("axes", info)
        axes = info["axes"]
        assert isinstance(axes, dict)  # Type narrowing for pyright
        self.assertEqual(axes["data"], 1)
        self.assertEqual(axes["model"], 1)
        self.assertEqual(len(axes), 2)


if __name__ == "__main__":
    unittest.main()
