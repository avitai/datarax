"""Tests for the NNX-based distributed training components."""

import unittest
from unittest import mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.distributed.data_parallel import DataParallel
from datarax.distributed.device_mesh import DeviceMeshManager
from datarax.distributed.metrics import DistributedMetrics


class TestDeviceMeshManager(unittest.TestCase):
    """Tests for the DeviceMeshManager class."""

    def test_init(self):
        """Test initializing a DeviceMeshManager."""
        mesh_manager = DeviceMeshManager()
        self.assertIsInstance(mesh_manager, DeviceMeshManager)

    def test_create_device_mesh_single_device_dict(self):
        """Test creating a single-device mesh with dict specification."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_device_mesh({"data": 1})
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("data",))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_device_mesh_single_device_list(self):
        """Test creating a single-device mesh with list specification."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_device_mesh([("batch", 1)])
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("batch",))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_device_mesh_explicit_devices(self):
        """Test creating mesh with explicitly provided devices."""
        mesh_manager = DeviceMeshManager()
        devices = jax.devices()
        mesh = mesh_manager.create_device_mesh([("data", 1)], devices=devices)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.devices[0], devices[0])

    def test_create_device_mesh_insufficient_devices_error(self):
        """Test that creating mesh with too few devices raises ValueError."""
        mesh_manager = DeviceMeshManager()
        # Request more devices than available
        with self.assertRaises(ValueError) as ctx:
            mesh_manager.create_device_mesh([("data", 100)])
        self.assertIn("Not enough devices", str(ctx.exception))
        self.assertIn("100 devices", str(ctx.exception))

    def test_create_data_parallel_mesh_default(self):
        """Test creating data-parallel mesh with all available devices."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_data_parallel_mesh()
        available_devices = len(jax.devices())
        self.assertEqual(mesh.devices.shape, (available_devices,))
        self.assertEqual(mesh.axis_names, ("data",))

    def test_create_data_parallel_mesh_single_device(self):
        """Test creating data-parallel mesh with single device."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_data_parallel_mesh(num_devices=1)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("data",))

    def test_create_model_parallel_mesh_single_device(self):
        """Test creating model-parallel mesh with single device."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_model_parallel_mesh(num_devices=1)
        self.assertEqual(mesh.devices.shape, (1,))
        self.assertEqual(mesh.axis_names, ("model",))

    def test_create_model_parallel_mesh_insufficient_devices_error(self):
        """Test that model-parallel mesh with too many devices raises ValueError."""
        mesh_manager = DeviceMeshManager()
        with self.assertRaises(ValueError) as ctx:
            mesh_manager.create_model_parallel_mesh(num_devices=100)
        self.assertIn("Not enough devices", str(ctx.exception))
        self.assertIn("100 devices", str(ctx.exception))

    def test_create_hybrid_mesh_single_device(self):
        """Test creating hybrid mesh with 1x1 configuration."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        self.assertEqual(mesh.devices.shape, (1, 1))
        self.assertEqual(mesh.axis_names, ("data", "model"))
        self.assertEqual(mesh.devices.size, 1)

    def test_create_hybrid_mesh_insufficient_devices_error(self):
        """Test that hybrid mesh with too many devices raises ValueError."""
        mesh_manager = DeviceMeshManager()
        with self.assertRaises(ValueError) as ctx:
            mesh_manager.create_hybrid_mesh(data_parallel_size=10, model_parallel_size=10)
        self.assertIn("Not enough devices", str(ctx.exception))
        self.assertIn("100 devices", str(ctx.exception))

    def test_get_mesh_info_single_device(self):
        """Test getting mesh information for single-device mesh."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_device_mesh([("data", 1)])
        info = mesh_manager.get_mesh_info(mesh)

        self.assertEqual(info["total_devices"], 1)
        axes = info["axes"]
        assert isinstance(axes, dict)
        self.assertEqual(axes["data"], 1)

    def test_get_mesh_info_hybrid_mesh(self):
        """Test getting mesh information for hybrid mesh."""
        mesh_manager = DeviceMeshManager()
        mesh = mesh_manager.create_hybrid_mesh(data_parallel_size=1, model_parallel_size=1)
        info = mesh_manager.get_mesh_info(mesh)

        self.assertEqual(info["total_devices"], 1)
        axes = info["axes"]
        assert isinstance(axes, dict)
        self.assertEqual(axes["data"], 1)
        self.assertEqual(axes["model"], 1)

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_create_device_mesh(self):
        """Test creating a device mesh."""
        mesh_module = DeviceMeshManager()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        self.assertEqual(mesh.devices.shape, (2,))
        self.assertEqual(mesh.axis_names, ("data",))

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_create_data_parallel_mesh(self):
        """Test creating a data-parallel mesh."""
        mesh_module = DeviceMeshManager()
        mesh = mesh_module.create_data_parallel_mesh(num_devices=2)
        self.assertEqual(mesh.devices.shape, (2,))
        self.assertEqual(mesh.axis_names, ("data",))

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_get_mesh_info(self):
        """Test getting mesh information."""
        mesh_module = DeviceMeshManager()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        info = mesh_module.get_mesh_info(mesh)
        self.assertEqual(info["total_devices"], 2)
        axes = info["axes"]
        assert isinstance(axes, dict)
        self.assertEqual(axes["data"], 2)


class TestDataParallel(unittest.TestCase):
    """Tests for the DataParallel class."""

    def test_init(self):
        """Test initializing a DataParallel."""
        dp_module = DataParallel()
        self.assertIsInstance(dp_module, nnx.Module)

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_create_data_parallel_sharding(self):
        """Test creating data-parallel sharding."""
        mesh_module = DeviceMeshManager()
        dp_module = DataParallel()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        sharding = dp_module.create_data_parallel_sharding(mesh)
        self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("data"))

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_shard_batch(self):
        """Test sharding a batch."""
        mesh_module = DeviceMeshManager()
        dp_module = DataParallel()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        sharding = dp_module.create_data_parallel_sharding(mesh)

        # Create a dummy batch
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        # Shard the batch
        sharded_batch = dp_module.shard_batch(batch, sharding)

        # Check that the batch was sharded
        self.assertEqual(sharded_batch["inputs"].shape, (4, 2))
        self.assertEqual(sharded_batch["targets"].shape, (4,))

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_create_data_parallel_sharding_static(self):
        """Test creating data-parallel sharding using static method."""
        mesh_module = DeviceMeshManager()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        sharding = DataParallel.create_data_parallel_sharding_static(mesh)
        self.assertEqual(sharding.spec, jax.sharding.PartitionSpec("data"))

    @pytest.mark.skipif(jax.device_count() < 2, reason="Test requires at least 2 devices")
    def test_shard_batch_static(self):
        """Test sharding a batch using static method."""
        mesh_module = DeviceMeshManager()
        mesh = mesh_module.create_device_mesh([("data", 2)])
        sharding = DataParallel.create_data_parallel_sharding_static(mesh)

        # Create a dummy batch
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        # Shard the batch using static method
        sharded_batch = DataParallel.shard_batch_static(batch, sharding)

        # Check that the batch was sharded
        self.assertEqual(sharded_batch["inputs"].shape, (4, 2))
        self.assertEqual(sharded_batch["targets"].shape, (4,))

    def test_all_reduce_gradients_static(self):
        """Test all-reduce gradients using static method."""
        # Mock the lax.pmean function
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            gradients = jnp.array(4.0)
            reduced = DataParallel.all_reduce_gradients_static(gradients, "mean")
            self.assertEqual(reduced, 2.0)

        # Mock the lax.psum function
        with mock.patch("jax.lax.psum", return_value=jnp.array(8.0)):
            gradients = jnp.array(4.0)
            reduced = DataParallel.all_reduce_gradients_static(gradients, "sum")
            self.assertEqual(reduced, 8.0)

    def test_all_reduce_gradients_instance(self):
        """Test all-reduce gradients using instance method."""
        dp_module = DataParallel()

        # Mock the lax.pmean function
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            gradients = jnp.array(4.0)
            reduced = dp_module.all_reduce_gradients(gradients, "mean")
            self.assertEqual(reduced, 2.0)

        # Mock the lax.psum function
        with mock.patch("jax.lax.psum", return_value=jnp.array(8.0)):
            gradients = jnp.array(4.0)
            reduced = dp_module.all_reduce_gradients(gradients, "sum")
            self.assertEqual(reduced, 8.0)


class TestDistributedMetrics(unittest.TestCase):
    """Tests for the DistributedMetrics class."""

    def test_init(self):
        """Test initializing a DistributedMetrics."""
        metrics_module = DistributedMetrics()
        self.assertIsInstance(metrics_module, nnx.Module)

    def test_reduce_mean(self):
        """Test reducing metrics using mean."""
        # Mock the lax.pmean function
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics_module = DistributedMetrics()
            metrics = {"loss": jnp.array(3.0), "accuracy": 0.5}
            reduced = metrics_module.reduce_mean(metrics)
            self.assertEqual(reduced["loss"], 2.0)
            self.assertEqual(reduced["accuracy"], 0.5)  # Unchanged

    def test_reduce_sum(self):
        """Test reducing metrics using sum."""
        # Mock the lax.psum function
        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics_module = DistributedMetrics()
            metrics = {"loss": jnp.array(3.0), "accuracy": 0.5}
            reduced = metrics_module.reduce_sum(metrics)
            self.assertEqual(reduced["loss"], 6.0)
            self.assertEqual(reduced["accuracy"], 0.5)  # Unchanged

    def test_reduce_custom(self):
        """Test reducing metrics using custom reductions."""
        # Mock the various lax reduction functions
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
                with mock.patch("jax.lax.pmax", return_value=jnp.array(4.0)):
                    metrics_module = DistributedMetrics()
                    metrics = {
                        "loss": jnp.array(3.0),
                        "accuracy": jnp.array(0.5),
                        "step": jnp.array(10),
                    }

                    # Use custom reduction operations
                    reduced = metrics_module.reduce_custom(
                        metrics,
                        reduce_fn={
                            "loss": "mean",
                            "accuracy": "sum",
                            "step": "max",
                        },
                    )

                    self.assertEqual(reduced["loss"], 2.0)
                    self.assertEqual(reduced["accuracy"], 6.0)
                    self.assertEqual(reduced["step"], 4.0)

    def test_reduce_mean_static(self):
        """Test reducing metrics using static mean method."""
        # Mock the lax.pmean function
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0), "accuracy": 0.5}
            reduced = DistributedMetrics.reduce_mean_static(metrics)
            self.assertEqual(reduced["loss"], 2.0)
            self.assertEqual(reduced["accuracy"], 0.5)  # Unchanged

    def test_reduce_sum_static(self):
        """Test reducing metrics using static sum method."""
        # Mock the lax.psum function
        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"loss": jnp.array(3.0), "accuracy": 0.5}
            reduced = DistributedMetrics.reduce_sum_static(metrics)
            self.assertEqual(reduced["loss"], 6.0)
            self.assertEqual(reduced["accuracy"], 0.5)  # Unchanged

    def test_reduce_custom_static(self):
        """Test reducing metrics using static custom reductions."""
        # Mock the various lax reduction functions
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
                with mock.patch("jax.lax.pmax", return_value=jnp.array(4.0)):
                    metrics = {
                        "loss": jnp.array(3.0),
                        "accuracy": jnp.array(0.5),
                        "step": jnp.array(10),
                    }

                    # Use custom reduction operations with static method
                    reduced = DistributedMetrics.reduce_custom_static(
                        metrics,
                        reduce_fn={
                            "loss": "mean",
                            "accuracy": "sum",
                            "step": "max",
                        },
                    )

                    self.assertEqual(reduced["loss"], 2.0)
                    self.assertEqual(reduced["accuracy"], 6.0)
                    self.assertEqual(reduced["step"], 4.0)

    def test_all_gather_static(self):
        """Test gathering metrics using static method."""
        # Mock the lax.all_gather function
        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0), "accuracy": 0.5}
            gathered = DistributedMetrics.all_gather_static(metrics)
            self.assertEqual(gathered["loss"].tolist(), [1.0, 2.0])
            self.assertEqual(gathered["accuracy"], 0.5)  # Unchanged

    def test_collect_from_devices_static(self):
        """Test collecting metrics from devices using static method."""
        metrics = {
            "loss": jnp.array([1.0, 2.0, 3.0]),  # Multi-device array
            "accuracy": 0.95,  # Scalar value
        }
        collected = DistributedMetrics.collect_from_devices_static(metrics)

        # Should collect device values for arrays
        self.assertEqual(len(collected["loss"]), 3)
        self.assertEqual(collected["loss"][0], 1.0)
        self.assertEqual(collected["loss"][1], 2.0)
        self.assertEqual(collected["loss"][2], 3.0)

        # Should keep scalar values unchanged
        self.assertEqual(collected["accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
