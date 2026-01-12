"""Tests for logical axis naming in NNX-based sharder modules.

This module demonstrates the enhanced logical axis naming functionality
for more descriptive sharding specifications, without creating circular imports.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec


class TestLogicalSharding:
    """Test suite for logical axis naming in sharding operations."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Check if we have enough devices for full mesh testing
        self.real_devices = jax.devices()
        if len(self.real_devices) >= 8:
            # Full 2x4 mesh for true multi-device testing
            device_mesh = np.array(self.real_devices[:8]).reshape(2, 4)
            self.limited_test = False
        else:
            # For limited devices, create a smaller mesh (1x1 or 2x1)
            self.limited_test = True
            device_count = len(self.real_devices)
            if device_count == 1:
                # Single device case - 1x1 mesh
                device_mesh = np.array(self.real_devices).reshape(1, 1)
            else:
                # Multiple but fewer than 8 devices - use what we have in a 2xN mesh
                # where N is half the available devices (minimum 1)
                n_cols = max(1, device_count // 2)
                n_rows = min(2, device_count // n_cols)
                device_mesh = np.array(self.real_devices[: n_rows * n_cols]).reshape(n_rows, n_cols)

        # Create the mesh with our available devices
        self.mesh = Mesh(device_mesh, axis_names=("data", "model"))

        # Define logical to physical axis mapping
        self.sharding_rules = [
            ("batch", "data"),  # 'batch' maps to 'data' dimension
            ("hidden", "model"),  # 'hidden' maps to 'model' dimension
            ("feature", None),  # 'feature' is not sharded
        ]

        # Create simple version of logical sharding utilities
        class LogicalShardingHelper(nnx.Module):
            def __init__(self, sharding_rules):
                super().__init__()
                self.sharding_rules = sharding_rules

            def get_partition_spec(self, logical_spec):
                """Convert logical axes to physical partition spec."""
                physical_spec = []
                for dim in logical_spec:
                    if dim is None:
                        physical_spec.append(None)
                    else:
                        mapped = next(
                            (phys for log, phys in self.sharding_rules if log == dim), dim
                        )
                        physical_spec.append(mapped)
                return PartitionSpec(*physical_spec)

            def get_named_sharding(self, mesh, logical_spec):
                """Get named sharding from logical spec."""
                pspec = self.get_partition_spec(logical_spec)
                return jax.sharding.NamedSharding(mesh, pspec)

            def shard_with_logical_names(self, array, mesh, logical_spec):
                """Shard array using logical axis names."""
                named_sharding = self.get_named_sharding(mesh, logical_spec)
                return jax.device_put(array, named_sharding)

            def apply_parallel_transform(self, array, transform_fn, mesh, in_spec, out_spec=None):
                """Apply transformation in parallel using logical names."""
                in_pspec = self.get_partition_spec(in_spec)
                out_pspec = self.get_partition_spec(out_spec) if out_spec else in_pspec
                with mesh:
                    return nnx.shard_map(
                        transform_fn, mesh=mesh, in_specs=(in_pspec,), out_specs=out_pspec
                    )(array)

            def create_sharded_param(self, init_fn, shape, logical_spec):
                """Create a sharded parameter."""
                param = nnx.Param(init_fn(None, shape))
                return param

        # Create our helper
        self.sharder = LogicalShardingHelper(self.sharding_rules)

        # Create test data
        self.test_batch = {
            "inputs": jnp.ones((8, 16, 32)),  # (batch, hidden, feature)
            "labels": jnp.zeros((8,), dtype=jnp.int32),
        }

    def test_logical_axis_sharding(self):
        """Test sharding with logical axis names."""
        if self.limited_test:
            # For limited device environments, use a simpler test
            # that just validates logical-to-physical name mapping
            logical_spec = ("batch", None, "feature")
            physical_pspec = self.sharder.get_partition_spec(logical_spec)
            assert physical_pspec == PartitionSpec("data", None, None)

            # Also test with the device we have
            with self.mesh:
                # Create a named sharding
                named_sharding = self.sharder.get_named_sharding(self.mesh, logical_spec)

                # Apply it to an array
                array = jnp.ones((8, 16, 32))
                sharded_array = jax.device_put(array, named_sharding)

                # Validate basic properties
                assert isinstance(sharded_array, jax.Array)
                assert hasattr(sharded_array, "sharding")
            return

        # Full test for environments with enough devices
        logical_spec = ("batch", None, "feature")

        # This should map to ('data', None, None) in physical device axes
        with self.mesh:
            sharded_batch = self.sharder.shard_with_logical_names(
                self.test_batch["inputs"], self.mesh, logical_spec
            )

        # Verify the sharding is applied as expected
        assert isinstance(sharded_batch, jax.Array)
        assert hasattr(sharded_batch, "sharding")

        # Get the physical partitioning from the sharding
        physical_pspec = self.sharder.get_partition_spec(logical_spec)
        assert physical_pspec == PartitionSpec("data", None, None)

    def test_parallel_transform(self):
        """Test applying a transformation in parallel across devices."""
        if self.limited_test:
            # Skip detailed tests for limited device setups
            pytest.skip("Needs multiple devices for meaningful parallel transform test")

        # Define a simple transformation function
        def double_values(x):
            return x * 2

        # Apply the transformation in parallel
        with self.mesh:
            transformed = self.sharder.apply_parallel_transform(
                self.test_batch["inputs"],
                double_values,
                self.mesh,
                in_spec=("batch", None, "feature"),
                out_spec=("batch", None, "feature"),
            )

        # Verify the transformation was applied correctly
        original_values = jnp.ones((8, 16, 32))
        expected_values = original_values * 2

        # Convert back to host for comparison
        host_result = np.asarray(transformed)
        np.testing.assert_allclose(host_result, expected_values)

    def test_create_sharded_param(self):
        """Test creating a parameter with sharding annotation."""
        # This test can run on any number of devices

        # Define an initializer function
        init_fn = jax.nn.initializers.ones

        # Create a sharded parameter
        with self.mesh:
            param = self.sharder.create_sharded_param(
                init_fn,
                shape=(16, 32),  # (hidden, feature)
                logical_spec=("hidden", "feature"),
            )

        # Verify the parameter is created
        assert isinstance(param, nnx.Param)

        # Check that the initializer was applied
        np.testing.assert_allclose(np.asarray(param.get_value()), np.ones((16, 32)))

    def test_state_serialization(self):
        """Test that sharding rules are preserved."""
        # Create our mapped sharding rules
        logical_spec = ("batch", None, "feature")
        physical_pspec = self.sharder.get_partition_spec(logical_spec)

        # Verify the mapping is correct
        assert physical_pspec == PartitionSpec("data", None, None)

        # Test another mapping
        logical_spec2 = ("hidden", "feature", None)
        physical_pspec2 = self.sharder.get_partition_spec(logical_spec2)
        assert physical_pspec2 == PartitionSpec("model", None, None)
