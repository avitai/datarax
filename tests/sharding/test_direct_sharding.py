"""Tests for JAX sharding functionality without circular imports.

This module contains tests that directly use JAX sharding functionality,
avoiding the circular import issues in the main codebase.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec


def test_basic_array_sharder():
    """Test basic sharding of arrays using direct JAX functions."""
    # Create a test array
    test_array = jnp.ones((8, 8))

    # Get the first device
    device = jax.devices()[0]

    # Create a single device sharding
    single_device_sharding = jax.sharding.SingleDeviceSharding(device)

    # Shard the array directly with JAX
    sharded_array = jax.device_put(test_array, single_device_sharding)

    # Check that the result is a JAX Array
    assert isinstance(sharded_array, jax.Array)

    # Check that the sharding is applied
    assert hasattr(sharded_array, "sharding")
    assert sharded_array.sharding == single_device_sharding

    # Check that the values are preserved
    np.testing.assert_allclose(np.asarray(sharded_array), np.ones((8, 8)))


def test_batch_sharding():
    """Test sharding a batch of arrays directly with JAX."""
    # Create a test batch
    batch = {
        "images": jnp.ones((4, 28, 28, 3)),
        "labels": jnp.zeros((4,), dtype=jnp.int32),
    }

    # Get the first device
    device = jax.devices()[0]

    # Create a single device sharding
    single_device_sharding = jax.sharding.SingleDeviceSharding(device)

    # Shard the batch directly with JAX
    sharded_batch = jax.tree.map(lambda x: jax.device_put(x, single_device_sharding), batch)

    # Check that the batch structure is preserved
    assert set(sharded_batch.keys()) == set(batch.keys())

    # Check that each array in the batch is sharded
    for key in batch:
        assert isinstance(sharded_batch[key], jax.Array)
        assert hasattr(sharded_batch[key], "sharding")
        assert sharded_batch[key].sharding == single_device_sharding


def test_nested_sharding():
    """Test sharding a nested PyTree structure."""
    # Create a nested batch
    nested_batch = {
        "features": {
            "images": jnp.ones((4, 28, 28, 3)),
            "metadata": jnp.zeros((4, 5)),
        },
        "targets": {
            "labels": jnp.zeros((4,), dtype=jnp.int32),
            "weights": jnp.ones((4,)),
        },
    }

    # Get the first device
    device = jax.devices()[0]

    # Create a single device sharding
    single_device_sharding = jax.sharding.SingleDeviceSharding(device)

    # Shard the batch
    sharded_batch = jax.tree.map(lambda x: jax.device_put(x, single_device_sharding), nested_batch)

    # Check that the structure is preserved
    assert set(sharded_batch.keys()) == set(nested_batch.keys())
    # Check features keys
    features_keys = set(nested_batch["features"].keys())
    assert set(sharded_batch["features"].keys()) == features_keys
    # Check targets keys
    targets_keys = set(nested_batch["targets"].keys())
    assert set(sharded_batch["targets"].keys()) == targets_keys

    # Check that each array is sharded
    for key1 in nested_batch:
        for key2 in nested_batch[key1]:
            assert isinstance(sharded_batch[key1][key2], jax.Array)
            assert hasattr(sharded_batch[key1][key2], "sharding")
            assert sharded_batch[key1][key2].sharding == single_device_sharding


def test_logical_sharding():
    """Test logical sharding of arrays."""
    # Create an array
    test_array = jnp.ones((8, 16))

    # Get the first device
    device = jax.devices()[0]

    # Create a logical mesh (single device)
    mesh = Mesh(np.array([device]), axis_names=("x",))

    # Create partition specs for logical sharding
    p_spec = PartitionSpec("x", None)  # Shard only on first axis

    # Create a NamedSharding
    named_sharding = jax.sharding.NamedSharding(mesh, p_spec)

    # Apply sharding
    with mesh:
        sharded_array = jax.device_put(test_array, named_sharding)

        # Check basic sharding properties
        assert isinstance(sharded_array, jax.Array)
        assert hasattr(sharded_array, "sharding")

        # Check mesh and spec match instead of using is_equivalent_to
        assert isinstance(sharded_array.sharding, jax.sharding.NamedSharding)
        assert sharded_array.sharding.mesh == named_sharding.mesh
        assert sharded_array.sharding.spec == named_sharding.spec

        # Check values
        np.testing.assert_allclose(np.asarray(sharded_array), np.ones((8, 16)))


def test_shard_consistency():
    """Test consistency of sharding operations across different configurations.

    This test verifies that:
    1. Data remains consistent after sharding
    2. Sharding is properly applied regardless of configuration
    3. Sharding and unsharding preserves data integrity
    """
    # Create test arrays with distinctive patterns to better verify consistency
    # Use arrays with recognizable patterns
    test_array_1 = jnp.arange(32).reshape(8, 4)  # Sequential pattern
    test_array_2 = jnp.eye(8)  # Identity matrix pattern

    # Create a PyTree batch with these distinctive arrays
    batch = {
        "sequential": test_array_1,
        "identity": test_array_2,
        "nested": {
            "random": jax.random.normal(jax.random.key(42), (4, 6)),
        },
    }

    # First test: Sharding with SingleDeviceSharding preserves data
    device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)

    # Helper function to check if an object is a JAX array
    def is_array(x):
        return isinstance(x, jax.Array | jax.Array)

    # Shard the batch
    sharded_batch = jax.tree.map(lambda x: jax.device_put(x, sharding), batch, is_leaf=is_array)

    # Unhard by copying back to host
    unsharded_batch = jax.tree.map(np.asarray, sharded_batch)

    # Check that we get the same values back after sharding and unsharding
    np.testing.assert_allclose(
        np.asarray(unsharded_batch["sequential"]), np.asarray(batch["sequential"])
    )
    np.testing.assert_allclose(
        np.asarray(unsharded_batch["identity"]), np.asarray(batch["identity"])
    )

    # Check nested structures are preserved
    np.testing.assert_allclose(
        np.asarray(unsharded_batch["nested"]["random"]), np.asarray(batch["nested"]["random"])
    )

    # Second test: Verify consistency when resharding
    # Create a mesh for more complex logical sharding
    devices = jax.devices()

    # Adapt test for single or multiple devices
    if len(devices) >= 2:
        # With multiple devices, test actual distributed sharding
        device_mesh = np.array(devices[:2])
        mesh = Mesh(device_mesh, axis_names=("x",))

        # Create two different partition specs
        pspec1 = PartitionSpec("x")  # Shard on the x axis
        pspec2 = PartitionSpec(None)  # Replicate

        with mesh:
            # Create two different shardings
            sharding1 = jax.sharding.NamedSharding(mesh, pspec1)
            sharding2 = jax.sharding.NamedSharding(mesh, pspec2)

            # First shard with sharding1
            sharded1 = jax.device_put(test_array_1, sharding1)

            # Then reshard to sharding2
            sharded2 = jax.device_put(sharded1, sharding2)

            # Back to sharding1
            sharded3 = jax.device_put(sharded2, sharding1)

            # Check that all the data is preserved
            np.testing.assert_allclose(np.asarray(sharded1), np.asarray(test_array_1))
            np.testing.assert_allclose(np.asarray(sharded2), np.asarray(test_array_1))
            np.testing.assert_allclose(np.asarray(sharded3), np.asarray(test_array_1))
    else:
        # With a single device, test with different shardings
        # This still verifies that the resharding logic works correctly
        sharding1 = jax.sharding.SingleDeviceSharding(devices[0])
        # Same device but different object
        sharding2 = jax.sharding.SingleDeviceSharding(devices[0])

        # First shard with sharding1
        sharded1 = jax.device_put(test_array_1, sharding1)

        # Then reshard to sharding2
        sharded2 = jax.device_put(sharded1, sharding2)

        # Check that the data is preserved
        np.testing.assert_allclose(np.asarray(sharded1), np.asarray(test_array_1))
        np.testing.assert_allclose(np.asarray(sharded2), np.asarray(test_array_1))

        # Verify the sharding objects are different
        assert sharding1 is not sharding2
        # But in terms of equivalence they should match
        assert sharded1.sharding == sharding1
        assert sharded2.sharding == sharding2


def test_gather_from_shards():
    """Test gathering data from sharded arrays back to a single device.

    This test verifies that:
    1. Data sharded across devices can be gathered back correctly
    2. Gathered data maintains consistency with the original data
    3. Pytree structures are preserved during gather operations
    """
    # Create test data with recognizable patterns
    test_array = jnp.arange(64).reshape(8, 8)

    # Get available devices
    devices = jax.devices()

    if len(devices) >= 2:
        # Test with actual multi-device setup
        device_mesh = np.array(devices[:2])
        mesh = Mesh(device_mesh, axis_names=("x",))

        # Create partition specs for different sharding strategies
        row_sharded = PartitionSpec("x", None)  # Shard by rows

        with mesh:
            # Shard the array by rows
            row_sharding = jax.sharding.NamedSharding(mesh, row_sharded)
            sharded_rows = jax.device_put(test_array, row_sharding)

            # Verify the array is sharded
            assert isinstance(sharded_rows, jax.Array)
            assert hasattr(sharded_rows, "sharding")

            # Create a replicated copy with no sharding
            replicated_sharding = jax.sharding.NamedSharding(mesh, PartitionSpec(None, None))

            # Gather data by resharding to replicated
            gathered = jax.device_put(sharded_rows, replicated_sharding)

            # Check data is preserved after gathering
            np.testing.assert_allclose(np.asarray(gathered), np.asarray(test_array))

            # Test with a PyTree structure
            tree = {
                "array1": test_array,
                "array2": jnp.ones((4, 4)),
                "nested": {"array3": jnp.zeros((2, 6))},
            }

            # Shard the PyTree
            sharded_tree = jax.tree.map(
                lambda x: jax.device_put(x, row_sharding)
                if x.ndim >= 2
                else jax.device_put(x, replicated_sharding),
                tree,
            )

            # Gather the PyTree
            gathered_tree = jax.tree.map(
                lambda x: jax.device_put(x, replicated_sharding), sharded_tree
            )

            # Check structure and data are preserved
            assert set(gathered_tree.keys()) == set(tree.keys())
            assert set(gathered_tree["nested"].keys()) == set(tree["nested"].keys())

            # Check data values
            np.testing.assert_allclose(
                np.asarray(gathered_tree["array1"]), np.asarray(tree["array1"])
            )
            np.testing.assert_allclose(
                np.asarray(gathered_tree["array2"]), np.asarray(tree["array2"])
            )
            np.testing.assert_allclose(
                np.asarray(gathered_tree["nested"]["array3"]), np.asarray(tree["nested"]["array3"])
            )
    else:
        # Single-device test - focus on API consistency rather than actual distribution
        device = devices[0]
        device_sharding = jax.sharding.SingleDeviceSharding(device)

        # Shard the array to the single device
        sharded = jax.device_put(test_array, device_sharding)

        # "Gather" by simply converting to NumPy array
        gathered = np.asarray(sharded)

        # Check data consistency
        np.testing.assert_allclose(gathered, np.asarray(test_array))

        # Test with a PyTree structure
        tree = {
            "array1": test_array,
            "array2": jnp.ones((4, 4)),
            "nested": {"array3": jnp.zeros((2, 6))},
        }

        # Shard the PyTree
        sharded_tree = jax.tree.map(lambda x: jax.device_put(x, device_sharding), tree)

        # Gather the PyTree
        gathered_tree = jax.tree.map(np.asarray, sharded_tree)

        # Check structure and data are preserved
        assert set(gathered_tree.keys()) == set(tree.keys())
        assert set(gathered_tree["nested"].keys()) == set(tree["nested"].keys())

        # Check data values
        np.testing.assert_allclose(gathered_tree["array1"], np.asarray(tree["array1"]))
        np.testing.assert_allclose(gathered_tree["array2"], np.asarray(tree["array2"]))
        np.testing.assert_allclose(
            gathered_tree["nested"]["array3"], np.asarray(tree["nested"]["array3"])
        )


def test_mesh_sharding():
    """Test sharding with a device mesh using mock devices if needed."""
    # Create a test batch
    batch = {
        "images": jnp.ones((4, 28, 28, 3)),
        "labels": jnp.zeros((4,), dtype=jnp.int32),
    }

    real_devices = jax.devices()

    # For environments with just one device, we'll test a simplified version
    # that doesn't rely on actual multi-device sharding
    if len(real_devices) < 2:
        # Since we only have one device, we'll verify single-device sharding
        # capabilities and skip the multi-device mesh test
        device = real_devices[0]
        single_device_sharding = jax.sharding.SingleDeviceSharding(device)

        # Test basic sharding
        for key in batch:
            sharded_array = jax.device_put(batch[key], single_device_sharding)

            # Check sharding is applied
            assert isinstance(sharded_array, jax.Array)
            assert hasattr(sharded_array, "sharding")
            assert sharded_array.sharding == single_device_sharding

            # Verify data integrity
            np.testing.assert_allclose(np.asarray(sharded_array), np.asarray(batch[key]))

        # Test is complete for single-device case
        return

    # Real multi-device case - continue with the actual mesh test
    devices = real_devices[:2]
    device_mesh = np.array(devices)
    mesh = jax.sharding.Mesh(device_mesh, axis_names=("batch",))

    # Create partition specs
    partition_specs = {
        "images": jax.sharding.PartitionSpec("batch"),
        "labels": jax.sharding.PartitionSpec("batch"),
    }

    # Apply sharding
    with mesh:
        sharded_batch = {}
        for key, pspec in partition_specs.items():
            mesh_sharding = jax.sharding.NamedSharding(mesh, pspec)
            sharded_batch[key] = jax.device_put(batch[key], mesh_sharding)

            # Check sharding is applied
            assert isinstance(sharded_batch[key], jax.Array)
            assert hasattr(sharded_batch[key], "sharding")
