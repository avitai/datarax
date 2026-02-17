"""Tests for device placement utilities.

This module contains tests for the DevicePlacement class and related utilities
for explicit device placement of JAX arrays and PyTrees.
"""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from datarax.distributed.device_placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    HardwareType,
    distribute_batch,
    get_batch_size_recommendation,
    place_on_device,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def placement() -> DevicePlacement:
    """Create a DevicePlacement instance."""
    return DevicePlacement()


@pytest.fixture
def sample_data() -> dict[str, jax.Array]:
    """Create sample data for testing."""
    return {
        "images": jnp.ones((4, 28, 28, 3)),
        "labels": jnp.zeros((4,), dtype=jnp.int32),
    }


@pytest.fixture
def single_device_mesh() -> Mesh:
    """Create a single-device mesh for testing."""
    devices = jax.devices()[:1]
    return Mesh(np.array(devices), axis_names=("data",))


# ============================================================================
# Test Classes
# ============================================================================


class TestDevicePlacementInitialization:
    """Tests for DevicePlacement initialization."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        placement = DevicePlacement()
        assert placement.default_device is not None
        assert placement.default_device in jax.devices()

    def test_initialization_with_device(self):
        """Test initialization with a specific device."""
        device = jax.devices()[0]
        placement = DevicePlacement(default_device=device)
        assert placement.default_device is device

    def test_lazy_init_does_not_call_jax_devices(self):
        """Test that constructing DevicePlacement(None) does not call jax.devices()."""
        with mock.patch("datarax.distributed.device_placement.jax.devices") as mock_devices:
            placement = DevicePlacement(default_device=None)
            mock_devices.assert_not_called()
            assert placement._default_device is None
            assert placement._hardware_type_cache is None

    def test_lazy_init_resolves_on_access(self):
        """Test that default_device is lazily resolved on first access."""
        placement = DevicePlacement(default_device=None)
        assert placement._default_device is None
        device = placement.default_device
        assert device is not None
        assert placement._default_device is not None

    def test_hardware_detection(self):
        """Test that hardware type is detected."""
        placement = DevicePlacement()
        assert placement.hardware_type is not None
        assert isinstance(placement.hardware_type, HardwareType)

    def test_num_devices(self):
        """Test num_devices property."""
        placement = DevicePlacement()
        assert placement.num_devices == len(jax.devices())
        assert placement.num_devices >= 1


class TestPlaceOnDevice:
    """Tests for place_on_device method."""

    def test_place_single_array(self, placement):
        """Test placing a single array on device."""
        data = jnp.ones((4, 8))
        device = jax.devices()[0]

        result = placement.place_on_device(data, device)

        assert isinstance(result, jax.Array)
        assert isinstance(result.sharding, SingleDeviceSharding)
        np.testing.assert_allclose(np.asarray(result), np.ones((4, 8)))

    def test_place_pytree(self, placement, sample_data):
        """Test placing a PyTree on device."""
        device = jax.devices()[0]

        result = placement.place_on_device(sample_data, device)

        assert isinstance(result, dict)
        assert "images" in result
        assert "labels" in result
        assert isinstance(result["images"], jax.Array)
        assert isinstance(result["labels"], jax.Array)

    def test_place_with_default_device(self, placement):
        """Test placing data on the default device."""
        data = jnp.ones((4, 8))

        result = placement.place_on_device(data)

        assert isinstance(result, jax.Array)
        # Should be on the default device
        assert result.sharding is not None

    def test_preserves_values(self, placement, sample_data):
        """Test that values are preserved after placement."""
        device = jax.devices()[0]

        result = placement.place_on_device(sample_data, device)

        np.testing.assert_allclose(np.asarray(result["images"]), np.asarray(sample_data["images"]))
        np.testing.assert_array_equal(
            np.asarray(result["labels"]), np.asarray(sample_data["labels"])
        )

    def test_preserves_dtypes(self, placement):
        """Test that dtypes are preserved after placement."""
        data = {
            "float32": jnp.ones((4,), dtype=jnp.float32),
            "int32": jnp.ones((4,), dtype=jnp.int32),
            "float16": jnp.ones((4,), dtype=jnp.float16),
        }
        device = jax.devices()[0]

        result = placement.place_on_device(data, device)

        assert result["float32"].dtype == jnp.float32
        assert result["int32"].dtype == jnp.int32
        assert result["float16"].dtype == jnp.float16


class TestDistributeBatch:
    """Tests for distribute_batch method."""

    def test_distribute_with_single_device_sharding(self, placement, sample_data):
        """Test distribution with single device sharding."""
        sharding = SingleDeviceSharding(jax.devices()[0])

        result = placement.distribute_batch(sample_data, sharding)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, jax.Array)

    def test_distribute_with_named_sharding(self, placement, single_device_mesh):
        """Test distribution with NamedSharding."""
        data = {"x": jnp.ones((4, 8))}
        sharding = NamedSharding(single_device_mesh, PartitionSpec("data", None))

        result = placement.distribute_batch(data, sharding)

        assert isinstance(result["x"], jax.Array)
        assert result["x"].sharding == sharding


class TestReplicateAcrossDevices:
    """Tests for replicate_across_devices method."""

    def test_replicate_single_array(self, placement):
        """Test replicating a single array."""
        data = jnp.ones((4, 8))

        result = placement.replicate_across_devices(data)

        assert isinstance(result, jax.Array)
        np.testing.assert_allclose(np.asarray(result), np.ones((4, 8)))

    def test_replicate_pytree(self, placement, sample_data):
        """Test replicating a PyTree."""
        result = placement.replicate_across_devices(sample_data)

        assert isinstance(result, dict)
        assert "images" in result
        assert "labels" in result


class TestShardBatchDim:
    """Tests for shard_batch_dim method."""

    def test_shard_single_array(self, placement, single_device_mesh):
        """Test sharding a single array along batch dim."""
        data = jnp.ones((4, 8))

        result = placement.shard_batch_dim(data, single_device_mesh)

        assert isinstance(result, jax.Array)
        assert result.shape == (4, 8)

    def test_shard_pytree(self, placement, single_device_mesh):
        """Test sharding a PyTree along batch dim."""
        data = {"x": jnp.ones((4, 8, 3)), "y": jnp.zeros((4,))}

        result = placement.shard_batch_dim(data, single_device_mesh)

        assert isinstance(result, dict)
        assert result["x"].shape == (4, 8, 3)
        assert result["y"].shape == (4,)

    def test_shard_custom_batch_axis(self, placement, single_device_mesh):
        """Test sharding with custom batch axis."""
        # Batch is second dimension
        data = jnp.ones((8, 4, 3))

        result = placement.shard_batch_dim(data, single_device_mesh, batch_axis=1)

        assert isinstance(result, jax.Array)
        assert result.shape == (8, 4, 3)

    def test_shard_scalar(self, placement, single_device_mesh):
        """Test sharding scalar values (should be replicated)."""
        data = {"scalar": jnp.array(1.0), "vector": jnp.ones((4,))}

        result = placement.shard_batch_dim(data, single_device_mesh)

        assert result["scalar"].shape == ()
        assert result["vector"].shape == (4,)


class TestBatchSizeRecommendation:
    """Tests for batch size recommendation."""

    def test_get_recommendation(self, placement):
        """Test getting batch size recommendation."""
        rec = placement.get_batch_size_recommendation()

        assert isinstance(rec, BatchSizeRecommendation)
        assert rec.min_batch_size > 0
        assert rec.optimal_batch_size >= rec.min_batch_size
        assert rec.critical_batch_size > 0

    def test_recommendation_for_specific_hardware(self):
        """Test getting recommendation for specific hardware."""
        placement = DevicePlacement()
        rec = placement.get_batch_size_recommendation(HardwareType.H100)

        assert rec.critical_batch_size == 298
        assert rec.optimal_batch_size == 320

    def test_recommendation_for_tpu(self):
        """Test getting recommendation for TPU."""
        placement = DevicePlacement()
        rec = placement.get_batch_size_recommendation(HardwareType.TPU_V5E)

        assert rec.critical_batch_size == 240
        assert rec.optimal_batch_size == 256


class TestValidateBatchSize:
    """Tests for batch size validation."""

    def test_validate_optimal_size(self, placement):
        """Test validating an optimal batch size."""
        rec = placement.get_batch_size_recommendation()
        is_valid, message = placement.validate_batch_size(rec.optimal_batch_size)

        assert is_valid is True
        assert "optimal" in message.lower()

    def test_validate_suboptimal_size(self, placement):
        """Test validating a suboptimal batch size."""
        rec = placement.get_batch_size_recommendation()
        # Use a size between min and critical
        suboptimal_size = max(rec.min_batch_size, rec.critical_batch_size - 1)
        if suboptimal_size < rec.critical_batch_size:
            is_valid, message = placement.validate_batch_size(suboptimal_size)
            assert is_valid is True
            assert "critical" in message.lower() or "acceptable" in message.lower()

    def test_validate_too_small_size(self, placement):
        """Test validating a batch size that's too small."""
        rec = placement.get_batch_size_recommendation()
        too_small = max(1, rec.min_batch_size - 1)
        if too_small < rec.min_batch_size:
            is_valid, message = placement.validate_batch_size(too_small)
            assert is_valid is False
            assert "below minimum" in message.lower()

    def test_validate_without_warning(self, placement):
        """Test validation without warnings for suboptimal sizes."""
        rec = placement.get_batch_size_recommendation()
        suboptimal_size = max(rec.min_batch_size, rec.critical_batch_size - 1)
        if suboptimal_size < rec.critical_batch_size:
            is_valid, message = placement.validate_batch_size(
                suboptimal_size, warn_suboptimal=False
            )
            assert is_valid is True


class TestDeviceInfo:
    """Tests for device information methods."""

    def test_get_device_info(self, placement):
        """Test getting device information."""
        info = placement.get_device_info()

        assert isinstance(info, dict)
        assert "num_devices" in info
        assert "hardware_type" in info
        assert "platforms" in info
        assert "devices" in info

        assert info["num_devices"] >= 1
        assert isinstance(info["devices"], list)

    def test_device_info_contents(self, placement):
        """Test device info contains expected fields."""
        info = placement.get_device_info()

        for device_info in info["devices"]:
            assert "id" in device_info
            assert "platform" in device_info


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_place_on_device_function(self, sample_data):
        """Test the place_on_device convenience function."""
        device = jax.devices()[0]
        result = place_on_device(sample_data, device)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, jax.Array)

    def test_distribute_batch_function(self, sample_data):
        """Test the distribute_batch convenience function."""
        sharding = SingleDeviceSharding(jax.devices()[0])
        result = distribute_batch(sample_data, sharding)

        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, jax.Array)

    def test_get_batch_size_recommendation_function(self):
        """Test the get_batch_size_recommendation convenience function."""
        rec = get_batch_size_recommendation()

        assert isinstance(rec, BatchSizeRecommendation)
        assert rec.min_batch_size > 0


class TestHardwareType:
    """Tests for HardwareType enum."""

    def test_all_types_have_recommendations(self):
        """Test that all hardware types have batch size recommendations."""
        for hw_type in HardwareType:
            rec = get_batch_size_recommendation(hw_type)
            assert isinstance(rec, BatchSizeRecommendation)
            assert rec.min_batch_size > 0

    def test_enum_values(self):
        """Test that enum values are strings."""
        for hw_type in HardwareType:
            assert isinstance(hw_type.value, str)


class TestPrefetchToDevice:
    """Tests for prefetch_to_device method."""

    def test_prefetch_simple_iterator(self, placement):
        """Test prefetching from a simple iterator."""
        data_list = [jnp.ones((4, 8)) * i for i in range(5)]
        device = jax.devices()[0]

        prefetched = list(placement.prefetch_to_device(iter(data_list), device))

        assert len(prefetched) == 5
        for i, item in enumerate(prefetched):
            assert isinstance(item, jax.Array)
            np.testing.assert_allclose(np.asarray(item), np.ones((4, 8)) * i)

    def test_prefetch_empty_iterator(self, placement):
        """Test prefetching from an empty iterator."""
        device = jax.devices()[0]

        prefetched = list(placement.prefetch_to_device(iter([]), device))

        assert len(prefetched) == 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_pytree(self, placement):
        """Test placing an empty PyTree."""
        device = jax.devices()[0]
        result = placement.place_on_device({}, device)
        assert result == {}

    def test_scalar_data(self, placement):
        """Test placing scalar data."""
        device = jax.devices()[0]
        result = placement.place_on_device(jnp.array(42.0), device)

        assert isinstance(result, jax.Array)
        assert result.shape == ()
        assert float(result) == 42.0

    def test_nested_pytree(self, placement):
        """Test placing a deeply nested PyTree."""
        device = jax.devices()[0]
        data = {
            "level1": {
                "level2": {
                    "level3": jnp.ones((4, 8)),
                },
            },
        }

        result = placement.place_on_device(data, device)

        assert result["level1"]["level2"]["level3"].shape == (4, 8)

    def test_mixed_pytree(self, placement):
        """Test placing a PyTree with mixed types."""
        device = jax.devices()[0]
        data = {
            "array": jnp.ones((4, 8)),
            "list": [jnp.zeros((2,)), jnp.ones((3,))],
            "tuple": (jnp.array(1.0), jnp.array(2.0)),
        }

        result = placement.place_on_device(data, device)

        assert isinstance(result["array"], jax.Array)
        assert isinstance(result["list"], list)
        assert isinstance(result["tuple"], tuple)
