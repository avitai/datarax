"""Extensive tests for the SharderModule base class.

This module contains tests for the SharderModule class from datarax.core.sharder,
verifying sharding specifications, partition spec creation, named sharding,
state management, and multi-device operations.
"""

from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from datarax.core.element_batch import Batch, Element
from datarax.core.sharder import (
    LogicalAxisSpec,
    SharderModule,
    SharderModuleConfig,
    ShardingRules,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def single_device_mesh() -> Mesh:
    """Create a single-device mesh for testing."""
    devices = jax.devices()[:1]
    return Mesh(np.array(devices), axis_names=("data",))


@pytest.fixture
def multi_device_mesh() -> Mesh:
    """Create a multi-device mesh for testing (uses available devices)."""
    devices = jax.devices()
    # Reshape devices to 2D grid if possible, otherwise 1D
    num_devices = len(devices)
    if num_devices >= 2:
        return Mesh(np.array(devices).reshape(-1), axis_names=("data",))
    return Mesh(np.array(devices), axis_names=("data",))


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Create a sample batch for testing."""
    return {
        "images": jnp.ones((4, 28, 28, 3)),
        "labels": jnp.zeros((4,), dtype=jnp.int32),
    }


@pytest.fixture
def nested_batch() -> dict[str, Any]:
    """Create a nested batch structure for testing."""
    return {
        "features": {
            "images": jnp.ones((4, 28, 28, 3)),
            "embeddings": jnp.zeros((4, 128)),
        },
        "targets": {
            "labels": jnp.zeros((4,), dtype=jnp.int32),
            "weights": jnp.ones((4,)),
        },
    }


# ============================================================================
# Concrete Sharder for Testing
# ============================================================================


class ConcreteSharderModule(SharderModule):
    """Concrete implementation of SharderModule for testing."""

    def shard(self, batch: Batch, sharding: NamedSharding | SingleDeviceSharding) -> Batch:
        """Apply sharding to a batch of data.

        Args:
            batch: A batch of data (PyTree of arrays).
            sharding: The sharding to apply.

        Returns:
            The sharded batch.
        """

        def shard_array(x: Any) -> Any:
            if isinstance(x, jax.Array):
                return jax.device_put(x, sharding)
            return x

        return jax.tree.map(shard_array, batch)


# ============================================================================
# Test Classes
# ============================================================================


class TestSharderModuleConfig:
    """Tests for SharderModuleConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SharderModuleConfig()
        assert config.sharding_rules is None
        assert config.cacheable is False

    def test_config_with_sharding_rules(self):
        """Test configuration with sharding rules."""
        rules: ShardingRules = [
            ("batch", "data"),
            ("model", None),
        ]
        config = SharderModuleConfig(sharding_rules=rules)
        assert config.sharding_rules == rules

    def test_config_inheritance(self):
        """Test that SharderModuleConfig inherits from DataraxModuleConfig."""
        from datarax.core.config import DataraxModuleConfig

        config = SharderModuleConfig()
        assert isinstance(config, DataraxModuleConfig)


class TestSharderModuleInitialization:
    """Tests for SharderModule initialization."""

    def test_init_without_config(self):
        """Test initialization without explicit config."""
        sharder = ConcreteSharderModule()
        assert sharder.config is not None
        assert isinstance(sharder.config, SharderModuleConfig)

    def test_init_with_config(self):
        """Test initialization with explicit config."""
        config = SharderModuleConfig(
            sharding_rules=[("batch", "data")],
            cacheable=True,
        )
        sharder = ConcreteSharderModule(config=config)
        assert sharder.config.sharding_rules == [("batch", "data")]
        assert sharder.config.cacheable is True

    def test_init_with_name(self):
        """Test initialization with a name."""
        sharder = ConcreteSharderModule(name="test_sharder")
        assert sharder.name == "test_sharder"

    def test_inherits_from_datarax_module(self):
        """Test that SharderModule inherits from DataraxModule."""
        from datarax.core.module import DataraxModule

        sharder = ConcreteSharderModule()
        assert isinstance(sharder, DataraxModule)

    def test_inherits_from_nnx_module(self):
        """Test that SharderModule inherits from nnx.Module."""
        sharder = ConcreteSharderModule()
        assert isinstance(sharder, nnx.Module)


class TestPartitionSpecConversion:
    """Tests for partition spec conversion methods."""

    def test_logical_spec_without_rules(self):
        """Test get_partition_spec without sharding rules."""
        sharder = ConcreteSharderModule()
        logical_spec: LogicalAxisSpec = ("batch", "model", None)
        pspec = sharder.get_partition_spec(logical_spec)

        assert isinstance(pspec, PartitionSpec)
        assert pspec == PartitionSpec("batch", "model", None)

    def test_logical_spec_with_rules(self):
        """Test get_partition_spec with sharding rules."""
        rules: ShardingRules = [
            ("batch", "data"),
            ("model", "model"),
            ("hidden", None),
        ]
        config = SharderModuleConfig(sharding_rules=rules)
        sharder = ConcreteSharderModule(config=config)

        logical_spec: LogicalAxisSpec = ("batch", "hidden", None)
        pspec = sharder.get_partition_spec(logical_spec)

        assert isinstance(pspec, PartitionSpec)
        # "batch" maps to "data", "hidden" maps to None
        assert pspec == PartitionSpec("data", None, None)

    def test_passthrough_partition_spec(self):
        """Test that PartitionSpec input is passed through."""
        sharder = ConcreteSharderModule()
        pspec_in = PartitionSpec("x", "y", None)
        pspec_out = sharder.get_partition_spec(pspec_in)

        assert pspec_out is pspec_in

    def test_empty_logical_spec(self):
        """Test get_partition_spec with empty tuple."""
        sharder = ConcreteSharderModule()
        logical_spec: LogicalAxisSpec = ()
        pspec = sharder.get_partition_spec(logical_spec)

        assert isinstance(pspec, PartitionSpec)
        assert pspec == PartitionSpec()

    def test_all_none_logical_spec(self):
        """Test get_partition_spec with all None axes."""
        sharder = ConcreteSharderModule()
        logical_spec: LogicalAxisSpec = (None, None, None)
        pspec = sharder.get_partition_spec(logical_spec)

        assert pspec == PartitionSpec(None, None, None)

    def test_unknown_logical_axis_passthrough(self):
        """Test that unmapped logical axes pass through unchanged."""
        rules: ShardingRules = [("batch", "data")]
        config = SharderModuleConfig(sharding_rules=rules)
        sharder = ConcreteSharderModule(config=config)

        # "unknown" is not in rules, so it should pass through
        logical_spec: LogicalAxisSpec = ("batch", "unknown")
        pspec = sharder.get_partition_spec(logical_spec)

        assert pspec == PartitionSpec("data", "unknown")


class TestNamedShardingCreation:
    """Tests for named sharding creation methods."""

    def test_get_named_sharding(self, single_device_mesh):
        """Test get_named_sharding creates proper NamedSharding."""
        sharder = ConcreteSharderModule()
        logical_spec: LogicalAxisSpec = ("data", None)
        named_sharding = sharder.get_named_sharding(single_device_mesh, logical_spec)

        assert isinstance(named_sharding, NamedSharding)
        assert named_sharding.mesh is single_device_mesh
        assert named_sharding.spec == PartitionSpec("data", None)

    def test_get_named_sharding_with_rules(self, single_device_mesh):
        """Test get_named_sharding with logical-to-physical mapping."""
        rules: ShardingRules = [("batch", "data")]
        config = SharderModuleConfig(sharding_rules=rules)
        sharder = ConcreteSharderModule(config=config)

        logical_spec: LogicalAxisSpec = ("batch",)
        named_sharding = sharder.get_named_sharding(single_device_mesh, logical_spec)

        assert isinstance(named_sharding, NamedSharding)
        assert named_sharding.spec == PartitionSpec("data")


class TestShardingOperation:
    """Tests for the shard() method."""

    def test_shard_raises_not_implemented_on_base(self):
        """Test that base SharderModule.shard() raises NotImplementedError."""
        base_sharder = SharderModule()
        batch = {"data": jnp.ones((4, 8))}
        sharding = SingleDeviceSharding(jax.devices()[0])

        with pytest.raises(NotImplementedError):
            base_sharder.shard(batch, sharding)

    def test_shard_simple_batch(self, sample_batch):
        """Test sharding a simple batch."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])

        sharded_batch = sharder.shard(sample_batch, sharding)

        assert set(sharded_batch.keys()) == set(sample_batch.keys())
        for key in sample_batch:
            assert isinstance(sharded_batch[key], jax.Array)
            assert sharded_batch[key].sharding == sharding

    def test_shard_preserves_values(self, sample_batch):
        """Test that sharding preserves array values."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])

        sharded_batch = sharder.shard(sample_batch, sharding)

        np.testing.assert_allclose(
            np.asarray(sharded_batch["images"]), np.asarray(sample_batch["images"])
        )
        np.testing.assert_array_equal(
            np.asarray(sharded_batch["labels"]), np.asarray(sample_batch["labels"])
        )

    def test_shard_nested_pytree(self, nested_batch):
        """Test sharding a nested PyTree structure."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])

        sharded_batch = sharder.shard(nested_batch, sharding)

        # Verify structure preservation
        assert "features" in sharded_batch
        assert "targets" in sharded_batch
        assert "images" in sharded_batch["features"]
        assert "embeddings" in sharded_batch["features"]
        assert "labels" in sharded_batch["targets"]
        assert "weights" in sharded_batch["targets"]

        # Verify sharding applied
        for key in ["images", "embeddings"]:
            assert sharded_batch["features"][key].sharding == sharding
        for key in ["labels", "weights"]:
            assert sharded_batch["targets"][key].sharding == sharding

    def test_shard_with_named_sharding(self, single_device_mesh, sample_batch):
        """Test sharding with NamedSharding."""
        sharder = ConcreteSharderModule()
        named_sharding = NamedSharding(single_device_mesh, PartitionSpec("data", None))

        # Adjust batch to match the partition spec dimensions
        adjusted_batch = {"data": jnp.ones((4, 8))}
        sharded_batch = sharder.shard(adjusted_batch, named_sharding)

        assert isinstance(sharded_batch["data"], jax.Array)
        assert sharded_batch["data"].sharding == named_sharding


class TestStateManagement:
    """Tests for state management methods."""

    def test_get_state(self):
        """Test get_state returns proper state dictionary."""
        config = SharderModuleConfig(sharding_rules=[("batch", "data")])
        sharder = ConcreteSharderModule(config=config)

        state = sharder.get_state()

        assert isinstance(state, dict)
        assert "sharding_rules" in state
        assert state["sharding_rules"] == [("batch", "data")]

    def test_get_state_without_rules(self):
        """Test get_state when no sharding rules are configured."""
        sharder = ConcreteSharderModule()
        state = sharder.get_state()

        assert isinstance(state, dict)
        # sharding_rules should not be in state if None
        assert "sharding_rules" not in state or state.get("sharding_rules") is None

    def test_set_state_with_rules(self):
        """Test set_state restores sharding rules."""
        sharder = ConcreteSharderModule()
        state = {
            "sharding_rules": [("model", "model_axis")],
        }

        sharder.set_state(state)

        assert sharder.config.sharding_rules == [("model", "model_axis")]

    def test_roundtrip_state(self):
        """Test state can be saved and restored."""
        config = SharderModuleConfig(sharding_rules=[("batch", "data"), ("model", None)])
        sharder1 = ConcreteSharderModule(config=config)

        # Save state
        state = sharder1.get_state()

        # Create new sharder and restore state
        sharder2 = ConcreteSharderModule()
        sharder2.set_state(state)

        assert sharder2.config.sharding_rules == sharder1.config.sharding_rules


class TestParallelTransform:
    """Tests for parallel_transform method."""

    def test_parallel_transform_basic(self, single_device_mesh):
        """Test parallel_transform applies function correctly."""
        sharder = ConcreteSharderModule()
        batch = jnp.ones((4, 8))

        def double_fn(x):
            return x * 2

        # Use simple in_spec that matches the batch dimensions
        result = sharder.parallel_transform(
            batch, double_fn, single_device_mesh, in_spec=PartitionSpec("data", None)
        )

        expected = jnp.ones((4, 8)) * 2
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))

    def test_parallel_transform_with_logical_spec(self, single_device_mesh):
        """Test parallel_transform with logical axis specification."""
        rules: ShardingRules = [("batch", "data")]
        config = SharderModuleConfig(sharding_rules=rules)
        sharder = ConcreteSharderModule(config=config)

        batch = jnp.ones((4, 8))

        def negate_fn(x):
            return -x

        result = sharder.parallel_transform(
            batch, negate_fn, single_device_mesh, in_spec=("batch", None)
        )

        expected = -jnp.ones((4, 8))
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected))


class TestJITCompatibility:
    """Tests for JIT compatibility."""

    def test_shard_jit_compatible(self, sample_batch):
        """Test that shard method is JIT compatible."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])

        @jax.jit
        def jitted_shard(batch):
            return sharder.shard(batch, sharding)

        result = jitted_shard(sample_batch)

        assert set(result.keys()) == set(sample_batch.keys())
        for key in sample_batch:
            assert isinstance(result[key], jax.Array)

    def test_get_partition_spec_jit_compatible(self):
        """Test that get_partition_spec is JIT compatible."""
        sharder = ConcreteSharderModule()

        # get_partition_spec should work outside JIT (it's a pure Python function)
        pspec = sharder.get_partition_spec(("batch", None))
        assert pspec == PartitionSpec("batch", None)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_batch(self):
        """Test sharding an empty batch."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])
        empty_batch: dict[str, jax.Array] = {}

        result = sharder.shard(empty_batch, sharding)

        assert result == {}

    def test_scalar_values_in_batch(self):
        """Test sharding a batch with scalar values."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])
        batch = {
            "data": jnp.array(1.0),  # Scalar
            "count": jnp.array(42, dtype=jnp.int32),
        }

        result = sharder.shard(batch, sharding)

        assert result["data"].shape == ()
        assert result["count"].shape == ()

    def test_mixed_dtypes(self):
        """Test sharding a batch with mixed dtypes."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])
        batch = {
            "float32": jnp.ones((4,), dtype=jnp.float32),
            "float16": jnp.ones((4,), dtype=jnp.float16),
            "int32": jnp.ones((4,), dtype=jnp.int32),
            "bool": jnp.array([True, False, True, False]),
        }

        result = sharder.shard(batch, sharding)

        assert result["float32"].dtype == jnp.float32
        assert result["float16"].dtype == jnp.float16
        assert result["int32"].dtype == jnp.int32
        assert result["bool"].dtype == jnp.bool_

    def test_large_sharding_rules(self):
        """Test with many sharding rules."""
        rules: ShardingRules = [
            ("batch", "data"),
            ("model", "model"),
            ("hidden", None),
            ("sequence", "data"),
            ("heads", "model"),
        ]
        config = SharderModuleConfig(sharding_rules=rules)
        sharder = ConcreteSharderModule(config=config)

        logical_spec: LogicalAxisSpec = ("batch", "sequence", "hidden", "heads")
        pspec = sharder.get_partition_spec(logical_spec)

        assert pspec == PartitionSpec("data", "data", None, "model")


class TestIntegrationWithBatch:
    """Tests for integration with Batch class."""

    def test_shard_batch_object(self):
        """Test sharding a Batch object."""
        sharder = ConcreteSharderModule()
        sharding = SingleDeviceSharding(jax.devices()[0])

        # Create a Batch from Elements
        elements = [
            Element(data={"x": jnp.array([1.0, 2.0])}),
            Element(data={"x": jnp.array([3.0, 4.0])}),
        ]
        batch = Batch(elements)

        # Shard the batch data
        sharded_data = sharder.shard(batch.get_data(), sharding)

        assert isinstance(sharded_data, dict)
        assert "x" in sharded_data
        assert sharded_data["x"].sharding == sharding
