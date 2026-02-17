"""Tests for data parallelism functions.

Tests both SPMD-based and legacy pmap-based data parallel utilities.
"""

from unittest import mock

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from datarax.distributed.data_parallel import (
    all_reduce_gradients,
    create_data_parallel_sharding,
    reduce_gradients,
    shard_batch,
    shard_model_state,
    spmd_train_step,
)
from datarax.distributed.device_mesh import DeviceMeshManager


class TestCreateDataParallelSharding:
    """Tests for create_data_parallel_sharding function."""

    def test_single_device_sharding(self):
        """Test creating sharding with single device mesh."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.spec == jax.sharding.PartitionSpec("data")

    def test_custom_data_axis(self):
        """Test creating sharding with custom axis name."""
        mesh = DeviceMeshManager.create_device_mesh([("batch", 1)])
        sharding = create_data_parallel_sharding(mesh, data_axis="batch")
        assert sharding.spec == jax.sharding.PartitionSpec("batch")

    @pytest.mark.skipif(jax.device_count() < 2, reason="Requires 2+ devices")
    def test_multi_device_sharding(self):
        """Test creating sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        assert sharding.spec == jax.sharding.PartitionSpec("data")


class TestShardBatch:
    """Tests for shard_batch function."""

    def test_shards_array_values(self):
        """Test that jax.Array values are sharded."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = shard_batch(batch, sharding)

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)

    def test_non_array_values_unchanged(self):
        """Test that non-array values pass through unchanged."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"data": jnp.ones((4, 2)), "label": "test_string"}

        result = shard_batch(batch, sharding)

        assert result["label"] == "test_string"

    @pytest.mark.skipif(jax.device_count() < 2, reason="Requires 2+ devices")
    def test_multi_device_shard(self):
        """Test sharding across multiple devices."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=2)
        sharding = create_data_parallel_sharding(mesh)
        batch = {"inputs": jnp.ones((4, 2)), "targets": jnp.zeros((4,))}

        result = shard_batch(batch, sharding)

        assert result["inputs"].shape == (4, 2)
        assert result["targets"].shape == (4,)


class TestShardModelState:
    """Tests for shard_model_state function."""

    def test_replicate_mode(self):
        """Test replication sharding mode."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        state = {"params": jnp.ones((4, 2))}

        result = shard_model_state(state, mesh, param_sharding="replicate")

        assert isinstance(result, dict)
        assert result["params"].shape == (4, 2)

    def test_default_replication(self):
        """Test that default (None) replicates parameters."""
        mesh = DeviceMeshManager.create_data_parallel_mesh(num_devices=1)
        state = {"params": jnp.ones((4, 2))}

        result = shard_model_state(state, mesh)

        assert isinstance(result, dict)
        assert result["params"].shape == (4, 2)


class TestAllReduceGradients:
    """Tests for all_reduce_gradients function."""

    def test_mean_reduction(self):
        """Test mean reduction of gradients."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            result = all_reduce_gradients(jnp.array(4.0), "mean")
            assert float(result) == 2.0

    def test_sum_reduction(self):
        """Test sum reduction of gradients."""
        with mock.patch("jax.lax.psum", return_value=jnp.array(8.0)):
            result = all_reduce_gradients(jnp.array(4.0), "sum")
            assert float(result) == 8.0

    def test_unsupported_reduction_raises(self):
        """Test that unsupported reduce_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported reduce_type"):
            all_reduce_gradients(jnp.array(4.0), "invalid")


class TestReduceGradients:
    """Tests for reduce_gradients (SPMD-compatible)."""

    def test_mean_reduction(self):
        """Test mean reduction on a gradient pytree."""
        grads = {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([4.0, 6.0])}
        result = reduce_gradients(grads, "mean")
        assert float(result["w"]) == 2.0
        assert float(result["b"]) == 5.0

    def test_sum_reduction(self):
        """Test sum reduction on a gradient pytree."""
        grads = {"w": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_gradients(grads, "sum")
        assert float(result["w"]) == 6.0

    def test_unsupported_reduction_raises(self):
        """Test that unsupported reduce_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported reduce_type"):
            reduce_gradients({"w": jnp.array([1.0])}, "invalid")


class TestSpmdTrainStep:
    """Tests for spmd_train_step function."""

    def _make_model_and_optimizer(self) -> tuple[nnx.Module, nnx.Optimizer]:
        """Create a minimal NNX model and optimizer for testing."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.sgd(0.01), wrt=nnx.Param)
        return model, optimizer

    def test_reduces_loss(self):
        """Test that a single training step produces a finite loss."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss = spmd_train_step(model, optimizer, loss_fn, batch)
        assert jnp.isfinite(loss)

    def test_updates_parameters(self):
        """Test that parameters change after a training step."""
        model, optimizer = self._make_model_and_optimizer()
        params_before = jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        spmd_train_step(model, optimizer, loss_fn, batch)
        params_after = nnx.state(model, nnx.Param)

        # At least one parameter leaf must have changed
        leaves_before = jax.tree.leaves(params_before)
        leaves_after = jax.tree.leaves(params_after)
        any_changed = any(
            not jnp.array_equal(b, a) for b, a in zip(leaves_before, leaves_after)
        )
        assert any_changed, "Parameters should change after a training step"

    def test_loss_decreases_over_steps(self):
        """Test that loss decreases over multiple training steps."""
        model, optimizer = self._make_model_and_optimizer()
        batch = {"x": jnp.ones((4, 2)), "y": jnp.zeros((4, 1))}

        def loss_fn(m: nnx.Module, b: dict) -> jax.Array:
            return jnp.mean((m(b["x"]) - b["y"]) ** 2)

        loss_first = spmd_train_step(model, optimizer, loss_fn, batch)
        for _ in range(10):
            loss_last = spmd_train_step(model, optimizer, loss_fn, batch)

        assert float(loss_last) < float(loss_first)
