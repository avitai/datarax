"""Tests for distributed metrics functions.

Tests both SPMD-compatible reductions (jnp.*) and collective reductions (lax.p*).
"""

from unittest import mock

import jax.numpy as jnp

from datarax.distributed.metrics import (
    all_gather,
    collect_from_devices,
    reduce_custom,
    reduce_max,
    reduce_mean,
    reduce_mean_collective,
    reduce_min,
    reduce_sum,
    reduce_sum_collective,
)


# ---------------------------------------------------------------------------
# SPMD-compatible reductions (jnp.* on global arrays)
# ---------------------------------------------------------------------------


class TestReduceMean:
    """Tests for reduce_mean (SPMD-compatible)."""

    def test_reduces_multidim_arrays(self):
        """Test that multi-dimensional arrays are reduced with jnp.mean."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_mean(metrics)
        assert float(result["loss"]) == 2.0

    def test_scalar_arrays_unchanged(self):
        """Test that scalar arrays pass through unchanged."""
        metrics = {"loss": jnp.array(3.0)}
        result = reduce_mean(metrics)
        assert float(result["loss"]) == 3.0

    def test_non_arrays_unchanged(self):
        """Test that non-array values pass through unchanged."""
        metrics = {"loss": jnp.array([1.0, 2.0]), "step": 10, "label": "test"}
        result = reduce_mean(metrics)
        assert result["step"] == 10
        assert result["label"] == "test"


class TestReduceSum:
    """Tests for reduce_sum (SPMD-compatible)."""

    def test_reduces_arrays(self):
        """Test that arrays are summed."""
        metrics = {"count": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_sum(metrics)
        assert float(result["count"]) == 6.0


class TestReduceMax:
    """Tests for reduce_max (SPMD-compatible)."""

    def test_reduces_arrays(self):
        """Test that arrays are reduced with max."""
        metrics = {"peak": jnp.array([1.0, 5.0, 3.0])}
        result = reduce_max(metrics)
        assert float(result["peak"]) == 5.0


class TestReduceMin:
    """Tests for reduce_min (SPMD-compatible)."""

    def test_reduces_arrays(self):
        """Test that arrays are reduced with min."""
        metrics = {"low": jnp.array([1.0, 5.0, 3.0])}
        result = reduce_min(metrics)
        assert float(result["low"]) == 1.0


class TestReduceCustom:
    """Tests for reduce_custom (SPMD-compatible)."""

    def test_custom_reductions(self):
        """Test applying different reductions per metric."""
        metrics = {
            "loss": jnp.array([1.0, 2.0, 3.0]),
            "count": jnp.array([1.0, 2.0, 3.0]),
            "peak": jnp.array([1.0, 5.0, 3.0]),
        }
        result = reduce_custom(
            metrics,
            reduce_fn={"loss": "mean", "count": "sum", "peak": "max"},
        )
        assert float(result["loss"]) == 2.0
        assert float(result["count"]) == 6.0
        assert float(result["peak"]) == 5.0

    def test_default_reduction_is_mean(self):
        """Test that default reduction is mean when reduce_fn is None."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0])}
        result = reduce_custom(metrics)
        assert float(result["loss"]) == 2.0

    def test_unknown_operation_passthrough(self):
        """Test that unknown reduction operations pass value through."""
        metrics = {"val": jnp.array([5.0, 10.0])}
        result = reduce_custom(metrics, reduce_fn={"val": "unknown_op"})
        # Unknown op doesn't reduce, value passes through
        assert result["val"].shape == (2,)


# ---------------------------------------------------------------------------
# Collective reductions (lax.p* — pmap/shard_map only)
# ---------------------------------------------------------------------------


class TestReduceMeanCollective:
    """Tests for reduce_mean_collective (pmap/shard_map context)."""

    def test_reduces_arrays_with_pmean(self):
        """Test that arrays are reduced with lax.pmean."""
        with mock.patch("jax.lax.pmean", return_value=jnp.array(2.0)):
            metrics = {"loss": jnp.array(3.0), "step": 10}
            result = reduce_mean_collective(metrics)
            assert float(result["loss"]) == 2.0
            assert result["step"] == 10


class TestReduceSumCollective:
    """Tests for reduce_sum_collective (pmap/shard_map context)."""

    def test_reduces_arrays_with_psum(self):
        """Test that arrays are reduced with lax.psum."""
        with mock.patch("jax.lax.psum", return_value=jnp.array(6.0)):
            metrics = {"loss": jnp.array(3.0)}
            result = reduce_sum_collective(metrics)
            assert float(result["loss"]) == 6.0


# ---------------------------------------------------------------------------
# Gather and collect utilities
# ---------------------------------------------------------------------------


class TestAllGather:
    """Tests for all_gather (pmap/shard_map context)."""

    def test_gathers_arrays(self):
        """Test that arrays are gathered."""
        with mock.patch("jax.lax.all_gather", return_value=jnp.array([1.0, 2.0])):
            metrics = {"loss": jnp.array(1.0), "step": 5}
            result = all_gather(metrics)
            assert result["loss"].tolist() == [1.0, 2.0]
            assert result["step"] == 5


class TestCollectFromDevices:
    """Tests for collect_from_devices."""

    def test_collects_multi_dim_arrays(self):
        """Test collecting values from multi-dimensional arrays."""
        metrics = {"loss": jnp.array([1.0, 2.0, 3.0]), "accuracy": 0.95}
        result = collect_from_devices(metrics)

        assert len(result["loss"]) == 3
        assert float(result["loss"][0]) == 1.0
        assert float(result["loss"][1]) == 2.0
        assert float(result["loss"][2]) == 3.0
        assert result["accuracy"] == 0.95

    def test_scalar_arrays_unchanged(self):
        """Test that scalar arrays are kept as-is."""
        metrics = {"scalar": jnp.array(1.0)}
        result = collect_from_devices(metrics)
        assert float(result["scalar"]) == 1.0

    def test_non_array_unchanged(self):
        """Test that non-array values pass through."""
        metrics = {"label": "test", "count": 42}
        result = collect_from_devices(metrics)
        assert result["label"] == "test"
        assert result["count"] == 42
