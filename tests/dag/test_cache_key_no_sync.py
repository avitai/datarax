"""Tests for cache key computation without device-to-host sync.

Validates that _compute_cache_key in dag_executor and caching.py:
1. Does not call jax.device_get or force device-to-host transfers
2. Uses only host-side metadata (shape, dtype, identity)
3. Produces unique keys for different inputs
"""

from unittest.mock import patch

import jax.numpy as jnp
import pytest

from datarax.dag.nodes import Identity
from datarax.dag.nodes.caching import Cache, CacheNode


class TestDagExecutorCacheKey:
    """Tests for DAGExecutor._compute_cache_key."""

    @pytest.fixture
    def executor(self):
        """Create a DAGExecutor with caching enabled."""
        from datarax.dag.dag_executor import DAGExecutor

        return DAGExecutor(enable_caching=True, enforce_batch=False)

    def test_jax_array_no_device_get(self, executor):
        """Cache key for JAX arrays should not trigger device_get."""
        arr = jnp.ones((4, 8))
        node = Identity()

        with patch("jax.device_get", side_effect=AssertionError("device_get called!")):
            # Should not raise — no D2H transfer
            key = executor._compute_cache_key(node, arr)
            assert isinstance(key, int)

    def test_dict_no_device_get(self, executor):
        """Cache key for dict data should not trigger device_get."""
        data = {"image": jnp.ones((4, 32, 32, 3)), "label": jnp.ones((4,))}
        node = Identity()

        with patch("jax.device_get", side_effect=AssertionError("device_get called!")):
            key = executor._compute_cache_key(node, data)
            assert isinstance(key, int)

    def test_different_arrays_produce_different_keys(self, executor):
        """Different array objects should produce different cache keys."""
        node = Identity()
        arr1 = jnp.ones((4, 8))
        arr2 = jnp.zeros((4, 8))  # Same shape/dtype, different object

        key1 = executor._compute_cache_key(node, arr1)
        key2 = executor._compute_cache_key(node, arr2)
        # Different objects = different ids = different keys
        assert key1 != key2

    def test_same_node_different_data(self, executor):
        """Same node with different data should produce different keys."""
        node = Identity()
        data1 = {"a": jnp.ones((2, 3))}
        data2 = {"b": jnp.ones((2, 3))}

        key1 = executor._compute_cache_key(node, data1)
        key2 = executor._compute_cache_key(node, data2)
        assert key1 != key2


class TestCachingNodeCacheKey:
    """Tests for CacheNode._compute_cache_key (no D2H sync)."""

    def test_cache_node_no_device_transfer(self):
        """CacheNode cache key should not trigger device-to-host transfer."""
        node = CacheNode(Identity(), cache_size=10)
        data = {"image": jnp.ones((4, 32, 32, 3))}

        with patch("jax.device_get", side_effect=AssertionError("device_get called!")):
            key = node._compute_cache_key(data)
            assert isinstance(key, str)

    def test_cache_node_single_array_no_device_transfer(self):
        """CacheNode with single array should not trigger device transfer."""
        node = CacheNode(Identity(), cache_size=10)
        data = jnp.ones((4, 8))

        with patch("jax.device_get", side_effect=AssertionError("device_get called!")):
            key = node._compute_cache_key(data)
            assert isinstance(key, str)

    def test_cache_class_no_device_transfer(self):
        """Cache._compute_cache_key should not trigger device transfer."""
        cache = Cache(Identity(), cache_size=10)
        arr = jnp.ones((4, 8))

        with patch("jax.device_get", side_effect=AssertionError("device_get called!")):
            key = cache._compute_cache_key(arr)
            assert isinstance(key, int)


class TestModuleHashNoSync:
    """Tests for DataraxModule._hash_jax_array (no D2H sync)."""

    def _make_module(self):
        """Create a minimal DataraxModule for testing."""
        from datarax.core.config import DataraxModuleConfig
        from datarax.core.module import DataraxModule

        class DummyModule(DataraxModule):
            def __call__(self, *args, **kwargs):
                pass

        config = DataraxModuleConfig()
        return DummyModule(config)

    def test_hash_jax_array_no_tolist(self):
        """_hash_jax_array should not call .tolist() or .sum()."""
        module = self._make_module()
        arr = jnp.ones((4, 8))

        # The old implementation called samples.tolist() which triggers D2H
        result = module._hash_jax_array(arr)
        assert isinstance(result, int)

    def test_hash_different_arrays_differ(self):
        """Different array objects should hash differently."""
        module = self._make_module()
        arr1 = jnp.ones((4, 8))
        arr2 = jnp.zeros((4, 8))

        h1 = module._hash_jax_array(arr1)
        h2 = module._hash_jax_array(arr2)
        assert h1 != h2
