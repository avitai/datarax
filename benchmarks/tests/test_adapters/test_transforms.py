"""Tests for shared adapter transform utilities.

TDD: Write tests first per Core Principle #2.
Verifies array-level transforms used by Grain and Datarax adapters.
"""

import numpy as np
import pytest

from benchmarks.adapters._utils import (
    apply_to_dict,
    cast_to_float32,
    normalize_uint8,
)


class TestNormalizeUint8:
    """Tests for normalize_uint8 array-level transform."""

    def test_converts_uint8_to_float32(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_uint8(arr)
        assert result.dtype == np.float32

    def test_scales_to_zero_one_range(self):
        arr = np.array([0, 255], dtype=np.uint8)
        result = normalize_uint8(arr)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_passthrough_non_uint8(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = normalize_uint8(arr)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_works_with_jax_arrays(self):
        jnp = pytest.importorskip("jax.numpy")
        arr = jnp.array([0, 128, 255], dtype=jnp.uint8)
        result = normalize_uint8(arr)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(result, [0.0, 128 / 255.0, 1.0], atol=1e-6)

    def test_jit_safe_with_tree_map(self):
        """Verify normalize_uint8 works inside jax.jit via tree.map — zero overhead."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        @jax.jit
        def apply_normalize(x):
            return jax.tree.map(normalize_uint8, x)

        data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.uint8) * 128}
        result = apply_normalize(data)
        assert result["image"].dtype == jnp.float32
        np.testing.assert_allclose(result["image"][0, 0, 0, 0], 128 / 255.0, atol=1e-6)


class TestCastToFloat32:
    """Tests for cast_to_float32 array-level transform."""

    def test_casts_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = cast_to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_noop_on_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = cast_to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr)

    def test_works_with_jax_arrays(self):
        jnp = pytest.importorskip("jax.numpy")
        arr = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = cast_to_float32(arr)
        assert result.dtype == jnp.float32

    def test_jit_safe_with_tree_map(self):
        """Verify cast_to_float32 works inside jax.jit via tree.map — zero overhead."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        @jax.jit
        def apply_cast(x):
            return jax.tree.map(cast_to_float32, x)

        data = {"tokens": jnp.array([1, 2, 3], dtype=jnp.int32)}
        result = apply_cast(data)
        assert result["tokens"].dtype == jnp.float32


class TestApplyToDict:
    """Tests for apply_to_dict dict-level wrapper."""

    def test_applies_fn_to_all_values(self):
        d = {"a": np.array([1, 2], dtype=np.int32), "b": np.array([3, 4], dtype=np.int32)}
        result = apply_to_dict(cast_to_float32, d)
        assert all(v.dtype == np.float32 for v in result.values())

    def test_preserves_keys(self):
        d = {"x": np.array([1]), "y": np.array([2])}
        result = apply_to_dict(cast_to_float32, d)
        assert set(result.keys()) == {"x", "y"}

    def test_returns_new_dict(self):
        d = {"a": np.array([1])}
        result = apply_to_dict(cast_to_float32, d)
        assert result is not d
