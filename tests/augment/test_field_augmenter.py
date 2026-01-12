"""Tests for field-specific operations using MapOperator (subtree mode).

This file tests MapOperator configured for field-specific transformations
that target specific data fields.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

from datarax.operators import MapOperator
from datarax.core.config import MapOperatorConfig
from datarax.core.element_batch import create_batch_from_arrays


class TestFieldOperations:
    """Tests for MapOperator in subtree (field-specific) mode."""

    def test_init_basic_deterministic(self):
        """Test basic initialization for deterministic field operation."""

        def simple_fn(data, key):
            return data

        config = MapOperatorConfig(
            subtree={"test_field": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=simple_fn)

        # MapOperator stores config, not individual fields
        assert operator.config.subtree == {"test_field": None}
        assert operator.fn == simple_fn
        assert operator.rngs is None
        assert not operator.stochastic

    def test_init_stochastic_with_rngs(self):
        """Test initialization for stochastic field operation."""

        def simple_fn(data, key):
            return data

        rngs = nnx.Rngs({"custom": 42})
        config = MapOperatorConfig(
            subtree={"field": None},
            stochastic=True,
            stream_name="custom",
        )
        operator = MapOperator(config, fn=simple_fn, rngs=rngs, name="TestOperator")

        assert operator.config.subtree == {"field": None}
        assert operator.config.stream_name == "custom"
        assert operator.rngs is rngs
        assert operator.stochastic
        assert operator.name == "TestOperator"

    def test_transform_existing_field(self):
        """Test transformation of existing field."""

        def add_one(data, key):
            return data + 1

        config = MapOperatorConfig(
            subtree={"value": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=add_one)

        batch = create_batch_from_arrays(
            {"value": jnp.array([1.0, 2.0]), "other": jnp.array([10.0, 20.0])}
        )

        result = operator(batch)
        result_data = result.get_data()

        np.testing.assert_array_equal(result_data["value"], [2.0, 3.0])
        np.testing.assert_array_equal(result_data["other"], [10.0, 20.0])  # Unchanged

    def test_transform_multiple_fields(self):
        """Test transformation of multiple fields simultaneously."""

        def multiply_by_two(data, key):
            return data * 2

        config = MapOperatorConfig(
            subtree={"value": None, "other": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=multiply_by_two)

        batch = create_batch_from_arrays(
            {"value": jnp.array([1.0, 2.0]), "other": jnp.array([10.0, 20.0])}
        )

        result = operator(batch)
        result_data = result.get_data()

        np.testing.assert_array_equal(result_data["value"], [2.0, 4.0])
        np.testing.assert_array_equal(result_data["other"], [20.0, 40.0])

    def test_transform_missing_field(self):
        """Test transformation when field doesn't exist (passes through)."""

        def add_one(data, key):
            return data + 1

        config = MapOperatorConfig(
            subtree={"missing": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=add_one)

        batch = create_batch_from_arrays({"value": jnp.array([1.0, 2.0])})

        result = operator(batch)
        result_data = result.get_data()

        # Should be unchanged since field doesn't exist
        np.testing.assert_array_equal(result_data["value"], [1.0, 2.0])
        assert "missing" not in result_data

    def test_stochastic_with_randomness(self):
        """Test stochastic field operation with actual randomness."""

        def add_noise(data, key):
            noise = jax.random.normal(key, data.shape) * 0.1
            return data + noise

        rngs = nnx.Rngs({"augment": 42})
        config = MapOperatorConfig(
            subtree={"value": None},
            stochastic=True,
            stream_name="augment",
        )
        operator = MapOperator(config, fn=add_noise, rngs=rngs)

        batch = create_batch_from_arrays({"value": jnp.array([1.0, 2.0])})

        result = operator(batch)
        result_data = result.get_data()

        # Results should be different from input due to noise
        assert not np.allclose(result_data["value"], [1.0, 2.0])

    def test_deterministic_reproducibility(self):
        """Test that deterministic operations are reproducible."""

        def scale(data, key):
            return data * 1.5

        config = MapOperatorConfig(
            subtree={"value": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=scale)

        batch = create_batch_from_arrays({"value": jnp.array([1.0, 2.0, 3.0])})

        result1 = operator(batch)
        result2 = operator(batch)

        # Deterministic operations should give same results
        np.testing.assert_array_equal(result1.get_data()["value"], result2.get_data()["value"])

    def test_with_batch_dimension(self):
        """Test operator with batch dimension."""

        def augment_fn(data, key):
            return data + 1

        config = MapOperatorConfig(
            subtree={"data": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=augment_fn)

        batch = create_batch_from_arrays(
            {"data": jnp.ones((4, 3))}
        )  # Batch of 4, each with 3 features
        result = operator(batch)

        assert result.batch_size == 4
        assert jnp.allclose(result.get_data()["data"], jnp.ones((4, 3)) + 1)

    def test_custom_stream_name(self):
        """Test operator with custom RNG stream name."""

        def augment_fn(data, key):
            noise = jax.random.normal(key, data.shape) * 0.01
            return data + noise

        rngs = nnx.Rngs(custom_stream=42)
        config = MapOperatorConfig(
            subtree={"data": None},
            stochastic=True,
            stream_name="custom_stream",
        )
        operator = MapOperator(config, fn=augment_fn, rngs=rngs)

        batch = create_batch_from_arrays({"data": jnp.array([1.0, 2.0, 3.0])})
        result = operator(batch)
        assert result.get_data()["data"].shape == batch.get_data()["data"].shape

    def test_nested_field_access(self):
        """Test that subtree mode filters by field name correctly."""

        def transform_fn(data, key):
            return data * 3

        config = MapOperatorConfig(
            subtree={"data": None},
            stochastic=False,
        )
        operator = MapOperator(config, fn=transform_fn)

        # Test that only 'data' field is transformed
        batch = create_batch_from_arrays(
            {
                "data": jnp.array([1.0, 2.0, 3.0]),
                "label": jnp.array([0, 1, 2]),
            }
        )

        result = operator(batch)
        result_data = result.get_data()

        np.testing.assert_array_equal(result_data["data"], [3.0, 6.0, 9.0])
        np.testing.assert_array_equal(result_data["label"], [0, 1, 2])  # Unchanged
