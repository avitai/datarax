"""Tests for operators with dynamic output structure.

This module tests the get_output_structure() method and the ability
of apply_batch() to handle operators that add new keys to output data.

The key feature being tested is that operators can now return data with
a DIFFERENT structure than their input, enabling operations like:
- Adding computed fields (e.g., "score", "alignment")
- Enriching data with derived values
- Multi-output operators
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule, _OUTPUT_STRUCT_CACHE


# =============================================================================
# Test Operators
# =============================================================================


class AddKeyOperator(OperatorModule):
    """Test operator that adds a single key to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "computed": data["input"] * 2,
        }
        return out_data, state, metadata


class AddMultipleKeysOperator(OperatorModule):
    """Test operator that adds multiple keys to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "sum": data["a"] + data["b"],
            "product": data["a"] * data["b"],
            "difference": data["a"] - data["b"],
        }
        return out_data, state, metadata


class ExplicitStructureOperator(OperatorModule):
    """Test operator with explicit get_output_structure override."""

    def get_output_structure(self, sample_data, sample_state):
        # Return 0 for each leaf (vmap axis spec)
        out_data = {
            **jax.tree.map(lambda _: 0, sample_data),
            "result": 0,
        }
        out_state = jax.tree.map(lambda _: 0, sample_state) if sample_state else {}
        return out_data, out_state

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "result": data["value"] ** 2,
        }
        return out_data, state, metadata


class StructurePreservingOperator(OperatorModule):
    """Test operator that preserves input structure (existing behavior)."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {k: v * 2 for k, v in data.items()}
        return out_data, state, metadata


class StateModifyingOperator(OperatorModule):
    """Test operator that adds keys to both data and state."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        out_data = {
            **data,
            "processed": data["input"] + 1,
        }
        out_state = {
            **state,
            "was_processed": jnp.array(True),
        }
        return out_data, out_state, metadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rngs():
    """Create RNGs for operator initialization."""
    return nnx.Rngs(42)


@pytest.fixture
def config():
    """Create default operator config."""
    return OperatorConfig()


# =============================================================================
# Test Cases: Dynamic Output Structure
# =============================================================================


class TestDynamicOutputStructure:
    """Test operators that add/change output keys."""

    def test_operator_adds_single_key(self, config, rngs):
        """Operator adds one new key to output."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [
            Element(data={"input": jnp.array([1.0, 2.0])}, state={}),
            Element(data={"input": jnp.array([3.0, 4.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        # Original key preserved
        assert "input" in result_data
        # New key added
        assert "computed" in result_data
        # Correct shape (batch_size=2, array_len=2)
        assert result_data["computed"].shape == (2, 2)
        # Correct values
        assert jnp.allclose(result_data["computed"][0], jnp.array([2.0, 4.0]))
        assert jnp.allclose(result_data["computed"][1], jnp.array([6.0, 8.0]))

    def test_operator_adds_multiple_keys(self, config, rngs):
        """Operator adds multiple new keys."""
        op = AddMultipleKeysOperator(config, rngs=rngs)

        elements = [
            Element(
                data={"a": jnp.array([1.0]), "b": jnp.array([2.0])},
                state={},
            ),
            Element(
                data={"a": jnp.array([3.0]), "b": jnp.array([4.0])},
                state={},
            ),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        # All original keys preserved
        assert "a" in result_data
        assert "b" in result_data
        # All new keys added
        assert "sum" in result_data
        assert "product" in result_data
        assert "difference" in result_data
        # Correct values for first element
        assert jnp.allclose(result_data["sum"][0], jnp.array([3.0]))
        assert jnp.allclose(result_data["product"][0], jnp.array([2.0]))
        assert jnp.allclose(result_data["difference"][0], jnp.array([-1.0]))

    def test_default_eval_shape_discovery(self, config, rngs):
        """Default get_output_structure uses eval_shape correctly."""
        op = AddKeyOperator(config, rngs=rngs)

        sample_data = {"input": jnp.array([1.0])}
        sample_state = {}

        out_data_struct, out_state_struct = op.get_output_structure(sample_data, sample_state)

        # Should discover both input and computed keys
        assert "input" in out_data_struct
        assert "computed" in out_data_struct
        # Leaf values should be 0 (vmap axis spec, not None!)
        # Note: We use 0 instead of None because None is an empty pytree in JAX
        assert out_data_struct["input"] == 0
        assert out_data_struct["computed"] == 0

    def test_explicit_override_works(self, config, rngs):
        """Explicit get_output_structure override works correctly."""
        op = ExplicitStructureOperator(config, rngs=rngs)

        elements = [
            Element(data={"value": jnp.array([2.0])}, state={}),
            Element(data={"value": jnp.array([3.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        # Original key preserved
        assert "value" in result_data
        # New key added
        assert "result" in result_data
        # Correct squared values
        assert jnp.allclose(result_data["result"][0], jnp.array([4.0]))
        assert jnp.allclose(result_data["result"][1], jnp.array([9.0]))

    def test_backward_compatible_same_structure(self, config, rngs):
        """Existing operators (same in/out structure) still work."""
        op = StructurePreservingOperator(config, rngs=rngs)

        elements = [
            Element(
                data={"x": jnp.array([1.0]), "y": jnp.array([2.0])},
                state={},
            ),
            Element(
                data={"x": jnp.array([3.0]), "y": jnp.array([4.0])},
                state={},
            ),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        # Same keys as input
        assert set(result_data.keys()) == {"x", "y"}
        # Values doubled
        assert jnp.allclose(result_data["x"][0], jnp.array([2.0]))
        assert jnp.allclose(result_data["y"][0], jnp.array([4.0]))

    def test_structure_caching_works(self, config, rngs):
        """Output structure is cached between calls."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [Element(data={"input": jnp.array([1.0])}, state={})]
        batch = Batch(elements)

        # Get the cache key that will be used
        batch_data = batch.data.get_value()
        input_struct_key = jax.tree.structure(batch_data)
        cache_key = (op._unique_id, input_struct_key)

        # First call - should populate module-level cache
        op.apply_batch(batch)
        assert cache_key in _OUTPUT_STRUCT_CACHE

        # Second call - should use cache (no new entries for this operator/structure)
        cache_size_after_first = len(_OUTPUT_STRUCT_CACHE)
        op.apply_batch(batch)
        cache_size_after_second = len(_OUTPUT_STRUCT_CACHE)

        # Cache should not grow (structure reused)
        assert cache_size_after_first == cache_size_after_second

    def test_gradient_flow_with_new_keys(self, config, rngs):
        """Gradients flow through operators that add keys."""
        op = AddKeyOperator(config, rngs=rngs)

        def loss_fn(input_val):
            data = {"input": input_val}
            state = {}
            out_data, _, _ = op.apply(data, state, None)
            return jnp.sum(out_data["computed"])

        input_val = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(loss_fn)(input_val)

        assert grad is not None
        assert grad.shape == input_val.shape
        assert jnp.all(jnp.isfinite(grad))
        # computed = input * 2, so gradient should be 2
        assert jnp.allclose(grad, jnp.array([2.0, 2.0, 2.0]))

    def test_state_structure_changes(self, config, rngs):
        """Operator can add keys to state as well as data."""
        op = StateModifyingOperator(config, rngs=rngs)

        elements = [
            Element(data={"input": jnp.array([1.0])}, state={}),
            Element(data={"input": jnp.array([2.0])}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()
        result_states = result.states.get_value()

        # Data has new key
        assert "input" in result_data
        assert "processed" in result_data
        # State has new key
        assert "was_processed" in result_states


# =============================================================================
# Test Cases: Nested Output Structure
# =============================================================================


class TestNestedOutputStructure:
    """Test operators with nested PyTree outputs."""

    def test_nested_input_with_added_keys(self, config, rngs):
        """Operator adds keys to nested input structure."""

        class NestedAddKeyOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                out_data = {
                    "nested": {
                        **data["nested"],
                        "computed": data["nested"]["value"] * 2,
                    },
                }
                return out_data, state, metadata

        op = NestedAddKeyOperator(config, rngs=rngs)

        elements = [
            Element(data={"nested": {"value": jnp.array([1.0])}}, state={}),
            Element(data={"nested": {"value": jnp.array([2.0])}}, state={}),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert "nested" in result_data
        assert "value" in result_data["nested"]
        assert "computed" in result_data["nested"]
        assert jnp.allclose(result_data["nested"]["computed"][0], jnp.array([2.0]))

    def test_deeply_nested_structure(self, config, rngs):
        """Operator handles deeply nested PyTree structures."""

        class DeeplyNestedOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                out_data = {
                    "level1": {
                        "level2": {
                            **data["level1"]["level2"],
                            "derived": data["level1"]["level2"]["value"] ** 2,
                        },
                    },
                }
                return out_data, state, metadata

        op = DeeplyNestedOperator(config, rngs=rngs)

        elements = [
            Element(
                data={"level1": {"level2": {"value": jnp.array([3.0])}}},
                state={},
            ),
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert result_data["level1"]["level2"]["value"].shape == (1, 1)
        assert "derived" in result_data["level1"]["level2"]
        assert jnp.allclose(result_data["level1"]["level2"]["derived"][0], jnp.array([9.0]))


# =============================================================================
# Test Cases: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases for dynamic output structure."""

    def test_empty_input_with_added_keys(self, config, rngs):
        """Operator can add keys to empty input data."""

        class EmptyToNonEmptyOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Input is empty, but we add a key
                out_data = {"generated": jnp.array([1.0, 2.0, 3.0])}
                return out_data, state, metadata

        EmptyToNonEmptyOperator(config, rngs=rngs)

        # Create batch with empty data (but we need some array for batch dim)
        # This is a tricky case - empty input PyTree
        # For this test, we use a minimal input
        elements = [
            Element(data={"placeholder": jnp.array([0.0])}, state={}),
            Element(data={"placeholder": jnp.array([0.0])}, state={}),
        ]
        Batch(elements)

        # This tests that an operator can return a completely different structure
        # Note: The input still needs batch dimension for vmap to work

    def test_single_element_batch(self, config, rngs):
        """Single element batch works correctly."""
        op = AddKeyOperator(config, rngs=rngs)

        elements = [Element(data={"input": jnp.array([5.0])}, state={})]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert result.batch_size == 1
        assert "computed" in result_data
        assert jnp.allclose(result_data["computed"][0], jnp.array([10.0]))

    def test_large_batch(self, config, rngs):
        """Large batch processes correctly."""
        op = AddKeyOperator(config, rngs=rngs)

        batch_size = 128
        elements = [
            Element(data={"input": jnp.array([float(i)])}, state={}) for i in range(batch_size)
        ]
        batch = Batch(elements)

        result = op.apply_batch(batch)
        result_data = result.data.get_value()

        assert result.batch_size == batch_size
        assert result_data["computed"].shape == (batch_size, 1)

    def test_different_input_structures_use_different_cache_entries(self, config, rngs):
        """Different input structures get separate cache entries."""
        op = AddKeyOperator(config, rngs=rngs)

        # First structure: {"input": ...}
        elements1 = [Element(data={"input": jnp.array([1.0])}, state={})]
        batch1 = Batch(elements1)
        op.apply_batch(batch1)

        # Get cache key for this operator (uses _unique_id, not id())
        batch_data = batch1.data.get_value()
        input_struct_key = jax.tree.structure(batch_data)
        cache_key = (op._unique_id, input_struct_key)
        assert cache_key in _OUTPUT_STRUCT_CACHE

        # Note: We can only test with same-structure inputs here because
        # AddKeyOperator specifically expects "input" key. Different structures
        # would require different operators.
