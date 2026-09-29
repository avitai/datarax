"""Tests for Batch with JAX PyTree states.

This test suite validates that Batch correctly handles states as JAX-compatible PyTrees,
enabling efficient vmap operations and batch processing.

Test Categories:
1. State PyTree validation - ensure states are JAX-compatible
2. State stacking - stack element states like data
3. State extraction - get_element returns proper state
4. batch_ops.from_arrays with PyTree states
5. vmap compatibility - states can be vmapped
6. Edge cases - empty states, mixed types, nested PyTrees
"""

import jax
import jax.numpy as jnp
import pytest

from datarax.core import batch_ops
from datarax.core.element_batch import Element


class TestStatePyTreeValidation:
    """Test that states must be JAX-compatible PyTrees."""

    def test_element_with_jax_array_state(self):
        """Test Element with JAX array values in state."""
        elem = Element(
            data={"x": jnp.array([1.0, 2.0])},
            state={"count": jnp.array(0), "flag": jnp.array(True)},
        )

        assert isinstance(elem.state["count"], jax.Array)
        assert isinstance(elem.state["flag"], jax.Array)

    def test_element_with_python_primitive_state(self):
        """Test Element with Python primitives in state (valid pytree leaves)."""
        elem = Element(
            data={"x": jnp.array([1.0, 2.0])},
            state={"count": 0, "flag": True, "ratio": 0.5, "none_val": None},
        )

        # Python primitives are valid pytree leaves
        assert elem.state["count"] == 0
        assert elem.state["flag"] is True
        assert elem.state["ratio"] == 0.5
        assert elem.state["none_val"] is None

    def test_element_with_nested_pytree_state(self):
        """Test Element with nested PyTree structure in state."""
        elem = Element(
            data={"x": jnp.array([1.0, 2.0])},
            state={
                "counters": {"augment": jnp.array(0), "transform": jnp.array(1)},
                "flags": {"processed": jnp.array(True), "validated": jnp.array(False)},
            },
        )

        assert isinstance(elem.state["counters"]["augment"], jax.Array)
        assert isinstance(elem.state["flags"]["processed"], jax.Array)

    def test_empty_state_is_valid_pytree(self):
        """Test that empty dict {} is a valid PyTree."""
        elem = Element(
            data={"x": jnp.array([1.0, 2.0])},
            state={},  # Empty dict is valid PyTree
        )

        assert elem.state == {}
        # Verify it's a pytree
        leaves, treedef = jax.tree.flatten(elem.state)
        assert len(leaves) == 0


class TestBatchStatePyTreeStacking:
    """Test that Batch stacks element states as PyTrees (like data)."""

    def test_batch_stacks_simple_array_states(self):
        """Test batch construction stacks simple array states."""
        elements = [
            Element(
                data={"x": jnp.array([1.0])},
                state={"count": jnp.array(0)},
            ),
            Element(
                data={"x": jnp.array([2.0])},
                state={"count": jnp.array(1)},
            ),
            Element(
                data={"x": jnp.array([3.0])},
                state={"count": jnp.array(2)},
            ),
        ]

        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        # States should be stacked like data
        assert batch.batch_size == 3
        assert "count" in batch.states
        assert batch.states["count"].shape == (3,)  # type: ignore[reportCallIssue]
        assert jnp.array_equal(batch.states["count"], jnp.array([0, 1, 2]))  # type: ignore[reportCallIssue]

    def test_batch_stacks_multiple_state_fields(self):
        """Test batch construction stacks multiple state fields."""
        elements = [
            Element(
                data={"x": jnp.array([1.0])},
                state={"count": jnp.array(0), "flag": jnp.array(True)},
            ),
            Element(
                data={"x": jnp.array([2.0])},
                state={"count": jnp.array(1), "flag": jnp.array(False)},
            ),
            Element(
                data={"x": jnp.array([3.0])},
                state={"count": jnp.array(2), "flag": jnp.array(True)},
            ),
        ]

        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        assert "count" in batch.states
        assert "flag" in batch.states
        assert batch.states["count"].shape == (3,)  # type: ignore[reportCallIssue]
        assert batch.states["flag"].shape == (3,)  # type: ignore[reportCallIssue]
        assert jnp.array_equal(batch.states["count"], jnp.array([0, 1, 2]))  # type: ignore[reportCallIssue]
        assert jnp.array_equal(batch.states["flag"], jnp.array([True, False, True]))  # type: ignore[reportCallIssue]

    def test_batch_stacks_nested_pytree_states(self):
        """Test batch construction stacks nested PyTree states."""
        elements = [
            Element(
                data={"x": jnp.array([1.0])},
                state={
                    "counters": {"augment": jnp.array(0), "transform": jnp.array(1)},
                    "score": jnp.array(0.5),
                },
            ),
            Element(
                data={"x": jnp.array([2.0])},
                state={
                    "counters": {"augment": jnp.array(1), "transform": jnp.array(2)},
                    "score": jnp.array(0.7),
                },
            ),
        ]

        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        # Nested structure should be preserved
        assert "counters" in batch.states
        assert "score" in batch.states
        assert "augment" in batch.states["counters"]  # type: ignore[reportCallIssue]
        assert "transform" in batch.states["counters"]  # type: ignore[reportCallIssue]

        # Values should be stacked
        assert batch.states["counters"]["augment"].shape == (2,)  # type: ignore[reportCallIssue]
        assert batch.states["counters"]["transform"].shape == (2,)  # type: ignore[reportCallIssue]
        assert batch.states["score"].shape == (2,)  # type: ignore[reportCallIssue]

    def test_batch_stacks_python_primitive_states(self):
        """Test batch construction with Python primitives in states."""
        elements = [
            Element(
                data={"x": jnp.array([1.0])},
                state={"id": 100, "ratio": 0.5, "flag": True},
            ),
            Element(
                data={"x": jnp.array([2.0])},
                state={"id": 200, "ratio": 0.7, "flag": False},
            ),
        ]

        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        # Python primitives should be stacked as JAX arrays
        assert isinstance(batch.states["id"], jax.Array)  # type: ignore[reportCallIssue]
        assert isinstance(batch.states["ratio"], jax.Array)  # type: ignore[reportCallIssue]
        assert isinstance(batch.states["flag"], jax.Array)  # type: ignore[reportCallIssue]


class TestBatchFromPartsWithPyTreeStates:
    """Test batch_ops.from_arrays() with PyTree states."""

    def test_from_arrays_with_stacked_states(self):
        """Test creating batch from pre-stacked state PyTree."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((4, 3))},
            states={"count": jnp.array([0, 1, 2, 3]), "flag": jnp.array([True] * 4)},
        )

        assert batch.batch_size == 4
        assert "count" in batch.states
        assert "flag" in batch.states
        assert batch.states["count"].shape == (4,)  # type: ignore[reportCallIssue]

    def test_from_arrays_validates_state_batch_size(self):
        """States must share the data's record axis."""
        with pytest.raises(ValueError, match="record axis"):
            batch_ops.from_arrays({"x": jnp.ones((4, 3))}, states={"count": jnp.array([0, 1, 2])})

    def test_from_arrays_with_nested_state_pytree(self):
        """Test from_arrays with nested state structure."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((3, 2))},
            states={
                "counters": {"augment": jnp.array([0, 1, 2]), "transform": jnp.array([1, 2, 3])},
                "score": jnp.array([0.5, 0.7, 0.9]),
            },
        )

        assert batch.batch_size == 3
        assert batch.states["counters"]["augment"].shape == (3,)  # type: ignore[reportCallIssue]
        assert batch.states["score"].shape == (3,)  # type: ignore[reportCallIssue]


class TestBatchGetElementWithPyTreeStates:
    """Test get_element correctly extracts PyTree states."""

    def test_get_element_extracts_state_slice(self):
        """Test get_element returns correct state for index."""
        batch = batch_ops.from_arrays(
            {"x": jnp.array([[1.0], [2.0], [3.0]])},
            states={"count": jnp.array([10, 20, 30]), "flag": jnp.array([True, False, True])},
        )

        elem = batch_ops.element(batch, 1)

        # State should have single values (no batch dimension)
        assert elem.state["count"] == 20
        assert not elem.state["flag"]

    def test_get_element_handles_nested_state_pytree(self):
        """Test get_element with nested state structure."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((2, 3))},
            states={
                "counters": {"augment": jnp.array([0, 1]), "transform": jnp.array([10, 20])},
                "score": jnp.array([0.5, 0.7]),
            },
        )

        elem = batch_ops.element(batch, 0)

        assert elem.state["counters"]["augment"] == 0
        assert elem.state["counters"]["transform"] == 10
        assert elem.state["score"] == 0.5


class TestVmapCompatibilityWithPyTreeStates:
    """Test that PyTree states work correctly with jax.vmap."""

    def test_vmap_over_batch_data_and_states(self):
        """Test vmap can iterate over both data and states."""
        batch = batch_ops.from_arrays(
            {"x": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
            states={"count": jnp.array([0, 1, 2])},
        )

        def process_element(data_dict, state_dict):
            # Add count to x values
            return {"x": data_dict["x"] + state_dict["count"]}

        # vmap with per-key axes
        data_axes = {k: 0 for k in batch.data.keys()}  # type: ignore[reportAttributeAccessIssue]
        states_axes = {k: 0 for k in batch.states.keys()}  # type: ignore[reportAttributeAccessIssue]

        result = jax.vmap(process_element, in_axes=(data_axes, states_axes), out_axes=data_axes)(
            batch.data, batch.states
        )

        # Check results
        assert result["x"].shape == (3, 2)
        assert jnp.allclose(result["x"][0], jnp.array([1.0, 2.0]))  # + 0
        assert jnp.allclose(result["x"][1], jnp.array([4.0, 5.0]))  # + 1
        assert jnp.allclose(result["x"][2], jnp.array([7.0, 8.0]))  # + 2

    def test_vmap_modifies_states(self):
        """Test vmap can transform and return modified states."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((4, 2))}, states={"count": jnp.array([0, 1, 2, 3])}
        )

        def increment_count(data_dict, state_dict):
            new_state = {"count": state_dict["count"] + 1}
            return data_dict, new_state

        data_axes = {k: 0 for k in batch.data.keys()}  # type: ignore[reportAttributeAccessIssue]
        states_axes = {k: 0 for k in batch.states.keys()}  # type: ignore[reportAttributeAccessIssue]

        result_data, result_states = jax.vmap(
            increment_count,
            in_axes=(data_axes, states_axes),
            out_axes=(data_axes, states_axes),
        )(batch.data, batch.states)

        # States should be incremented
        assert jnp.array_equal(result_states["count"], jnp.array([1, 2, 3, 4]))

    def test_vmap_with_nested_state_pytree(self):
        """Test vmap with nested state structures."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((3, 2))},
            states={
                "counters": {"augment": jnp.array([0, 1, 2]), "transform": jnp.array([10, 20, 30])},
                "score": jnp.array([0.5, 0.7, 0.9]),
            },
        )

        def process(data_dict, state_dict):
            # Increment all counters
            new_state = {
                "counters": {
                    "augment": state_dict["counters"]["augment"] + 1,
                    "transform": state_dict["counters"]["transform"] + 1,
                },
                "score": state_dict["score"] * 2,
            }
            return data_dict, new_state

        data_axes = {k: 0 for k in batch.data.keys()}
        # For nested dict, use recursive dict specification
        states_axes = {
            "counters": {"augment": 0, "transform": 0},
            "score": 0,
        }

        _, result_states = jax.vmap(
            process, in_axes=(data_axes, states_axes), out_axes=(data_axes, states_axes)
        )(batch.data, batch.states)

        assert jnp.array_equal(result_states["counters"]["augment"], jnp.array([1, 2, 3]))
        assert jnp.array_equal(result_states["score"], jnp.array([1.0, 1.4, 1.8]))


class TestEdgeCasesWithPyTreeStates:
    """Test edge cases and error conditions."""

    def test_empty_state_batch(self):
        """Test batch with all empty states."""
        elements = [Element(data={"x": jnp.array([1.0])}, state={}) for _ in range(3)]

        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        assert batch.batch_size == 3
        assert batch.states == {}

    def test_inconsistent_state_keys_raises_error(self):
        """Test that inconsistent state keys across elements raise error."""
        elements = [
            Element(data={"x": jnp.array([1.0])}, state={"count": jnp.array(0)}),
            Element(data={"x": jnp.array([2.0])}, state={"different_key": jnp.array(1)}),
        ]

        # Should raise error during stacking (mismatched structures)
        with pytest.raises((ValueError, KeyError, TypeError)):
            batch_ops.from_stacked(batch_ops.stack(elements))

    def test_batch_state_is_pytree_not_list(self):
        """Test that batch_state is a PyTree, not a list."""
        batch = batch_ops.from_arrays(
            {"x": jnp.ones((2, 3))},
            states={"count": jnp.array([0, 1])},
            batch_state={"total_count": jnp.array(1), "flag": jnp.array(True)},
        )

        # batch_state should be a PyTree (no batch dimension)
        assert isinstance(batch.batch_state, dict)
        assert "total_count" in batch.batch_state
        assert batch.batch_state["total_count"].shape == ()  # Scalar

    def test_from_arrays_with_mismatched_state_structure(self):
        """Test from_arrays with inconsistent state structure fails."""
        with pytest.raises((ValueError, TypeError)):
            batch_ops.from_arrays(
                {"x": jnp.ones((3, 2))},
                states={
                    "field1": jnp.array([0, 1, 2]),
                    "field2": jnp.array([0, 1]),  # Wrong size!
                },
            )
