"""Unit tests for merging strategies."""

import jax.numpy as jnp
from datarax.operators.strategies.merging import merge_outputs, merge_outputs_conditional


class TestMergeOutputs:
    def test_merge_concat(self):
        outputs = [jnp.array([1, 2]), jnp.array([3, 4])]
        merged = merge_outputs(outputs, "concat", merge_axis=0)
        assert jnp.array_equal(merged, jnp.array([1, 2, 3, 4]))

    def test_merge_stack(self):
        outputs = [jnp.array([1, 2]), jnp.array([3, 4])]
        merged = merge_outputs(outputs, "stack", merge_axis=0)
        expected = jnp.array([[1, 2], [3, 4]])
        assert jnp.array_equal(merged, expected)

    def test_merge_sum(self):
        outputs = [jnp.array([1, 2]), jnp.array([3, 4])]
        merged = merge_outputs(outputs, "sum")
        assert jnp.array_equal(merged, jnp.array([4, 6]))

    def test_merge_mean(self):
        outputs = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
        merged = merge_outputs(outputs, "mean")
        assert jnp.array_equal(merged, jnp.array([2.0, 3.0]))

    def test_merge_dict(self):
        outputs = [jnp.array([1]), jnp.array([2])]
        merged = merge_outputs(outputs, "dict")
        assert isinstance(merged, dict)
        assert jnp.array_equal(merged["operator_0"], jnp.array([1]))
        assert jnp.array_equal(merged["operator_1"], jnp.array([2]))

    def test_merge_dict_pytree(self):
        # Test with PyTree structure
        outputs = [
            {"a": jnp.array([1]), "b": jnp.array([10])},
            {"a": jnp.array([2]), "b": jnp.array([20])},
        ]
        merged = merge_outputs(outputs, "dict")

        # Structure should be preserved (keys 'a' and 'b')
        assert "a" in merged
        assert "b" in merged

        # Each leaf should be a dict of operator outputs
        assert isinstance(merged["a"], dict)
        assert jnp.array_equal(merged["a"]["operator_0"], jnp.array([1]))
        assert jnp.array_equal(merged["a"]["operator_1"], jnp.array([2]))

        assert isinstance(merged["b"], dict)
        assert jnp.array_equal(merged["b"]["operator_0"], jnp.array([10]))
        assert jnp.array_equal(merged["b"]["operator_1"], jnp.array([20]))

    def test_custom_merge_fn(self):
        outputs = [jnp.array([1]), jnp.array([2])]

        def custom_fn(outs):
            return outs[0] * 10 + outs[1]

        merged = merge_outputs(outputs, None, merge_fn=custom_fn)
        assert jnp.array_equal(merged, jnp.array([12]))


class TestMergeOutputsConditional:
    def test_merge_conditional_sum(self):
        outputs = [jnp.array([1, 2]), jnp.array([3, 4])]  # Assume identity returned for False
        conditions = [jnp.array(True), jnp.array(False)]

        # In actual conditional execution, if condition is False, the operator output
        # depends on the 'noop_fn'. In ConditionalParallelStrategy, noop_fn returns original input.
        # But merge_outputs_conditional takes whatever valid outputs are passed.
        # It masks them based on conditions.

        merged = merge_outputs_conditional(outputs, conditions, "sum")

        # Output 2 is masked (False condition)
        # So sum is just output 1: [1, 2]
        # BUT wait, the mask multiplies the stack.
        # Stack: [[1, 2], [3, 4]]
        # Mask: [1, 0] broadcasting to [[1, 1], [0, 0]]
        # Product: [[1, 2], [0, 0]]
        # Sum: [1, 2]
        assert jnp.array_equal(merged, jnp.array([1, 2]))

    def test_merge_conditional_mean(self):
        outputs = [jnp.array([10.0]), jnp.array([20.0]), jnp.array([30.0])]
        conditions = [jnp.array(True), jnp.array(False), jnp.array(True)]

        # True count is 2.
        # Masked values: 10, 0, 30
        # Sum: 40
        # Mean: 40 / 2 = 20

        merged = merge_outputs_conditional(outputs, conditions, "mean")
        assert jnp.array_equal(merged, jnp.array([20.0]))

    def test_merge_conditional_no_true_conditions(self):
        outputs = [jnp.array([1.0]), jnp.array([2.0])]
        conditions = [jnp.array(False), jnp.array(False)]

        # Sum should be 0
        merged_sum = merge_outputs_conditional(outputs, conditions, "sum")
        assert jnp.array_equal(merged_sum, jnp.array([0.0]))

        # Mean should be 0 (avoid div by zero)
        merged_mean = merge_outputs_conditional(outputs, conditions, "mean")
        assert jnp.array_equal(merged_mean, jnp.array([0.0]))
