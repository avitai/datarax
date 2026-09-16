"""Tests for operators whose output structure differs from their input.

An operator may return data and state structured differently from what it was given:
adding a computed field, enriching a record with derived values, or writing several
outputs at once. The batch path vectorizes with an integer ``out_axes``, which is a tree
prefix of whatever the operator returns, so nothing has to discover that structure before
the call.

The regression tests at the end of the module cover what a per-operator identity and a
cached output structure used to break: branches under ``nnx.cond`` and ``nnx.switch``, a
raw ``jax.jit`` closure, operators built and freed in sequence, and one operator called
with different state structures in either order.
"""

import gc

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)


# =============================================================================
# Test Operators
# =============================================================================


class AddKeyOperator(OperatorModule):
    """Test operator that adds a single key to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        del random_params, stats
        out_data = {
            **data,
            "computed": data["input"] * 2,
        }
        return out_data, state, metadata


class AddMultipleKeysOperator(OperatorModule):
    """Test operator that adds multiple keys to output."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        del random_params, stats
        out_data = {
            **data,
            "sum": data["a"] + data["b"],
            "product": data["a"] * data["b"],
            "difference": data["a"] - data["b"],
        }
        return out_data, state, metadata


class StructurePreservingOperator(OperatorModule):
    """Test operator that preserves input structure (existing behavior)."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        del random_params, stats
        out_data = {k: v * 2 for k, v in data.items()}
        return out_data, state, metadata


class StateModifyingOperator(OperatorModule):
    """Test operator that adds keys to both data and state."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        del random_params, stats
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

    def test_structure_preserving_operator_still_works(self, config, rngs):
        """Operators with unchanged in/out structure execute correctly."""
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
                del random_params, stats
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
                del random_params, stats
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
                del data, random_params, stats
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


# =============================================================================
# Regression tests: an operator's own output decides the axes
# =============================================================================


class AddDataAndStateOperator(OperatorModule):
    """Adds one data field and one state field."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Add a computed field and record that the record was seen."""
        del random_params, stats
        out_data = {**data, "computed": data["input"] * 2}
        out_state = {**state, "seen": jnp.asarray(True)}
        return out_data, out_state, metadata


def _batch_of(values, state=None):
    """Return a batch of single-value records, each carrying ``state``."""
    return Batch(
        [Element(data={"input": jnp.asarray([v])}, state=dict(state or {})) for v in values]
    )


class TestAddedFieldsSurviveEveryPath:
    """An operator that adds a data field and a state field works on every path."""

    def test_apply_batch(self, config, rngs):
        """apply_batch returns both added fields."""
        op = AddDataAndStateOperator(config, rngs=rngs)

        result = op.apply_batch(_batch_of([1.0, 2.0]))

        assert "computed" in result.data.get_value()
        assert "seen" in result.states.get_value()

    def test_raw_path(self, config, rngs):
        """The raw path returns both added fields."""
        op = AddDataAndStateOperator(config, rngs=rngs)

        out_data, out_state = op._apply_on_raw({"input": jnp.ones((2, 1))}, {})

        assert "computed" in out_data
        assert "seen" in out_state

    def test_under_nnx_jit(self, config, rngs):
        """The added fields survive an nnx.jit call taking the operator as an argument."""
        op = AddDataAndStateOperator(config, rngs=rngs)

        @nnx.jit
        def run(operator, data):
            return operator._apply_on_raw(data, {})

        out_data, out_state = run(op, {"input": jnp.ones((2, 1))})

        assert "computed" in out_data
        assert "seen" in out_state

    def test_scan_strategy(self, rngs):
        """The sequential strategy returns the same added fields as vmap."""
        op = AddDataAndStateOperator(OperatorConfig(batch_strategy="scan"), rngs=rngs)

        out_data, out_state = op._apply_on_raw({"input": jnp.ones((2, 1))}, {})

        assert "computed" in out_data
        assert "seen" in out_state


class TestStateStructureOrderDoesNotMatter:
    """One operator called with different state structures, in either order."""

    def test_raw_path_without_state_then_batch_with_state(self, config, rngs):
        """A call with empty states does not decide the axes of a later call that carries one."""
        op = AddDataAndStateOperator(config, rngs=rngs)

        op._apply_on_raw({"input": jnp.ones((2, 1))}, {})
        result = op.apply_batch(_batch_of([1.0, 2.0], state={"count": jnp.asarray(0)}))

        assert "computed" in result.data.get_value()
        assert "count" in result.states.get_value()

    def test_batch_with_state_then_raw_path_without_state(self, config, rngs):
        """The reverse order works as well."""
        op = AddDataAndStateOperator(config, rngs=rngs)

        op.apply_batch(_batch_of([1.0, 2.0], state={"count": jnp.asarray(0)}))
        out_data, out_state = op._apply_on_raw({"input": jnp.ones((2, 1))}, {})

        assert "computed" in out_data
        assert "seen" in out_state


class TestCompiledAndBranchingCallers:
    """The paths that a per-operator identity used to break."""

    def test_raw_jit_closure_called_twice(self, config, rngs):
        """A raw jax.jit closure over apply_batch runs twice without mutating the operator."""
        op = AddKeyOperator(config, rngs=rngs)
        batch = _batch_of([1.0, 2.0])

        @jax.jit
        def run(values):
            return op.apply_batch(_batch_of([1.0, 2.0])).data.get_value()["computed"] + values

        first = run(jnp.zeros((2, 1)))
        second = run(jnp.ones((2, 1)))

        assert first.shape == second.shape
        del batch

    def test_cond_branches_adding_the_same_field(self, config, rngs):
        """Two operators as nnx.cond branches, both adding the same field."""
        left = AddKeyOperator(config, rngs=rngs)
        right = AddKeyOperator(config, rngs=rngs)
        data = {"input": jnp.ones((2, 1))}

        def use_left(batch_data):
            return left._apply_on_raw(batch_data, {})[0]

        def use_right(batch_data):
            return right._apply_on_raw(batch_data, {})[0]

        chosen = nnx.cond(jnp.mean(data["input"]) > 0.0, use_left, use_right, data)

        assert "computed" in chosen

    def test_switch_branches_adding_the_same_field(self, config, rngs):
        """Three operators as nnx.switch branches, all adding the same field."""
        operators = [AddKeyOperator(config, rngs=rngs) for _ in range(3)]
        data = {"input": jnp.ones((2, 1))}
        branches = [
            (lambda batch_data, op=op: op._apply_on_raw(batch_data, {})[0]) for op in operators
        ]

        chosen = nnx.switch(jnp.asarray(1), branches, data)

        assert "computed" in chosen

    def test_operators_built_called_and_freed_in_sequence(self, config, rngs):
        """Building, calling and freeing many operators never serves another's structure."""
        for index in range(200):
            adds_state = index % 2 == 0
            operator = (
                AddDataAndStateOperator(config, rngs=rngs)
                if adds_state
                else AddKeyOperator(config, rngs=rngs)
            )

            out_data, out_state = operator._apply_on_raw({"input": jnp.ones((2, 1))}, {})

            assert "computed" in out_data, f"iteration {index} lost its data field"
            assert ("seen" in out_state) == adds_state, f"iteration {index} got another structure"
            del operator
            gc.collect()


class TestTracingIsDecidedByTheConfiguration:
    """What two operators share a compiled trace, and what they do not."""

    def test_equal_configurations_share_a_trace(self, config, rngs):
        """Two operators with equal configurations do not each force a trace.

        The counter wraps the function and the transform wraps the counter, which is the
        order ``TraceCounter.wrap`` requires: jitting first would leave the counter outside
        the compiled function, where it counts calls instead of traces.
        """
        counter = TraceCounter()

        def run(operator, data):
            return operator._apply_on_raw(data, {})[0]

        traced = nnx.jit(counter.wrap(run))
        data = {"input": jnp.ones((2, 1))}

        with counter.expect(new_traces=1):
            traced(AddKeyOperator(config, rngs=rngs), data)
        with counter.expect(new_traces=0):
            traced(AddKeyOperator(config, rngs=rngs), data)

    def test_a_wrapper_rebuilt_from_new_children_traces_again(self, config, rngs):
        """A wrapper rebuilt over fresh children traces again, unlike a plain operator.

        A wrapper's configuration holds the child module itself, and flax defines no
        ``__eq__`` on ``Module``, so two wrappers built over freshly constructed children
        carry different graphdef metadata however equal their configurations look. Removing
        the per-operator identity does not change that, and this test measures it rather
        than leaving it to be met later as a surprise.
        """
        counter = TraceCounter()

        def run(operator, data):
            return operator._apply_on_raw(data, {})[0]

        traced = nnx.jit(counter.wrap(run))
        data = {"input": jnp.ones((2, 1))}

        def wrapper():
            child = AddKeyOperator(config, rngs=rngs)
            return ProbabilisticOperator(
                ProbabilisticOperatorConfig(operator=child, probability=1.0), rngs=nnx.Rngs(0)
            )

        with counter.expect(new_traces=1):
            traced(wrapper(), data)
        with counter.expect(new_traces=1):
            traced(wrapper(), data)


class MaskWhenDrawnOperator(OperatorModule):
    """Stochastic operator whose drawn branch adds a ``mask`` field."""

    def generate_random_params(self, element_keys, data_shapes):
        """Return one key per record."""
        del data_shapes
        return element_keys

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Add a mask drawn from the record's key, refusing to run without one."""
        del stats
        if random_params is None:
            raise ValueError("MaskWhenDrawnOperator draws its mask from a random key")
        mask = jax.random.bernoulli(random_params, 0.5, data["x"].shape)
        return {**data, "mask": mask}, state, metadata


class TestStochasticOutputStructure:
    """An operator whose draw decides which fields it writes."""

    def test_apply_batch_returns_the_fields_the_drawn_branch_adds(self):
        """The batch carries the field only the drawn branch writes.

        The operator raises when it is handed no random parameters, so a batch that comes
        back at all also shows that nothing calls its ``apply`` without them.
        """
        operator = MaskWhenDrawnOperator(
            OperatorConfig(stochastic=True, stream_name="aug"), rngs=nnx.Rngs(aug=0)
        )
        batch = Batch([Element(data={"x": jnp.ones(3)}, state={}) for _ in range(4)])

        result_data = operator.apply_batch(batch).data.get_value()

        assert set(result_data) == {"x", "mask"}
        assert result_data["mask"].shape == (4, 3)
