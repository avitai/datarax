"""Tests for MapOperator - unified operator for mapping functions over array leaves.

This test suite validates MapOperator which applies user-provided array transformation
functions (fn: Array, Array -> Array) to leaves in element.data PyTree.

BREAKING CHANGE: All functions now MUST accept key parameter: fn(x, key) -> x
- Deterministic mode: ignore key parameter
- Stochastic mode: use key for randomness

Test Categories:
1. Config validation (MapOperatorConfig)
2. Full-tree mode (subtree=None) - applies fn to all array leaves
3. Subtree mode (subtree specified) - applies fn only to specified leaves
4. Stochastic mode (stochastic=True) - applies fn with RNG keys
5. Edge cases (empty PyTree, nested structures, non-array leaves)
6. Integration with batch processing (vmap)
7. JIT compilation compatibility
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

# Import guard for TDD RED phase
try:
    from datarax.core.config import MapOperatorConfig
    from datarax.operators.map_operator import MapOperator
    from datarax.core.element_batch import Batch, Element
except ImportError:
    MapOperatorConfig = None
    MapOperator = None
    Batch = None
    Element = None


pytestmark = pytest.mark.skipif(
    MapOperator is None,
    reason="MapOperator not implemented yet (RED phase)",
)


# ========================================================================
# Test Helper: Batch Creation
# ========================================================================


def create_test_batch(data, states=None, metadata_list=None, batch_size=None):
    """Helper to create batch using Batch.from_parts() API."""
    if batch_size is None:
        first_array = jax.tree.leaves(data)[0]
        batch_size = first_array.shape[0]

    if states is None:
        states = {}
    if metadata_list is None:
        metadata_list = [None] * batch_size

    return Batch.from_parts(data, states, metadata_list, validate=False)


# ========================================================================
# Test Category 1: Config Validation
# ========================================================================


class TestMapOperatorConfig:
    """Test MapOperatorConfig validation."""

    def test_config_inheritance(self):
        """Config inherits from OperatorConfig."""
        from datarax.core.config import OperatorConfig

        config = MapOperatorConfig(stochastic=False)
        assert isinstance(config, OperatorConfig)

    def test_subtree_none_by_default(self):
        """Subtree defaults to None (full-tree mode)."""
        config = MapOperatorConfig(stochastic=False)
        assert config.subtree is None

    def test_subtree_can_be_dict(self):
        """Subtree can be a dict structure."""
        config = MapOperatorConfig(subtree={"image": None}, stochastic=False)
        assert config.subtree == {"image": None}

    def test_subtree_can_be_nested(self):
        """Subtree can be nested dict structure."""
        config = MapOperatorConfig(
            subtree={"features": {"image": None, "depth": None}}, stochastic=False
        )
        assert config.subtree == {"features": {"image": None, "depth": None}}

    def test_stochastic_mode_accepted(self):
        """stochastic=True is now supported (no longer raises NotImplementedError)."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment")

        def fn(x, key):
            return x

        rngs = nnx.Rngs(0, augment=1)

        # Should NOT raise - stochastic mode is implemented
        op = MapOperator(config, fn=fn, rngs=rngs)
        assert op.stochastic is True
        assert op.stream_name == "augment"


# ========================================================================
# Test Category 2: Path-Based Filtering Helper
# ========================================================================


class TestPathInSubtree:
    """Test _path_in_subtree static method independently."""

    def test_simple_path_match(self):
        """Path matches single-level dict with None leaf."""
        from jax.tree_util import DictKey

        subtree_mask = {"image": None}
        keypath = (DictKey(key="image"),)

        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is True

    def test_simple_path_mismatch(self):
        """Path doesn't match when key not in mask."""
        from jax.tree_util import DictKey

        subtree_mask = {"image": None}
        keypath = (DictKey(key="label"),)

        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is False

    def test_nested_path_match(self):
        """Path matches nested dict structure."""
        from jax.tree_util import DictKey

        subtree_mask = {"features": {"image": None, "depth": None}}
        keypath = (DictKey(key="features"), DictKey(key="image"))

        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is True

    def test_nested_path_partial_match(self):
        """Path fails when only partially matching nested structure."""
        from jax.tree_util import DictKey

        subtree_mask = {"features": {"image": None}}
        keypath = (DictKey(key="features"), DictKey(key="depth"))

        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is False

    def test_path_too_short(self):
        """Path fails when shorter than mask structure."""
        from jax.tree_util import DictKey

        subtree_mask = {"features": {"image": None}}
        keypath = (DictKey(key="features"),)

        # Path ends at dict, not None leaf
        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is False

    def test_path_too_long(self):
        """Path fails when longer than mask structure."""
        from jax.tree_util import DictKey

        subtree_mask = {"image": None}
        keypath = (DictKey(key="image"), DictKey(key="extra"))

        # Can't navigate past None leaf
        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is False

    def test_empty_keypath(self):
        """Empty keypath matches if mask is None."""
        keypath = ()

        # Root is None - matches
        result = MapOperator._path_in_subtree(keypath, None)
        assert result is True

        # Root is dict - doesn't match
        result = MapOperator._path_in_subtree(keypath, {"image": None})
        assert result is False

    def test_multiple_fields_in_mask(self):
        """Multiple fields in mask - only specified ones match."""
        from jax.tree_util import DictKey

        subtree_mask = {"image": None, "mask": None}

        # image matches
        keypath_image = (DictKey(key="image"),)
        assert MapOperator._path_in_subtree(keypath_image, subtree_mask) is True

        # mask matches
        keypath_mask = (DictKey(key="mask"),)
        assert MapOperator._path_in_subtree(keypath_mask, subtree_mask) is True

        # label doesn't match
        keypath_label = (DictKey(key="label"),)
        assert MapOperator._path_in_subtree(keypath_label, subtree_mask) is False

    def test_deeply_nested_structure(self):
        """Deep nesting works correctly."""
        from jax.tree_util import DictKey

        subtree_mask = {"a": {"b": {"c": {"d": None}}}}
        keypath = (
            DictKey(key="a"),
            DictKey(key="b"),
            DictKey(key="c"),
            DictKey(key="d"),
        )

        result = MapOperator._path_in_subtree(keypath, subtree_mask)
        assert result is True

    def test_mixed_nested_structure(self):
        """Mixed structure with some branches having None, others having nested dicts."""
        from jax.tree_util import DictKey

        subtree_mask = {"image": None, "features": {"depth": None, "normals": None}}

        # Direct field matches
        assert MapOperator._path_in_subtree((DictKey(key="image"),), subtree_mask) is True

        # Nested field matches
        assert (
            MapOperator._path_in_subtree(
                (DictKey(key="features"), DictKey(key="depth")), subtree_mask
            )
            is True
        )
        assert (
            MapOperator._path_in_subtree(
                (DictKey(key="features"), DictKey(key="normals")), subtree_mask
            )
            is True
        )

        # Non-existent nested field doesn't match
        assert (
            MapOperator._path_in_subtree(
                (DictKey(key="features"), DictKey(key="albedo")), subtree_mask
            )
            is False
        )

        # Non-existent top-level field doesn't match
        assert MapOperator._path_in_subtree((DictKey(key="label"),), subtree_mask) is False


# ========================================================================
# Test Category 3: Full-Tree Mode (subtree=None)
# ========================================================================


class TestMapOperatorFullTree:
    """Test MapOperator in full-tree mode (applies fn to all leaves)."""

    def test_apply_to_single_field(self):
        """Full-tree mode applies fn to single field."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        # Single element batch
        batch_data = {"image": jnp.array([[1.0, 2.0, 3.0]])}  # Shape (1, 3)
        batch = create_test_batch(batch_data)

        result_batch = op(batch)

        expected = jnp.array([[2.0, 4.0, 6.0]])
        assert jnp.allclose(result_batch.get_data()["image"], expected)
        # State and metadata preserved (empty in this case)
        assert result_batch.get_states() == {}
        assert result_batch._metadata_list[0] is None

    def test_apply_to_multiple_fields(self):
        """Full-tree mode applies fn to all fields."""

        def add_one(x, key):
            return x + 1.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=add_one, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0, 2.0]]), "mask": jnp.array([[0.0, 1.0]])}
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        assert jnp.allclose(result_data["image"], jnp.array([[2.0, 3.0]]))
        assert jnp.allclose(result_data["mask"], jnp.array([[1.0, 2.0]]))

    def test_apply_to_nested_structure(self):
        """Full-tree mode applies fn to nested structures."""

        def negate(x, key):
            return -x

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=negate, rngs=rngs)

        batch_data = {
            "features": {"image": jnp.array([[1.0, 2.0]]), "depth": jnp.array([[3.0, 4.0]])}
        }
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        assert jnp.allclose(result_data["features"]["image"], jnp.array([[-1.0, -2.0]]))
        assert jnp.allclose(result_data["features"]["depth"], jnp.array([[-3.0, -4.0]]))

    def test_batch_processing(self):
        """Full-tree mode works with batch processing."""

        def normalize(x, key):
            return (x - 0.5) / 0.5

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=normalize, rngs=rngs)

        # Batch of 3 elements
        batch_data = {"image": jnp.ones((3, 4, 4, 1))}
        batch = create_test_batch(batch_data)

        result = op(batch)

        expected = (jnp.ones((3, 4, 4, 1)) - 0.5) / 0.5
        assert jnp.allclose(result.data.get_value()["image"], expected)


# ========================================================================
# Test Category 3: Subtree Mode (subtree specified)
# ========================================================================


class TestMapOperatorSubtree:
    """Test MapOperator in subtree mode (selective transformation)."""

    def test_apply_to_single_field_only(self):
        """Subtree mode applies fn only to specified field."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree={"image": None}, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0, 2.0]]), "label": jnp.array([[3.0, 4.0]])}
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        # Only "image" should be doubled
        assert jnp.allclose(result_data["image"], jnp.array([[2.0, 4.0]]))
        # "label" should be unchanged
        assert jnp.allclose(result_data["label"], jnp.array([[3.0, 4.0]]))

    def test_apply_to_multiple_fields_only(self):
        """Subtree mode applies fn to multiple specified fields."""

        def add_ten(x, key):
            return x + 10.0

        config = MapOperatorConfig(subtree={"image": None, "mask": None}, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=add_ten, rngs=rngs)

        batch_data = {
            "image": jnp.array([[1.0]]),
            "mask": jnp.array([[2.0]]),
            "label": jnp.array([[3.0]]),
        }
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        # "image" and "mask" transformed
        assert jnp.allclose(result_data["image"], jnp.array([[11.0]]))
        assert jnp.allclose(result_data["mask"], jnp.array([[12.0]]))
        # "label" unchanged
        assert jnp.allclose(result_data["label"], jnp.array([[3.0]]))

    def test_apply_to_nested_subtree(self):
        """Subtree mode works with nested structures."""

        def square(x, key):
            return x**2

        config = MapOperatorConfig(subtree={"features": {"image": None}}, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=square, rngs=rngs)

        batch_data = {
            "features": {"image": jnp.array([[2.0, 3.0]]), "depth": jnp.array([[4.0, 5.0]])},
            "label": jnp.array([[6.0, 7.0]]),
        }
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        # Only features.image transformed
        assert jnp.allclose(result_data["features"]["image"], jnp.array([[4.0, 9.0]]))
        # features.depth unchanged
        assert jnp.allclose(result_data["features"]["depth"], jnp.array([[4.0, 5.0]]))
        # label unchanged
        assert jnp.allclose(result_data["label"], jnp.array([[6.0, 7.0]]))

    def test_subtree_preserves_data_structure(self):
        """Subtree mode preserves overall PyTree structure."""

        def identity(x, key):
            return x

        config = MapOperatorConfig(subtree={"a": None}, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=identity, rngs=rngs)

        batch_data = {"a": jnp.array([[1.0]]), "b": jnp.array([[2.0]]), "c": jnp.array([[3.0]])}
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        # Structure preserved
        assert set(result_data.keys()) == {"a", "b", "c"}


# ========================================================================
# Test Category 4: Stochastic Mode (NEW)
# ========================================================================


class TestMapOperatorStochastic:
    """Test MapOperator in stochastic mode with RNG key generation."""

    def test_generate_random_params_structure(self):
        """generate_random_params returns PyTree matching data structure."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=1)

        def add_noise(x, key):
            return x + jax.random.normal(key, x.shape) * 0.1

        op = MapOperator(config, fn=add_noise, rngs=rngs)

        # Create data shapes
        batch_size = 4
        data_shapes = {
            "image": (batch_size, 32, 32, 3),
            "mask": (batch_size, 32, 32, 1),
        }

        # Generate random params
        rng = jax.random.PRNGKey(42)
        random_params = op.generate_random_params(rng, data_shapes)

        # Verify structure matches
        assert set(random_params.keys()) == {"image", "mask"}
        # Verify each leaf is array of keys (one per batch element)
        assert random_params["image"].shape == (batch_size, 2)  # PRNGKey shape is (2,)
        assert random_params["mask"].shape == (batch_size, 2)

    def test_generate_random_params_batch_size_extraction(self):
        """generate_random_params correctly extracts batch size from shapes."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=1)

        def fn(x, key):
            return x

        op = MapOperator(config, fn=fn, rngs=rngs)

        # Different batch sizes
        for batch_size in [1, 2, 8, 16]:
            data_shapes = {"data": (batch_size, 10)}
            rng = jax.random.PRNGKey(0)
            random_params = op.generate_random_params(rng, data_shapes)
            assert random_params["data"].shape == (batch_size, 2)

    def test_stochastic_full_tree_adds_noise(self):
        """Stochastic mode adds randomness in full-tree mode."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment", subtree=None)
        rngs = nnx.Rngs(0, augment=1)

        def add_noise(x, key):
            noise = jax.random.normal(key, x.shape) * 0.1
            return x + noise

        op = MapOperator(config, fn=add_noise, rngs=rngs)

        # Create batch
        batch_data = {"image": jnp.ones((3, 4, 4, 1))}
        batch = create_test_batch(batch_data)

        # Apply operator
        result = op(batch)
        result_data = result.get_data()

        # Output should differ from input (noise added)
        assert not jnp.allclose(result_data["image"], jnp.ones((3, 4, 4, 1)))
        # But should be close (noise is small)
        assert jnp.allclose(result_data["image"], jnp.ones((3, 4, 4, 1)), atol=0.5)

    def test_stochastic_subtree_selective(self):
        """Stochastic mode only affects specified subtree fields."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment", subtree={"image": None})
        rngs = nnx.Rngs(0, augment=1)

        def add_noise(x, key):
            noise = jax.random.normal(key, x.shape) * 0.1
            return x + noise

        op = MapOperator(config, fn=add_noise, rngs=rngs)

        # Batch with multiple fields
        original_image = jnp.ones((2, 3, 3, 1))
        original_label = jnp.array([[0.0], [1.0]])
        batch_data = {"image": original_image, "label": original_label}
        batch = create_test_batch(batch_data)

        # Apply operator
        result = op(batch)
        result_data = result.get_data()

        # image should be modified (in subtree)
        assert not jnp.allclose(result_data["image"], original_image)
        # label should be unchanged (not in subtree)
        assert jnp.allclose(result_data["label"], original_label)

    def test_stochastic_reproducibility(self):
        """Same RNG seed produces same stochastic output."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment", subtree=None)

        def add_noise(x, key):
            return x + jax.random.normal(key, x.shape) * 0.1

        # Create two operators with same seed
        rngs1 = nnx.Rngs(0, augment=42)
        op1 = MapOperator(config, fn=add_noise, rngs=rngs1)

        rngs2 = nnx.Rngs(0, augment=42)
        op2 = MapOperator(config, fn=add_noise, rngs=rngs2)

        # Same input batch
        batch_data = {"image": jnp.ones((2, 4, 4, 1))}
        batch = create_test_batch(batch_data)

        # Apply both operators
        result1 = op1(batch)
        result2 = op2(batch)

        # Should produce identical outputs (same seed)
        assert jnp.allclose(result1.get_data()["image"], result2.get_data()["image"])

    def test_stochastic_different_seeds_differ(self):
        """Different RNG seeds produce different stochastic outputs."""
        config = MapOperatorConfig(stochastic=True, stream_name="augment", subtree=None)

        def add_noise(x, key):
            return x + jax.random.normal(key, x.shape) * 0.1

        # Create two operators with different seeds
        rngs1 = nnx.Rngs(0, augment=1)
        op1 = MapOperator(config, fn=add_noise, rngs=rngs1)

        rngs2 = nnx.Rngs(0, augment=999)
        op2 = MapOperator(config, fn=add_noise, rngs=rngs2)

        # Same input batch
        batch_data = {"image": jnp.ones((2, 4, 4, 1))}
        batch = create_test_batch(batch_data)

        # Apply both operators
        result1 = op1(batch)
        result2 = op2(batch)

        # Should produce different outputs (different seeds)
        assert not jnp.allclose(result1.get_data()["image"], result2.get_data()["image"])

    def test_deterministic_mode_ignores_key(self):
        """Deterministic mode ignores key parameter (always same output)."""
        config = MapOperatorConfig(stochastic=False, subtree=None)

        def multiply_by_two(x, key):
            # Ignore key - deterministic
            return x * 2.0

        # Create operators with different seeds (shouldn't matter)
        rngs1 = nnx.Rngs(0)
        op1 = MapOperator(config, fn=multiply_by_two, rngs=rngs1)

        rngs2 = nnx.Rngs(999)
        op2 = MapOperator(config, fn=multiply_by_two, rngs=rngs2)

        # Same input
        batch_data = {"image": jnp.ones((2, 3, 3, 1))}
        batch = create_test_batch(batch_data)

        # Both should produce identical deterministic outputs
        result1 = op1(batch)
        result2 = op2(batch)

        expected = jnp.ones((2, 3, 3, 1)) * 2.0
        assert jnp.allclose(result1.get_data()["image"], expected)
        assert jnp.allclose(result2.get_data()["image"], expected)


# ========================================================================
# Test Category 5: Edge Cases
# ========================================================================


class TestMapOperatorEdgeCases:
    """Test MapOperator edge cases."""

    def test_empty_data(self):
        """Handles empty data dict."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        # Create batch with empty data using Element constructor
        batch = Batch([Element(data={}, state={}, metadata=None)])

        result_batch = op(batch)

        assert result_batch.get_data() == {}
        assert result_batch.get_states() == {}
        assert result_batch._metadata_list == [None]

    def test_state_unchanged(self):
        """State PyTree is never modified."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0]])}
        batch_states = {"model_state": jnp.array([[100.0]])}
        batch = Batch.from_parts(batch_data, batch_states, [None], validate=False)

        result_batch = op(batch)

        # State completely unchanged
        result_states = result_batch.get_states()
        assert jnp.allclose(result_states["model_state"], jnp.array([[100.0]]))

    def test_metadata_unchanged(self):
        """Metadata dict is never modified."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0]])}
        metadata_list = [{"filename": "test.jpg", "index": 42}]
        batch = Batch.from_parts(batch_data, {}, metadata_list, validate=False)

        result_batch = op(batch)

        # Metadata completely unchanged
        assert result_batch._metadata_list[0] == {"filename": "test.jpg", "index": 42}

    def test_different_array_shapes(self):
        """Handles arrays of different shapes."""

        def add_one(x, key):
            return x + 1.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=add_one, rngs=rngs)

        batch_data = {
            "small": jnp.array([[1.0]]),
            "medium": jnp.array([[[2.0, 3.0]]]),
            "large": jnp.array([[[[4.0, 5.0], [6.0, 7.0]]]]),
        }
        batch = create_test_batch(batch_data)

        result_batch = op(batch)
        result_data = result_batch.get_data()

        assert jnp.allclose(result_data["small"], jnp.array([[2.0]]))
        assert jnp.allclose(result_data["medium"], jnp.array([[[3.0, 4.0]]]))
        assert jnp.allclose(result_data["large"], jnp.array([[[[5.0, 6.0], [7.0, 8.0]]]]))


# ========================================================================
# Test Category 6: JIT Compilation
# ========================================================================


class TestMapOperatorJIT:
    """Test MapOperator JIT compilation compatibility."""

    def test_jit_full_tree_mode(self):
        """JIT compilation works in full-tree mode."""

        def normalize(x, key):
            return (x - 0.5) / 0.5

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=normalize, rngs=rngs)

        batch_data = {"image": jnp.ones((2, 3, 3, 1))}
        batch = create_test_batch(batch_data)

        # JIT compile with nnx.jit (pass module as argument, not closure)
        @nnx.jit
        def jitted_apply(model, batch):
            return model(batch)

        # Transform
        result = jitted_apply(op, batch)

        # Verify
        expected = (jnp.ones((2, 3, 3, 1)) - 0.5) / 0.5
        assert jnp.allclose(result.data.get_value()["image"], expected)

    def test_jit_subtree_mode(self):
        """JIT compilation works in subtree mode."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree={"image": None}, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0], [2.0]]), "label": jnp.array([[3.0], [4.0]])}
        batch = create_test_batch(batch_data)

        # JIT compile with nnx.jit (pass module as argument, not closure)
        @nnx.jit
        def jitted_apply(model, batch):
            return model(batch)

        # Transform
        result = jitted_apply(op, batch)

        # Verify: only image doubled
        assert jnp.allclose(result.data.get_value()["image"], jnp.array([[2.0], [4.0]]))
        assert jnp.allclose(result.data.get_value()["label"], jnp.array([[3.0], [4.0]]))

    def test_jit_stochastic_mode(self):
        """JIT compilation works with stochastic operators and produces deterministic results."""

        def add_noise(x, key):
            return x + jax.random.normal(key, x.shape) * 0.1

        config = MapOperatorConfig(stochastic=True, stream_name="augment", subtree=None)

        batch_data = {"image": jnp.ones((2, 3, 3, 1))}
        batch = create_test_batch(batch_data)

        # Create two operators with same seed - should produce identical results
        rngs1 = nnx.Rngs(0, augment=42)
        op1 = MapOperator(config, fn=add_noise, rngs=rngs1)

        rngs2 = nnx.Rngs(0, augment=42)
        op2 = MapOperator(config, fn=add_noise, rngs=rngs2)

        # ✅ CORRECT: Pass operator as argument (not closure)
        @nnx.jit
        def apply_op(op, batch):
            return op(batch)

        # Apply both operators
        result1 = apply_op(op1, batch)
        result2 = apply_op(op2, batch)

        # Verify: output differs from input due to noise
        assert not jnp.allclose(result1.data.get_value()["image"], jnp.ones((2, 3, 3, 1)))

        # Verify: same seed produces same output (deterministic)
        assert jnp.allclose(result1.data.get_value()["image"], result2.data.get_value()["image"])

    def test_jit_empty_pytree(self):
        """JIT compilation handles empty PyTree edge case."""

        def double(x, key):
            return x * 2.0

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=double, rngs=rngs)

        # Create batch with empty data
        batch = Batch([Element(data={}, state={}, metadata=None)])

        # JIT compile with nnx.jit (pass module as argument, not closure)
        @nnx.jit
        def jitted_apply(model, batch):
            return model(batch)

        # Transform
        result = jitted_apply(op, batch)

        # Verify: batch passes through unchanged
        assert result.get_data() == {}
        assert result.get_states() == {}

    def test_jit_apply_batch_directly(self):
        """JIT compilation of apply_batch method directly."""

        def normalize(x, key):
            return (x - 0.5) / 0.5

        config = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=normalize, rngs=rngs)

        batch_data = {"image": jnp.ones((2, 3, 3, 1))}
        batch = create_test_batch(batch_data)

        # JIT compile apply_batch directly
        jitted_apply_batch = jax.jit(lambda b: op.apply_batch(b))

        # Transform
        result = jitted_apply_batch(batch)

        # Verify
        expected = (jnp.ones((2, 3, 3, 1)) - 0.5) / 0.5
        assert jnp.allclose(result.data.value["image"], expected)


# ========================================================================
# Test Category 7: Integration Tests
# ========================================================================


class TestMapOperatorIntegration:
    """Test MapOperator integration with other Datarax components."""

    def test_chaining_multiple_operators(self):
        """Can chain multiple MapOperators."""

        def normalize(x, key):
            return (x - 0.5) / 0.5

        def clip(x, key):
            return jnp.clip(x, -1.0, 1.0)

        config1 = MapOperatorConfig(subtree=None, stochastic=False)
        config2 = MapOperatorConfig(subtree=None, stochastic=False)
        rngs = nnx.Rngs(0)

        op1 = MapOperator(config1, fn=normalize, rngs=rngs)
        op2 = MapOperator(config2, fn=clip, rngs=rngs)

        batch_data = {"image": jnp.array([[0.0, 0.5, 1.0]])}
        batch = create_test_batch(batch_data)

        # Chain operations
        batch1 = op1(batch)
        batch2 = op2(batch1)

        # First op normalizes: [0.0, 0.5, 1.0] -> [-1.0, 0.0, 1.0]
        # Second op clips (already in range)
        expected = jnp.array([[-1.0, 0.0, 1.0]])
        assert jnp.allclose(batch2.get_data()["image"], expected)

    def test_different_functions_per_field(self):
        """Can apply different functions to different fields using multiple operators."""

        def double(x, key):
            return x * 2.0

        def square(x, key):
            return x**2

        config_image = MapOperatorConfig(subtree={"image": None}, stochastic=False)
        config_mask = MapOperatorConfig(subtree={"mask": None}, stochastic=False)
        rngs = nnx.Rngs(0)

        op_image = MapOperator(config_image, fn=double, rngs=rngs)
        op_mask = MapOperator(config_mask, fn=square, rngs=rngs)

        batch_data = {"image": jnp.array([[2.0]]), "mask": jnp.array([[3.0]])}
        batch = create_test_batch(batch_data)

        # Apply different transformations
        batch1 = op_image(batch)
        batch2 = op_mask(batch1)

        result_data = batch2.get_data()
        assert jnp.allclose(result_data["image"], jnp.array([[4.0]]))  # doubled
        assert jnp.allclose(result_data["mask"], jnp.array([[9.0]]))  # squared
