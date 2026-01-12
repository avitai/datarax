"""Test suite for ModalityOperator.

Tests the base class for modality-specific operators.
Follows TDD approach - tests written first (RED phase).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig


class ConcreteModalityOperator(ModalityOperator):
    """Concrete implementation for testing."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Simple test implementation - multiply image by factor."""
        factor = random_params if random_params is not None else 1.0
        field = self._extract_field(data, self.config.field_key)
        transformed = field * factor
        transformed = self._apply_clip_range(transformed)
        result = self._remap_field(data, transformed)
        return result, state, metadata


class StochasticModalityOperator(ModalityOperator):
    """Stochastic implementation for testing."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Apply random brightness adjustment."""
        brightness_factor = random_params if random_params is not None else 1.0
        field = self._extract_field(data, self.config.field_key)
        transformed = field * brightness_factor
        transformed = self._apply_clip_range(transformed)
        result = self._remap_field(data, transformed)
        return result, state, metadata

    def generate_random_params(self, rng, data_shapes):
        """Generate random brightness factors."""
        batch_size = data_shapes[self.config.field_key][0]
        return jax.random.uniform(rng, (batch_size,), minval=0.8, maxval=1.2)


class TestModalityOperatorInitialization:
    """Test ModalityOperator initialization."""

    def test_deterministic_initialization(self):
        """Deterministic operator initialization should succeed."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        assert operator.config.field_key == "image"
        assert operator.stochastic is False
        assert operator.stream_name is None

    def test_stochastic_initialization(self):
        """Stochastic operator initialization should succeed."""
        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = StochasticModalityOperator(config, rngs=rngs)

        assert operator.config.field_key == "image"
        assert operator.stochastic is True
        assert operator.stream_name == "augment"

    def test_stochastic_without_rngs_raises_error(self):
        """Stochastic operator without rngs should raise ValueError."""
        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        with pytest.raises(ValueError, match="Stochastic operators require rngs"):
            StochasticModalityOperator(config, rngs=None)

    def test_initialization_with_name(self):
        """Initialization with custom name should work."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs, name="brightness_op")

        assert operator.name == "brightness_op"


class TestModalityOperatorAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_base_class_apply_raises_not_implemented(self):
        """Calling apply() on base class should raise NotImplementedError."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ModalityOperator(config, rngs=rngs)

        data = {"image": jnp.ones((224, 224, 3))}
        state = {}
        metadata = None

        with pytest.raises(NotImplementedError, match="must implement apply"):
            operator.apply(data, state, metadata)

    def test_stochastic_without_generate_random_params_raises_error(self):
        """Stochastic operator without generate_random_params should raise error."""

        class IncompleteStochasticOperator(ModalityOperator):
            """Missing generate_random_params implementation."""

            def apply(self, data, state, metadata, random_params=None, stats=None):
                return data, state, metadata

        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = IncompleteStochasticOperator(config, rngs=rngs)

        data_shapes = {"image": (4, 224, 224, 3)}
        rng = jax.random.PRNGKey(0)

        with pytest.raises(NotImplementedError, match="does not implement generate_random_params"):
            operator.generate_random_params(rng, data_shapes)


class TestModalityOperatorHelperMethods:
    """Test helper methods for field extraction and transformation."""

    def test_extract_field_success(self):
        """_extract_field should extract correct field from data."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1, 2, 3]), "mask": jnp.array([1, 1, 0])}
        result = operator._extract_field(data, "image")

        assert jnp.array_equal(result, jnp.array([1, 2, 3]))

    def test_extract_field_missing_raises_key_error(self):
        """_extract_field should raise KeyError for missing field."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"mask": jnp.array([1, 1, 0])}

        with pytest.raises(KeyError, match="Field 'image' not found"):
            operator._extract_field(data, "image")

    def test_apply_clip_range_clips_values(self):
        """_apply_clip_range should clip values to configured range."""
        config = ModalityOperatorConfig(field_key="image", clip_range=(0.0, 1.0))
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        values = jnp.array([-0.5, 0.3, 0.7, 1.5])
        clipped = operator._apply_clip_range(values)

        expected = jnp.array([0.0, 0.3, 0.7, 1.0])
        assert jnp.allclose(clipped, expected)

    def test_apply_clip_range_none_no_clipping(self):
        """_apply_clip_range with None should not clip."""
        config = ModalityOperatorConfig(field_key="image", clip_range=None)
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        values = jnp.array([-0.5, 0.3, 0.7, 1.5])
        result = operator._apply_clip_range(values)

        assert jnp.array_equal(result, values)

    def test_remap_field_overwrites_source_when_target_is_none(self):
        """_remap_field should overwrite source field when target_key is None."""
        config = ModalityOperatorConfig(field_key="image", target_key=None)
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1, 2, 3]), "mask": jnp.array([1, 1, 0])}
        transformed_value = jnp.array([4, 5, 6])
        result = operator._remap_field(data, transformed_value)

        assert jnp.array_equal(result["image"], jnp.array([4, 5, 6]))
        assert jnp.array_equal(result["mask"], jnp.array([1, 1, 0]))
        assert len(result) == 2

    def test_remap_field_creates_new_field_when_target_different(self):
        """_remap_field should create new field when target_key differs from field_key."""
        config = ModalityOperatorConfig(field_key="image", target_key="processed")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1, 2, 3])}
        transformed_value = jnp.array([4, 5, 6])
        result = operator._remap_field(data, transformed_value)

        assert jnp.array_equal(result["image"], jnp.array([1, 2, 3]))
        assert jnp.array_equal(result["processed"], jnp.array([4, 5, 6]))
        assert len(result) == 2


class TestModalityOperatorApply:
    """Test apply() method implementation."""

    def test_apply_deterministic(self):
        """Apply should work for deterministic operator."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1.0, 2.0, 3.0])}
        state = {}
        metadata = None

        result_data, result_state, result_metadata = operator.apply(
            data, state, metadata, random_params=2.0
        )

        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result_data["image"], expected)
        assert result_state == {}
        assert result_metadata is None

    def test_apply_with_clip_range(self):
        """Apply should clip values when clip_range is set."""
        config = ModalityOperatorConfig(field_key="image", clip_range=(0.0, 5.0))
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1.0, 2.0, 3.0])}
        state = {}
        metadata = None

        result_data, result_state, result_metadata = operator.apply(
            data, state, metadata, random_params=3.0
        )

        expected = jnp.array([3.0, 5.0, 5.0])  # 6.0 and 9.0 clipped to 5.0
        assert jnp.allclose(result_data["image"], expected)

    def test_apply_with_target_key(self):
        """Apply should write to target_key when specified."""
        config = ModalityOperatorConfig(
            field_key="image",
            target_key="processed_image",
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1.0, 2.0, 3.0])}
        state = {}
        metadata = None

        result_data, result_state, result_metadata = operator.apply(
            data, state, metadata, random_params=2.0
        )

        # Original should be preserved
        assert jnp.allclose(result_data["image"], jnp.array([1.0, 2.0, 3.0]))
        # New field should be created
        assert jnp.allclose(result_data["processed_image"], jnp.array([2.0, 4.0, 6.0]))


class TestModalityOperatorStochastic:
    """Test stochastic operator behavior."""

    def test_stochastic_operator_generates_random_params(self):
        """Stochastic operator should generate random parameters."""
        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = StochasticModalityOperator(config, rngs=rngs)

        data_shapes = {"image": (4, 224, 224, 3)}
        rng = jax.random.PRNGKey(42)

        random_params = operator.generate_random_params(rng, data_shapes)

        # Should generate 4 random factors (batch size = 4)
        assert random_params.shape == (4,)
        # Values should be in [0.8, 1.2] range
        assert jnp.all(random_params >= 0.8)
        assert jnp.all(random_params <= 1.2)

    def test_stochastic_operator_apply_uses_random_params(self):
        """Stochastic operator apply should use provided random params."""
        config = ModalityOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = StochasticModalityOperator(config, rngs=rngs)

        data = {"image": jnp.array([1.0, 2.0, 3.0])}
        state = {}
        metadata = None

        result_data, _, _ = operator.apply(data, state, metadata, random_params=1.5)

        expected = jnp.array([1.5, 3.0, 4.5])
        assert jnp.allclose(result_data["image"], expected)


class TestModalityOperatorJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_operator_is_jit_compatible(self):
        """Operator apply should be JIT compatible."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        @jax.jit
        def jitted_apply(data, state, metadata):
            return operator.apply(data, state, metadata, random_params=2.0)

        data = {"image": jnp.array([1.0, 2.0, 3.0])}
        state = {}
        metadata = None

        result_data, _, _ = jitted_apply(data, state, metadata)

        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(result_data["image"], expected)

    def test_operator_is_vmap_compatible(self):
        """Operator apply should be vmap compatible."""
        config = ModalityOperatorConfig(field_key="image")
        rngs = nnx.Rngs(0)
        operator = ConcreteModalityOperator(config, rngs=rngs)

        # Batch of 3 images
        batch_data = {"image": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
        batch_state = [{}, {}, {}]
        batch_random_params = jnp.array([1.0, 2.0, 3.0])

        def apply_single(data, state, random_param):
            data_dict = {"image": data}
            result_data, result_state, _ = operator.apply(
                data_dict, state, None, random_params=random_param
            )
            return result_data["image"], result_state

        vmapped_apply = jax.vmap(apply_single, in_axes=(0, 0, 0))
        result_images, result_states = vmapped_apply(
            batch_data["image"], batch_state, batch_random_params
        )

        expected = jnp.array([[1.0, 2.0], [6.0, 8.0], [15.0, 18.0]])
        assert jnp.allclose(result_images, expected)
