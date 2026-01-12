import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from datarax.core.batcher import Batch
from datarax.utils.external import (
    ExternalAdapterConfig,
    ExternalLibraryAdapter,
    PureJaxAdapter,
    to_datarax_operator,
    with_jax_key,
    with_jax_key_wrapper,
)


def test_external_adapter_config():
    """Test ExternalAdapterConfig initialization and defaults."""
    config = ExternalAdapterConfig()
    assert config.stochastic is True
    assert config.stream_name == "augment"

    config_custom = ExternalAdapterConfig(stochastic=False, stream_name=None)
    assert config_custom.stochastic is False
    assert config_custom.stream_name is None


class TestExternalLibraryAdapter:
    @pytest.fixture
    def mock_external_fn(self):
        """A simple external function mimicking an augmentation that adds noise."""

        def add_noise(data: dict, key: jax.Array) -> dict:
            noise = jax.random.normal(key, data["x"].shape)
            return {**data, "x": data["x"] + noise}

        return add_noise

    def test_initialization(self, mock_external_fn):
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        assert adapter.config == config
        assert adapter.fn == mock_external_fn
        assert adapter.stream_name == "augment"

    def test_generate_random_params(self, mock_external_fn):
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        master_key = jax.random.key(0)
        batch_size = 5
        # Simulate data_shapes for a batch of 5 images of shape (32, 32, 3)
        data_shapes = {"x": (batch_size, 32, 32, 3)}

        keys = adapter.generate_random_params(master_key, data_shapes)

        # JAX keys can be (N, 2) or (N,) depending on implementation/version
        assert keys.shape[0] == batch_size

        # Ensure keys are different
        # Compare first two keys
        assert not jnp.array_equal(keys[0], keys[1])

    def test_apply_single_element(self, mock_external_fn):
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        data = {"x": jnp.ones((10,))}
        state = {}
        metadata = None
        key = jax.random.key(123)

        transformed_data, new_state, new_metadata = adapter.apply(
            data, state, metadata, random_params=key
        )

        # Check that noise was added (unlikely to equal exactly 1.0 everywhere)
        assert not jnp.allclose(transformed_data["x"], data["x"])
        # Check output shape preserved
        assert transformed_data["x"].shape == data["x"].shape
        # State and metadata passed through
        assert new_state == state
        assert new_metadata == metadata

    def test_apply_raises_without_params(self, mock_external_fn):
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        with pytest.raises(ValueError, match="requires random_params"):
            adapter.apply({"x": jnp.zeros(1)}, {}, None, random_params=None)

    def test_integration_with_batch(self, mock_external_fn):
        """Test full pipeline execution via __call__ with a Batch object."""
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        batch_size = 4
        # Create a batch of data
        data = {"x": jnp.zeros((batch_size, 10))}
        batch = Batch.from_parts(data=data, states={}, validate=False)

        # Apply adapter
        output_batch = adapter(batch)

        output_data = output_batch.data.get_value()
        # Verify noise added
        assert not jnp.allclose(output_data["x"], 0.0)
        # Verify shape preserved
        assert output_data["x"].shape == (batch_size, 10)

    def test_jit_compatibility(self, mock_external_fn):
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        @nnx.jit
        def jitted_apply(adapter, batch):
            return adapter(batch)

        data = {"x": jnp.zeros((2, 5))}
        batch = Batch.from_parts(data=data, states={}, validate=False)

        # Should run without error
        output_batch = jitted_apply(adapter, batch)
        assert output_batch.data.get_value()["x"].shape == (2, 5)

    def test_stochastic_false_config(self, mock_external_fn):
        """Test with stochastic=False.

        Note: Even if stochastic=False, the logic in ExternalLibraryAdapter.generate_random_params
        doesn't explicitly forbid splitting keys if called. However, OperatorModule.apply_batch
        handles the conditional logic. If stochastic=False, it passes a dummy RNG.
        The external adapter will still split this dummy RNG and pass keys to the function.
        Effectively, it becomes deterministic noise.
        """
        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        # rngs not required if stochastic=False
        adapter = ExternalLibraryAdapter(config, mock_external_fn)

        batch_size = 2
        data = {"x": jnp.zeros((batch_size, 5))}
        batch = Batch.from_parts(data=data, states={}, validate=False)

        # Should run without error
        output_batch = adapter(batch)
        output_data = output_batch.data.get_value()

        # Verify result is deterministic
        adapter2 = ExternalLibraryAdapter(config, mock_external_fn)
        output_batch2 = adapter2(batch)
        output_data2 = output_batch2.data.get_value()

        assert jnp.array_equal(output_data["x"], output_data2["x"])

    def test_missing_rngs_error(self, mock_external_fn):
        """Test that missing rngs raises ValueError when stochastic=True."""
        config = ExternalAdapterConfig(stochastic=True, stream_name="augment")
        with pytest.raises(ValueError, match="require rngs"):
            ExternalLibraryAdapter(config, mock_external_fn, rngs=None)

    def test_empty_batch(self, mock_external_fn):
        """Test handling of empty batch."""
        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, mock_external_fn, rngs=rngs)

        # Empty data with valid structure but 0 batch size
        data = {"x": jnp.zeros((0, 10))}
        # Note: Batch.from_parts might validate batch size consistency, assuming it passes
        batch = Batch.from_parts(data=data, states={}, validate=False)

        output_batch = adapter(batch)
        assert output_batch.data.get_value()["x"].shape == (0, 10)

    def test_nested_data_structure(self):
        """Test with deep nested data structure."""

        def nested_fn(data, key):
            # Access nested field
            val = data["a"]["b"]
            noise = jax.random.normal(key, val.shape)
            new_val = val + noise
            return {"a": {"b": new_val}}

        config = ExternalAdapterConfig()
        rngs = nnx.Rngs(augment=42)
        adapter = ExternalLibraryAdapter(config, nested_fn, rngs=rngs)

        batch_size = 3
        data = {"a": {"b": jnp.zeros((batch_size, 4))}}
        batch = Batch.from_parts(data=data, states={}, validate=False)

        output = adapter(batch)
        out_data = output.data.get_value()

        assert "a" in out_data
        assert "b" in out_data["a"]
        assert out_data["a"]["b"].shape == (batch_size, 4)
        assert not jnp.allclose(out_data["a"]["b"], 0.0)


class TestPureJaxAdapter:
    @pytest.fixture
    def mock_pure_fn(self):
        """A simple pure function."""

        def add_one(data: dict) -> dict:
            return {**data, "x": data["x"] + 1}

        return add_one

    def test_initialization(self, mock_pure_fn):
        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        adapter = PureJaxAdapter(config, mock_pure_fn)

        assert adapter.config.stochastic is False
        assert adapter.fn == mock_pure_fn

    def test_apply_single_element(self, mock_pure_fn):
        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        adapter = PureJaxAdapter(config, mock_pure_fn)

        data = {"x": jnp.array([1.0])}
        transformed_data, _, _ = adapter.apply(data, {}, None)

        assert transformed_data["x"][0] == 2.0

    def test_integration_with_batch(self, mock_pure_fn):
        config = ExternalAdapterConfig(stochastic=False, stream_name=None)
        adapter = PureJaxAdapter(config, mock_pure_fn)

        data = {"x": jnp.zeros((3, 1))}
        batch = Batch.from_parts(data=data, states={}, validate=False)

        output = adapter(batch)
        assert jnp.all(output.data.get_value()["x"] == 1.0)


class TestWrappers:
    def test_with_jax_key_wrapper(self):
        def external_fn(data, key):
            return data + jax.random.normal(key, data.shape)

        wrapped = with_jax_key_wrapper(external_fn)

        rngs = nnx.Rngs(augment=42)
        data = jnp.zeros((5,))

        # Call with stream
        result = wrapped(data, rngs.augment)

        assert result.shape == (5,)
        assert not jnp.allclose(result, 0.0)

    def test_with_jax_key_decorator(self):
        @with_jax_key
        def decorated_fn(data, key):
            return data + 1.0  # Simple op to verify call

        rngs = nnx.Rngs(augment=42)
        result = decorated_fn(jnp.zeros(1), rngs.augment)
        assert result[0] == 1.0


class TestToDataraxOperator:
    def test_pure_function(self):
        def pure_fn(data):
            return data

        op = to_datarax_operator(pure_fn, stochastic=False)
        assert isinstance(op, PureJaxAdapter)
        assert op.config.stochastic is False

    def test_stochastic_function(self):
        def stochastic_fn(data, key):
            return data

        rngs = nnx.Rngs(augment=42)
        op = to_datarax_operator(stochastic_fn, stochastic=True, stream_name="augment", rngs=rngs)
        assert isinstance(op, ExternalLibraryAdapter)
        assert op.config.stochastic is True
        assert op.stream_name == "augment"
