"""Tests for the DataraxModule base class."""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from datarax.core.config import DataraxModuleConfig
from datarax.core.module import DataraxModule


class SimpleDataraxModule(DataraxModule):
    """A simple DataraxModule for testing."""

    def __init__(
        self,
        config: DataraxModuleConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        # Use default config if not provided
        if config is None:
            config = DataraxModuleConfig()
        super().__init__(config, rngs=rngs)
        # Initialize the counter variable and dense layer in __init__
        self.counter = nnx.Variable(0)
        self.dense = nnx.Linear(8, 4, rngs=rngs)  # Specify in_features and out_features

    def __call__(self, x, increment: bool = True):
        """Apply the module to input x."""
        if increment:
            # Use set_value/get_value for non-array Variable (new NNX API)
            self.counter.set_value(self.counter.get_value() + 1)
        return self.dense(x)

    def requires_rng_streams(self):
        """Override to require a test RNG stream."""
        return ["test"]


class RNGRequiringModule(DataraxModule):
    """A DataraxModule that requires specific RNG streams."""

    def __init__(
        self,
        config: DataraxModuleConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        # Use default config if not provided
        if config is None:
            config = DataraxModuleConfig()
        super().__init__(config, rngs=rngs)

    def requires_rng_streams(self):
        """Override to require specific RNG streams."""
        return ["stream1", "stream2"]

    def __call__(self, x):
        """Apply the module to input x."""
        if self.rngs is not None:
            self.rngs.stream1()
            self.rngs.stream2()
            # Use the RNG keys (they will be created automatically)
        return x


def test_serialization():
    """Test serialization and deserialization of DataraxModule."""
    # Create and initialize a module
    rngs = nnx.Rngs(0)
    module = SimpleDataraxModule(rngs=rngs)
    x = jnp.ones((2, 8))
    module(x)

    # Increment counter a few times
    module(x)
    module(x)
    assert module.counter.get_value() == 3

    # Get serializable state
    state = module.get_state()

    # Create a new module and restore state
    new_rngs = nnx.Rngs(1)  # Different seed to ensure state restoration works
    new_module = SimpleDataraxModule(rngs=new_rngs)
    new_module.set_state(state)

    # Check that the state was restored correctly
    assert new_module.counter.get_value() == 3

    # Test that the weights are the same by checking output
    assert jnp.allclose(module(x, increment=False), new_module(x, increment=False))


def test_clone():
    """Test cloning a DataraxModule."""
    # Create and initialize a module
    rngs = nnx.Rngs(0)
    module = SimpleDataraxModule(rngs=rngs)
    x = jnp.ones((2, 8))
    module(x)

    # Increment counter a few times
    module(x)
    module(x)
    assert module.counter.get_value() == 3

    # Clone the module
    cloned_module = module.clone()

    # Check that the state was cloned correctly
    assert cloned_module.counter.get_value() == 3

    # Check that the cloned module works independently
    cloned_module(x)
    assert cloned_module.counter.get_value() == 4
    assert module.counter.get_value() == 3

    # Test that the weights are the same by checking output
    assert jnp.allclose(module(x, increment=False), cloned_module(x, increment=False))


def test_requires_rng_streams():
    """Test the requires_rng_streams method."""
    # Default implementation should return None
    default_module = DataraxModule(DataraxModuleConfig())
    assert default_module.requires_rng_streams() is None

    # Custom implementation should return the expected streams
    rngs = nnx.Rngs(0)
    custom_module = SimpleDataraxModule(rngs=rngs)
    assert custom_module.requires_rng_streams() == ["test"]

    # Multiple streams
    multi_stream_module = RNGRequiringModule()
    assert multi_stream_module.requires_rng_streams() == ["stream1", "stream2"]


def test_ensure_rng_streams():
    """Test the ensure_rng_streams method."""
    # Default implementation should not raise an error
    default_module = DataraxModule(DataraxModuleConfig())
    default_module.ensure_rng_streams(["anything"])

    # Custom implementation should check streams
    rngs = nnx.Rngs(0)
    custom_module = SimpleDataraxModule(rngs=rngs)
    custom_module.ensure_rng_streams(["test", "other"])

    # Should raise an error if a required stream is missing
    with pytest.raises(ValueError):
        custom_module.ensure_rng_streams(["other"])

    # Test with module requiring multiple streams
    multi_stream_module = RNGRequiringModule()
    multi_stream_module.ensure_rng_streams(["stream1", "stream2", "extra"])

    # Should raise an error if any required stream is missing
    with pytest.raises(ValueError):
        multi_stream_module.ensure_rng_streams(["stream1"])
    with pytest.raises(ValueError):
        multi_stream_module.ensure_rng_streams(["stream2"])
    with pytest.raises(ValueError):
        multi_stream_module.ensure_rng_streams(["extra"])


class TestEnhancedDataraxModule:
    """Test suite for enhanced DataraxModule functionality."""

    @pytest.fixture
    def rngs(self):
        """Standard RNG fixture."""
        return nnx.Rngs(42)

    @pytest.fixture
    def basic_module(self, rngs):
        """Basic enhanced DataraxModule for testing."""
        config = DataraxModuleConfig(cacheable=True)
        return DataraxModule(config, rngs=rngs, name="test_module")

    def test_enhanced_initialization(self, rngs):
        """Test enhanced DataraxModule initialization with new features."""
        # Test with all new parameters via config
        # Note: batch_stats_fn and precomputed_stats are mutually exclusive
        # Testing precomputed_stats path
        config = DataraxModuleConfig(
            cacheable=True,
            precomputed_stats={"constant": 42},
        )
        module = DataraxModule(config, rngs=rngs, name="test_module")

        assert module.name == "test_module"
        assert module.config.cacheable is True
        assert module.config.precomputed_stats == {"constant": 42}
        assert module._cache is not None

    def test_caching_functionality(self, basic_module):
        """Test caching functionality."""
        # Test cache initialization
        assert basic_module._cache is not None
        assert len(basic_module._cache) == 0

        # Test cache operations
        test_data = jnp.array([1, 2, 3])
        cache_key = basic_module._compute_cache_key(test_data)

        # Cache should be empty initially
        assert cache_key not in basic_module._cache

        # Add to cache
        result = jnp.array([2, 4, 6])
        basic_module._cache[cache_key] = result

        # Verify cached result
        assert cache_key in basic_module._cache
        assert jnp.array_equal(basic_module._cache[cache_key], result)

        # Test cache reset
        basic_module.reset_cache()
        assert len(basic_module._cache) == 0

    def test_cache_key_computation(self, basic_module):
        """Test cache key computation for different data types."""
        # Test with hashable types (strings, tuples)
        key1 = basic_module._compute_cache_key("test_string")
        key2 = basic_module._compute_cache_key("test_string")
        key3 = basic_module._compute_cache_key("different_string")

        # Same hashable data should produce same key
        assert key1 == key2
        # Different data should produce different key
        assert key1 != key3

        # Test with JAX arrays - content-based hashing
        # Same content should produce same key regardless of array identity
        data1 = jnp.array([1, 2, 3])
        array_key1 = basic_module._compute_cache_key(data1)
        # Same array object should produce same key
        array_key2 = basic_module._compute_cache_key(data1)
        assert array_key1 == array_key2

        # Different array objects with SAME content should have SAME key (content-based)
        data2 = jnp.array([1, 2, 3])  # Same content, different object
        array_key3 = basic_module._compute_cache_key(data2)
        assert array_key1 == array_key3  # Content-based: same content = same key

        # Different content should produce different key
        data3 = jnp.array([4, 5, 6])  # Different content
        array_key4 = basic_module._compute_cache_key(data3)
        assert array_key1 != array_key4  # Different content = different key

        # Test with PyTree structures
        pytree1 = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}
        pytree2 = {"a": jnp.array([1, 2]), "b": jnp.array([3, 4])}  # Same content
        pytree3 = {"a": jnp.array([9, 9]), "b": jnp.array([3, 4])}  # Different content
        assert basic_module._compute_cache_key(pytree1) == basic_module._compute_cache_key(pytree2)
        assert basic_module._compute_cache_key(pytree1) != basic_module._compute_cache_key(pytree3)

    def test_statistics_computation(self, rngs):
        """Test batch statistics computation."""

        # Test with batch_stats_fn
        def compute_stats(batch):
            return {"mean": jnp.mean(batch), "std": jnp.std(batch)}

        config = DataraxModuleConfig(batch_stats_fn=compute_stats)
        module = DataraxModule(config, rngs=rngs)

        batch = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = module.compute_statistics(batch)

        assert stats is not None
        assert "mean" in stats
        assert "std" in stats
        assert jnp.isclose(stats["mean"], 3.0)

        # Test with static stats - using get_statistics for precomputed
        static_config = DataraxModuleConfig(precomputed_stats={"constant_mean": 10.0})
        static_module = DataraxModule(static_config, rngs=rngs)

        # Precomputed stats are returned via get_statistics, not compute_statistics
        static_stats = static_module.get_statistics()
        assert static_stats == {"constant_mean": 10.0}

    def test_enhanced_state_management(self, basic_module):
        """Test enhanced state management including new features."""
        basic_module._cache["test_key"] = "test_value"

        # Get state
        state = basic_module.get_state()

        # _cache is not included as it's not an nnx.Variable
        assert isinstance(state, dict)

        # Test state restoration
        new_config = DataraxModuleConfig(cacheable=True)
        new_module = DataraxModule(new_config)
        new_module.set_state(state)

    def test_module_without_caching(self, rngs):
        """Test module behavior when caching is disabled."""
        config = DataraxModuleConfig(cacheable=False)
        module = DataraxModule(config, rngs=rngs)

        assert module._cache is None

        # reset_cache should work even when cache is None
        module.reset_cache()  # Should not raise error

    def test_batch_stats_with_none_function(self, rngs):
        """Test statistics computation when no function is provided."""
        config = DataraxModuleConfig()  # No batch_stats_fn
        module = DataraxModule(config, rngs=rngs)

        batch = jnp.array([1, 2, 3])
        stats = module.compute_statistics(batch)

        assert stats is None

    def test_requires_rng_streams_override(self, rngs):
        """Test that subclasses can override required RNG streams."""

        class CustomModule(DataraxModule):
            def __init__(self, config=None, *, rngs=None):
                if config is None:
                    config = DataraxModuleConfig()
                super().__init__(config, rngs=rngs)

            def requires_rng_streams(self):
                return ["custom_stream", "another_stream"]

        module = CustomModule(rngs=rngs)
        required = module.requires_rng_streams()

        assert required == ["custom_stream", "another_stream"]

    def test_ensure_rng_streams_validation(self, rngs):
        """Test RNG stream validation."""

        class CustomModule(DataraxModule):
            def __init__(self, config=None, *, rngs=None):
                if config is None:
                    config = DataraxModuleConfig()
                super().__init__(config, rngs=rngs)

            def requires_rng_streams(self):
                return ["required_stream"]

        module = CustomModule(rngs=rngs)

        # Should raise error if required stream is not available
        with pytest.raises(ValueError, match="RNG stream 'required_stream' is required"):
            module.ensure_rng_streams(["other_stream"])

        # Should not raise error if required stream is available
        module.ensure_rng_streams(["required_stream", "other_stream"])
