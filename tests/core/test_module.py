"""Tests for the DataraxModule base class."""

import flax.nnx as nnx
import jax
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
        assert rngs is not None
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


def test_base_module_carries_no_operation_counters():
    """Every compiled step carries a module's variables, so the base adds none nothing updates."""
    module = RNGRequiringModule()

    top_level_names = {path[0] for path, _ in nnx.to_flat_state(nnx.state(module))}

    assert top_level_names.isdisjoint({"_applied_count", "_skipped_count"})
    assert not hasattr(module, "get_operation_stats")


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

    # The RNG streams are state too: a resumed module draws the same numbers
    assert new_module.rngs is not None and module.rngs is not None
    assert jnp.array_equal(
        jax.random.key_data(new_module.rngs.default.key.get_value()),
        jax.random.key_data(module.rngs.default.key.get_value()),
    )
    assert new_module.rngs.default.count.get_value() == module.rngs.default.count.get_value()


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
    assert cloned_module.counter.get_value() == 3  # type: ignore[reportAttributeAccessIssue]

    # Check that the cloned module works independently
    cloned_module(x)
    assert cloned_module.counter.get_value() == 4  # type: ignore[reportAttributeAccessIssue]
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
        config = DataraxModuleConfig()
        return DataraxModule(config, rngs=rngs, name="test_module")

    def test_enhanced_initialization(self, rngs):
        """Test DataraxModule initialization records its config, rngs and name."""
        config = DataraxModuleConfig()
        module = DataraxModule(config, rngs=rngs, name="test_module")

        assert module.name == "test_module"
        assert module.config is config
        assert module.rngs is rngs

    def test_a_module_has_no_statistics_of_its_own(self, rngs):
        """Statistics belong to operators, which hold them in their own store.

        A source, sampler or batcher carried an empty statistics variable through every
        compiled step and had no way to fill it.
        """
        module = DataraxModule(DataraxModuleConfig(), rngs=rngs)

        assert not hasattr(module, "get_statistics")
        assert not hasattr(module, "set_statistics")
        assert not hasattr(module, "compute_statistics")

    def test_enhanced_state_management(self, basic_module):
        """Test enhanced state management including new features."""
        # Get state
        state = basic_module.get_state()

        assert isinstance(state, dict)

        # Test state restoration
        new_config = DataraxModuleConfig()
        # State restoration is strict: target must have compatible RNG structure.
        new_module = DataraxModule(new_config, rngs=nnx.Rngs(0))
        new_module.set_state(state)

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
