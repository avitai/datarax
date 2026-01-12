"""Tests for the enhanced Datarax component registry system.

This module tests the enhanced registry functionality that supports
Flax NNX modules with proper RNG and Variable handling.
"""

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from datarax.config.registry import (
    _initialize_nnx_variables,
    _is_nnx_module,
    _prepare_config_for_nnx,
    _prepare_rngs_for_nnx,
    create_component_from_config,
    get_component_info,
    list_registered_components,
    register_component,
)


class SimpleClass:
    """A simple non-NNX class for testing."""

    def __init__(self, value: int = 42):
        self.value = value


class SimpleNNXModule(nnx.Module):
    """A simple NNX module for testing."""

    def __init__(self, size: int = 10, *, rngs: nnx.Rngs | None = None):
        super().__init__()
        self.size = nnx.Variable(size)
        self.count = nnx.Variable(0)
        self.rngs = rngs


class ComplexNNXModule(nnx.Module):
    """A complex NNX module with various Variable types."""

    def __init__(
        self, hidden_dim: int = 64, learning_rate: float = 0.01, *, rngs: nnx.Rngs | None = None
    ):
        super().__init__()
        self.hidden_dim = nnx.Variable(hidden_dim)
        self.learning_rate = nnx.Variable(learning_rate)
        self.weights = nnx.Param(jnp.ones((hidden_dim, hidden_dim)))
        self.step_count = nnx.Variable(0)
        self.rngs = rngs


class TestNNXModuleDetection:
    """Test NNX module detection functionality."""

    def test_is_nnx_module_with_nnx_class(self):
        """Test that NNX modules are properly detected."""
        assert _is_nnx_module(SimpleNNXModule)
        assert _is_nnx_module(ComplexNNXModule)

    def test_is_nnx_module_with_regular_class(self):
        """Test that regular classes are not detected as NNX modules."""
        assert not _is_nnx_module(SimpleClass)
        assert not _is_nnx_module(int)
        assert not _is_nnx_module(str)

    def test_is_nnx_module_with_non_class(self):
        """Test behavior with non-class objects."""
        assert not _is_nnx_module(lambda x: x)
        assert not _is_nnx_module(42)
        assert not _is_nnx_module("string")


class TestRNGPreparation:
    """Test RNG preparation for NNX modules."""

    def test_prepare_rngs_from_dict(self):
        """Test preparing RNGs from dictionary configuration."""
        config = {"rngs": {"params": 42, "dropout": 123, "default": 456}}

        rngs = _prepare_rngs_for_nnx(config)
        assert isinstance(rngs, nnx.Rngs)
        assert "params" in rngs
        assert "dropout" in rngs
        assert "default" in rngs

    def test_prepare_rngs_from_empty_dict(self):
        """Test preparing RNGs from empty dictionary."""
        config = {"rngs": {}}

        rngs = _prepare_rngs_for_nnx(config)
        assert isinstance(rngs, nnx.Rngs)

    def test_prepare_rngs_with_seed(self):
        """Test preparing RNGs with global seed."""
        config = {"seed": 42}

        rngs = _prepare_rngs_for_nnx(config)
        assert isinstance(rngs, nnx.Rngs)

    def test_prepare_rngs_with_existing_rngs(self):
        """Test preparing RNGs when already provided."""
        existing_rngs = nnx.Rngs(42)
        config = {"rngs": existing_rngs}

        rngs = _prepare_rngs_for_nnx(config)
        assert rngs is existing_rngs


class TestConfigPreparation:
    """Test configuration preparation for NNX modules."""

    def test_prepare_config_for_nnx_with_rngs_param(self):
        """Test config preparation when constructor has rngs parameter."""
        config = {
            "size": 100,
            "rngs": {"params": 42},
            "variables": {"count": 5},
            "load_state_from": "/path/to/state",
        }

        prepared = _prepare_config_for_nnx(SimpleNNXModule, config)

        assert "size" in prepared
        assert "rngs" in prepared
        assert isinstance(prepared["rngs"], nnx.Rngs)
        assert "variables" not in prepared
        assert "load_state_from" not in prepared

    def test_prepare_config_for_regular_class(self):
        """Test config preparation for regular classes."""
        config = {"value": 123, "rngs": {"params": 42}, "variables": {"count": 5}}

        prepared = _prepare_config_for_nnx(SimpleClass, config)

        assert "value" in prepared
        # Regular classes should not get rngs filtered


class TestVariableInitialization:
    """Test NNX Variable initialization from configuration."""

    def test_initialize_simple_variables(self):
        """Test initializing simple Variables."""
        module = SimpleNNXModule(rngs=nnx.Rngs(42))
        config = {"variables": {"size": 50, "count": 10}}

        _initialize_nnx_variables(module, config)

        assert module.size.get_value() == 50
        assert module.count.get_value() == 10

    def test_initialize_nested_variables(self):
        """Test initializing nested Variables (not yet implemented)."""
        module = ComplexNNXModule(rngs=nnx.Rngs(42))
        config = {"variables": {"hidden_dim": 128, "learning_rate": 0.001}}

        _initialize_nnx_variables(module, config)

        assert module.hidden_dim.get_value() == 128
        assert module.learning_rate.get_value() == 0.001

    def test_initialize_nonexistent_variable(self):
        """Test error handling for nonexistent variables."""
        module = SimpleNNXModule(rngs=nnx.Rngs(42))
        config = {"variables": {"nonexistent": 123}}

        with pytest.raises(ValueError, match="Variable.*not found"):
            _initialize_nnx_variables(module, config)


class TestRegistryIntegration:
    """Test the integrated registry functionality."""

    def setup_method(self):
        """Register test components."""
        register_component("test", "SimpleClass")(SimpleClass)
        register_component("test", "SimpleNNX")(SimpleNNXModule)
        register_component("test", "ComplexNNX")(ComplexNNXModule)

    def test_create_regular_component(self):
        """Test creating a regular component."""
        config = {"value": 123}

        component = create_component_from_config("test", "SimpleClass", config)

        assert isinstance(component, SimpleClass)
        assert component.value == 123

    def test_create_simple_nnx_component(self):
        """Test creating a simple NNX component."""
        config = {"size": 50, "rngs": {"params": 42}}

        component = create_component_from_config("test", "SimpleNNX", config)

        assert isinstance(component, SimpleNNXModule)
        assert component.size.get_value() == 50
        assert isinstance(component.rngs, nnx.Rngs)

    def test_create_nnx_component_with_variables(self):
        """Test creating NNX component with Variable initialization."""
        config = {
            "hidden_dim": 32,
            "rngs": {"params": 42},
            "variables": {"hidden_dim": 64, "step_count": 100},
        }

        component = create_component_from_config("test", "ComplexNNX", config)

        assert isinstance(component, ComplexNNXModule)
        assert component.hidden_dim.get_value() == 64  # Overridden by variables
        assert component.step_count.get_value() == 100

    def test_get_component_info_regular_class(self):
        """Test getting component info for regular class."""
        info = get_component_info("test", "SimpleClass")

        assert not info["is_nnx_module"]
        assert not info["requires_rngs"]
        assert "value" in info["parameters"]

    def test_get_component_info_nnx_module(self):
        """Test getting component info for NNX module."""
        info = get_component_info("test", "SimpleNNX")

        assert info["is_nnx_module"]
        assert info["requires_rngs"]
        assert "size" in info["parameters"]
        assert "rngs" in info["parameters"]

    def test_list_registered_components(self):
        """Test listing registered components."""
        components = list_registered_components("test")

        assert "SimpleClass" in components
        assert "SimpleNNX" in components
        assert "ComplexNNX" in components


class TestErrorHandling:
    """Test error handling in the enhanced registry."""

    def test_create_nonexistent_component(self):
        """Test error when creating nonexistent component."""
        with pytest.raises(KeyError, match="Component not registered"):
            create_component_from_config("test", "NonExistent", {})

    def test_create_component_with_invalid_config(self):
        """Test error when config is invalid."""
        register_component("test", "InvalidConfig")(SimpleClass)

        # Pass invalid parameter
        config = {"invalid_param": 123}

        with pytest.raises(TypeError, match="Failed to create component"):
            create_component_from_config("test", "InvalidConfig", config)

    def test_initialize_variables_invalid_path(self):
        """Test error when variable path is invalid."""
        module = SimpleNNXModule(rngs=nnx.Rngs(42))
        config = {"variables": {"nonexistent.nested": 123}}

        with pytest.raises(ValueError, match="Variable path.*not found"):
            _initialize_nnx_variables(module, config)


@pytest.fixture
def clear_registry():
    """Fixture to clear the registry after tests."""
    yield
    # Clear test components from registry
    from datarax.config.registry import _COMPONENT_REGISTRY

    keys_to_remove = [k for k in _COMPONENT_REGISTRY.keys() if k.startswith("test.")]
    for key in keys_to_remove:
        del _COMPONENT_REGISTRY[key]


# Use the fixture for cleanup
pytestmark = pytest.mark.usefixtures("clear_registry")
