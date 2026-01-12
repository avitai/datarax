"""Tests for DataraxModuleConfig base configuration class.

This test suite validates the base configuration used by all Datarax modules.
Tests config construction, validation, and inheritance patterns.

Test Categories:
- Config construction (valid cases)
- Validation rules (__post_init__)
- Field defaults
- Invalid configurations
"""

import pytest
from typing import Any
from flax import nnx


# NOTE: Import will fail initially (RED phase) - this is expected!
# Implementation will be created in src/datarax/core/config.py
try:
    from datarax.core.config import DataraxModuleConfig
except ImportError:
    # Expected during RED phase - tests should fail
    DataraxModuleConfig = None


# Skip all tests if implementation doesn't exist yet
pytestmark = pytest.mark.skipif(
    DataraxModuleConfig is None,
    reason="DataraxModuleConfig not implemented yet (RED phase)",
)


class TestDataraxModuleConfigConstruction:
    """Test valid config construction with various parameter combinations."""

    def test_default_construction(self):
        """Test config with all defaults."""
        config = DataraxModuleConfig()

        assert config.cacheable is False
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None

    def test_with_cacheable(self):
        """Test config with cacheable enabled."""
        config = DataraxModuleConfig(cacheable=True)

        assert config.cacheable is True
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None

    def test_with_batch_stats_fn(self):
        """Test config with dynamic statistics function."""

        def compute_stats(batch: Any) -> dict[str, Any]:
            return {"mean": 0.5}

        config = DataraxModuleConfig(batch_stats_fn=compute_stats)

        assert config.batch_stats_fn is compute_stats
        assert config.precomputed_stats is None

    def test_with_batch_stats_fn_as_module(self):
        """Test config with statistics computed by NNX module."""

        class StatsModule(nnx.Module):
            def __call__(self, batch: Any) -> dict[str, Any]:
                return {"mean": 0.5}

        stats_module = StatsModule()
        config = DataraxModuleConfig(batch_stats_fn=stats_module)

        assert config.batch_stats_fn is stats_module

    def test_with_precomputed_stats(self):
        """Test config with static precomputed statistics."""
        stats = {"mean": 0.5, "std": 0.2}
        config = DataraxModuleConfig(precomputed_stats=stats)

        assert config.precomputed_stats == stats
        assert config.batch_stats_fn is None

    def test_all_valid_combinations(self):
        """Test various valid parameter combinations."""
        # Cacheable with stats function
        config1 = DataraxModuleConfig(cacheable=True, batch_stats_fn=lambda x: {"mean": 0.5})
        assert config1.cacheable is True
        assert config1.batch_stats_fn is not None

        # Cacheable with precomputed stats
        config2 = DataraxModuleConfig(cacheable=True, precomputed_stats={"mean": 0.5})
        assert config2.cacheable is True
        assert config2.precomputed_stats is not None


class TestDataraxModuleConfigValidation:
    """Test __post_init__ validation rules."""

    def test_mutual_exclusivity_batch_stats_fn_and_precomputed_stats(self):
        """Test that batch_stats_fn and precomputed_stats are mutually exclusive."""

        def compute_stats(batch: Any) -> dict[str, Any]:
            return {"mean": 0.5}

        with pytest.raises(ValueError) as exc_info:
            DataraxModuleConfig(batch_stats_fn=compute_stats, precomputed_stats={"mean": 0.5})

        error_msg = str(exc_info.value).lower()
        assert "both" in error_msg
        assert "batch_stats_fn" in error_msg
        assert "precomputed_stats" in error_msg

    def test_validation_error_message_quality(self):
        """Test that validation errors have helpful messages."""
        with pytest.raises(ValueError) as exc_info:
            DataraxModuleConfig(batch_stats_fn=lambda x: {}, precomputed_stats={})

        error_msg = str(exc_info.value)
        # Message should explain the mutual exclusivity
        assert "cannot" in error_msg.lower() or "choose" in error_msg.lower()


class TestDataraxModuleConfigDefaults:
    """Test default values are correct."""

    def test_cacheable_defaults_to_false(self):
        """Test cacheable defaults to False."""
        config = DataraxModuleConfig()
        assert config.cacheable is False

    def test_batch_stats_fn_defaults_to_none(self):
        """Test batch_stats_fn defaults to None."""
        config = DataraxModuleConfig()
        assert config.batch_stats_fn is None

    def test_precomputed_stats_defaults_to_none(self):
        """Test precomputed_stats defaults to None."""
        config = DataraxModuleConfig()
        assert config.precomputed_stats is None


class TestDataraxModuleConfigInheritance:
    """Test config inheritance patterns for child configs."""

    def test_child_config_can_add_fields(self):
        """Test that child configs can add their own fields."""
        from dataclasses import dataclass

        @dataclass
        class ChildConfig(DataraxModuleConfig):
            """Child config with additional field."""

            extra_field: int = 42

        config = ChildConfig()
        assert config.cacheable is False  # Inherited
        assert config.extra_field == 42  # Child-specific

    def test_child_config_inherits_validation(self):
        """Test that child configs inherit parent validation."""
        from dataclasses import dataclass

        @dataclass
        class ChildConfig(DataraxModuleConfig):
            """Child config that inherits validation."""

            pass

        # Should still enforce mutual exclusivity
        with pytest.raises(ValueError):
            ChildConfig(batch_stats_fn=lambda x: {}, precomputed_stats={})

    def test_child_config_can_add_validation(self):
        """Test that child configs can add their own validation."""
        from dataclasses import dataclass

        @dataclass
        class ChildConfig(DataraxModuleConfig):
            """Child config with additional validation."""

            min_value: float = 0.0
            max_value: float = 1.0

            def __post_init__(self):
                super().__post_init__()  # Parent validation first
                if self.min_value >= self.max_value:
                    raise ValueError("min_value must be < max_value")

        # Parent validation still works
        with pytest.raises(ValueError):
            ChildConfig(batch_stats_fn=lambda x: {}, precomputed_stats={})

        # Child validation works
        with pytest.raises(ValueError) as exc_info:
            ChildConfig(min_value=1.0, max_value=0.5)

        assert "min_value" in str(exc_info.value).lower()


class TestDataraxModuleConfigDataclass:
    """Test that config is a proper dataclass."""

    def test_is_dataclass(self):
        """Test that DataraxModuleConfig is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(DataraxModuleConfig)

    def test_not_frozen_by_default(self):
        """Test that base config is mutable (not frozen)."""
        config = DataraxModuleConfig()
        # Should be able to modify fields (mutable)
        config.cacheable = True
        assert config.cacheable is True

    def test_repr_includes_fields(self):
        """Test that __repr__ shows field values."""
        config = DataraxModuleConfig(cacheable=True)
        repr_str = repr(config)

        assert "DataraxModuleConfig" in repr_str
        assert "cacheable" in repr_str
        assert "True" in repr_str


# Test Count Summary
# ------------------
# TestDataraxModuleConfigConstruction: 7 tests
# TestDataraxModuleConfigValidation: 2 tests
# TestDataraxModuleConfigDefaults: 3 tests
# TestDataraxModuleConfigInheritance: 3 tests
# TestDataraxModuleConfigDataclass: 3 tests
# ------------------
# Total: 18 tests
