"""Tests for OperatorConfig configuration class.

This test suite validates OperatorConfig used by OperatorModule for parametric,
differentiable data transformations. Tests stochastic/deterministic validation,
config inheritance, and operator-specific parameters.

Test Categories:
- Valid config construction (stochastic and deterministic)
- Validation rules (__post_init__)
- Child config inheritance
- Field access and defaults
"""

import pytest
from typing import Any


# NOTE: Import will fail initially (RED phase) - this is expected!
try:
    from datarax.core.config import DataraxModuleConfig, OperatorConfig
except ImportError:
    DataraxModuleConfig = None
    OperatorConfig = None


pytestmark = pytest.mark.skipif(
    OperatorConfig is None,
    reason="OperatorConfig not implemented yet (RED phase)",
)


class TestOperatorConfigStochasticConstruction:
    """Test valid stochastic operator config construction."""

    def test_stochastic_with_stream_name(self):
        """Test valid stochastic config with stream_name."""
        config = OperatorConfig(stochastic=True, stream_name="augment")

        assert config.stochastic is True
        assert config.stream_name == "augment"
        assert config.cacheable is False  # Inherited default

    def test_stochastic_with_stream_name_and_cacheable(self):
        """Test stochastic config with caching enabled."""
        config = OperatorConfig(stochastic=True, stream_name="augment", cacheable=True)

        assert config.stochastic is True
        assert config.stream_name == "augment"
        assert config.cacheable is True

    def test_stochastic_with_batch_stats_fn(self):
        """Test stochastic config with statistics function."""

        def compute_stats(batch: Any) -> dict[str, Any]:
            return {"mean": 0.5}

        config = OperatorConfig(
            stochastic=True, stream_name="augment", batch_stats_fn=compute_stats
        )

        assert config.stochastic is True
        assert config.batch_stats_fn is compute_stats

    def test_stochastic_with_precomputed_stats(self):
        """Test stochastic config with precomputed statistics."""
        stats = {"mean": 0.5, "std": 0.2}
        config = OperatorConfig(stochastic=True, stream_name="augment", precomputed_stats=stats)

        assert config.stochastic is True
        assert config.precomputed_stats == stats


class TestOperatorConfigDeterministicConstruction:
    """Test valid deterministic operator config construction."""

    def test_deterministic_without_stream_name(self):
        """Test valid deterministic config without stream_name."""
        config = OperatorConfig(stochastic=False)

        assert config.stochastic is False
        assert config.stream_name is None

    def test_deterministic_default(self):
        """Test that stochastic=False is the default."""
        config = OperatorConfig()

        assert config.stochastic is False
        assert config.stream_name is None

    def test_deterministic_with_cacheable(self):
        """Test deterministic config with caching enabled."""
        config = OperatorConfig(stochastic=False, cacheable=True)

        assert config.stochastic is False
        assert config.cacheable is True

    def test_deterministic_with_precomputed_stats(self):
        """Test deterministic config with precomputed statistics."""
        stats = {"mean": 0.5, "std": 0.2}
        config = OperatorConfig(stochastic=False, precomputed_stats=stats)

        assert config.stochastic is False
        assert config.precomputed_stats == stats


class TestOperatorConfigValidationRules:
    """Test __post_init__ validation rules for OperatorConfig."""

    def test_stochastic_requires_stream_name(self):
        """Test that stochastic=True requires stream_name."""
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(stochastic=True)  # Missing stream_name

        error_msg = str(exc_info.value).lower()
        assert "stochastic" in error_msg
        assert "stream_name" in error_msg
        assert "require" in error_msg

    def test_deterministic_forbids_stream_name(self):
        """Test that stochastic=False forbids stream_name."""
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(
                stochastic=False,
                stream_name="augment",  # Should not be specified
            )

        error_msg = str(exc_info.value).lower()
        assert "deterministic" in error_msg or "should not" in error_msg
        assert "stream_name" in error_msg

    def test_inherits_parent_validation(self):
        """Test that OperatorConfig inherits DataraxModuleConfig validation."""
        # Should still enforce mutual exclusivity of statistics
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(
                stochastic=True,
                stream_name="augment",
                batch_stats_fn=lambda x: {},
                precomputed_stats={},
            )

        error_msg = str(exc_info.value).lower()
        assert "both" in error_msg

    def test_validation_error_messages_are_helpful(self):
        """Test that validation errors provide actionable guidance."""
        # Error 1: Stochastic without stream_name
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(stochastic=True)

        msg = str(exc_info.value)
        # Should explain what's needed
        assert "stream_name" in msg.lower()

        # Error 2: Deterministic with stream_name
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(stochastic=False, stream_name="augment")

        msg = str(exc_info.value)
        # Should explain what to do
        assert "remove" in msg.lower() or "should not" in msg.lower()


class TestOperatorConfigInheritance:
    """Test OperatorConfig inheritance from DataraxModuleConfig."""

    def test_inherits_all_base_fields(self):
        """Test that OperatorConfig has all DataraxModuleConfig fields."""
        config = OperatorConfig()

        # Base fields should be accessible
        assert hasattr(config, "cacheable")
        assert hasattr(config, "batch_stats_fn")
        assert hasattr(config, "precomputed_stats")

        # Operator fields should be accessible
        assert hasattr(config, "stochastic")
        assert hasattr(config, "stream_name")

    def test_base_field_defaults_preserved(self):
        """Test that base field defaults are preserved."""
        config = OperatorConfig()

        # Base defaults
        assert config.cacheable is False
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None

        # Operator defaults
        assert config.stochastic is False
        assert config.stream_name is None

    def test_is_instance_of_base(self):
        """Test that OperatorConfig is instance of DataraxModuleConfig."""
        config = OperatorConfig()
        assert isinstance(config, DataraxModuleConfig)


class TestOperatorConfigChildInheritance:
    """Test child configs inheriting from OperatorConfig."""

    def test_child_config_adds_fields(self):
        """Test child config can add operator-specific fields."""
        from dataclasses import dataclass

        @dataclass
        class RandomBrightnessConfig(OperatorConfig):
            """Child config with brightness-specific parameters."""

            min_factor: float = 0.8
            max_factor: float = 1.2

            def __post_init__(self):
                super().__post_init__()  # Parent validation
                if self.min_factor >= self.max_factor:
                    raise ValueError("min_factor must be < max_factor")

        # Valid construction
        config = RandomBrightnessConfig(
            stochastic=True, stream_name="augment", min_factor=0.8, max_factor=1.2
        )

        assert config.stochastic is True
        assert config.stream_name == "augment"
        assert config.min_factor == 0.8
        assert config.max_factor == 1.2

    def test_child_config_inherits_validation(self):
        """Test child config inherits all parent validation rules."""
        from dataclasses import dataclass

        @dataclass
        class ChildConfig(OperatorConfig):
            """Simple child config."""

            extra_param: float = 1.0

        # Should enforce stochastic requires stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=True)  # Missing stream_name

        # Should enforce deterministic forbids stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=False, stream_name="augment")

        # Should enforce statistics mutual exclusivity
        with pytest.raises(ValueError):
            ChildConfig(batch_stats_fn=lambda x: {}, precomputed_stats={})

    def test_child_config_custom_validation(self):
        """Test child config can add its own validation."""
        from dataclasses import dataclass

        @dataclass
        class MixupConfig(OperatorConfig):
            """Mixup augmentation config."""

            alpha: float = 1.0
            num_classes: int = 10

            def __post_init__(self):
                super().__post_init__()  # Parent validation first

                if self.alpha <= 0:
                    raise ValueError("Mixup alpha must be positive")

                if self.num_classes < 2:
                    raise ValueError("Mixup requires at least 2 classes")

        # Parent validation works
        with pytest.raises(ValueError):
            MixupConfig(stochastic=True)  # Missing stream_name

        # Child validation works
        with pytest.raises(ValueError) as exc_info:
            MixupConfig(
                stochastic=True,
                stream_name="augment",
                alpha=-0.5,  # Invalid
            )

        assert "alpha" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            MixupConfig(
                stochastic=True,
                stream_name="augment",
                num_classes=1,  # Invalid
            )

        assert "classes" in str(exc_info.value).lower()

    def test_child_config_validation_order(self):
        """Test that parent validation runs before child validation."""
        from dataclasses import dataclass

        @dataclass
        class OrderTestConfig(OperatorConfig):
            """Config to test validation order."""

            test_field: float = 1.0

            def __post_init__(self):
                # Parent validation should run first
                super().__post_init__()

                # This should only run if parent validation passed
                if self.test_field < 0:
                    raise ValueError("test_field must be non-negative")

        # Parent validation should fail first (missing stream_name)
        with pytest.raises(ValueError) as exc_info:
            OrderTestConfig(
                stochastic=True,  # Missing stream_name
                test_field=-1.0,  # Also invalid
            )

        # Error should be about stream_name (parent), not test_field (child)
        error_msg = str(exc_info.value).lower()
        assert "stream_name" in error_msg


class TestOperatorConfigDataclass:
    """Test OperatorConfig dataclass properties."""

    def test_is_dataclass(self):
        """Test that OperatorConfig is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(OperatorConfig)

    def test_not_frozen(self):
        """Test that OperatorConfig is mutable (not frozen)."""
        config = OperatorConfig()

        # Should be able to modify (mutable for learnable parameters)
        config.cacheable = True
        assert config.cacheable is True

    def test_repr_shows_all_fields(self):
        """Test that __repr__ includes all field values."""
        config = OperatorConfig(stochastic=True, stream_name="augment", cacheable=True)
        repr_str = repr(config)

        assert "OperatorConfig" in repr_str
        assert "stochastic" in repr_str
        assert "stream_name" in repr_str
        assert "cacheable" in repr_str


class TestOperatorConfigDefaults:
    """Test OperatorConfig default values."""

    def test_stochastic_defaults_to_false(self):
        """Test stochastic defaults to False."""
        config = OperatorConfig()
        assert config.stochastic is False

    def test_stream_name_defaults_to_none(self):
        """Test stream_name defaults to None."""
        config = OperatorConfig()
        assert config.stream_name is None

    def test_inherited_defaults(self):
        """Test inherited defaults from DataraxModuleConfig."""
        config = OperatorConfig()
        assert config.cacheable is False
        assert config.batch_stats_fn is None
        assert config.precomputed_stats is None


# Test Count Summary
# ------------------
# TestOperatorConfigStochasticConstruction: 4 tests
# TestOperatorConfigDeterministicConstruction: 4 tests
# TestOperatorConfigValidationRules: 4 tests
# TestOperatorConfigInheritance: 3 tests
# TestOperatorConfigChildInheritance: 4 tests
# TestOperatorConfigDataclass: 3 tests
# TestOperatorConfigDefaults: 3 tests
# ------------------
# Total: 25 tests
