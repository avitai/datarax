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

from dataclasses import fields

import pytest

from datarax.core.config import DataraxModuleConfig, OperatorConfig


class TestOperatorConfigStochasticConstruction:
    """Test valid stochastic operator config construction."""

    def test_stochastic_with_stream_name(self):
        """Test valid stochastic config with stream_name."""
        config = OperatorConfig(stochastic=True, stream_name="augment")

        assert config.stochastic is True
        assert config.stream_name == "augment"


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
        """Test that OperatorConfig runs its parent's __post_init__."""
        # The parent hook runs before the operator's own rules, which reject this config
        with pytest.raises(ValueError) as exc_info:
            OperatorConfig(stochastic=False, stream_name="augment")

        error_msg = str(exc_info.value).lower()
        assert "stream_name" in error_msg

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

    def test_declares_only_operator_fields(self):
        """An operator config says how to run the operator, never what it fitted to data.

        Statistics live on the operator itself, so a config stays static metadata that every
        transform can compare.
        """
        assert {field.name for field in fields(OperatorConfig)} == {
            "stochastic",
            "stream_name",
            "batch_strategy",
        }

    def test_operator_field_defaults(self):
        """Test that operator field defaults are preserved."""
        config = OperatorConfig()

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

        @dataclass(frozen=True)
        class RandomBrightnessConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
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

        @dataclass(frozen=True)
        class ChildConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
            """Simple child config."""

            extra_param: float = 1.0

        # Should enforce stochastic requires stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=True)  # Missing stream_name

        # Should enforce deterministic forbids stream_name
        with pytest.raises(ValueError):
            ChildConfig(stochastic=False, stream_name="augment")

    def test_child_config_custom_validation(self):
        """Test child config can add its own validation."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MixupConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
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

        @dataclass(frozen=True)
        class OrderTestConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
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

    def test_is_frozen(self):
        """Test that OperatorConfig is frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        config = OperatorConfig()

        # Should NOT be able to modify fields (frozen)
        with pytest.raises(FrozenInstanceError):
            config.stochastic = True  # type: ignore[reportAttributeAccessIssue]

    def test_repr_shows_all_fields(self):
        """Test that __repr__ includes all field values."""
        config = OperatorConfig(stochastic=True, stream_name="augment")
        repr_str = repr(config)

        assert "OperatorConfig" in repr_str
        assert "stochastic" in repr_str
        assert "stream_name" in repr_str


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

    def test_batch_strategy_defaults_to_vmap(self):
        """Test batch_strategy defaults to 'vmap'."""
        config = OperatorConfig()
        assert config.batch_strategy == "vmap"
