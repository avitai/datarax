"""Tests for DataraxModuleConfig base configuration class.

This test suite validates the base configuration used by all Datarax modules.
Tests config construction, validation, and inheritance patterns.

Test Categories:
- Config construction (valid cases)
- Validation rules (__post_init__)
- Field defaults
- Invalid configurations
"""

from dataclasses import fields

import pytest

from datarax.core.config import DataraxModuleConfig


class TestDataraxModuleConfigConstruction:
    """Test valid config construction with various parameter combinations."""

    def test_default_construction(self):
        """Test config with all defaults."""
        config = DataraxModuleConfig()

        assert isinstance(config, DataraxModuleConfig)

    def test_the_base_config_declares_no_fields(self):
        """A configuration is static metadata, so it carries no fitted values.

        Statistics moved to the operator that applies them, where they are state rather than
        part of the graphdef every transform compares.
        """
        assert [field.name for field in fields(DataraxModuleConfig)] == []


class TestDataraxModuleConfigInheritance:
    """Test config inheritance patterns for child configs."""

    def test_child_config_can_add_fields(self):
        """Test that child configs can add their own fields."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ChildConfig(DataraxModuleConfig):  # type: ignore[reportGeneralTypeIssues]
            """Child config with additional field."""

            extra_field: int = 42

        config = ChildConfig()
        assert config.extra_field == 42  # Child-specific

    def test_child_config_can_add_validation(self):
        """Test that child configs can add their own validation."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ChildConfig(DataraxModuleConfig):  # type: ignore[reportGeneralTypeIssues]
            """Child config with additional validation."""

            min_value: float = 0.0
            max_value: float = 1.0

            def __post_init__(self):
                super().__post_init__()  # Parent validation first
                if self.min_value >= self.max_value:
                    raise ValueError("min_value must be < max_value")

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

    def test_is_frozen(self):
        """Test that the base is frozen, so every config below it is immutable too."""
        from dataclasses import dataclass, FrozenInstanceError

        @dataclass(frozen=True)
        class ChildConfig(DataraxModuleConfig):  # type: ignore[reportGeneralTypeIssues]
            """Child config with one field."""

            extra_field: int = 42

        config = ChildConfig()
        with pytest.raises(FrozenInstanceError):
            config.extra_field = 7  # type: ignore[reportAttributeAccessIssue]

    def test_repr_includes_fields(self):
        """Test that __repr__ shows field values."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class ChildConfig(DataraxModuleConfig):  # type: ignore[reportGeneralTypeIssues]
            """Child config with one field."""

            mean: float = 0.5

        repr_str = repr(ChildConfig())

        assert "ChildConfig" in repr_str
        assert "mean=0.5" in repr_str
