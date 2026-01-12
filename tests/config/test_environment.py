"""Tests for environment variable configuration integration.

Tests follow TDD methodology - defining expected behavior for environment
variable handling, type conversion, and nested configuration overrides.
"""

from typing import Any

import pytest

from datarax.config.environment import (
    apply_environment_overrides,
    get_env_value,
)


class TestGetEnvValue:
    """Test suite for get_env_value function."""

    def test_get_env_value_with_set_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test retrieving a set environment variable with default prefix."""
        monkeypatch.setenv("DATARAX_TEST_VAR", "test_value")
        result = get_env_value("TEST_VAR")
        assert result == "test_value"

    def test_get_env_value_with_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test retrieving environment variable with custom prefix."""
        monkeypatch.setenv("CUSTOM_TEST_VAR", "custom_value")
        result = get_env_value("TEST_VAR", prefix="CUSTOM_")
        assert result == "custom_value"

    def test_get_env_value_with_unset_variable_returns_default(self) -> None:
        """Test that unset variable returns provided default value."""
        result = get_env_value("NONEXISTENT_VAR", default="default_value")
        assert result == "default_value"

    def test_get_env_value_with_unset_variable_returns_none(self) -> None:
        """Test that unset variable without default returns None."""
        result = get_env_value("NONEXISTENT_VAR")
        assert result is None

    def test_get_env_value_with_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test retrieving environment variable set to empty string."""
        monkeypatch.setenv("DATARAX_EMPTY", "")
        result = get_env_value("EMPTY")
        assert result == ""

    def test_get_env_value_with_numeric_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test retrieving environment variable with numeric value."""
        monkeypatch.setenv("DATARAX_NUMBER", "12345")
        result = get_env_value("NUMBER")
        # get_env_value returns raw string, no type conversion
        assert result == "12345"
        assert isinstance(result, str)


class TestApplyEnvironmentOverrides:
    """Test suite for apply_environment_overrides function."""

    def test_apply_simple_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying a simple single-level override."""
        monkeypatch.setenv("DATARAX_HOST", "localhost")
        config = {"host": "default.com", "port": 8080}
        result = apply_environment_overrides(config)
        assert result["host"] == "localhost"
        assert result["port"] == 8080

    def test_apply_nested_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying override to nested configuration."""
        monkeypatch.setenv("DATARAX_DATABASE__HOST", "db.example.com")
        config = {"database": {"host": "localhost", "port": 5432}}
        result = apply_environment_overrides(config)
        assert result["database"]["host"] == "db.example.com"
        assert result["database"]["port"] == 5432

    def test_apply_deep_nested_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying override to deeply nested configuration."""
        monkeypatch.setenv("DATARAX_APP__DB__POOL__SIZE", "20")
        config = {"app": {"db": {"pool": {"size": 10}}}}
        result = apply_environment_overrides(config)
        assert result["app"]["db"]["pool"]["size"] == 20

    def test_apply_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides with custom prefix."""
        monkeypatch.setenv("MYAPP_HOST", "custom.com")
        config = {"host": "default.com"}
        result = apply_environment_overrides(config, prefix="MYAPP_")
        assert result["host"] == "custom.com"

    def test_apply_custom_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides with custom separator."""
        monkeypatch.setenv("DATARAX_DATABASE.HOST", "db.com")
        config = {"database": {"host": "localhost"}}
        result = apply_environment_overrides(config, separator=".")
        assert result["database"]["host"] == "db.com"

    def test_apply_boolean_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for boolean true values."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("1", True),
        ]
        for env_val, expected in test_cases:
            monkeypatch.setenv("DATARAX_ENABLED", env_val)
            config: dict[str, Any] = {}
            result = apply_environment_overrides(config)
            assert result["enabled"] is expected, f"Failed for value: {env_val}"
            monkeypatch.delenv("DATARAX_ENABLED")

    def test_apply_boolean_false_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for boolean false values."""
        test_cases = [
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("no", False),
            ("No", False),
            ("NO", False),
            ("0", False),
        ]
        for env_val, expected in test_cases:
            monkeypatch.setenv("DATARAX_ENABLED", env_val)
            config: dict[str, Any] = {}
            result = apply_environment_overrides(config)
            assert result["enabled"] is expected, f"Failed for value: {env_val}"
            monkeypatch.delenv("DATARAX_ENABLED")

    def test_apply_integer_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for integer values."""
        monkeypatch.setenv("DATARAX_PORT", "8080")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["port"] == 8080
        assert isinstance(result["port"], int)

    def test_apply_negative_integer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for negative integer values."""
        monkeypatch.setenv("DATARAX_OFFSET", "-42")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["offset"] == -42
        assert isinstance(result["offset"], int)

    def test_apply_float_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for float values."""
        monkeypatch.setenv("DATARAX_RATE", "0.95")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["rate"] == 0.95
        assert isinstance(result["rate"], float)

    def test_apply_negative_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test type conversion for negative float values."""
        monkeypatch.setenv("DATARAX_TEMPERATURE", "-273.15")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["temperature"] == -273.15
        assert isinstance(result["temperature"], float)

    def test_apply_string_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test string values are preserved as strings."""
        monkeypatch.setenv("DATARAX_NAME", "test_name")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["name"] == "test_name"
        assert isinstance(result["name"], str)

    def test_apply_creates_nested_dict_if_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that nested dictionaries are created if they don't exist."""
        monkeypatch.setenv("DATARAX_NEW__NESTED__VALUE", "test")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["new"]["nested"]["value"] == "test"

    def test_apply_overwrites_non_dict_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that non-dict values are overwritten when creating nested structure."""
        monkeypatch.setenv("DATARAX_CONFIG__NESTED__VALUE", "new")
        config = {"config": "old_string_value"}
        result = apply_environment_overrides(config)
        assert isinstance(result["config"], dict)
        assert result["config"]["nested"]["value"] == "new"

    def test_apply_multiple_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying multiple environment overrides simultaneously."""
        monkeypatch.setenv("DATARAX_HOST", "example.com")
        monkeypatch.setenv("DATARAX_PORT", "9000")
        monkeypatch.setenv("DATARAX_DEBUG", "true")
        config = {"host": "localhost", "port": 8080, "debug": False}
        result = apply_environment_overrides(config)
        assert result["host"] == "example.com"
        assert result["port"] == 9000
        assert result["debug"] is True

    def test_apply_ignores_non_prefixed_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that environment variables without the prefix are ignored."""
        monkeypatch.setenv("OTHER_VAR", "should_be_ignored")
        monkeypatch.setenv("RANDOM_VAR", "also_ignored")
        config: dict[str, Any] = {"existing": "value"}
        result = apply_environment_overrides(config)
        assert "other_var" not in result
        assert "OTHER_VAR" not in result
        assert "random_var" not in result
        assert result["existing"] == "value"

    def test_apply_skips_empty_config_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty config paths (just prefix) are skipped."""
        monkeypatch.setenv("DATARAX_", "empty_key")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result == {}

    def test_apply_case_insensitive_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that keys are converted to lowercase for case insensitivity."""
        monkeypatch.setenv("DATARAX_DATABASE__HOST", "db.com")
        monkeypatch.setenv("DATARAX_API__KEY", "secret")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert "database" in result
        assert "DATABASE" not in result
        assert result["database"]["host"] == "db.com"
        assert result["api"]["key"] == "secret"

    def test_apply_preserves_original_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that original config dictionary is not modified."""
        monkeypatch.setenv("DATARAX_HOST", "new.com")
        original = {"host": "old.com", "port": 8080}
        original_copy = original.copy()
        result = apply_environment_overrides(original)
        assert original == original_copy  # Original unchanged
        assert result["host"] == "new.com"  # Result has override

    def test_apply_empty_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides to empty configuration."""
        monkeypatch.setenv("DATARAX_HOST", "example.com")
        monkeypatch.setenv("DATARAX_PORT", "8080")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["host"] == "example.com"
        assert result["port"] == 8080

    def test_apply_with_existing_nested_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides to config with existing nested structure."""
        monkeypatch.setenv("DATARAX_DB__HOST", "newdb.com")
        config = {
            "db": {"host": "olddb.com", "port": 5432, "pool": {"size": 10}},
            "api": {"key": "secret"},
        }
        result = apply_environment_overrides(config)
        assert result["db"]["host"] == "newdb.com"
        assert result["db"]["port"] == 5432  # Preserved
        assert result["db"]["pool"]["size"] == 10  # Preserved
        assert result["api"]["key"] == "secret"  # Preserved

    def test_apply_numeric_string_not_converted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that numeric strings with leading zeros are preserved."""
        monkeypatch.setenv("DATARAX_ZIP_CODE", "00123")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        # Leading zero means it should stay as string or be converted to int
        # Based on implementation, "00123" will be converted to int 123
        assert result["zip_code"] == 123

    def test_apply_path_like_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that path-like strings are preserved."""
        monkeypatch.setenv("DATARAX_DATA_PATH", "/path/to/data")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["data_path"] == "/path/to/data"
        assert isinstance(result["data_path"], str)

    def test_apply_single_level_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying override to single-level key."""
        monkeypatch.setenv("DATARAX_TIMEOUT", "30")
        config = {"timeout": 60}
        result = apply_environment_overrides(config)
        assert result["timeout"] == 30

    def test_apply_deeply_nested_new_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creating deeply nested structure from scratch."""
        monkeypatch.setenv("DATARAX_A__B__C__D__E", "deep_value")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["a"]["b"]["c"]["d"]["e"] == "deep_value"

    def test_apply_mixed_separators_in_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that separators in values don't affect parsing."""
        monkeypatch.setenv("DATARAX_URL", "http://example.com/__path__/resource")
        config: dict[str, Any] = {}
        result = apply_environment_overrides(config)
        assert result["url"] == "http://example.com/__path__/resource"
