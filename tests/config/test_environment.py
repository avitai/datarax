"""Environment variables override values of a loaded configuration.

An override lives under its own prefix, ``DATARAX_CONFIG__``, apart from the operational
``DATARAX_*`` variables (test, benchmark and activation settings), and names a value the
configuration already has: the string is read as that value's type. Only ``true``/``false`` are
booleans, so ``DATARAX_CONFIG__BATCH_SIZE=1`` is the integer 1. A value that does not read as
its type, or a name the configuration does not have, is refused rather than turned into a new
key. The caller's configuration is never modified.
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
    """``apply_environment_overrides`` over an explicit ``environ`` mapping."""

    def test_a_top_level_value_is_replaced(self) -> None:
        result = apply_environment_overrides(
            {"host": "default.com", "port": 8080}, environ={"DATARAX_CONFIG__HOST": "localhost"}
        )
        assert result == {"host": "localhost", "port": 8080}

    def test_a_nested_value_is_replaced_and_its_siblings_kept(self) -> None:
        config = {"db": {"host": "old", "port": 5432, "pool": {"size": 10}}, "api": {"key": "k"}}
        result = apply_environment_overrides(
            config,
            environ={"DATARAX_CONFIG__DB__HOST": "new", "DATARAX_CONFIG__DB__POOL__SIZE": "20"},
        )
        assert result == {
            "db": {"host": "new", "port": 5432, "pool": {"size": 20}},
            "api": {"key": "k"},
        }

    def test_the_callers_configuration_is_not_modified_at_any_depth(self) -> None:
        config = {"sources": {"images": {"path": "/data"}}, "batch_size": 32}
        result = apply_environment_overrides(
            config, environ={"DATARAX_CONFIG__SOURCES__IMAGES__PATH": "/elsewhere"}
        )
        assert config == {"sources": {"images": {"path": "/data"}}, "batch_size": 32}
        assert result["sources"]["images"]["path"] == "/elsewhere"
        assert result["sources"] is not config["sources"]

    @pytest.mark.parametrize(
        ("current", "text", "expected"),
        [
            (8, "1", 1),
            (8, "0", 0),
            (8, "-42", -42),
            (8, "00123", 123),
            (0.1, "0.95", 0.95),
            (0.1, "1", 1.0),
            (0.1, "-273.15", -273.15),
            (False, "true", True),
            (False, "TRUE", True),
            (True, "false", False),
            (True, "False", False),
            ("name", "1", "1"),
            ("name", "true", "true"),
            ("/data", "/path/to/data", "/path/to/data"),
            ("u", "http://example.com/__path__/resource", "http://example.com/__path__/resource"),
            ([1, 2], "[3, 4, 5]", [3, 4, 5]),
            (["a"], '["b", "c"]', ["b", "c"]),
        ],
    )
    def test_the_string_is_read_as_the_type_of_the_value_it_replaces(
        self, current: Any, text: str, expected: Any
    ) -> None:
        """The replaced value's type decides: ``"1"`` for an integer is 1, not ``True``.

        Reading ``"1"``/``"0"``/``"yes"``/``"no"`` as booleans regardless of the value would set
        a batch size of ``True`` from ``DATARAX_CONFIG__BATCH_SIZE=1``; only ``true``/``false``
        are booleans, and only where the replaced value is one.
        """
        result = apply_environment_overrides(
            {"value": current}, environ={"DATARAX_CONFIG__VALUE": text}
        )
        assert result["value"] == expected
        assert type(result["value"]) is type(expected)

    @pytest.mark.parametrize(
        ("current", "text"),
        [
            (8, "yes"),
            (8, "1.5"),
            (8, "true"),
            (0.1, "fast"),
            (False, "1"),
            (False, "yes"),
            ([1, 2], "3"),
            ([1, 2], "[1, "),
        ],
    )
    def test_a_string_that_does_not_read_as_the_type_is_refused(
        self, current: Any, text: str
    ) -> None:
        with pytest.raises(ValueError, match="DATARAX_CONFIG__VALUE"):
            apply_environment_overrides({"value": current}, environ={"DATARAX_CONFIG__VALUE": text})

    def test_a_table_is_not_replaced_by_a_scalar(self) -> None:
        with pytest.raises(ValueError, match="DATARAX_CONFIG__DB"):
            apply_environment_overrides({"db": {"host": "x"}}, environ={"DATARAX_CONFIG__DB": "y"})

    @pytest.mark.parametrize(
        "name", ["DATARAX_CONFIG__EPOCHS", "DATARAX_CONFIG__DB__USER", "DATARAX_CONFIG__HOST__NAME"]
    )
    def test_a_name_the_configuration_does_not_have_is_refused(self, name: str) -> None:
        with pytest.raises(KeyError, match=name):
            apply_environment_overrides({"host": "h", "db": {"host": "x"}}, environ={name: "1"})

    def test_operational_variables_outside_the_prefix_are_not_configuration(self) -> None:
        environ = {
            "DATARAX_EXAMPLE_TIMEOUT_SECONDS": "300",
            "DATARAX_BACKEND": "cuda12",
            "DATARAX_TEST_JAX_PLATFORMS": "cpu",
            "OTHER_VAR": "x",
        }
        assert apply_environment_overrides({"host": "h"}, environ=environ) == {"host": "h"}

    def test_names_are_matched_case_insensitively(self) -> None:
        result = apply_environment_overrides(
            {"database": {"host": "x"}}, environ={"DATARAX_CONFIG__DATABASE__HOST": "db.com"}
        )
        assert result == {"database": {"host": "db.com"}}

    def test_a_custom_prefix_and_separator(self) -> None:
        result = apply_environment_overrides(
            {"database": {"host": "x"}},
            prefix="MYAPP_",
            separator=".",
            environ={"MYAPP_DATABASE.HOST": "db.com"},
        )
        assert result == {"database": {"host": "db.com"}}

    def test_the_process_environment_is_read_when_no_mapping_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATARAX_CONFIG__PORT", "9000")
        assert apply_environment_overrides({"port": 8080}) == {"port": 9000}
