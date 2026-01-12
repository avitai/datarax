"""Tests for TOML configuration loaders."""

import tomllib
from pathlib import Path

import pytest

from datarax.config.loaders import (
    deep_merge_dict,
    load_config_with_includes,
    load_toml,
    save_toml,
)


class TestLoadToml:
    """Tests for load_toml function."""

    def test_load_valid_toml(self, tmp_path: Path):
        """Test loading a valid TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[section]\nkey = "value"\nnumber = 42\n')

        result = load_toml(config_file)

        assert result == {"section": {"key": "value", "number": 42}}

    def test_load_toml_with_str_path(self, tmp_path: Path):
        """Test loading TOML with string path."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[test]\nvalue = "hello"\n')

        result = load_toml(str(config_file))

        assert result == {"test": {"value": "hello"}}

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Test loading non-existent file raises FileNotFoundError."""
        nonexistent = tmp_path / "missing.toml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_toml(nonexistent)

        assert "Configuration file not found" in str(exc_info.value)
        assert str(nonexistent) in str(exc_info.value)

    def test_load_invalid_toml(self, tmp_path: Path):
        """Test loading invalid TOML raises TOMLDecodeError."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("[section\nkey = value")  # Malformed TOML

        with pytest.raises(tomllib.TOMLDecodeError):
            load_toml(config_file)

    def test_load_empty_toml(self, tmp_path: Path):
        """Test loading empty TOML file."""
        config_file = tmp_path / "empty.toml"
        config_file.write_text("")

        result = load_toml(config_file)

        assert result == {}

    def test_load_complex_toml(self, tmp_path: Path):
        """Test loading TOML with nested structures."""
        config_file = tmp_path / "complex.toml"
        config_content = """
[database]
host = "localhost"
port = 5432

[database.credentials]
username = "admin"
password = "secret"

[[servers]]
name = "server1"
ip = "192.168.1.1"

[[servers]]
name = "server2"
ip = "192.168.1.2"
"""
        config_file.write_text(config_content)

        result = load_toml(config_file)

        assert result["database"]["host"] == "localhost"
        assert result["database"]["credentials"]["username"] == "admin"
        assert len(result["servers"]) == 2
        assert result["servers"][0]["name"] == "server1"


class TestSaveToml:
    """Tests for save_toml function."""

    def test_save_simple_config(self, tmp_path: Path):
        """Test saving a simple configuration."""
        config = {"key": "value", "number": 42}
        config_file = tmp_path / "output.toml"

        save_toml(config, config_file)

        assert config_file.exists()
        loaded = load_toml(config_file)
        assert loaded == config

    def test_save_with_str_path(self, tmp_path: Path):
        """Test saving with string path."""
        config = {"test": "data"}
        config_file = tmp_path / "output.toml"

        save_toml(config, str(config_file))

        assert config_file.exists()
        loaded = load_toml(config_file)
        assert loaded == config

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Test that save_toml creates parent directories if they don't exist."""
        config = {"data": "value"}
        config_file = tmp_path / "nested" / "deep" / "config.toml"

        save_toml(config, config_file)

        assert config_file.exists()
        assert config_file.parent.exists()
        loaded = load_toml(config_file)
        assert loaded == config

    def test_save_nested_config(self, tmp_path: Path):
        """Test saving nested configuration."""
        config = {
            "section": {"key": "value", "nested": {"deep": "data"}},
            "list": [1, 2, 3],
        }
        config_file = tmp_path / "nested.toml"

        save_toml(config, config_file)

        loaded = load_toml(config_file)
        assert loaded == config

    def test_save_overwrites_existing(self, tmp_path: Path):
        """Test that save_toml overwrites existing files."""
        config_file = tmp_path / "existing.toml"
        config_file.write_text("old = 'data'\n")

        new_config = {"new": "data"}
        save_toml(new_config, config_file)

        loaded = load_toml(config_file)
        assert loaded == new_config
        assert "old" not in loaded


class TestDeepMergeDict:
    """Tests for deep_merge_dict function."""

    def test_merge_flat_dicts(self):
        """Test merging flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = deep_merge_dict(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_does_not_modify_originals(self):
        """Test that merge does not modify original dictionaries."""
        base = {"a": 1}
        override = {"b": 2}

        result = deep_merge_dict(base, override)

        assert base == {"a": 1}
        assert override == {"b": 2}
        assert result == {"a": 1, "b": 2}

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {"level1": {"a": 1, "b": 2}}
        override = {"level1": {"b": 3, "c": 4}}

        result = deep_merge_dict(base, override)

        assert result == {"level1": {"a": 1, "b": 3, "c": 4}}

    def test_merge_deep_nested_dicts(self):
        """Test merging deeply nested dictionaries."""
        base = {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}
        override = {"l1": {"l2": {"l3": {"b": 3, "c": 4}}}}

        result = deep_merge_dict(base, override)

        assert result == {"l1": {"l2": {"l3": {"a": 1, "b": 3, "c": 4}}}}

    def test_merge_override_with_non_dict(self):
        """Test that non-dict values override dict values."""
        base = {"key": {"nested": "value"}}
        override = {"key": "simple"}

        result = deep_merge_dict(base, override)

        assert result == {"key": "simple"}

    def test_merge_empty_dicts(self):
        """Test merging with empty dictionaries."""
        base = {"a": 1}
        override = {}

        result = deep_merge_dict(base, override)
        assert result == {"a": 1}

        result = deep_merge_dict({}, base)
        assert result == {"a": 1}

    def test_merge_with_lists(self):
        """Test merging with list values."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        result = deep_merge_dict(base, override)

        # Lists are replaced, not merged
        assert result == {"items": [4, 5]}


class TestLoadConfigWithIncludes:
    """Tests for load_config_with_includes function."""

    def test_load_without_includes(self, tmp_path: Path):
        """Test loading config without include directives."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[section]\nkey = "value"\n')

        result = load_config_with_includes(config_file)

        assert result == {"section": {"key": "value"}}

    def test_load_with_single_include(self, tmp_path: Path):
        """Test loading config with single include."""
        # Create included file
        included_file = tmp_path / "base.toml"
        included_file.write_text("[defaults]\nport = 8080\n")

        # Create main file with include
        main_file = tmp_path / "config.toml"
        main_file.write_text('include = "base.toml"\n[app]\nname = "test"\n')

        result = load_config_with_includes(main_file)

        assert result == {"defaults": {"port": 8080}, "app": {"name": "test"}}

    def test_load_with_multiple_includes(self, tmp_path: Path):
        """Test loading config with multiple includes."""
        # Create included files
        base1 = tmp_path / "base1.toml"
        base1.write_text("[config1]\nvalue1 = 1\n")

        base2 = tmp_path / "base2.toml"
        base2.write_text("[config2]\nvalue2 = 2\n")

        # Create main file
        main_file = tmp_path / "config.toml"
        main_file.write_text('include = ["base1.toml", "base2.toml"]\n[app]\nname = "test"\n')

        result = load_config_with_includes(main_file)

        assert result["config1"]["value1"] == 1
        assert result["config2"]["value2"] == 2
        assert result["app"]["name"] == "test"

    def test_include_overrides_included_values(self, tmp_path: Path):
        """Test that main config overrides included values."""
        included = tmp_path / "base.toml"
        included.write_text('[section]\nkey = "from_base"\n')

        main = tmp_path / "config.toml"
        main.write_text('include = "base.toml"\n[section]\nkey = "from_main"\n')

        result = load_config_with_includes(main)

        assert result["section"]["key"] == "from_main"

    def test_nested_includes(self, tmp_path: Path):
        """Test loading configs with nested includes."""
        # Level 2
        level2 = tmp_path / "level2.toml"
        level2.write_text("[level2]\nvalue = 2\n")

        # Level 1 includes level 2
        level1 = tmp_path / "level1.toml"
        level1.write_text('include = "level2.toml"\n[level1]\nvalue = 1\n')

        # Main includes level 1
        main = tmp_path / "main.toml"
        main.write_text('include = "level1.toml"\n[main]\nvalue = 0\n')

        result = load_config_with_includes(main)

        assert result["level2"]["value"] == 2
        assert result["level1"]["value"] == 1
        assert result["main"]["value"] == 0

    def test_circular_include_detection(self, tmp_path: Path):
        """Test that circular includes are detected."""
        # Create circular includes
        config1 = tmp_path / "config1.toml"
        config1.write_text('include = "config2.toml"\n[c1]\nvalue = 1\n')

        config2 = tmp_path / "config2.toml"
        config2.write_text('include = "config1.toml"\n[c2]\nvalue = 2\n')

        with pytest.raises(RecursionError) as exc_info:
            load_config_with_includes(config1)

        assert "Circular include detected" in str(exc_info.value)

    def test_include_with_custom_key(self, tmp_path: Path):
        """Test using custom include key."""
        included = tmp_path / "base.toml"
        included.write_text("[base]\nvalue = 1\n")

        main = tmp_path / "config.toml"
        main.write_text('imports = "base.toml"\n[main]\nvalue = 2\n')

        result = load_config_with_includes(main, include_key="imports")

        assert result["base"]["value"] == 1
        assert result["main"]["value"] == 2

    def test_include_removes_include_key(self, tmp_path: Path):
        """Test that include key is removed from final config."""
        included = tmp_path / "base.toml"
        included.write_text("[base]\nvalue = 1\n")

        main = tmp_path / "config.toml"
        main.write_text('include = "base.toml"\n[main]\nvalue = 2\n')

        result = load_config_with_includes(main)

        assert "include" not in result
        assert result["base"]["value"] == 1

    def test_include_missing_file(self, tmp_path: Path):
        """Test including a non-existent file raises error."""
        main = tmp_path / "config.toml"
        main.write_text('include = "missing.toml"\n')

        with pytest.raises(FileNotFoundError):
            load_config_with_includes(main)
