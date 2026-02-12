"""Tests for TOML configuration loading utility.

Validates that scenarios and hardware profiles load correctly from TOML files,
and that error handling works for missing files/profiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.core.config_loader import load_hardware_profile, load_scenarios


class TestLoadScenarios:
    """Tests for load_scenarios()."""

    def test_loads_default_scenarios(self):
        """Loading with no args reads benchmarks/config/scenarios.toml."""
        scenarios = load_scenarios()
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0

    def test_contains_expected_scenario_ids(self):
        """Default scenarios include known IDs from the benchmark report."""
        scenarios = load_scenarios()
        for scenario_id in ("CV-1", "NLP-1", "TAB-1"):
            assert scenario_id in scenarios, f"Missing {scenario_id}"

    def test_scenario_has_name_field(self):
        """Each scenario entry must have a 'name' field."""
        scenarios = load_scenarios()
        for sid, cfg in scenarios.items():
            assert "name" in cfg, f"{sid} missing 'name'"

    def test_custom_path(self, tmp_path: Path):
        """Loading from a custom path reads the specified file."""
        toml_content = b'[scenario.TEST-1]\nname = "Test"\n'
        toml_file = tmp_path / "custom.toml"
        toml_file.write_bytes(toml_content)

        scenarios = load_scenarios(config_path=toml_file)
        assert "TEST-1" in scenarios
        assert scenarios["TEST-1"]["name"] == "Test"

    def test_missing_file_raises(self, tmp_path: Path):
        """Loading from a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scenarios(config_path=tmp_path / "nonexistent.toml")

    def test_empty_toml_returns_empty(self, tmp_path: Path):
        """A TOML file with no [scenario] section returns empty dict."""
        toml_file = tmp_path / "empty.toml"
        toml_file.write_bytes(b"")
        scenarios = load_scenarios(config_path=toml_file)
        assert scenarios == {}


class TestLoadHardwareProfile:
    """Tests for load_hardware_profile()."""

    def test_loads_ci_cpu_profile(self):
        """The ci_cpu profile must exist and load successfully."""
        profile = load_hardware_profile("ci_cpu")
        assert isinstance(profile, dict)
        assert "profile" in profile

    def test_ci_cpu_has_required_settings(self):
        """ci_cpu profile must have num_batches and warmup_batches."""
        profile = load_hardware_profile("ci_cpu")
        settings = profile["profile"]
        assert "num_batches" in settings
        assert "warmup_batches" in settings
        assert isinstance(settings["num_batches"], int)
        assert isinstance(settings["warmup_batches"], int)

    def test_missing_profile_raises(self):
        """Loading a nonexistent profile raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_hardware_profile("nonexistent_profile_xyz")
