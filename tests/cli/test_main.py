"""Tests for the Datarax CLI main module."""

from unittest.mock import patch

import pytest
from substrax.testing import restored_jax_config

from datarax.cli.main import main


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_help_command(self, capsys):
        """Test that help command works."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Datarax" in captured.out
        assert "high-performance data pipeline" in captured.out.lower()

    def test_version_command(self, capsys):
        """Test version command."""
        exit_code = main(["version"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Datarax version" in captured.out

    def test_no_command(self, capsys):
        """Test behavior when no command is provided."""
        exit_code = main([])

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should print help when no command given
        assert "usage:" in captured.out.lower() or "help" in captured.out.lower()

    def test_validate_command(self, tmp_path):
        """Test validate command for config validation."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test_pipeline"

[sources.train]
type = "memory"
        """)

        with patch("datarax.cli.main.is_config_valid") as mock_validate:
            mock_validate.return_value = True
            exit_code = main(["validate", "--config-path", str(config_file)])

        assert exit_code == 0
        mock_validate.assert_called_once()

    def test_invalid_command(self, capsys):
        """Test behavior with invalid command."""
        with pytest.raises(SystemExit) as exc_info:
            main(["invalid_command"])

        # argparse exits with code 2 for invalid arguments
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err.lower()

    def test_list_command(self, capsys):
        """Test list command to show available components."""
        del capsys
        with patch("datarax.cli.main.list_components") as mock_list:
            mock_list.return_value = {
                "sources": ["memory", "tfds", "huggingface"],
                "transforms": ["normalize", "augment", "batch"],
                "augmenters": ["brightness", "rotate", "noise"],
            }
            exit_code = main(["list", "--type", "sources"])

        assert exit_code == 0
        mock_list.assert_called_once()

    def test_create_command(self, tmp_path):
        """Test create command for generating pipeline templates."""
        output_file = tmp_path / "new_pipeline.toml"

        with patch("datarax.cli.main.is_pipeline_template_written_to_path") as mock_create:
            mock_create.return_value = True
            exit_code = main(
                ["create", "--output", str(output_file), "--template", "image_classification"]
            )

        assert exit_code == 0
        mock_create.assert_called_once()


class TestCreateValidateRoundtrip:
    """A freshly created template must pass validation."""

    def test_created_basic_template_validates(self, tmp_path):
        output = tmp_path / "pipeline.toml"
        assert main(["create", "--output", str(output), "--template", "basic"]) == 0
        assert main(["validate", "--config-path", str(output)]) == 0

    def test_created_image_template_validates(self, tmp_path):
        output = tmp_path / "pipeline.toml"
        assert main(["create", "--output", str(output), "--template", "image_classification"]) == 0
        assert main(["validate", "--config-path", str(output)]) == 0


class TestRemovedCommands:
    """The config-driven run/profile commands are gone, not stubbed."""

    def test_run_command_removed(self):
        """'datarax run' is rejected by the parser (exit code 2)."""
        with pytest.raises(SystemExit) as excinfo:
            main(["run", "--config-path", "pipeline.toml"])
        assert excinfo.value.code == 2

    def test_profile_command_removed(self):
        """'datarax profile' is rejected by the parser (exit code 2)."""
        with pytest.raises(SystemExit) as excinfo:
            main(["profile", "--config-path", "pipeline.toml"])
        assert excinfo.value.code == 2


class TestCLIIntegration:
    """Integration tests for CLI with actual Datarax components."""

    def test_validate_complex_config(self, tmp_path):
        """Test validation of a complex pipeline configuration."""
        config_file = tmp_path / "complex.toml"
        config_file.write_text("""
[pipeline]
name = "complex_pipeline"
version = "2.0.0"

[dag]
batch_size = 32
prefetch_size = 2

[[dag.nodes]]
name = "train_source"
type = "source"
source_type = "tfds"
dataset_name = "mnist"
split = "train"

[[dag.nodes]]
name = "augment"
type = "augment"
augment_type = "brightness"
max_delta = 0.2
inputs = ["train_source"]

[[dag.nodes]]
name = "batch"
type = "batch"
batch_size = 32
drop_remainder = true
inputs = ["augment"]

[checkpoint]
enabled = true
directory = "./checkpoints"
save_interval = 1000
        """)

        with patch("datarax.cli.main.is_config_valid") as mock_validate:
            mock_validate.return_value = True
            exit_code = main(["validate", "--config-path", str(config_file)])

        assert exit_code == 0

    def test_running_the_cli_changes_no_global_jax_configuration(self, tmp_path, monkeypatch):
        """The CLI leaves backend selection to ``JAX_PLATFORMS``, which jax reads when it starts.

        ``DATARAX_DEVICE`` set the deprecated ``jax_platform_name`` option after jax was imported,
        which cannot choose the backends jax starts; setting it must change nothing.
        """
        config_file = tmp_path / "env_test.toml"
        config_file.write_text("""
[pipeline]
name = "env_test"

[sources.train]
type = "memory"
        """)
        monkeypatch.setenv("DATARAX_DEVICE", "cpu")

        with restored_jax_config() as changed:
            exit_code = main(["validate", "--config-path", str(config_file)])

        assert exit_code == 0
        assert changed == []
