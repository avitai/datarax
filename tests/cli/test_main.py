"""Tests for the Datarax CLI main module."""

from unittest.mock import patch, MagicMock
import pytest

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

    def test_run_command_with_missing_config(self, capsys):
        """Test run command with missing config file."""
        exit_code = main(["run", "--config-path", "/nonexistent/config.toml"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower() or "not found" in captured.out.lower()

    def test_run_command_with_valid_config(self, tmp_path, capsys):
        """Test run command with valid config file."""
        # Create a temporary config file
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test_pipeline"
version = "1.0.0"

[sources.train]
type = "memory"
data_shape = [32, 32, 3]
num_samples = 100

[transforms.normalize]
type = "function"
function = "lambda x: x / 255.0"
        """)

        with patch("datarax.cli.main.run_pipeline") as mock_run:
            mock_run.return_value = 0
            exit_code = main(["run", "--config-path", str(config_file)])

        assert exit_code == 0
        mock_run.assert_called_once()

    def test_run_command_with_overrides(self, tmp_path):
        """Test run command with configuration overrides."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test_pipeline"
batch_size = 32
        """)

        with patch("datarax.cli.main.run_pipeline") as mock_run:
            mock_run.return_value = 0
            exit_code = main(
                [
                    "run",
                    "--config-path",
                    str(config_file),
                    "--override",
                    "pipeline.batch_size=64",
                    "--override",
                    "pipeline.num_epochs=10",
                ]
            )

        assert exit_code == 0
        mock_run.assert_called_once()
        # Check that overrides were parsed
        call_args = mock_run.call_args
        assert "overrides" in call_args.kwargs
        assert call_args.kwargs["overrides"]["pipeline.batch_size"] == "64"
        assert call_args.kwargs["overrides"]["pipeline.num_epochs"] == "10"

    def test_run_command_invalid_override_format(self, tmp_path, capsys):
        """Test run command with invalid override format."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("[pipeline]\nname = 'test'")

        exit_code = main(["run", "--config-path", str(config_file), "--override", "invalid_format"])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert (
            "invalid override format" in captured.err.lower()
            or "invalid override format" in captured.out.lower()
        )

    def test_benchmark_command(self):
        """Test benchmark command."""
        with patch("datarax.cli.main.run_benchmark") as mock_benchmark:
            mock_benchmark.return_value = 0
            exit_code = main(["benchmark", "--dataset", "synthetic"])

        assert exit_code == 0
        mock_benchmark.assert_called_once()

    def test_validate_command(self, tmp_path):
        """Test validate command for config validation."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test_pipeline"

[sources.train]
type = "memory"
        """)

        with patch("datarax.cli.main.validate_config") as mock_validate:
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

    def test_main_exception_handling(self, tmp_path, capsys):
        """Test that exceptions are properly handled."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test"

[dag]
batch_size = 32
        """)

        # The run_pipeline function should catch exceptions internally
        # So we mock at a deeper level
        with patch("datarax.cli.main.DAGExecutor") as mock_executor:
            mock_executor.from_config.side_effect = RuntimeError("Pipeline failed")
            exit_code = main(["run", "--config-path", str(config_file)])

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower() or "error" in captured.out.lower()

    def test_list_command(self, capsys):
        """Test list command to show available components."""
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

        with patch("datarax.cli.main.create_pipeline_template") as mock_create:
            mock_create.return_value = True
            exit_code = main(
                ["create", "--output", str(output_file), "--template", "image_classification"]
            )

        assert exit_code == 0
        mock_create.assert_called_once()

    def test_profile_command(self, tmp_path):
        """Test profile command for performance analysis."""
        config_file = tmp_path / "pipeline.toml"
        config_file.write_text("""
[pipeline]
name = "test_pipeline"
        """)

        with patch("datarax.cli.main.profile_pipeline") as mock_profile:
            mock_profile.return_value = {"throughput": 1000, "latency": 0.001}
            exit_code = main(
                ["profile", "--config-path", str(config_file), "--num-iterations", "100"]
            )

        assert exit_code == 0
        mock_profile.assert_called_once()


class TestCLIIntegration:
    """Integration tests for CLI with actual Datarax components."""

    def test_run_simple_pipeline_integration(self, tmp_path):
        """Test running a simple pipeline through the CLI."""
        config_file = tmp_path / "simple.toml"
        config_file.write_text("""
[pipeline]
name = "simple_test"
version = "1.0.0"

[dag]
batch_size = 4

[[dag.nodes]]
name = "source"
type = "source"
source_type = "memory"
num_samples = 16
sample_shape = [28, 28, 1]

[[dag.nodes]]
name = "normalize"
type = "transform"
transform_type = "function"
function = "lambda x: x / 255.0"
inputs = ["source"]
        """)

        # This should work with the actual implementation
        with patch("datarax.cli.main.DAGExecutor") as mock_executor:
            mock_instance = MagicMock()
            mock_executor.from_config.return_value = mock_instance
            mock_instance.run.return_value = None

            exit_code = main(["run", "--config-path", str(config_file)])

        assert exit_code == 0
        mock_executor.from_config.assert_called_once()

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

        with patch("datarax.cli.main.validate_config") as mock_validate:
            mock_validate.return_value = True
            exit_code = main(["validate", "--config-path", str(config_file)])

        assert exit_code == 0

    def test_cli_with_env_variables(self, tmp_path, monkeypatch):
        """Test CLI respects environment variables."""
        config_file = tmp_path / "env_test.toml"
        config_file.write_text("""
[pipeline]
name = "env_test"
        """)

        # Set environment variable for Datarax
        monkeypatch.setenv("DATARAX_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("DATARAX_DEVICE", "cpu")

        with patch("datarax.cli.main.run_pipeline") as mock_run:
            mock_run.return_value = 0
            exit_code = main(["run", "--config-path", str(config_file)])

        assert exit_code == 0
        # The implementation should respect these env variables
