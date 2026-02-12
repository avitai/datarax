"""Tests for Datarax CLI benchmark commands.

Following TDD principles - tests define expected behavior.
Updated to match refactored cli/benchmark.py (TimingCollector-based).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datarax.cli.benchmark import (
    main,
    run_pipeline_benchmark,
    save_benchmark_results,
)


class TestSaveBenchmarkResults:
    """Test the save_benchmark_results utility function."""

    def test_save_valid_dict_creates_file(self, tmp_path: Path) -> None:
        """Test saving a valid dictionary creates a JSON file."""
        output_path = tmp_path / "results.json"
        results = {"metric": "throughput", "value": 123.45}

        save_benchmark_results(results, str(output_path))

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["metric"] == "throughput"
        assert loaded["value"] == 123.45
        assert "timestamp" in loaded

    def test_save_creates_nested_directories(self, tmp_path: Path) -> None:
        """Test that nested directories are created if they don't exist."""
        output_path = tmp_path / "nested" / "dir" / "results.json"
        results = {"test": "data"}

        save_benchmark_results(results, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_adds_timestamp(self, tmp_path: Path) -> None:
        """Test that a timestamp field is automatically added."""
        output_path = tmp_path / "results.json"
        results = {"metric": "latency"}

        save_benchmark_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)

        assert "timestamp" in loaded
        timestamp = loaded["timestamp"]
        assert isinstance(timestamp, str)
        assert len(timestamp) == 19  # "2025-10-28 12:34:56"
        assert timestamp[4] == "-"
        assert timestamp[10] == " "

    def test_save_handles_non_serializable_objects(self, tmp_path: Path) -> None:
        """Test that non-JSON-serializable objects are converted to strings."""
        output_path = tmp_path / "results.json"

        class CustomObject:
            def __str__(self) -> str:
                return "custom_repr"

        results = {
            "normal": 42,
            "custom": CustomObject(),
            "nested": {"obj": CustomObject()},
        }

        save_benchmark_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["normal"] == 42
        assert loaded["custom"] == "custom_repr"
        assert loaded["nested"]["obj"] == "custom_repr"

    def test_save_handles_lists_and_tuples(self, tmp_path: Path) -> None:
        """Test serialization of lists and tuples."""
        output_path = tmp_path / "results.json"
        results = {
            "list": [1, 2, 3],
            "tuple": (4, 5, 6),
            "nested": [[1, 2], [3, 4]],
        }

        save_benchmark_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["list"] == [1, 2, 3]
        assert loaded["tuple"] == [4, 5, 6]  # Tuples become lists in JSON
        assert loaded["nested"] == [[1, 2], [3, 4]]

    def test_save_wraps_non_dict_results(self, tmp_path: Path) -> None:
        """Test that non-dict results are wrapped in a 'results' key."""
        output_path = tmp_path / "results.json"
        results = [1, 2, 3]  # Not a dict

        save_benchmark_results(results, str(output_path))

        with open(output_path) as f:
            loaded = json.load(f)

        assert "results" in loaded
        assert loaded["results"] == [1, 2, 3]
        assert "timestamp" in loaded


class TestCLIArgumentParsing:
    """Test CLI argument parsing and command dispatch."""

    def test_pipeline_command_requires_module_and_setup(self) -> None:
        """Test that 'pipeline' subcommand requires --module-path and --setup-function."""
        with patch("sys.argv", ["datarax-benchmark", "pipeline"]):
            with pytest.raises(SystemExit):
                main()

    @patch("datarax.cli.benchmark.run_pipeline_benchmark")
    def test_pipeline_command_dispatches_correctly(self, mock_run: MagicMock) -> None:
        """Test that 'pipeline' command dispatches to run_pipeline_benchmark."""
        test_args = [
            "datarax-benchmark",
            "pipeline",
            "--module-path",
            "test_module.py",
            "--setup-function",
            "create_pipeline",
            "--num-batches",
            "100",
        ]

        with patch("sys.argv", test_args):
            main()

        assert mock_run.call_count == 1

        args = mock_run.call_args[0][0]
        assert args.module_path == "test_module.py"
        assert args.setup_function == "create_pipeline"
        assert args.num_batches == 100

    def test_no_command_shows_help(self) -> None:
        """Test that running without a subcommand shows help and exits."""
        with patch("sys.argv", ["datarax-benchmark"]):
            with pytest.raises(SystemExit):
                main()


class TestRunPipelineBenchmark:
    """Test the run_pipeline_benchmark function."""

    def test_run_pipeline_benchmark_with_valid_module(self, tmp_path: Path) -> None:
        """Test running pipeline benchmark with a valid module."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("""
def create_pipeline():
    class MockStream:
        def __iter__(self):
            for i in range(10):
                yield {"data": i}
    return MockStream()
""")

        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "create_pipeline"
        args.num_batches = 5
        args.warmup_batches = 1
        args.seed = 42
        args.output = None

        with patch("datarax.cli.benchmark.TimingCollector") as MockCollector:
            mock_sample = MagicMock()
            mock_sample.wall_clock_sec = 1.0
            mock_sample.num_batches = 5
            mock_sample.num_elements = 5
            mock_sample.first_batch_time = 0.2
            mock_collector = MagicMock()
            mock_collector.measure_iteration.return_value = mock_sample
            MockCollector.return_value = mock_collector

            run_pipeline_benchmark(args)

            assert MockCollector.call_count == 1
            assert mock_collector.measure_iteration.call_count == 1

    def test_run_pipeline_benchmark_module_not_found(self, tmp_path: Path) -> None:
        """Test error handling when module file doesn't exist."""
        args = MagicMock()
        args.module_path = str(tmp_path / "nonexistent.py")
        args.setup_function = "create_pipeline"

        with pytest.raises((SystemExit, FileNotFoundError)):
            run_pipeline_benchmark(args)

    def test_run_pipeline_benchmark_function_not_found(self, tmp_path: Path) -> None:
        """Test error handling when setup function doesn't exist in module."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("def other_function(): pass")

        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "nonexistent_function"

        with pytest.raises(SystemExit):
            run_pipeline_benchmark(args)

    def test_run_pipeline_benchmark_custom_output_path(self, tmp_path: Path) -> None:
        """Test running with custom output path."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("""
def create_pipeline():
    class MockStream:
        def __iter__(self):
            yield {"data": 1}
    return MockStream()
""")

        output_path = tmp_path / "custom_results.json"
        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "create_pipeline"
        args.num_batches = 2
        args.warmup_batches = 0
        args.seed = 123
        args.output = str(output_path)

        with patch("datarax.cli.benchmark.TimingCollector") as MockCollector:
            mock_sample = MagicMock()
            mock_sample.wall_clock_sec = 0.5
            mock_sample.num_batches = 2
            mock_sample.num_elements = 2
            mock_sample.first_batch_time = 0.25
            mock_collector = MagicMock()
            mock_collector.measure_iteration.return_value = mock_sample
            MockCollector.return_value = mock_collector

            run_pipeline_benchmark(args)

            assert output_path.exists()
