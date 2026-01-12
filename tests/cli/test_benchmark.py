"""Tests for Datarax CLI benchmark commands.

Following TDD principles - tests define expected behavior.
Target: 20% minimum coverage of cli/benchmark.py (68+ lines).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datarax.cli.benchmark import (
    main,
    run_batch_size_benchmark,
    run_pipeline_benchmark,
    run_profile,
    save_benchmark_results,
)


class TestSaveBenchmarkResults:
    """Test the save_benchmark_results utility function.

    This function handles result serialization and file I/O.
    Target: Lines 17-53 (~37 lines coverage).
    """

    def test_save_valid_dict_creates_file(self, tmp_path: Path) -> None:
        """Test saving a valid dictionary creates a JSON file."""
        output_path = tmp_path / "results.json"
        results = {"metric": "throughput", "value": 123.45}

        save_benchmark_results(results, str(output_path))

        # Verify file exists
        assert output_path.exists()

        # Verify content is valid JSON
        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["metric"] == "throughput"
        assert loaded["value"] == 123.45
        assert "timestamp" in loaded  # Should add timestamp

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
        # Timestamp should be string in format YYYY-MM-DD HH:MM:SS
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

        # Should be wrapped
        assert "results" in loaded
        assert loaded["results"] == [1, 2, 3]
        assert "timestamp" in loaded


class TestCLIArgumentParsing:
    """Test CLI argument parsing and command dispatch.

    Target: Lines 244-337 (~94 lines) - we only need ~30 for 20% total.
    Focus on main entry point and subcommand structure.
    """

    def test_pipeline_command_requires_module_and_setup(self) -> None:
        """Test that 'pipeline' subcommand requires --module-path and --setup-function."""
        with patch("sys.argv", ["datarax-benchmark", "pipeline"]):
            with pytest.raises(SystemExit):
                main()

    def test_profile_command_requires_module_and_setup(self) -> None:
        """Test that 'profile' subcommand requires --module-path and --setup-function."""
        with patch("sys.argv", ["datarax-benchmark", "profile"]):
            with pytest.raises(SystemExit):
                main()

    def test_batch_size_command_requires_module_and_batch_sizes(self) -> None:
        """Test 'batch-size' subcommand requires --module-path, --setup-function, --batch-sizes."""
        with patch("sys.argv", ["datarax-benchmark", "batch-size", "-m", "test.py", "-f", "setup"]):
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

        # Should have called run_pipeline_benchmark once
        assert mock_run.call_count == 1

        # Check arguments passed
        args = mock_run.call_args[0][0]
        assert args.module_path == "test_module.py"
        assert args.setup_function == "create_pipeline"
        assert args.num_batches == 100

    @patch("datarax.cli.benchmark.run_profile")
    def test_profile_command_dispatches_correctly(self, mock_run: MagicMock) -> None:
        """Test that 'profile' command dispatches to run_profile."""
        test_args = [
            "datarax-benchmark",
            "profile",
            "--module-path",
            "test_module.py",
            "--setup-function",
            "create_pipeline",
        ]

        with patch("sys.argv", test_args):
            main()

        assert mock_run.call_count == 1
        args = mock_run.call_args[0][0]
        assert args.module_path == "test_module.py"
        assert args.setup_function == "create_pipeline"

    @patch("datarax.cli.benchmark.run_batch_size_benchmark")
    def test_batch_size_command_dispatches_correctly(self, mock_run: MagicMock) -> None:
        """Test that 'batch-size' command dispatches to run_batch_size_benchmark."""
        test_args = [
            "datarax-benchmark",
            "batch-size",
            "--module-path",
            "test_module.py",
            "--setup-function",
            "create_pipeline",
            "--batch-sizes",
            "8,16,32",
        ]

        with patch("sys.argv", test_args):
            main()

        assert mock_run.call_count == 1
        args = mock_run.call_args[0][0]
        assert args.module_path == "test_module.py"
        assert args.setup_function == "create_pipeline"
        assert args.batch_sizes == "8,16,32"

    def test_no_command_shows_help(self) -> None:
        """Test that running without a subcommand shows help and exits."""
        with patch("sys.argv", ["datarax-benchmark"]):
            with pytest.raises(SystemExit):
                main()


class TestRunPipelineBenchmark:
    """Test the run_pipeline_benchmark function.

    Target: Lines 56-119 (~64 lines) to reach 60% total coverage.
    """

    def test_run_pipeline_benchmark_with_valid_module(self, tmp_path: Path) -> None:
        """Test running pipeline benchmark with a valid module."""
        # Create a test module
        module_path = tmp_path / "test_module.py"
        module_path.write_text("""
def create_pipeline():
    # Return a mock data stream
    class MockStream:
        def __iter__(self):
            for i in range(10):
                yield {"data": i}
    return MockStream()
""")

        # Create mock args
        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "create_pipeline"
        args.num_batches = 5
        args.warmup_batches = 1
        args.seed = 42
        args.output = None  # Test default output path

        # Mock the benchmark classes
        with patch("datarax.cli.benchmark.PipelineBenchmark") as MockBenchmark:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {"throughput": 100.0}
            MockBenchmark.return_value = mock_instance

            # Run the function
            run_pipeline_benchmark(args)

            # Verify benchmark was created and run
            assert MockBenchmark.call_count == 1
            assert mock_instance.run.call_count == 1
            assert mock_instance.print_results.call_count == 1

    def test_run_pipeline_benchmark_module_not_found(self, tmp_path: Path) -> None:
        """Test error handling when module file doesn't exist."""
        args = MagicMock()
        args.module_path = str(tmp_path / "nonexistent.py")
        args.setup_function = "create_pipeline"

        # importlib raises FileNotFoundError, not SystemExit
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

        with patch("datarax.cli.benchmark.PipelineBenchmark") as MockBenchmark:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {"latency": 50.0}
            MockBenchmark.return_value = mock_instance

            run_pipeline_benchmark(args)

            # Verify output file was created
            assert output_path.exists()


class TestRunProfile:
    """Test the run_profile function.

    Target: Lines 121-176 (~56 lines) to reach 70% total coverage.
    """

    def test_run_profile_with_valid_module(self, tmp_path: Path) -> None:
        """Test running profile with a valid module."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("""
def create_pipeline():
    class MockStream:
        def __iter__(self):
            for i in range(5):
                yield {"data": i}
    return MockStream()
""")

        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "create_pipeline"
        args.num_batches = 3
        args.seed = 42
        args.output = None

        with patch("datarax.cli.benchmark.ProfileReport") as MockProfile:
            mock_instance = MagicMock()
            mock_instance.metrics = {"mean_time": 0.1, "std_time": 0.01}
            MockProfile.return_value = mock_instance

            run_profile(args)

            assert MockProfile.call_count == 1
            assert mock_instance.run.call_count == 1
            assert mock_instance.print_report.call_count == 1

    def test_run_profile_module_not_found(self, tmp_path: Path) -> None:
        """Test error handling when module doesn't exist."""
        args = MagicMock()
        args.module_path = str(tmp_path / "missing.py")
        args.setup_function = "setup"

        # importlib raises FileNotFoundError, not SystemExit
        with pytest.raises((SystemExit, FileNotFoundError)):
            run_profile(args)

    def test_run_profile_function_not_found(self, tmp_path: Path) -> None:
        """Test error handling when function doesn't exist."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("def wrong_name(): pass")

        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "correct_name"

        with pytest.raises(SystemExit):
            run_profile(args)


class TestRunBatchSizeBenchmark:
    """Test the run_batch_size_benchmark function.

    Target: Lines 178-242 (~65 lines) to reach 80%+ total coverage.
    """

    def test_run_batch_size_benchmark_with_valid_module(self, tmp_path: Path) -> None:
        """Test running batch size benchmark with valid module."""
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
        args.batch_sizes = "8,16,32"
        args.num_batches = 5
        args.warmup_batches = 1
        args.seed = 42
        args.output = None

        with patch("datarax.cli.benchmark.BatchSizeBenchmark") as MockBenchmark:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {
                8: {"throughput": 80.0},
                16: {"throughput": 150.0},
                32: {"throughput": 280.0},
            }
            MockBenchmark.return_value = mock_instance

            run_batch_size_benchmark(args)

            # Verify benchmark was created with parsed batch sizes
            assert MockBenchmark.call_count == 1
            call_kwargs = MockBenchmark.call_args[1]
            assert call_kwargs["batch_sizes"] == [8, 16, 32]
            assert mock_instance.run.call_count == 1
            assert mock_instance.print_results.call_count == 1

    def test_run_batch_size_benchmark_module_not_found(self, tmp_path: Path) -> None:
        """Test error handling when module doesn't exist."""
        args = MagicMock()
        args.module_path = str(tmp_path / "nonexistent.py")
        args.setup_function = "setup"
        args.batch_sizes = "8,16"

        # importlib raises FileNotFoundError, not SystemExit
        with pytest.raises((SystemExit, FileNotFoundError)):
            run_batch_size_benchmark(args)

    def test_run_batch_size_benchmark_function_not_found(self, tmp_path: Path) -> None:
        """Test error handling when function doesn't exist."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("def other(): pass")

        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "missing"
        args.batch_sizes = "16,32"

        with pytest.raises(SystemExit):
            run_batch_size_benchmark(args)

    def test_run_batch_size_benchmark_custom_output(self, tmp_path: Path) -> None:
        """Test batch size benchmark with custom output path."""
        module_path = tmp_path / "test_module.py"
        module_path.write_text("""
def factory():
    class MockStream:
        def __iter__(self):
            yield {"data": 1}
    return MockStream()
""")

        output_path = tmp_path / "batch_results.json"
        args = MagicMock()
        args.module_path = str(module_path)
        args.setup_function = "factory"
        args.batch_sizes = "4,8"
        args.num_batches = 2
        args.warmup_batches = 0
        args.seed = 99
        args.output = str(output_path)

        with patch("datarax.cli.benchmark.BatchSizeBenchmark") as MockBenchmark:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {4: {"throughput": 40.0}, 8: {"throughput": 75.0}}
            MockBenchmark.return_value = mock_instance

            run_batch_size_benchmark(args)

            assert output_path.exists()

            # Verify saved content
            with open(output_path) as f:
                data = json.load(f)

            assert data["type"] == "batch_size_benchmark"
            assert data["config"]["batch_sizes"] == [4, 8]
