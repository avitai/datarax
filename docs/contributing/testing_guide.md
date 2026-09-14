# Datarax Testing Guide

This guide explains how to run tests in the Datarax codebase, with special attention to CPU and GPU testing configurations.

## Test Setup Overview

Datarax has a flexible testing setup that:

1. Runs all tests on CPU by default (both locally and in CI)
2. Automatically runs tests on GPU as well if a GPU is available locally
3. Never attempts GPU tests in GitHub workflows

## Running Tests

### Quick Start

The simplest way to run tests is with the `run_tests.sh` script:

```bash
# Run core tests (tests/ only)
uv run pytest

# Run ALL test suites (tests/ + benchmarks/tests/)
uv run pytest --all-suites

# Run all tests, automatically using both CPU and GPU if available
./run_tests.sh

# Run only CPU tests regardless of GPU availability
./run_tests.sh --device=cpu

# Run only GPU tests (will fail if no GPU is available)
./run_tests.sh --device=gpu

# Run specific test categories
./run_tests.sh --integration  # Run integration tests
./run_tests.sh --end-to-end   # Run end-to-end tests
./run_tests.sh --benchmark    # Run benchmark tests
./run_tests.sh --all          # Run all test categories
```

### Unified GPU/CPU Test Runner

The main test runner at `scripts/run_tests.sh` (also available via `./run_tests.sh` at the root) handles everything:

```bash
./run_tests.sh                    # Auto-detect GPU, run on CPU (and GPU if available)
./run_tests.sh --device=cpu       # Force CPU only
./run_tests.sh --device=gpu       # Force GPU only
```

This script:
1. Checks if `uv` is installed
2. Safely detects GPU availability
3. Runs all tests on CPU first
4. If a GPU is available, also runs on GPU
5. Tracks exit codes properly across both runs

### GPU-Specific Testing

To run only GPU-specific tests with CUDA configuration:

```bash
./scripts/run_gpu_tests.sh
```

This script:
1. Checks if a GPU is available
2. Sets up the proper environment variables for GPU testing
3. Runs the GPU test suite via pytest

## Test Configuration

Tests are configured with several pytest markers and command-line options:

### Test Suite Selection

- `--all-suites`: Collect all test suites (`tests/`, `benchmarks/tests/`). Without this flag, only `tests/` is collected (configured in `pyproject.toml` via `testpaths`).

### Test Backend

- Tests run on eight emulated CPU devices by default
- `DATARAX_TEST_JAX_PLATFORMS=cuda`: run the tests on the GPU
- `DATARAX_TEST_DEVICE_COUNT=N`: emulate `N` CPU devices

### Test Categories

- `--integration`: Run integration tests
- `--end-to-end`: Run end-to-end tests
- `--benchmark`: Run benchmark tests
- `--no-integration`: Skip integration tests
- `--no-end-to-end`: Skip end-to-end tests

### Test Markers

- `@pytest.mark.accelerator(kind="gpu")`: Test needs a GPU backend (substrax plugin)
- `@pytest.mark.devices(count)`: Test needs at least `count` devices (substrax plugin)
- `@pytest.mark.integration`: Integration test
- `@pytest.mark.end_to_end`: End-to-end test
- `@pytest.mark.benchmark`: Performance benchmark test
- `@pytest.mark.tfds`: Test requires TensorFlow Datasets
- `@pytest.mark.hf`: Test requires HuggingFace Datasets

## CI/GitHub Workflow Testing

All GitHub workflow tests run exclusively on CPU regardless of the availability of GPU instances. This ensures consistent test results and avoids issues with GPU availability in CI environments.

The following workflows are configured to run tests on CPU:

- `ci.yml`: Main CI workflow, including unit, integration, end-to-end,
  performance, and combined coverage jobs

## Writing Device-Specific Tests

Declare what a test needs, and the substrax pytest plugin skips it when the run cannot provide it:

```python
import pytest

# Runs on whatever backend the run selected
def test_basic_functionality():
    ...

# Runs only when the run selected a GPU backend
@pytest.mark.accelerator(kind="gpu")
def test_gpu_specific_functionality():
    ...

# Runs only with at least two visible devices
@pytest.mark.devices(2)
def test_multi_device_sharding():
    ...
```

## Troubleshooting

### GPU Tests Failing

If GPU tests are failing but CPU tests pass:

1. Check if your GPU is properly detected: `uv run python scripts/check_gpu.py`
2. Ensure you have the correct JAX CUDA version installed
3. Try setting `XLA_CLIENT_MEM_FRACTION=0.5` to limit memory usage
4. Check for CUDA version mismatches between JAX and your system
5. Regenerate the backend configuration in `.datarax.env` (run `./setup.sh --backend cuda12` to configure the CUDA 12 backend)

### Test Selection Issues

If tests aren't being selected correctly:

1. Check that you're using the correct markers
2. Ensure pytest is correctly interpreting command-line arguments
3. Try running with `-v` for verbose output to see which tests are selected
