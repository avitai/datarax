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

For more control, you can use the `scripts/run_tests_with_gpu_auto.sh` script directly:

```bash
./scripts/run_tests_with_gpu_auto.sh [pytest_args]
```

This script:
1. Automatically detects if a GPU is available
2. Runs all tests on CPU first
3. If a GPU is available, runs all tests on GPU second
4. Passes any additional arguments to pytest

### GPU-Specific Testing

To run only GPU-specific tests:

```bash
./scripts/run_gpu_tests.sh
```

This script:
1. Checks if a GPU is available
2. Sets up the proper environment variables for GPU testing
3. Runs the GPU test suite defined in `run_gpu_tests.py`

## Test Configuration

Tests are configured with several pytest markers and command-line options:

### Device Selection

- `--device=cpu`: Run tests only on CPU
- `--device=gpu`: Run tests only on GPU
- `--device=all`: Run tests on all available devices (default)

### Test Categories

- `--integration`: Run integration tests
- `--end-to-end`: Run end-to-end tests
- `--benchmark`: Run benchmark tests
- `--no-integration`: Skip integration tests
- `--no-end-to-end`: Skip end-to-end tests

### Test Markers

- `@pytest.mark.gpu`: Test requires a GPU
- `@pytest.mark.integration`: Integration test
- `@pytest.mark.e2e`: End-to-end test
- `@pytest.mark.benchmark`: Performance benchmark test

## CI/GitHub Workflow Testing

All GitHub workflow tests run exclusively on CPU regardless of the availability of GPU instances. This ensures consistent test results and avoids issues with GPU availability in CI environments.

The following workflows are configured to run tests on CPU:

- `ci.yml`: Main CI workflow
- `test-coverage.yml`: Test coverage reporting

## Writing Device-Specific Tests

To write tests that run on specific devices:

```python
import pytest

# Test runs on all devices
def test_basic_functionality():
    ...

# Test only runs when GPU is available and selected
@pytest.mark.gpu
def test_gpu_specific_functionality():
    ...
```

For tests that should behave differently on different devices, use the device detection utilities:

```python
from tests.test_common.device_detection import has_gpu

def test_device_specific_behavior():
    if has_gpu():
        # GPU-specific test code
        ...
    else:
        # CPU-specific test code
        ...
```

## Troubleshooting

### GPU Tests Failing

If GPU tests are failing but CPU tests pass:

1. Check if your GPU is properly detected: `python check_gpu.py`
2. Ensure you have the correct JAX CUDA version installed
3. Try setting `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to limit memory usage
4. Check for CUDA version mismatches between JAX and your system

### Test Selection Issues

If tests aren't being selected correctly:

1. Check that you're using the correct markers
2. Ensure pytest is correctly interpreting command-line arguments
3. Try running with `-v` for verbose output to see which tests are selected
