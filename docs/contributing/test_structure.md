# Datarax Testing Guide

## Running Tests

Datarax tests can be run using different configurations depending on your environment and needs.

### CPU-Only Testing (Recommended for Development)

For stable testing without GPU/TPU dependencies:

```bash
# Use the project script (from project root)
./run_tests.sh

# Or run specific tests with CPU-only JAX
JAX_PLATFORMS=cpu uv run pytest tests/sources/test_memory_source_module.py -v

# Run all tests on CPU
JAX_PLATFORMS=cpu uv run pytest tests/ -v
```

### GPU Testing

For GPU-accelerated testing (requires CUDA setup):

```bash
# Use the GPU test script
bash scripts/run_gpu_tests.sh

# Or manually with device selection
JAX_PLATFORMS=cuda uv run pytest --device=gpu tests/ -v
```

### Full Test Suite

To run the complete test suite:

```bash
uv run pytest
```

## Test Directory Structure

The test directory structure mirrors the `src/datarax` package structure for easier navigation and maintenance:

```text
tests/
├── augment/         # Tests for augmentation functionality
├── batching/        # Tests for batch processing
├── benchmarking/    # Benchmarking infrastructure tests
├── benchmarks/      # Performance benchmarks
├── checkpoint/      # Tests for checkpoint functionality
├── cli/             # Tests for CLI tools
├── config/          # Tests for configuration handling
├── control/         # Tests for control flow
├── core/            # Tests for core functionality
├── dag/             # Tests for DAG execution
├── data/            # Test data and fixtures
├── distributed/     # Tests for distributed processing
├── examples/        # Tests for example code validation
├── integration/     # End-to-end integration tests
├── memory/          # Tests for memory management
├── monitoring/      # Tests for monitoring functionality
├── operators/       # Tests for pipeline operators
├── performance/     # Performance testing
├── samplers/        # Tests for sampling functionality
├── scripts/         # Utility scripts for test management
├── sharding/        # Tests for sharding functionality
├── sources/         # Tests for data sources
├── test_common/     # Common testing utilities
├── transforms/      # Tests for data transformations (neural network ops)
├── utils/           # Tests for utility functions
├── conftest.py      # Pytest configuration and custom markers
└── README.md        # Test directory overview
```

## Test Categories

Tests are organized using pytest markers defined in `conftest.py`:

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.unit` | Basic unit tests | Default for most tests |
| `@pytest.mark.integration` | Component interaction tests | `test_*_integration.py` files |
| `@pytest.mark.end_to_end` | Complete workflow tests | `integration/` directory |
| `@pytest.mark.benchmark` | Performance measurement | `benchmarks/` directory |
| `@pytest.mark.gpu` | Requires GPU hardware | Use `--device=gpu` to run |
| `@pytest.mark.tpu` | Requires TPU hardware | Currently skipped for stability |
| `@pytest.mark.tfds` | Requires TensorFlow Datasets | TFDS integration tests |
| `@pytest.mark.hf` | Requires HuggingFace Datasets | HF integration tests |

### Running Specific Test Types

```bash
# Run only unit tests
uv run pytest -m unit

# Run integration tests
uv run pytest -m integration

# Skip slow tests
uv run pytest -m "not slow"

# Run GPU tests only
uv run pytest -m gpu --device=gpu

# Run HuggingFace integration tests
uv run pytest -m hf
```

!!! note "TPU Tests"
    TPU tests are currently skipped unconditionally for stability reasons.
    They can be enabled by modifying `conftest.py` if TPU hardware is available.

## Adding New Tests

When adding new tests:

1. Place tests in the directory corresponding to the module they test
2. Name test files according to the specific component they test (`test_component_name.py`)
3. Follow the naming convention `test_*` for all test functions
4. Create one test file per source component when possible
5. Use appropriate markers for hardware requirements (`gpu`, `tpu`, etc.)
6. Create standalone test units that don't depend on other test files

## Test Dependencies

Test dependencies can be installed using:

```bash
# Using uv sync (recommended)
uv sync --extra test

# Or with pip-style installation
uv pip install -e ".[test]"

# For complete development setup including tests
uv sync --extra all
```

## Pytest Configuration

The `tests/conftest.py` file provides:

- **Custom markers** for test categorization
- **Fixtures** for common test data and setup
- **Command-line options** like `--device` for hardware selection
- **Automatic test skipping** based on available hardware

### Key Command-Line Options

| Option | Values | Description |
|--------|--------|-------------|
| `--device` | `cpu`, `gpu`, `tpu`, `all` | Select device type for tests (default: `all`) |
| `--integration` | flag | Include integration tests |
| `--end-to-end` | flag | Include end-to-end tests |
| `--benchmark` | flag | Include benchmark tests |
| `--no-integration` | flag | Exclude integration tests |
| `--no-end-to-end` | flag | Exclude end-to-end tests |

## Related Documentation

- [Developer Guide - Testing Section](dev_guide.md#testing)
- [Testing Guide](testing_guide.md) - Detailed testing practices
- [GPU Testing Guide](gpu_testing.md) - GPU-specific testing
