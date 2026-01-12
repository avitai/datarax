# Datarax Testing Guide

## Running Tests

Datarax tests can be run using different configurations depending on your environment and needs.

### CPU-Only Testing

For stable testing without GPU/TPU dependencies:

```bash
# Use our custom script
./run_tests.sh

# Or run specific tests using custom environment variables
JAX_PLATFORMS=cpu uv run pytest tests/sources/test_memory_source.py -v
```

### Full Test Suite

To run the full test suite (requires appropriate GPU/TPU setup):

```bash
uv run pytest
```

## Test Directory Structure

The test directory structure mirrors the `src/datarax` package structure for easier navigation and maintenance:

```
tests/
├── augment/         # Tests for augmentation functionality
├── batching/        # Tests for batch processing
├── benchmarking/    # Benchmarking infrastructure
├── benchmarks/      # Performance benchmarks
├── checkpoint/      # Tests for checkpoint functionality
├── cli/             # Tests for CLI tools
├── config/          # Tests for configuration handling
├── control/         # Tests for control flow
├── core/            # Tests for core functionality
├── dag/             # Tests for DAG execution
├── data/            # Test data and fixtures
├── distributed/     # Tests for distributed processing
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
├── transforms/      # Tests for data transformations
├── utils/           # Tests for utility functions
├── conftest.py      # Pytest configuration
└── README.md        # This file
```

## Test Categories

Tests are organized into the following categories:

- **Unit tests**: Basic functionality tests for individual components
- **Integration tests**: Tests for component interactions (named `test_*_integration.py`)
- **GPU tests**: Tests marked with `@pytest.mark.gpu` requiring GPU hardware
- **TPU tests**: Tests marked with `@pytest.mark.tpu` requiring TPU hardware
- **End-to-end tests**: Complete workflow tests (in `integration/` directory)
- **Benchmarks**: Performance measurement tests (in `benchmarks/` directory)

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
uv sync --extra test
# OR
pip install -e ".[test]"
```


For more information, refer to our [testing documentation](https://datarax.readthedocs.io/en/latest/testing/).
