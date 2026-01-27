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

```
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
└── README.md        # This file
```

## Test Categories

Tests are organized using pytest markers defined in `conftest.py`:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Basic unit tests (default) |
| `@pytest.mark.integration` | Component interaction tests |
| `@pytest.mark.end_to_end` | Complete workflow tests |
| `@pytest.mark.benchmark` | Performance benchmarks |
| `@pytest.mark.gpu` | Requires GPU hardware |
| `@pytest.mark.tpu` | Requires TPU hardware (currently skipped) |
| `@pytest.mark.tfds` | Requires TensorFlow Datasets |
| `@pytest.mark.hf` | Requires HuggingFace Datasets |

### Running Specific Test Types

```bash
# Run only GPU tests
uv run pytest -m gpu --device=gpu

# Run integration tests
uv run pytest -m integration

# Run HuggingFace tests
uv run pytest -m hf
```

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

# For complete development setup
uv sync --extra all
```

## Related Documentation

For more information, see:

- `docs/contributing/test_structure.md` - Detailed test structure guide
- `docs/contributing/testing_guide.md` - Testing practices
- `docs/contributing/gpu_testing.md` - GPU-specific testing
