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

# Or run the suite on the GPU directly
DATARAX_TEST_JAX_PLATFORMS=cuda uv run pytest tests/ -v
```

### Full Test Suite

By default, `uv run pytest` runs only the core test suite in `tests/`:

```bash
uv run pytest                    # core tests only (tests/)
```

To run **all** test suites — including benchmark application tests — use the `--all-suites` flag:

```bash
uv run pytest --all-suites       # tests/ + benchmarks/tests/
```

You can also run individual suites directly:

```bash
uv run pytest tests/               # core test suite
uv run pytest benchmarks/tests/    # benchmark adapter & runner tests
```

## Test Directory Structure

Datarax has two independent test suites. The core suite (`tests/`) mirrors the `src/datarax` package structure. The second suite tests the benchmark application layer.

```text
tests/                           # Core test suite (default)
├── augment/                     #   Augmentation functionality
├── batching/                    #   Batch processing
├── benchmarks/                  #   Performance-focused tests
├── checkpoint/                  #   Checkpoint functionality
├── cli/                         #   CLI tools
├── config/                      #   Configuration handling
├── control/                     #   Control flow
├── core/                        #   Core functionality
├── data/                        #   Test data and fixtures
├── distributed/                 #   Distributed processing
├── examples/                    #   Example validation tests
├── fixtures/                    #   Shared pytest fixtures
├── integration/                 #   End-to-end integration tests
├── memory/                      #   Memory management
├── monitoring/                  #   Monitoring functionality
├── operators/                   #   Pipeline operators
├── performance/                 #   Performance tests
├── pipeline/                    #   Pipeline / DAG execution
├── samplers/                    #   Sampling functionality
├── scripts/                     #   Test helper scripts
├── sharding/                    #   Data sharding
├── sources/                     #   Data sources
├── test_common/                 #   Common testing utilities
├── transforms/                  #   Data transformations (neural network ops)
├── utils/                       #   Utility functions
└── conftest.py                  #   Pytest configuration and custom markers

benchmarks/tests/                # Benchmark application suite (--all-suites)
├── test_adapters/               #   Per-framework adapter tests (16 adapters, incl. Datarax-scan)
├── test_analysis/               #   Comparison reports, gap detection, stability
├── test_automation/             #   SkyPilot / Vast orchestration templates
├── test_runners/                #   CI and full-runner tests
├── test_scenarios/              #   Scenario suites (vision, NLP, tabular, ...)
├── test_visualization/          #   Chart generation
├── test_baselines.py            #   Baseline store
├── test_cli.py                  #   Benchmark CLI
├── test_config_loader.py        #   TOML config loading
├── test_integration.py          #   Runner + adapter integration
├── test_synthetic_data.py       #   Synthetic data generation
├── ...                          #   Additional determinism, profiling, sharding tests
└── conftest.py                  #   Adapter-specific fixtures
```

The `--all-suites` flag (defined in the root `conftest.py`) collects both suites in a single pytest run.

## Test Categories

Tests are organized using pytest markers defined in `conftest.py`:

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.unit` | Basic unit tests | Default for most tests |
| `@pytest.mark.integration` | Component interaction tests | `test_*_integration.py` files |
| `@pytest.mark.end_to_end` | Complete workflow tests | `integration/` directory |
| `@pytest.mark.benchmark` | Performance measurement | `benchmarks/` directory |
| `@pytest.mark.accelerator(kind="gpu")` | Requires a GPU backend | Skips unless `DATARAX_TEST_JAX_PLATFORMS=cuda` selects one |
| `@pytest.mark.devices(count)` | Requires `count` devices | Skips below `count` visible devices |
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

# Run the tests that need a GPU backend
DATARAX_TEST_JAX_PLATFORMS=cuda uv run pytest -m accelerator

# Run HuggingFace integration tests
uv run pytest -m hf
```

## Adding New Tests

When adding new tests:

1. Place tests in the directory corresponding to the module they test
2. Name test files according to the specific component they test (`test_component_name.py`)
3. Follow the naming convention `test_*` for all test functions
4. Create one test file per source component when possible
5. Declare hardware requirements with the `accelerator` and `devices` markers
6. Create standalone test units that don't depend on other test files

## Test Dependencies

Test dependencies can be installed using:

```bash
# Using uv sync (recommended)
uv sync --extra test

# For complete development setup including tests
uv sync --extra all
```

## Pytest Configuration

The `tests/conftest.py` file provides:

- **Custom markers** for test categorization
- **Fixtures** for common test data and setup
- **Command-line options** for test categories
- **The test backend**, chosen through `DATARAX_TEST_JAX_PLATFORMS` before JAX is imported

### Key Command-Line Options

| Option | Values | Description |
|--------|--------|-------------|
| `--all-suites` | flag | Collect all test suites: `tests/`, `benchmarks/tests/` |
| `--integration` | flag | Include integration tests |
| `--end-to-end` | flag | Include end-to-end tests |
| `--benchmark` | flag | Include benchmark tests |
| `--no-integration` | flag | Exclude integration tests |
| `--no-end-to-end` | flag | Exclude end-to-end tests |

## Related Documentation

- [Developer Guide - Testing Section](dev_guide.md#testing)
- [Testing Guide](testing_guide.md) - Detailed testing practices
- [GPU Testing Guide](gpu_testing.md) - GPU-specific testing
