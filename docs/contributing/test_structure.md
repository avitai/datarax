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

By default, `uv run pytest` runs only the core test suite in `tests/`:

```bash
uv run pytest                    # core tests only (tests/)
```

To run **all** test suites — including benchmark application tests and benchkit library tests — use the `--all-suites` flag:

```bash
uv run pytest --all-suites       # tests/ + benchmarks/tests/ + tools/benchkit/tests/
```

You can also run individual suites directly:

```bash
uv run pytest benchmarks/tests/           # benchmark adapter & runner tests
uv run pytest tools/benchkit/tests/       # benchkit analysis library tests
```

## Test Directory Structure

Datarax has three independent test suites. The core suite (`tests/`) mirrors the `src/datarax` package structure. The other two suites test the benchmark application layer and the benchkit analysis library respectively.

```text
tests/                           # Core test suite (default)
├── augment/                     #   Augmentation functionality
├── batching/                    #   Batch processing
├── benchmarking/                #   Benchmarking engine (src/datarax/benchmarking/)
├── benchmarks/                  #   Performance benchmarks
├── checkpoint/                  #   Checkpoint functionality
├── cli/                         #   CLI tools
├── config/                      #   Configuration handling
├── control/                     #   Control flow
├── core/                        #   Core functionality
├── dag/                         #   DAG execution
├── distributed/                 #   Distributed processing
├── integration/                 #   End-to-end integration tests
├── monitoring/                  #   Monitoring functionality
├── operators/                   #   Pipeline operators
├── samplers/                    #   Sampling functionality
├── sources/                     #   Data sources
├── test_common/                 #   Common testing utilities
├── transforms/                  #   Data transformations (neural network ops)
├── utils/                       #   Utility functions
└── conftest.py                  #   Pytest configuration and custom markers

benchmarks/tests/                # Benchmark application suite (--all-suites)
├── test_adapters/               #   Per-framework adapter tests (15 adapters)
├── test_integration.py          #   Runner + adapter integration
├── test_synthetic_data.py       #   Synthetic data generation
├── test_config_loader.py        #   TOML config loading
├── test_baselines.py            #   Baseline store
└── conftest.py                  #   Adapter-specific fixtures

tools/benchkit/tests/            # benchkit library suite (--all-suites)
├── test_models.py               #   Data model serde round-trip
├── test_store.py                #   JSON store and baselines
├── test_analysis.py             #   Regression detection, CIs, ranking
├── test_wandb_exporter.py       #   W&B export (WANDB_MODE=offline)
└── test_cli.py                  #   CLI commands via CliRunner
```

The `--all-suites` flag (defined in the root `conftest.py`) collects all three suites in a single pytest run.

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
| `--all-suites` | flag | Collect all test suites: `tests/`, `benchmarks/tests/`, `tools/benchkit/tests/` |
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
